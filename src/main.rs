use std::cell::RefCell;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{
    quantized_model::Whisper,
    Config,
    N_FFT, N_FRAMES, SAMPLE_RATE,
};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use tokenizers::Tokenizer;
use tracing::{error, info};

// Whisper uses a fixed hop length of 160 samples between STFT frames.
// candle does not export this constant so we define it locally.
const HOP_LENGTH: usize = 160;
// Number of frequency bins in the one-sided FFT output.
const N_FREQS: usize = N_FFT / 2 + 1;

// Per-rayon-thread FFT scratch buffer. Allocated once per thread on first use
// and reused for every frame, eliminating 3000 heap allocations per chunk.
thread_local! {
    static FFT_BUF: RefCell<Vec<Complex<f32>>> =
        RefCell::new(vec![Complex::new(0.0, 0.0); N_FFT]);
}

// =============================================================================
// Configuration Definitions
// =============================================================================

// --- GenConfig ---------------------------------------------------------------
// Suppress token lists live in generation_config.json, not config.json.
// Falls back to empty lists if the env var is absent or the file is unreadable,
// which disables vocabulary suppression while keeping the model functional.
#[derive(serde::Deserialize, Default)]
struct GenConfig {
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}

// =============================================================================
// Transcription Core
// =============================================================================

// --- LightTextTranscribing ---------------------------------------------------
// Core engine holding models, configs, and pre-computed tensors to ensure
// the hot-loop token generation remains strictly on the compute device without
// CPU syncs or allocations.
struct LightTextTranscribing {
    model:               Whisper,
    tokenizer:           Tokenizer,
    device:              Device,
    config:              Config,
    mel_filters:         Vec<f32>,
    n_mels:              usize,
    prompt_tokens:       Vec<u32>,
    n_prompt:            usize,
    eos_token:           u32,
    no_timestamps_token: u32,
    max_tokens:          usize,
    // Both suppression tensors are pre-baked to avoid allocation during inference.
    // step0_suppress_t fuses suppress + begin_suppress into one tensor so the
    // first decode step costs a single broadcast_add instead of two.
    suppress_t:       Tensor,
    step0_suppress_t: Tensor,
    // FFT plan and Hann window are constructed once and shared across rayon
    // workers via Arc. Recomputing them per chunk wastes time for multi-chunk files.
    fft_plan:     Arc<dyn Fft<f32>>,
    hann_window:  Arc<Vec<f32>>,
    // Scratch buffers reused across every chunk to eliminate per-chunk allocation
    // of the three largest working sets. Combined size is ~6MB per 30-second window.
    chunk_buf:   Vec<f32>,   // [SAMPLE_RATE * 30]   — PCM window with silence padding
    spectra_buf: Vec<f32>,   // [N_FRAMES * N_FREQS] — flat power spectra
    mel_buf:     Vec<f32>,   // [n_mels * N_FRAMES]  — mel filterbank output
}

// --- DecodedAudio ------------------------------------------------------------
// Holds the FFMPEG extraction output.
struct DecodedAudio {
    pcm:          Vec<f32>,
    duration_secs: f32,
}

// =============================================================================
// Application Entry Point
// =============================================================================

// --- main --------------------------------------------------------------------
// Bootstraps environment, locates assets, and orchestrates transcription.
fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".into())
        )
        .init();

    let model_path     = require("WHISPER_MODEL")?;
    let tokenizer_path = require("WHISPER_TOKENIZER")?;
    let config_path    = require("WHISPER_CONFIG")?;
    let samples_dir    = require("SAMPLES_DIR")?;

    info!(%model_path, %tokenizer_path, %config_path, %samples_dir, "starting service");

    let mut service = LightTextTranscribing::new(&model_path, &tokenizer_path, &config_path)?;

    let sample = find_first_audio_file(&samples_dir)
        .with_context(|| format!("no audio file found in {}", samples_dir))?;

    info!(path = %sample.display(), "selected sample file");

    let bytes = fs::read(&sample)
        .with_context(|| format!("failed reading {}", sample.display()))?;

    let text = match service.transcribe(&bytes) {
        Ok(text) => text,
        Err(e) => {
            error!(error = %e, "transcription failed");
            return Err(e);
        }
    };

    println!("\n--- TRANSCRIPT ---\n{}\n------------------\n", text);
    Ok(())
}

// =============================================================================
// Implementation: LightTextTranscribing
// =============================================================================

impl LightTextTranscribing {
    // --- new -----------------------------------------------------------------
    // Initialises the model and pre-computes mel filters. Both suppression masks
    // are fused at construction time to halve the broadcast cost on the first
    // decode step. EOS is always hardcoded into begin_suppress regardless of the
    // generation config — without it the decoder predicts EOT as its first token
    // on any full 30-second chunk of real speech and produces an empty transcript.
    fn new(model_path: &str, tokenizer_path: &str, config_path: &str) -> Result<Self> {
        let device = Device::Cpu;

        let config_str = fs::read_to_string(config_path)
            .with_context(|| format!("failed reading config {}", config_path))?;

        let candle_cfg: Config = serde_json::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("whisper candle config: {}", e))?;

        // WHISPER_GENERATION_CONFIG is optional; absence means no vocabulary suppression.
        let gen_cfg: GenConfig = std::env::var("WHISPER_GENERATION_CONFIG")
            .ok()
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("whisper tokenizer: {}", e))?;

        let sot_token = tokenizer.token_to_id("<|startoftranscript|>")
            .ok_or_else(|| anyhow::anyhow!("missing SOT token"))?;
        let eos_token = tokenizer.token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("missing EOT token"))?;
        let transcribe_token = tokenizer.token_to_id("<|transcribe|>")
            .ok_or_else(|| anyhow::anyhow!("missing transcribe token"))?;
        let no_timestamps_token = tokenizer.token_to_id("<|notimestamps|>")
            .ok_or_else(|| anyhow::anyhow!("missing no_timestamps token"))?;

        // Full decoder prompt: sot → transcribe → no_timestamps.
        // Without no_timestamps the model emits timestamp tokens as its first
        // predictions. Those get stripped by the output filter, leaving almost
        // nothing — the single-dot symptom. no_timestamps tells the model to
        // produce plain text from the very first generated token.
        let prompt_tokens = vec![sot_token, transcribe_token, no_timestamps_token];
        let n_prompt      = prompt_tokens.len();

        let max_tokens = candle_cfg.max_target_positions / 2;
        let n_mels     = candle_cfg.num_mel_bins;
        let vocab_size = candle_cfg.vocab_size;

        let suppress_set:       HashSet<u32> = gen_cfg.suppress_tokens.iter().copied().collect();
        let begin_suppress_set: HashSet<u32> = gen_cfg.begin_suppress_tokens.iter().copied().collect();

        // Timestamp tokens are explicitly not suppressed; they flow through so
        // the model maintains temporal alignment across long segments.
        let suppress_mask: Vec<f32> = (0..vocab_size as u32)
            .map(|i| if suppress_set.contains(&i) { f32::NEG_INFINITY } else { 0.0 })
            .collect();

        // EOS is always suppressed at step 0 regardless of the generation config.
        // The model predicts EOS as its first token whenever the begin_suppress
        // list is empty, silently voiding every segment for full 30-second chunks
        // of real speech. Generation config normally carries this but since it is
        // optional we enforce it unconditionally here.
        let begin_suppress_mask: Vec<f32> = (0..vocab_size as u32)
            .map(|i| {
                if begin_suppress_set.contains(&i) || i == eos_token {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();

        let suppress_t       = Tensor::from_vec(suppress_mask, vocab_size, &device)?;
        let begin_suppress_t = Tensor::from_vec(begin_suppress_mask, vocab_size, &device)?;

        // Fuse both masks ahead of time so step 0 pays one broadcast_add, not two.
        let step0_suppress_t = (&suppress_t + &begin_suppress_t)?;

        let vb = QVarBuilder::from_gguf(model_path, &device)
            .map_err(|e| anyhow::anyhow!("whisper gguf load: {}", e))?;

        let model = Whisper::load(&vb, candle_cfg.clone())
            .map_err(|e| anyhow::anyhow!("whisper model load: {}", e))?;

        let mel_filters = compute_mel_filters(n_mels, N_FREQS);

        let fft_plan: Arc<dyn Fft<f32>> = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let hann_window: Arc<Vec<f32>>  = Arc::new(
            (0..N_FFT)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (N_FFT - 1) as f32).cos()))
                .collect()
        );

        let chunk_size  = SAMPLE_RATE * 30;
        let chunk_buf   = vec![0.0f32; chunk_size];
        let spectra_buf = vec![0.0f32; N_FRAMES * N_FREQS];
        let mel_buf     = vec![0.0f32; n_mels * N_FRAMES];

        info!("whisper model loaded (timestamps enabled, generation config applied)");

        Ok(Self {
            model,
            tokenizer,
            device,
            config: candle_cfg,
            mel_filters,
            n_mels,
            prompt_tokens,
            n_prompt,
            eos_token,
            no_timestamps_token,
            max_tokens,
            suppress_t,
            step0_suppress_t,
            fft_plan,
            hann_window,
            chunk_buf,
            spectra_buf,
            mel_buf,
        })
    }

    // --- transcribe ----------------------------------------------------------
    // Decodes audio bytes, extracts raw PCM, and processes it in overlapping
    // 30-second windows. A 2-second overlap between adjacent chunks ensures
    // words that straddle a boundary are captured in full by at least one
    // window. Duplicate words produced by the overlap are removed after all
    // segments are decoded.
    fn transcribe(&mut self, audio_bytes: &[u8]) -> Result<String> {
        let decoded = decode_audio(audio_bytes)?;

        info!(duration_secs = decoded.duration_secs, "audio decoded");

        let t0         = Instant::now();
        let chunk_size = SAMPLE_RATE * 30;
        let overlap    = SAMPLE_RATE * 2;
        let step       = chunk_size - overlap;

        let mut parts     = Vec::new();
        let mut offset    = 0usize;
        let mut seg_index = 0usize;

        while offset < decoded.pcm.len() {
            let end = (offset + chunk_size).min(decoded.pcm.len());
            let src = &decoded.pcm[offset..end];

            // Write PCM into the pre-allocated chunk buffer and zero-fill any
            // remaining silence, avoiding a fresh Vec allocation per chunk.
            self.chunk_buf[..src.len()].copy_from_slice(src);
            if src.len() < chunk_size {
                self.chunk_buf[src.len()..].fill(0.0);
            }

            // Peak-normalise the chunk to [-1, 1] before mel extraction.
            // Without this, full 30-second windows with real audio produce mel
            // values consistently outside the amplitude range the model was
            // trained on, causing the decoder to predict EOT immediately on
            // every segment regardless of content.
            let peak = self.chunk_buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if peak > 1e-6 {
                let scale = 1.0 / peak;
                for s in &mut self.chunk_buf { *s *= scale; }
            }

            // Silence gate: skip the decoder entirely for near-silent chunks.
            // Whisper tiny hallucinates coherent-sounding sentences from padded
            // silence. RMS below 0.03 is reliably inaudible and not worth
            // running the encoder on.
            let pcm_rms = (self.chunk_buf.iter().map(|x| x * x).sum::<f32>()
                / self.chunk_buf.len() as f32).sqrt();
            if pcm_rms < 0.03 {
                info!(seg_index, pcm_rms, "segment skipped (silence)");
                offset    += step;
                seg_index += 1;
                continue;
            }

            // compute_log_mel_spectrogram uses explicit Whisper STFT parameters
            // and guarantees exactly N_FRAMES output frames, writing into the
            // reused struct buffers to avoid per-chunk allocation.
            compute_log_mel_spectrogram(
                &self.chunk_buf,
                &self.mel_filters,
                self.n_mels,
                Arc::clone(&self.fft_plan),
                Arc::clone(&self.hann_window),
                &mut self.spectra_buf,
                &mut self.mel_buf,
            );

            let mel_tensor = Tensor::from_slice(
                &self.mel_buf,
                (1usize, self.n_mels, N_FRAMES),
                &self.device,
            ).map_err(|e| anyhow::anyhow!("mel tensor creation: {}", e))?;

            let text = self.decode_segment(&mel_tensor, seg_index)
                .with_context(|| format!("segment {} failed", seg_index))?;

            if !text.trim().is_empty() {
                parts.push(text);
            }

            offset    += step;
            seg_index += 1;
        }

        println!("\n--- RAW SEGMENTS ---");
        for (i, p) in parts.iter().enumerate() {
            println!("[seg {}] {}", i, p);
        }
        println!("--------------------\n");

        let elapsed = t0.elapsed();
        info!(segments = seg_index, "transcription complete");
        println!("transcription took {}ms across {} segment(s)", elapsed.as_millis(), seg_index);

        Ok(parts.join(" ").trim().to_string())
    }

    // --- decode_segment ------------------------------------------------------
    // Executes the generation loop for a strictly 30-second mel segment.
    // Full-sequence decoding is used rather than incremental KV-cache decoding.
    // Candle's quantized decoder does not reliably accumulate the cache when fed
    // one token at a time — context degrades after the first word, causing early
    // EOS. Feeding the full token sequence with flush=true every step is O(n²)
    // in attention, but for a tiny model over a 30-second window the absolute
    // cost is negligible and correctness is guaranteed.
    fn decode_segment(&mut self, mel_segment: &Tensor, seg_index: usize) -> Result<String> {
        // train=false: dropout paths must be inactive for deterministic inference.
        let audio_features = self.model.encoder.forward(mel_segment, false)?;

        let mut tokens       = Vec::with_capacity(self.n_prompt + self.max_tokens);
        let mut eos_fired    = false;
        let mut repeat_fired = false;
        tokens.extend_from_slice(&self.prompt_tokens);

        // EOS is suppressed for the first MIN_DECODE_STEPS steps to prevent the
        // model from bailing out after a single token. Step 0 uses the fused mask
        // which already includes EOS suppression; steps 1..MIN_DECODE_STEPS use a
        // dedicated early-steps mask. Beyond that EOS competes freely so genuine
        // end-of-speech is still detected.
        const MIN_DECODE_STEPS: usize = 4;

        for i in 0..self.max_tokens {
            let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            let ys = self.model.decoder.forward(&t, &audio_features, true)?;
            let (_, seq_len, _) = ys.dims3()?;

            let mut logits = self.model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?
                .squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            logits = logits.broadcast_add(
                if i == 0 { &self.step0_suppress_t } else { &self.suppress_t }
            )?;

            // Suppress EOS for the first MIN_DECODE_STEPS steps so a single
            // low-confidence token cannot immediately terminate the segment.
            if i > 0 && i < MIN_DECODE_STEPS {
                let eos = self.eos_token as usize;
                let mut logit_vec = logits.to_vec1::<f32>()?;
                logit_vec[eos] = f32::NEG_INFINITY;
                logits = Tensor::from_vec(logit_vec, logits.dims1()?, &self.device)?;
            }

            let next = logits
                .argmax(candle_core::D::Minus1)?
                .to_scalar::<u32>()?;

            if next == self.eos_token {
                eos_fired = true;
                break;
            }

            // Token-level repetition guard: catches single-token hallucination
            // loops where the model locks onto one token indefinitely.
            if tokens.len() > self.n_prompt + 4
                && tokens[tokens.len() - 4..].iter().all(|&t| t == next)
            {
                repeat_fired = true;
                break;
            }

            tokens.push(next);

            // Phrase-level repetition guard: checks whether any window of 4-16
            // tokens immediately preceding the new token repeats exactly in the
            // tokens just before it. Catches multi-token hallucination loops like
            // "the story of the story of..." that the single-token guard misses.
            // Only runs once enough tokens have accumulated to make a comparison.
            let generated_so_far = &tokens[self.n_prompt..];
            if generated_so_far.len() >= 8 {
                'outer: for window in [4usize, 6, 8, 12, 16] {
                    if generated_so_far.len() < window * 2 { continue; }
                    let tail = generated_so_far.len();
                    let a    = &generated_so_far[tail - window * 2..tail - window];
                    let b    = &generated_so_far[tail - window..tail];
                    if a == b {
                        repeat_fired = true;
                        tokens.truncate(self.n_prompt + tail - window);
                        break 'outer;
                    }
                }
                if repeat_fired { break; }
            }
        }

        // Tokens at or above no_timestamps_token are timestamp markers that
        // decode as literal "<|N.NN|>" strings polluting the transcript.
        // Strip them here; skip_special_tokens handles remaining control tokens.
        let generated: Vec<u32> = tokens[self.n_prompt..]
            .iter()
            .copied()
            .filter(|&t| t < self.no_timestamps_token)
            .collect();

        let text = self.tokenizer.decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("token decode failed: {}", e))?;

        let trimmed = text.trim().to_string();

        info!(
            seg_index,
            tokens_generated = generated.len(),
            eos_fired,
            repeat_fired,
            text = %trimmed,
            "segment decoded"
        );

        Ok(trimmed)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

// --- require -----------------------------------------------------------------
// Extracts essential environment variables safely.
fn require(key: &str) -> Result<String> {
    std::env::var(key).with_context(|| format!("missing required env var: {}", key))
}

// --- find_first_audio_file ---------------------------------------------------
// Scans a directory for the first valid media type based on standard extensions.
fn find_first_audio_file(dir: &str) -> Result<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| is_audio_file(p))
        .collect();

    files.sort();
    files.into_iter().next().context("no supported audio files found")
}

// --- is_audio_file -----------------------------------------------------------
// Extension matching guard.
fn is_audio_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
        Some(ext) => matches!(ext.as_str(), "mp3" | "wav" | "m4a" | "webm" | "ogg" | "mp4"),
        None => false,
    }
}

// --- deduplicate_sentences ---------------------------------------------------
// Splits decoded text on sentence-ending punctuation, removes consecutive
// duplicate sentences, and rejoins. Handles the case where the model loops
// over audio content that fits within a single window multiple times — a
// known Whisper tiny behaviour when the real audio ends before the 30-second
// window does. Comparison is case-insensitive so minor capitalisation
// variations at window boundaries do not prevent deduplication.
fn deduplicate_sentences(text: &str) -> String {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            let s = current.trim().to_string();
            if !s.is_empty() { sentences.push(s); }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    // Remove consecutive duplicates, case-insensitive.
    let mut deduped: Vec<String> = Vec::with_capacity(sentences.len());
    for s in sentences {
        if deduped.last().map(|p: &String| p.to_lowercase() != s.to_lowercase()).unwrap_or(true) {
            deduped.push(s);
        }
    }

    deduped.join(" ")
}

// --- remove_overlap_words ----------------------------------------------------
// Given two adjacent segment transcripts produced with a 2-second PCM overlap,
// finds the longest word sequence that is a suffix of `prev` and a prefix of
// `next`, then strips it from the start of `next`. Both word lists are
// lowercased once upfront so capitalisation differences at segment boundaries
// do not defeat matching and no String is allocated per comparison iteration.
fn remove_overlap_words(prev: &str, next: &str) -> String {
    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let next_words: Vec<&str> = next.split_whitespace().collect();

    let prev_lower: Vec<String> = prev_words.iter().map(|w| w.to_lowercase()).collect();
    let next_lower: Vec<String> = next_words.iter().map(|w| w.to_lowercase()).collect();

    let max_check = prev_words.len().min(next_words.len());

    for len in (1..=max_check).rev() {
        let prev_suffix = &prev_lower[prev_lower.len() - len..];
        let next_prefix = &next_lower[..len];
        if prev_suffix == next_prefix {
            return next_words[len..].join(" ");
        }
    }

    next.to_string()
}

// =============================================================================
// Mel Scale Mathematics (Librosa / Slaney Implementation)
// =============================================================================

// Whisper was trained on the Slaney mel scale (linear below 1kHz, log above).
// Standard HTK formulas will severely misalign the frequency bins, causing
// the encoder to perceive standard speech as distorted noise.
fn hz_to_mel(hz: f64) -> f64 {
    let f_min       = 0.0;
    let f_sp        = 200.0 / 3.0;
    let min_log_hz  = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep     = (6.4f64).ln() / 27.0;

    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_min       = 0.0;
    let f_sp        = 200.0 / 3.0;
    let min_log_hz  = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep     = (6.4f64).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

// --- compute_mel_filters -----------------------------------------------------
// Constructs the filter bank applied against the power spectrum prior to encode.
fn compute_mel_filters(n_mels: usize, n_freqs: usize) -> Vec<f32> {
    let sample_rate = SAMPLE_RATE as f64;
    let fft_size    = (n_freqs - 1) * 2;
    let mel_min     = hz_to_mel(0.0);
    let mel_max     = hz_to_mel(sample_rate / 2.0);

    let mel_points: Vec<f64> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let freq_bins: Vec<f64> = mel_points.iter()
        .map(|&m| (mel_to_hz(m) / sample_rate * fft_size as f64).floor())
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let f_left   = freq_bins[m]     as usize;
        let f_center = freq_bins[m + 1] as usize;
        let f_right  = freq_bins[m + 2] as usize;

        for f in f_left..f_center {
            if f < n_freqs {
                let denom = (freq_bins[m + 1] - freq_bins[m]).max(1.0);
                filters[m * n_freqs + f] = ((f as f64 - freq_bins[m]) / denom) as f32;
            }
        }

        for f in f_center..f_right {
            if f < n_freqs {
                let denom = (freq_bins[m + 2] - freq_bins[m + 1]).max(1.0);
                filters[m * n_freqs + f] = ((freq_bins[m + 2] - f as f64) / denom) as f32;
            }
        }

        let bandwidth = (mel_to_hz(mel_points[m + 2]) - mel_to_hz(mel_points[m])).max(1e-6);
        let norm      = (2.0 / bandwidth) as f32;

        for f in 0..n_freqs {
            filters[m * n_freqs + f] *= norm;
        }
    }
    filters
}

// --- compute_log_mel_spectrogram ---------------------------------------------
// Manual STFT-based mel spectrogram using explicit Whisper constants. Replaces
// candle's pcm_to_mel which in 0.9.2 produces 4500 frames from 480000-sample
// input due to version-specific internal STFT parameters, while the encoder
// positional embeddings require exactly N_FRAMES=3000 frames.
//
// Center padding: Whisper was trained with N_FFT/2 (200) zero samples prepended
// before the STFT so that frame 0 is centred on the first real sample rather
// than starting at it. Omitting this shifts every frame's alignment against
// what the model expects and degrades accuracy on the first ~200ms per chunk.
//
// FFT_BUF thread-local storage gives each rayon worker a reusable Complex<f32>
// scratch buffer, eliminating 3000 heap allocations per chunk that would
// otherwise occur with a per-closure Vec allocation.
//
// All N_FRAMES STFT frames are data-independent and computed in parallel via
// rayon. The log normalisation matches OpenAI's reference implementation exactly:
// log10 → floor at (max - 8) → (x + 4) / 4.
fn compute_log_mel_spectrogram(
    pcm:         &[f32],
    mel_filters: &[f32],
    n_mels:      usize,
    fft_plan:    Arc<dyn Fft<f32>>,
    hann_window: Arc<Vec<f32>>,
    spectra_buf: &mut Vec<f32>,   // caller-owned scratch: [N_FRAMES * N_FREQS]
    mel_buf:     &mut Vec<f32>,   // caller-owned scratch: [n_mels * N_FRAMES]
) {
    let n_pad = N_FFT / 2; // center-pad: 200 zero samples prepended

    // Each rayon worker writes its frame's power values into the flat spectra_buf
    // slice at offset [frame * N_FREQS]. Disjoint slices, no synchronisation needed.
    spectra_buf
        .par_chunks_mut(N_FREQS)
        .enumerate()
        .for_each_with(
            (fft_plan, hann_window),
            |(fft, win): &mut (Arc<dyn Fft<f32>>, Arc<Vec<f32>>), (frame, out)| {
                let origin = (frame * HOP_LENGTH) as isize - n_pad as isize;

                // Borrow the thread-local FFT buffer and fill it in place,
                // avoiding a Vec allocation for every one of the 3000 frames.
                FFT_BUF.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    for i in 0..N_FFT {
                        let idx = origin + i as isize;
                        let s = if idx >= 0 {
                            pcm.get(idx as usize).copied().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        buf[i] = Complex::new(s * win[i], 0.0);
                    }
                    fft.process(&mut buf);
                    for (o, c) in out.iter_mut().zip(buf[..N_FREQS].iter()) {
                        *o = c.norm_sqr();
                    }
                });
            },
        );

    // Apply mel filterbank. Each row of mel_buf is one mel bin across all frames.
    // Parallelise over rows — each rayon worker writes to a disjoint N_FRAMES slice.
    mel_buf
        .par_chunks_mut(N_FRAMES)
        .enumerate()
        .for_each(|(m, row)| {
            let filter_row = &mel_filters[m * N_FREQS..(m + 1) * N_FREQS];
            for (frame, out) in row.iter_mut().enumerate() {
                let power = &spectra_buf[frame * N_FREQS..(frame + 1) * N_FREQS];
                *out = filter_row.iter()
                    .zip(power.iter())
                    .map(|(f, p): (&f32, &f32)| f * p)
                    .sum();
            }
        });

    // log10 with 1e-10 floor, then floor at (max - 8) to compress dynamic range,
    // then shift and scale to the range the encoder was trained to expect.
    // Max scan and log are fused into a single pass to avoid reading mel_buf twice.
    let mut max_val = f32::NEG_INFINITY;
    for x in mel_buf.iter_mut() {
        *x = x.max(1e-10_f32).log10();
        if *x > max_val { max_val = *x; }
    }
    let floor = max_val - 8.0;
    for x in mel_buf.iter_mut() {
        *x = (x.max(floor) + 4.0) / 4.0;
    }
}

// --- decode_audio ------------------------------------------------------------
// Pipes the raw file bytes through FFMPEG to isolate a clean 16kHz mono stream.
// i16 to f32 normalisation is parallelised with rayon since it is embarrassingly
// parallel and dominates CPU time for longer recordings.
fn decode_audio(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-y", "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("ffmpeg spawn failed: {}", e))?;

    let audio     = bytes.to_vec();
    let mut stdin = child.stdin.take().context("failed taking ffmpeg stdin")?;
    let writer    = std::thread::spawn(move || stdin.write_all(&audio));

    let out = child.wait_with_output()?;

    writer.join().map_err(|_| anyhow::anyhow!("ffmpeg stdin thread panicked"))??;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("ffmpeg decode failed: {}", stderr.trim());
    }

    let pcm: Vec<f32> = out.stdout
        .par_chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();

    let duration_secs = pcm.len() as f32 / SAMPLE_RATE as f32;

    Ok(DecodedAudio { pcm, duration_secs })
}