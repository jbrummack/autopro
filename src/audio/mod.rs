//! Audio to tensors: waveforms and (log-)mel spectrograms.
pub mod mel;

use std::f64::consts::PI;

use ndarray::Array1;

use crate::error::{Error, Result};
pub use mel::{LogMode, MelNorm, MelScale, MelSpectrogram, WhisperFeatures};

/// Mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Audio {
        Audio { samples, sample_rate }
    }

    /// Decodes a WAV file. Integer samples are scaled to [-1, 1) like
    /// soundfile does (e.g. i16 / 32768); channels are averaged to mono.
    pub fn from_wav_bytes(bytes: &[u8]) -> Result<Audio> {
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| Error::Audio(e.to_string()))?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>(),
            hound::SampleFormat::Int => {
                let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()
            }
        }
        .map_err(|e| Error::Audio(e.to_string()))?;
        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect()
        };
        Ok(Audio { samples, sample_rate: spec.sample_rate })
    }

    pub fn from_wav_file(path: impl AsRef<std::path::Path>) -> Result<Audio> {
        Self::from_wav_bytes(&std::fs::read(path)?)
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Band-limited resampling with a Hann-windowed sinc (32 zero crossings).
    /// High quality, but not bit-compatible with librosa/soxr.
    pub fn resample(&self, rate: u32) -> Audio {
        if rate == self.sample_rate || self.samples.is_empty() {
            return Audio { samples: self.samples.clone(), sample_rate: rate };
        }
        let ratio = rate as f64 / self.sample_rate as f64;
        // Low-pass at the lower Nyquist frequency, slightly inside it.
        let cutoff = ratio.min(1.0) * 0.97;
        let zeros = 32.0;
        let half_width = zeros / cutoff;
        let out_len = (self.samples.len() as f64 * ratio).round() as usize;
        let n = self.samples.len() as i64;
        let samples = (0..out_len)
            .map(|i| {
                let t = i as f64 / ratio;
                let first = (t - half_width).ceil().max(0.0) as i64;
                let last = ((t + half_width).floor() as i64).min(n - 1);
                let mut acc = 0.0;
                for k in first..=last {
                    let x = t - k as f64;
                    let sinc = if x == 0.0 { 1.0 } else { (PI * cutoff * x).sin() / (PI * cutoff * x) };
                    let window = 0.5 + 0.5 * (PI * x / half_width).cos();
                    acc += self.samples[k as usize] as f64 * cutoff * sinc * window;
                }
                acc as f32
            })
            .collect();
        Audio { samples, sample_rate: rate }
    }

    /// Zero-pads or truncates to exactly `len` samples.
    pub fn pad_or_truncate(&self, len: usize) -> Audio {
        let mut samples = self.samples.clone();
        samples.resize(len, 0.0);
        Audio { samples, sample_rate: self.sample_rate }
    }
}

/// Raw waveform input (Wav2Vec2FeatureExtractor-style).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformProcessor {
    pub sample_rate: u32,
    /// Zero mean, unit variance: (x - mean) / sqrt(var + 1e-7).
    pub normalize: bool,
    /// Pad (with `padding_value`) or truncate to this many samples.
    pub length: Option<usize>,
    pub padding_value: f32,
}

impl WaveformProcessor {
    /// (values, attention mask) for one clip; the audio must already have
    /// `sample_rate` (see [`Audio::resample`]).
    pub fn process(&self, audio: &Audio) -> Result<(Array1<f32>, Array1<i64>)> {
        if audio.sample_rate != self.sample_rate {
            return Err(Error::Audio(format!(
                "audio is {} Hz, the processor expects {} Hz; resample first",
                audio.sample_rate, self.sample_rate
            )));
        }
        let mut values = audio.samples.clone();
        if self.normalize && !values.is_empty() {
            let n = values.len() as f64;
            let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let scale = 1.0 / (var + 1e-7).sqrt();
            values.iter_mut().for_each(|v| *v = ((*v as f64 - mean) * scale) as f32);
        }
        let valid = values.len();
        let mut mask = vec![1i64; valid];
        if let Some(len) = self.length {
            values.resize(len, self.padding_value);
            mask.resize(len, 0);
        }
        Ok((Array1::from(values), Array1::from(mask)))
    }
}
