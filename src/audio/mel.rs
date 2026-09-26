//! Spectrograms following transformers' `audio_utils` (numpy reference
//! implementation) and `WhisperFeatureExtractor`.
use ndarray::Array2;
use realfft::RealFftPlanner;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Htk,
    Slaney,
    Kaldi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    None,
    /// Divide each triangle by its width in Hz (constant energy per band).
    Slaney,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogMode {
    None,
    Ln,
    Log10,
    /// 10 * log10(x / reference), floored at `min_value` and limited to `range` dB below the peak.
    Db { reference: f64, min_value: f64, range: Option<f64> },
}

fn hz_to_mel(f: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + f / 700.0).log10(),
        MelScale::Kaldi => 1127.0 * (1.0 + f / 700.0).ln(),
        MelScale::Slaney => {
            let logstep = 27.0 / 6.4f64.ln();
            if f >= 1000.0 { 15.0 + (f / 1000.0).ln() * logstep } else { 3.0 * f / 200.0 }
        }
    }
}

fn mel_to_hz(m: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(m / 2595.0) - 1.0),
        MelScale::Kaldi => 700.0 * ((m / 1127.0).exp() - 1.0),
        MelScale::Slaney => {
            let logstep = 6.4f64.ln() / 27.0;
            if m >= 15.0 { 1000.0 * (logstep * (m - 15.0)).exp() } else { 200.0 * m / 3.0 }
        }
    }
}

/// numpy.linspace: start + i * step, with the last value exactly `stop`.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
    v[n - 1] = stop;
    v
}

/// Mel spectrogram configuration. `compute` returns `[n_mels, frames]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MelSpectrogram {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    /// Window length (<= n_fft); a periodic Hann window.
    pub win_length: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: f64,
    /// Exponent of the magnitude (2.0: power spectrum).
    pub power: f64,
    /// Reflect-pad n_fft/2 on both sides so frames are centered.
    pub center: bool,
    pub mel_scale: MelScale,
    pub norm: MelNorm,
    /// Lower bound before the log.
    pub mel_floor: f64,
    pub log: LogMode,
}

impl MelSpectrogram {
    /// Whisper's settings for `n_mels` (80 or 128).
    pub fn whisper(n_mels: usize) -> MelSpectrogram {
        MelSpectrogram {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels,
            f_min: 0.0,
            f_max: 8000.0,
            power: 2.0,
            center: true,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            mel_floor: 1e-10,
            log: LogMode::Log10,
        }
    }

    /// `[n_fft/2 + 1, n_mels]` triangular filters (transformers' mel_filter_bank).
    pub fn filter_bank(&self) -> Array2<f64> {
        let bins = self.n_fft / 2 + 1;
        let mel_freqs = linspace(hz_to_mel(self.f_min, self.mel_scale), hz_to_mel(self.f_max, self.mel_scale), self.n_mels + 2);
        let filter_freqs: Vec<f64> = mel_freqs.iter().map(|&m| mel_to_hz(m, self.mel_scale)).collect();
        let fft_freqs = linspace(0.0, (self.sample_rate / 2) as f64, bins);
        let diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
        Array2::from_shape_fn((bins, self.n_mels), |(b, m)| {
            let slope = |i: usize| filter_freqs[i] - fft_freqs[b];
            let down = -slope(m) / diff[m];
            let up = slope(m + 2) / diff[m + 1];
            let w = 0f64.max(down.min(up));
            match self.norm {
                MelNorm::Slaney => w * 2.0 / (filter_freqs[m + 2] - filter_freqs[m]),
                MelNorm::None => w,
            }
        })
    }

    /// Log-mel (or mel) spectrogram `[n_mels, frames]` of mono samples.
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let filters = self.filter_bank();
        let power = self.power_spectrogram(samples);
        let (bins, frames) = power.dim();
        let mut mel = Array2::<f64>::zeros((self.n_mels, frames));
        for m in 0..self.n_mels {
            for t in 0..frames {
                let mut acc = 0.0;
                for b in 0..bins {
                    acc += filters[[b, m]] * power[[b, t]];
                }
                mel[[m, t]] = acc.max(self.mel_floor);
            }
        }
        match self.log {
            LogMode::None => {}
            LogMode::Ln => mel.mapv_inplace(f64::ln),
            LogMode::Log10 => mel.mapv_inplace(f64::log10),
            LogMode::Db { reference, min_value, range } => {
                mel.mapv_inplace(|v| 10.0 * v.max(min_value).log10() - 10.0 * reference.max(min_value).log10());
                if let Some(range) = range {
                    let peak = mel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    mel.mapv_inplace(|v| v.max(peak - range));
                }
            }
        }
        mel.mapv(|v| v as f32)
    }

    /// `|STFT|^power`, `[n_fft/2 + 1, frames]`, with numpy's complex64 storage.
    fn power_spectrogram(&self, samples: &[f32]) -> Array2<f64> {
        let mut signal: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        if self.center {
            let pad = self.n_fft / 2;
            let n = signal.len();
            // numpy 'reflect': mirror without repeating the edge sample.
            let left: Vec<f64> = (1..=pad).rev().map(|i| signal[i.min(n - 1)]).collect();
            let right: Vec<f64> = (1..=pad).map(|i| signal[n.saturating_sub(1 + i)]).collect();
            signal = left.into_iter().chain(signal).chain(right).collect();
        }
        // Periodic Hann window of win_length, zero-padded (centered) to n_fft.
        let hann: Vec<f64> = (0..self.win_length)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / self.win_length as f64).cos())
            .collect();
        let offset = (self.n_fft - self.win_length) / 2;
        let frames = if signal.len() >= self.n_fft { 1 + (signal.len() - self.n_fft) / self.hop_length } else { 0 };
        let bins = self.n_fft / 2 + 1;
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(self.n_fft);
        let mut input = fft.make_input_vec();
        let mut spectrum = fft.make_output_vec();
        let mut out = Array2::<f64>::zeros((bins, frames));
        for t in 0..frames {
            let frame = &signal[t * self.hop_length..t * self.hop_length + self.n_fft];
            input.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..self.win_length {
                input[offset + i] = frame[offset + i] * hann[i];
            }
            fft.process(&mut input, &mut spectrum).expect("fft sizes");
            for (b, c) in spectrum.iter().enumerate() {
                let (re, im) = (c.re as f32 as f64, c.im as f32 as f64);
                out[[b, t]] = (re * re + im * im).sqrt().powf(self.power);
            }
        }
        out
    }
}

/// WhisperFeatureExtractor: pad/truncate to `n_samples`, log10 mel, drop the
/// last frame, clamp to 8 below the max, scale to (x + 4) / 4.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperFeatures {
    pub mel: MelSpectrogram,
    /// Samples per input (30 s at 16 kHz = 480000).
    pub n_samples: usize,
}

impl WhisperFeatures {
    pub fn new(n_mels: usize) -> WhisperFeatures {
        WhisperFeatures { mel: MelSpectrogram::whisper(n_mels), n_samples: 480_000 }
    }

    /// `[n_mels, n_samples / hop]` features of 16 kHz mono samples.
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let mut padded = samples.to_vec();
        padded.resize(self.n_samples, 0.0);
        let mut log_spec = self.mel.compute(&padded);
        let frames = log_spec.ncols();
        log_spec = log_spec.slice(ndarray::s![.., ..frames.saturating_sub(1)]).to_owned();
        let max = log_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        log_spec.mapv(|v| (v.max(max - 8.0) + 4.0) / 4.0)
    }
}
