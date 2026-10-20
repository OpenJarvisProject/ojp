//! Small audio filters for synthesized speech.
//!
//! [`clarity`] makes consonants easier to hear: a gentle high-pass, a lift
//! around 3.5 kHz and 6.5 kHz where sibilants and plosive bursts live, and a
//! light compressor that evens out quiet word endings. It is what turns
//! slurred-sounding output crisp, and it is plain DSP, so it behaves the same
//! on every device and with every speech engine.

/// One second-order filter section (RBJ cookbook), direct form I.
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    x: [f32; 2],
    y: [f32; 2],
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Biquad {
        Biquad { b: [b0 / a0, b1 / a0, b2 / a0], a: [a1 / a0, a2 / a0], x: [0.0; 2], y: [0.0; 2] }
    }

    fn high_pass(rate: f32, freq: f32, q: f32) -> Biquad {
        let w = std::f32::consts::TAU * freq / rate;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        Biquad::new((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    fn peaking(rate: f32, freq: f32, q: f32, gain_db: f32) -> Biquad {
        let a = 10f32.powf(gain_db / 40.0);
        let w = std::f32::consts::TAU * freq / rate;
        let (s, c) = w.sin_cos();
        let alpha = s / (2.0 * q);
        Biquad::new(1.0 + alpha * a, -2.0 * c, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * c, 1.0 - alpha / a)
    }

    fn process(&mut self, input: f32) -> f32 {
        let out = self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x = [input, self.x[0]];
        self.y = [out, self.y[0]];
        out
    }
}

/// Make speech crisper. `strength` is 0.0 (untouched) to 1.0 (strong); about
/// 0.65 is a natural-sounding default. Samples are mono at `rate` Hz.
pub fn clarity(samples: &mut [f32], rate: u32, strength: f32) {
    let s = strength.clamp(0.0, 1.0);
    if s < 0.01 || samples.is_empty() {
        return;
    }
    let r = rate as f32;
    let mut filters = [
        Biquad::high_pass(r, 70.0, 0.707),
        Biquad::peaking(r, 3500.0, 1.2, 2.5 * s),
        Biquad::peaking(r, 6500.0, 1.5, 2.0 * s),
    ];
    // Feed-forward compressor on the peak level, in dB.
    let threshold_db = -24.0;
    let ratio = 1.0 + 1.5 * s;
    let makeup = 10f32.powf(2.0 * s / 20.0);
    let attack = (-1.0 / (0.008 * r)).exp();
    let release = (-1.0 / (0.120 * r)).exp();
    let mut env = 0.0_f32;
    for sample in samples.iter_mut() {
        let mut x = *sample;
        for f in filters.iter_mut() {
            x = f.process(x);
        }
        let level = x.abs();
        env = if level > env { attack * env + (1.0 - attack) * level } else { release * env + (1.0 - release) * level };
        let env_db = 20.0 * env.max(1e-6).log10();
        let gain_db = if env_db > threshold_db { (threshold_db - env_db) * (1.0 - 1.0 / ratio) } else { 0.0 };
        *sample = (x * 10f32.powf(gain_db / 20.0) * makeup).clamp(-0.99, 0.99);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, rate: u32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn zero_strength_changes_nothing() {
        let mut x = tone(440.0, 24_000, 0.2, 0.3);
        let before = x.clone();
        clarity(&mut x, 24_000, 0.0);
        assert_eq!(x, before);
    }

    #[test]
    fn presence_band_is_lifted_more_than_the_low_band() {
        // Compare gain on 200 Hz vs 3.5 kHz at the same input level.
        let (mut low, mut mid) = (tone(200.0, 24_000, 0.5, 0.05), tone(3500.0, 24_000, 0.5, 0.05));
        let (low_in, mid_in) = (rms(&low), rms(&mid));
        clarity(&mut low, 24_000, 1.0);
        clarity(&mut mid, 24_000, 1.0);
        let (low_gain, mid_gain) = (rms(&low) / low_in, rms(&mid) / mid_in);
        assert!(mid_gain > low_gain * 1.2, "mid {mid_gain} low {low_gain}");
    }

    #[test]
    fn output_never_clips() {
        let mut x = tone(1000.0, 24_000, 0.3, 0.98);
        clarity(&mut x, 24_000, 1.0);
        assert!(x.iter().all(|v| v.abs() <= 0.99));
    }
}
