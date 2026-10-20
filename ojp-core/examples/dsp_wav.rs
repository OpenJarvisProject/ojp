//! Apply the clarity filter to a WAV: ... --example dsp_wav -- IN.wav OUT.wav STRENGTH
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut reader = hound_free::read(&a[1]);
    ojp_core::dsp::clarity(&mut reader.samples, reader.rate, a[3].parse().expect("strength 0..1"));
    hound_free::write(&a[2], &reader.samples, reader.rate);
}

/// Minimal 16-bit mono WAV reader/writer so the example needs no extra crate.
mod hound_free {
    pub struct Wav {
        pub samples: Vec<f32>,
        pub rate: u32,
    }

    pub fn read(path: &str) -> Wav {
        let b = std::fs::read(path).expect("read wav");
        let rate = u32::from_le_bytes(b[24..28].try_into().unwrap());
        let data = b.windows(4).position(|w| w == b"data").expect("data chunk") + 8;
        let samples = b[data..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        Wav { samples, rate }
    }

    pub fn write(path: &str, samples: &[f32], rate: u32) {
        let mut out = Vec::new();
        let bytes = (samples.len() * 2) as u32;
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + bytes).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&bytes.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        std::fs::write(path, out).expect("write wav");
    }
}
