#!/usr/bin/env python3
"""Make the reference clip for OJP's "My own reference clip" MOSS voice.

  make-reference.py RECORDING [--feminize] [--formant 1.15] [--pitch 210]
                    [--max-seconds 12] [--moss-dir DIR] [--out FILE]

Takes a recording of your own voice (WAV/FLAC/OGG; 10-20 seconds of clear
speech, one speaker, no music or background noise), trims the silence, caps
the length, levels it, and writes <moss dir>/reference.wav (24 kHz mono).
With --feminize it first shifts the voice with Praat's "Change gender":
formants up by --formant (1.0 = unchanged; about 1.1-1.2 is male to
female) and the pitch median to --pitch Hz (default 210). Use only voices
you have the right to use. Needs the MOSS environment from setup-moss.sh
(it installs praat-parselmouth, soundfile and numpy).

Then pick "Voice engine: MOSS" and "MOSS voice: My own reference clip" in
Settings. Re-run with other --formant/--pitch values to compare.
"""
import argparse
import os
import sys
import tomllib

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

RATE = 24000


def moss_dir_from_config() -> str:
    path = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "ojp", "config.toml")
    try:
        with open(path, "rb") as f:
            configured = tomllib.load(f).get("tts", {}).get("moss_dir", "")
    except OSError:
        configured = ""
    if configured:
        return configured
    return os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "ojp", "moss")


def trim_silence(samples: np.ndarray, rate: int) -> np.ndarray:
    frame = int(rate * 0.02)
    n = len(samples) // frame
    if n == 0:
        return samples
    energy = np.sqrt((samples[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(energy > max(energy.max() * 0.03, 1e-4))
    if len(loud) == 0:
        return samples
    pad = int(rate * 0.1)
    return samples[max(loud[0] * frame - pad, 0) : min((loud[-1] + 1) * frame + pad, len(samples))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--feminize", action="store_true", help="shift the voice male -> female with Praat")
    ap.add_argument("--formant", type=float, default=1.15, help="formant shift ratio (with --feminize)")
    ap.add_argument("--pitch", type=float, default=210.0, help="new pitch median in Hz (with --feminize)")
    ap.add_argument("--max-seconds", type=float, default=12.0, help="longer clips make every sentence slower to start")
    ap.add_argument("--moss-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, rate = sf.read(args.recording, dtype="float64", always_2d=True)
    sound = parselmouth.Sound(data.mean(axis=1), sampling_frequency=rate).resample(RATE)
    samples = trim_silence(sound.values[0], RATE)
    if len(samples) < RATE * 3:
        sys.exit("The usable speech is under 3 seconds; record a longer, clearer clip.")
    samples = samples[: int(args.max_seconds * RATE)]
    sound = parselmouth.Sound(samples, sampling_frequency=RATE)

    median = call(sound.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
    print(f"clip: {len(samples) / RATE:.1f} s, pitch median {median:.0f} Hz")
    if args.feminize:
        sound = call(sound, "Change gender", 75, 600, args.formant, args.pitch, 1.0, 1.0)
        after = call(sound.to_pitch(), "Get quantile", 0, 0, 0.5, "Hertz")
        print(f"feminized: formants x{args.formant}, pitch median {after:.0f} Hz (asked for {args.pitch:.0f})")

    out = np.asarray(sound.values[0])
    out = out * (0.2 / max(np.sqrt((out**2).mean()), 1e-6))  # level to a steady loudness
    out = np.clip(out, -0.98, 0.98)
    dest = args.out or os.path.join(args.moss_dir or moss_dir_from_config(), "reference.wav")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    sf.write(dest, out, RATE, subtype="PCM_16")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
