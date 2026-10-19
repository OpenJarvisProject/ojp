#!/usr/bin/env python3
"""Sidecar for OJP's experimental MOSS-TTS-Nano voice (see setup-moss.sh).

Usage: moss_server.py REPO_DIR [--check]

Reads one JSON request per line on stdin, {"text": "...", "voice": "Bella"}
(optionally "reference": "/path/to/clip.wav" to imitate that clip instead),
and answers on stdout with either `ERR <message>\\n` or `OK <sample_rate>
<byte_count>\\n` followed by that many bytes of mono little-endian float32
samples. Prints `READY\\n` once the model is loaded. Nothing else is ever
written to stdout (library chatter goes to stderr), and there is no network
listener: it talks only to the app that started it.

TEMPORARY: this bridges to MOSS's Python reference code so the voice can be
auditioned. A native Rust port would replace it.
"""
import json
import os
import sys
import tempfile

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
os.chdir(repo)
proto = os.fdopen(os.dup(1), "wb", buffering=0)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import soundfile  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _load_with_soundfile(path, *args, **kwargs):
    """torchaudio.load without TorchCodec/ffmpeg, for reading reference clips."""
    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), rate


torchaudio.load = _load_with_soundfile

from onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402

runtime = OnnxTtsRuntime(
    model_dir=None,
    thread_count=int(os.environ.get("OJP_MOSS_THREADS", "4")),
    max_new_frames=375,
    do_sample=True,
    sample_mode="fixed",
    execution_provider="cpu",
)
# Encoding a reference clip is slow, so do it once per clip (and per change to it).
_resolve_codes = runtime.resolve_prompt_audio_codes
_codes_cache = {}


def _cached_codes(**kwargs):
    path = kwargs.get("prompt_audio_path")
    stamp = os.stat(path).st_mtime_ns if path else None
    key = (str(kwargs.get("voice")), str(path), stamp)
    if key not in _codes_cache:
        _codes_cache.clear()
        _codes_cache[key] = _resolve_codes(**kwargs)
    return _codes_cache[key]


runtime.resolve_prompt_audio_codes = _cached_codes

if "--check" in sys.argv:
    proto.write(b"READY\n")
    sys.exit(0)
proto.write(b"READY\n")

scratch = tempfile.mkdtemp(prefix="ojp-moss-")
for line in sys.stdin:
    try:
        request = json.loads(line)
        text = str(request["text"]).strip()
        if not text:
            raise ValueError("empty text")
        # Optional sampling overrides (lower temperatures = steadier pronunciation).
        defaults = runtime.manifest["generation_defaults"]
        for key in ("text_temperature", "text_top_p", "text_top_k", "audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty"):
            if key in request:
                defaults[key] = type(defaults[key])(request[key]) if key in defaults else request[key]
        reference = request.get("reference") or None  # a clip to imitate instead of a built-in voice
        result = runtime.synthesize(
            text=text,
            voice=str(request.get("voice") or "Bella"),
            prompt_audio_path=reference,
            output_audio_path=os.path.join(scratch, "out.wav"),
            sample_mode=request.get("sample_mode", "fixed"),
            do_sample=True,
            streaming=True,
            max_new_frames=375,
            voice_clone_max_text_tokens=75,
            enable_wetext=False,
            enable_normalize_tts_text=False,
            seed=request.get("seed"),
        )
        wave = np.asarray(result["waveform"], dtype=np.float32)
        if wave.ndim == 2:
            wave = wave.mean(axis=1 if wave.shape[0] > wave.shape[1] else 0)
        data = np.ascontiguousarray(wave, dtype="<f4").tobytes()
        proto.write(f"OK {int(result['sample_rate'])} {len(data)}\n".encode() + data)
    except Exception as exc:  # report, keep serving
        proto.write(f"ERR {str(exc).replace(chr(10), ' ')}\n".encode())
