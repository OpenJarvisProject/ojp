# Making a voice of your own (experimental)

Goal: a voice for the assistant that starts from your own speech, with the
timbre you choose, without anyone else's voice.

1. `scripts/setup-moss.sh DIR` sets up the MOSS engine (Python, about 1.5 GB).
2. Set up Seed-VC (GPL-3.0, an offline tool; nothing from it ships with the
   app) in `/media/ojp-seedvc` or wherever `$SEEDVC` points: clone
   https://github.com/Plachtaa/seed-vc into `repo/`, create a Python 3.10 venv
   in `venv/` (its pinned requirements need 3.10; `uv python install 3.10`
   works), and install CPU PyTorch 2.4 plus `requirements.txt` without gradio,
   funasr and modelscope. Checkpoints download on first run (about 2 GB).
3. Record 15-60 seconds of clear speech. For each timbre you want to try:
   `scripts/make-voice-variant.sh recording.m4a NAME "af_heart.6+af_bella.4"`
   (the last argument is any Kokoro voice or blend, the same syntax as
   `[tts] voice`). It converts your speech to that timbre and stores
   `<moss dir>/references/NAME.wav`.
4. Restart the app. Settings → Voice engine → MOSS lists each one as
   "Mine: NAME". `scripts/make-reference.py` is the simpler, lower-quality
   route (Praat pitch and formant shifting); it writes "My own reference clip".

Use only voices you have the right to use. A converted clip is only a
reference for MOSS to imitate; how closely it follows depends on MOSS.
