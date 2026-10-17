# Speech engines

The app speaks through one small interface (`SpeechEngine` in
`ojp-core/src/tts.rs`): given a chunk of text, a voice name and a speed, return
mono samples and their sample rate. The app handles everything else: splitting
a reply into sentences, queueing and stopping playback, settings, downloads.
An engine can be linked in (Kokoro is) or run as a separate process (MOSS is).

## Sidecar protocol

An engine in its own process talks to the app over its stdin and stdout; it
opens no network port. Everything else it prints must go to stderr.

- On start, once it is ready: `READY\n` on stdout.
- Request, one JSON object per line on stdin:
  `{"text": "...", "voice": "Bella", "reference": null}`
  (`reference` is an optional path to a recording to imitate.)
- Reply on stdout: `ERR <message>\n`, or `OK <sample_rate> <byte_count>\n`
  followed by `byte_count` bytes of mono little-endian float32 samples.
- The app ends the engine by closing its stdin or killing the process.

`scripts/moss_server.py` is the reference implementation. Adding another
engine means writing one of these (any language) or implementing the trait.
