# Devices working together

Pair your devices once, by hand, and they behave like one assistant.

## What you get

- **A bigger brain, when you need it.** A phone with only the small model answers chat with your
  computer's model (when it can reach it), and falls back to its own when it can't. (`[link] brain` in
  config.toml: `auto`, `local`, `remote`.)
- **"On my phone ...".** *"Remind me on my phone in 10 minutes to call Mom"*, *"ask my server how much
  disk space is left"*, *"tell pixel to turn the volume down"*. The request, with the device wording
  removed, goes to that device's own assistant, which does it with its own tools. If the device can't be
  reached you are told; nothing is run here instead.
- **Shared memory.** What you tell one device is remembered by all. Forgetting is shared too (a forgotten
  memory can't be brought back by a device that was offline).
- **Messages.** One device can show a message on another (`ojp-cli link-notify`).
- **Reachable from anywhere** your devices can reach each other: the LAN, or a VPN such as Tailscale or
  Netbird. At pairing each device tells the other every address it has.

## Pairing and security

1. On one device: Settings > Devices > *Show a pairing code* (or `ojp-cli link-pair-host`).
2. On the other: type the 8-digit code, tap *Pair* (or `ojp-cli link-pair-join 1234-5678`).

- The code is the password of a SPAKE2 key exchange, so a listener on the network learns nothing it can
  test guesses against, one wrong code gets one try, and the host gives up after three failed
  connections or two minutes. The key from that exchange protects the swap of each device's own
  public key.
- After that, every connection is Noise XX (X25519, ChaCha20-Poly1305) between the two devices' own keys.
  A device that is not in `peers.json` is dropped right after the handshake. Keys live in
  `<data>/ojp/link/` (mode 0600) and never leave the device.
- Each paired device has permissions you can change in Settings (or `ojp-cli link-allow`):
  - *Use its AI model* (chat);
  - *May ask this device to do things*: **Nothing**, **Basic** (timers, reminders, notes, memory, volume,
    media, time, status, calculator) or **Basic + files and web**. Anything that asks the user first
    (writing files, running commands, forgetting) is never done on another device's say-so;
    it has to be done on the device itself. Typed `/commands` are not accepted from other devices;
  - *May show messages here*;
  - *Share memories*.
- Requests from another device don't touch the conversation on this one.

## Finding each other

Known addresses are tried all at once; then a quick look at this network's /24 for the link port
(47600); then mDNS. Pairing looks for the waiting device the same way (port 47620), so it works even when
multicast doesn't (on the development PC another program holds the mDNS port).

## Not done yet

- Android: the device link runs, but the phone's own tools aren't available yet (so "remind me on my
  phone" can't run there until timers/notifications exist on Android), and the app only serves while it
  is open (no background service yet).
- Streaming a file or the clipboard to another device; showing messages in the Android notification shade.
- A request from a device that needs your yes ("delete these files on my desktop") could be shown on the
  device that asked, then confirmed there; today it is refused with a pointer to do it on the device.
- Several phones of the same kind need to be told apart by name ("on pixel"), not by "my phone".
- Routing by load: when two paired devices both have big models, the first reachable one is used.
