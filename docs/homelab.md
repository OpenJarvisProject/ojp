# Homelab: design notes (nothing built yet)

Goal: OJP/Zelia eventually helps run a small homelab: OpenWrt routers, DHCP and DNS, a Raspberry Pi
acting as a mobile router, and services such as OxiCloud or Stalwart mail. This page records what that
means for decisions made now, so the device link and the tool system don't have to be redone.

## What already fits

- **Every managed machine runs its own agent** (the same `ojp-core`, headless: `ojp-cli link-serve` is the
  seed of it), paired by hand like a phone. "Ask my router ...", "tell the pi to ..." already work through
  `kind` names (`server`, `router`, `pi`) and per-device permission levels.
- **Risky changes are approved on the device you are holding** (permission level *Everything*), never
  done unasked, and expire. For infrastructure this stays the rule, not an option.
- **Reachable from anywhere** the devices can reach each other (LAN or the VPN already in use); the link
  port is never meant to face the internet.

## What it should add, in this order

1. **Read-only first.** Status and health for each host and service: is Stalwart up, how full is the
   disk, who holds which DHCP lease, is the WAN link up. A failed check sends a message to the phone
   (the `Notify` request exists). No change tools until this is solid.
2. **Changes as plans.** Any change is shown as a diff (the file, UCI or service setting before and after),
   approved on the asking device, applied, then verified. For OpenWrt and anything that can cut the
   network: apply with an automatic rollback timer (revert in 60 s unless confirmed from a second path),
   as OpenWrt's own safe mode does. Every change goes in an append-only audit log on the host.
3. **Tool packs, not a bigger default list.** The 0.8B planner already struggles past ~30 tools. Host
   tools live in separate, declarative packs loaded only on the host that has them (an OpenWrt pack over
   `ubus`/`uci`, a Stalwart pack over its admin API, an OxiCloud pack ...), with a per-pack probe or a
   routing step that first picks the pack. Compatibility with MCP-style tool descriptions is worth keeping
   in mind so existing servers can be reused.
4. **A secrets store separate from memory.** API keys and passwords never go into `memory.jsonl` or a prompt;
   a tool gets a secret by name at run time and its output is scrubbed before the model sees it.

## Things to respect

- You want to **write some configs yourself to learn them** (for example Kea for DHCP and hickory-dns). The
  assistant should explain, review and diff, and only write a config when asked.
- Models this small make mistakes: nothing destructive runs without a yes, and "dry run" is the default.
- Prompt injection from the network is real here (DNS names, DHCP hostnames, mail subjects are text that
  strangers write). Treat anything read from a service as untrusted, as the engine already does for web
  pages and files.
- The home server has 7.6 GB of RAM and an old CPU: the agent on it should be the router-sized model only.
