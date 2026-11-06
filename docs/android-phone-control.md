# Phone control on Android

The assistant on the phone has tools (`ojp-core/src/tools_android.rs`, Java in `ojp-android/android/java/`).

**Without anything extra** (plain Android APIs, no helper app): volume and mute, media keys, open an app by the name
under its icon, open a web link or a page of the system settings, reminders and timers (an alarm that posts a
notification, even with the app closed; Android 13+ asks once to allow notifications), alarms in the clock app,
the flashlight, calls and texts (the dialer or messages app opens with the number or text filled in and the user presses
call or send; a contact's name is looked up after a one-time contacts permission), brightness (after the one-time
"modify system settings" page), battery, and `phone_read` lookups of Wi-Fi (band, signal, speed; the network name needs a
one-time location permission because Android hides it otherwise), IP address, storage, Android version, phone model,
uptime, installed apps and `settings get` values. Plus the tools that need no device access (time, calculator, notes,
memory, web).

**With Shizuku (optional):** Wi-Fi, Bluetooth and airplane mode flipped by the assistant itself (without Shizuku the
tool opens that settings page and says to tap the switch: Android doesn't let apps do it), screenshots and `adb_shell`,
any shell command with adb's rights. `adb_shell` always asks first. `phone_read` is the lookup tool: the assistant picks
a read-only command itself (for example `cmd wifi status`, `ip -f inet addr show wlan0`, `dumpsys battery`, `getprop`)
and it runs without asking, but only if it passes a read-only check (no redirects, chaining, or anything that changes
settings, packages or files); everything else goes through `adb_shell`. With Shizuku the command really runs; without it
the common ones are answered from what Android gives any app (`tools_android::without_shizuku`) and the rest say they
need Shizuku. Shizuku is a separate free app; start it once (wireless debugging, no PC needed on Android 11+), then
Settings > Phone control > *Allow with Shizuku*. Nothing in OJP needs it.

**Phone check:** Settings > Phone control > *Check what works* / *Full check* (or typing "test everything", "run a phone
check") runs `tools_android::self_check`: every lookup, volume, flashlight, and with Shizuku the shell, app info,
screenshot and (full check only) Wi-Fi, Bluetooth and airplane mode switched off and back to how they were. The report
lists OK / FAILED / skipped per item and is saved as a chat message. The small model is not asked to plan this: given
"test everything" it ran three steps and the answer sounded as if all were done. The answer step is also told what was
actually run and not to claim anything else. A `phone_read` call whose command isn't a read-only lookup becomes an
`adb_shell` call (asks first). *Copy chat* in a chat's three-dot menu puts the whole chat, with each reply's grey line, on
the clipboard.

**Running in the background:** Settings > Phone control > *Keep running in the background* (on by default) starts a
foreground service (`KeepAliveService`, type `specialUse`, one silent low-priority notification with a Stop button).
The Back button at the top of the app then moves the app to the background instead of closing it, so the loaded model
stays in memory and reopening is instant; replies keep being written with the screen off (a partial wake lock is held
while one is written, at most 10 minutes), and a notification shows when a reply finishes while the app isn't in front.
It does not start at boot (the model lives in the app's process, which only starts when the app opens). Swiping the
app out of the recent apps list leaves the service (and so the process) running; reopening was quick in the emulator,
not yet checked on a real phone.

Another device can ask the phone for these things ("turn off the wifi on my phone") within the permission
level you gave it (see devices.md); `adb_shell` and `phone_read` need the *Everything* level and your yes on the asking device.

How it is wired: `ojp_core::platform` holds one `(method, argument) -> text` function that the app points at
`PhoneActions.invoke`, so ojp-core has no JNI. The Shizuku API classes (Apache-2.0) are fetched by
`scripts/fetch-shizuku-api.sh` with pinned checksums.

Not done / not tested on a device yet: file tools on Android, clipboard (Android blocks background reads),
reading the screen and tapping (a UI-automation tool set), the tool-choice probe knows only the desktop tool
names (new phone tools are chosen by the planner, slower), alarms don't survive a reboot.
