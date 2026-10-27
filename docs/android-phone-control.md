# Phone control on Android

The assistant on the phone has tools (`ojp-core/src/tools_android.rs`, Java in `ojp-android/android/java/`).

**Without anything extra** (plain Android APIs): volume and mute, media keys, open an app by the name under
its icon, open a web link, reminders and timers (an alarm that posts a notification, even with the app
closed; Android 13+ asks once to allow notifications), battery, plus the tools that need no device access
(time, calculator, notes, memory, web).

**With Shizuku (optional):** Wi-Fi, Bluetooth, airplane mode, brightness, screenshots and `adb_shell`, any
shell command with adb's rights. `adb_shell` always asks first. Shizuku is a separate free app; start it once
(wireless debugging, no PC needed on Android 11+), then Settings > Phone control > *Allow with Shizuku*.
Nothing in OJP needs it, and without it those tools just say what to do.

Another device can ask the phone for these things ("turn off the wifi on my phone") within the permission
level you gave it (see devices.md); `adb_shell` needs the *Everything* level and your yes on the asking device.

How it is wired: `ojp_core::platform` holds one `(method, argument) -> text` function that the app points at
`PhoneActions.invoke`, so ojp-core has no JNI. The Shizuku API classes (Apache-2.0) are fetched by
`scripts/fetch-shizuku-api.sh` with pinned checksums.

Not done / not tested on a device yet: file tools on Android, clipboard (Android blocks background reads),
reading the screen and tapping (a UI-automation tool set), the tool-choice probe knows only the desktop tool
names (new phone tools are chosen by the planner, slower), alarms don't survive a reboot.
