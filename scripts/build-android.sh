#!/usr/bin/env bash
# Build ojp.apk (arm64-v8a) with only the Android SDK command-line tools:
# no Gradle, no Google Maven downloads.
#
#   scripts/build-android.sh                     # -> ojp.apk
#   OJP_APP_LABEL=Zelia scripts/build-android.sh # launcher label
#   OJP_ABIS="arm64-v8a x86_64" scripts/...       # + x86_64 for the emulator
#
# Needs: Android SDK (platform 35, build-tools 35), NDK, JDK 17+,
# cargo-ndk, rustup target aarch64-linux-android.
# Signs with $OJP_KEYSTORE (default ~/.android/ojp-debug.keystore, created
# on first run, password "android"). Use your own keystore for releases.
set -euo pipefail
cd "$(dirname "$0")/.."

SDK=${ANDROID_HOME:-$HOME/Android/Sdk}
NDK=${ANDROID_NDK_HOME:-$(ls -d "$SDK"/ndk/*/ | sort -V | tail -1)}
BT=$SDK/build-tools/35.0.0
export ANDROID_HOME=$SDK ANDROID_NDK_HOME=$NDK ANDROID_NDK_ROOT=$NDK
export ANDROID_JAR=$SDK/platforms/android-35/android.jar ANDROID_D8_JAR=$BT/lib/d8.jar
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
SRC=ojp-android/android
OUT=ojp-android/build

ABIS=${OJP_ABIS:-arm64-v8a}
targets=()
for abi in $ABIS; do targets+=(-t "$abi"); done
rm -rf "$OUT-jni"
cargo ndk "${targets[@]}" -P 26 -o "$OUT-jni" build --release -p ojp-android

rm -rf "$OUT" && mkdir -p "$OUT"/{classes,dex}
cp -r "$SRC/res" "$OUT/res"
if [[ -n "${OJP_APP_LABEL:-}" ]]; then
    sed -i "s|<string name=\"app_name\">[^<]*</string>|<string name=\"app_name\">$OJP_APP_LABEL</string>|" "$OUT/res/values/strings.xml"
fi

javac --release 11 -classpath "$ANDROID_JAR" -d "$OUT/classes" $(find "$SRC/java" -name '*.java')
java -cp "$ANDROID_D8_JAR" com.android.tools.r8.D8 --release --min-api 26 --lib "$ANDROID_JAR" \
    --output "$OUT/dex" $(find "$OUT/classes" -name '*.class')

"$BT/aapt2" compile --dir "$OUT/res" -o "$OUT/res.zip"
"$BT/aapt2" link -o "$OUT/unsigned.apk" -I "$ANDROID_JAR" --manifest "$SRC/AndroidManifest.xml" \
    --min-sdk-version 26 --target-sdk-version 35 --version-code 1 --version-name "$VERSION" "$OUT/res.zip"

cp -r "$OUT-jni" "$OUT/lib"
# aapt (v1) adds files under their path relative to the working dir.
(cd "$OUT/dex" && "$BT/aapt" add ../unsigned.apk classes.dex)
(cd "$OUT" && "$BT/aapt" add unsigned.apk $(find lib -name '*.so'))
"$BT/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"

KS=${OJP_KEYSTORE:-$HOME/.android/ojp-debug.keystore}
if [[ ! -f "$KS" ]]; then
    mkdir -p "$(dirname "$KS")"
    keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias ojp \
        -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=OJP debug" >/dev/null
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --out ojp.apk "$OUT/aligned.apk"
"$BT/apksigner" verify ojp.apk
echo "built $(du -h ojp.apk | cut -f1) ojp.apk"
