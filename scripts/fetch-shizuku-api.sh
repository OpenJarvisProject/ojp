#!/usr/bin/env bash
# Download the Shizuku API (Apache-2.0, https://github.com/RikkaApps/Shizuku-API) from Maven Central,
# check each file against a pinned SHA-256, and unpack the class files the Android build links against
# into ojp-android/android/libs/. The build script runs this when the jars are missing.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=ojp-android/android/libs
VERSION=13.1.5
declare -A SHA=(
    [api]=4def9bde498ef8626614c2fc5db9af4749c86f16f6c33e3f5658d35e70bab59b
    [provider]=b0f18cd9812464ec171c53cac93a819fe411718a3965c311f01eb4de265381b3
    [aidl]=33fe7191cdd69fcb66d649264f3b0c47acb2f3d6343afc05b98dbbff6f221963
    [shared]=4659642c9339be0a26e9c65bb8648f7ad6d8f4a465f557993ccbc78802381635
)
mkdir -p "$OUT"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for m in api provider aidl shared; do
    [[ -f "$OUT/shizuku-$m.jar" ]] && continue
    curl -sfL -o "$tmp/$m.aar" "https://repo1.maven.org/maven2/dev/rikka/shizuku/$m/$VERSION/$m-$VERSION.aar"
    echo "${SHA[$m]}  $tmp/$m.aar" | sha256sum -c --quiet -
    unzip -qo -j "$tmp/$m.aar" classes.jar -d "$tmp/$m"
    mv "$tmp/$m/classes.jar" "$OUT/shizuku-$m.jar"
done
echo "Shizuku API $VERSION ready in $OUT"
