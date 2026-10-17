#!/usr/bin/env bash
# Package a Linux build as openjarvis_<version>_<arch>.deb (Debian 12+),
# without needing dpkg-deb: a .deb is an `ar` archive of debian-binary,
# control.tar.xz and data.tar.xz, in that order.
#   scripts/build-deb.sh                # arm64: run build-aarch64-linux.sh first
#   ARCH=amd64 scripts/build-deb.sh     # amd64: run `cargo build --release` first
# (an amd64 build with the default features needs libvulkan1 and libasound2).
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
ARCH=${ARCH:-arm64}
case "$ARCH" in
    arm64) BIN=target/aarch64-unknown-linux-gnu/release; EXTRA_DEPS="" ;;
    amd64) BIN=target/release; EXTRA_DEPS=", libvulkan1, libasound2" ;;
    *) echo "ARCH must be arm64 or amd64"; exit 1 ;;
esac
OUT=target/deb
PKG=$OUT/pkg
rm -rf "$OUT" && mkdir -p "$PKG/usr/bin" "$PKG/usr/share/applications" "$OUT/control" \
    "$PKG/usr/share/icons/hicolor/scalable/apps" "$PKG/usr/share/icons/hicolor/256x256/apps"
install -m644 assets/openjarvis.svg "$PKG/usr/share/icons/hicolor/scalable/apps/openjarvis.svg"
install -m644 assets/openjarvis-256.png "$PKG/usr/share/icons/hicolor/256x256/apps/openjarvis.png"

install -m755 "$BIN/openjarvis" "$BIN/ojp-cli" "$PKG/usr/bin/"
cat > "$PKG/usr/share/applications/openjarvis.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=OpenJarvis
Comment=Local AI assistant
Exec=openjarvis
Icon=openjarvis
Terminal=false
Categories=Utility;
DESKTOP

SIZE_KB=$(du -sk "$PKG" | cut -f1)
cat > "$OUT/control/control" <<CONTROL
Package: openjarvis
Version: $VERSION
Architecture: $ARCH
Maintainer: OpenJarvisProject <https://github.com/OpenJarvisProject/ojp>
Installed-Size: $SIZE_KB
Depends: libc6 (>= 2.35), libfontconfig1, libegl1 | libgl1, libxkbcommon0$EXTRA_DEPS
Recommends: libwayland-client0 | libx11-6
Section: utils
Priority: optional
Homepage: https://github.com/OpenJarvisProject/ojp
Description: OpenJarvis local AI assistant
 Offline assistant that runs small language models on this device. On
 first run it downloads its models (about 1.5 GB) and verifies them.
CONTROL
(cd "$PKG" && find usr -type f -exec md5sum {} + > ../control/md5sums)

echo 2.0 > "$OUT/debian-binary"
tar -C "$OUT/control" --owner=0 --group=0 --numeric-owner -cJf "$OUT/control.tar.xz" .
tar -C "$PKG" --owner=0 --group=0 --numeric-owner -cJf "$OUT/data.tar.xz" ./usr
DEB="openjarvis_${VERSION}_${ARCH}.deb"
rm -f "$DEB"
ar rc "$DEB" "$OUT/debian-binary" "$OUT/control.tar.xz" "$OUT/data.tar.xz"
echo "built $(du -h "$DEB" | cut -f1) $DEB"
