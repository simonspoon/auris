#!/bin/sh
# Build auris in release mode and install the binary.
#
# The binary is self-contained: ONNX Runtime is statically linked, so there is
# no runtime dylib to fetch. The only thing downloaded on first run is the
# Parakeet model itself (~661 MB, once) into ~/.cache/auris, so installing is
# just a matter of copying one file onto PATH.
#
# Usage:
#   scripts/install.sh                  # install to ~/.local/bin
#   PREFIX=/usr/local scripts/install.sh  # install to /usr/local/bin
#   BINDIR=/opt/bin scripts/install.sh    # install to an exact directory

set -eu

cd "$(dirname "$0")/.."

bindir=${BINDIR:-${PREFIX:-$HOME/.local}/bin}
name=auris

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found; install Rust from https://rustup.rs" >&2
    exit 1
fi

echo "building $name (release)..."
cargo build --release

built=${CARGO_TARGET_DIR:-target}/release/$name
if [ ! -x "$built" ]; then
    echo "error: $built missing after build" >&2
    exit 1
fi

echo "installing to $bindir/$name"
mkdir -p "$bindir"
# Install to a temporary name and rename, so a running copy of the old binary
# is not overwritten in place.
tmp=$bindir/.$name.new.$$
trap 'rm -f "$tmp"' EXIT INT TERM
cp "$built" "$tmp"
chmod 755 "$tmp"
mv "$tmp" "$bindir/$name"
trap - EXIT INT TERM

case ":${PATH:-}:" in
    *":$bindir:"*) ;;
    *) echo "note: $bindir is not on your PATH; add it to your shell profile" >&2 ;;
esac

"$bindir/$name" --version
