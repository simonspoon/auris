#!/usr/bin/env bash
# Generates tts-nonspeech-case.wav (mesa task 968): a kokoro-rs speech
# clip, resampled to auris's 16 kHz mono 16-bit PCM. See README.md's entry
# for this file for what it is for and why it is not a pass/fail asset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/tts-nonspeech-case.wav"
TMP="$(mktemp -t auris_tts_case).wav"

kokoro-rs -q -o "$TMP" "The quick brown fox jumps over the lazy dog."
ffmpeg -y -loglevel error -i "$TMP" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT"
rm -f "$TMP"
echo "made $(basename "$OUT")"
