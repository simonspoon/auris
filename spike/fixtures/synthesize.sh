#!/usr/bin/env bash
# Synthesize spike/fixtures/utterances.tsv into 16kHz mono WAVs using macOS `say`,
# rotating across a few built-in voices so the fixture set isn't one uniform voice.
# Idempotent: skips any wav that already exists. Re-run with --force to regenerate all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSV="$SCRIPT_DIR/utterances.tsv"
OUT_DIR="$SCRIPT_DIR/wav"
mkdir -p "$OUT_DIR"

VOICES=(Daniel Samantha Karen)
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

i=0
while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    voice="${VOICES[$((i % ${#VOICES[@]}))]}"
    i=$((i + 1))

    out="$OUT_DIR/$id.wav"
    if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
        echo "skip $id (exists)"
        continue
    fi

    tmp_aiff="$(mktemp -t "auris_${id}").aiff"
    say -v "$voice" -o "$tmp_aiff" "$text"
    ffmpeg -y -loglevel error -i "$tmp_aiff" -ar 16000 -ac 1 -c:a pcm_s16le "$out"
    rm -f "$tmp_aiff"
    echo "made $id ($voice)"
done < "$TSV"

echo
echo "Duration table:"
printf '%-6s %-10s %s\n' "id" "seconds" "voice"
i=0
while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    voice="${VOICES[$((i % ${#VOICES[@]}))]}"
    i=$((i + 1))
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/$id.wav")
    printf '%-6s %-10.2f %s\n' "$id" "$dur" "$voice"
done < "$TSV"
