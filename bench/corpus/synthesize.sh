#!/usr/bin/env bash
# Synthesize bench/corpus/utterances.tsv into 16kHz mono WAVs using macOS `say`,
# rotating across a wider voice set than the spike fixtures for speaker
# variety — a single voice was the spike's stated weakness. These are
# synthetic stand-ins, not the owner's dictated voice: they exercise the
# pipeline end to end, but absolute WER/RTF numbers from them are a floor,
# not a ceiling — bench/corpus/record.sh re-records the same sentences for
# real. Idempotent: skips any wav that already exists. Re-run with --force
# to regenerate all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSV="$SCRIPT_DIR/utterances.tsv"
OUT_DIR="$SCRIPT_DIR/wav"
mkdir -p "$OUT_DIR"

# Requested set was Daniel Samantha Karen Alex Moira Tessa; Alex is not
# installed on this machine (`say -v '?'` doesn't list it), so it's dropped.
# Pinned explicitly, not auto-discovered, so regeneration is reproducible.
VOICES=(Daniel Samantha Karen Moira Tessa)

# Cycled alongside voice, independently, so pacing isn't uniform either.
RATES=(175 195 160)

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

i=0
while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    voice="${VOICES[$((i % ${#VOICES[@]}))]}"
    rate="${RATES[$((i % ${#RATES[@]}))]}"
    i=$((i + 1))

    out="$OUT_DIR/$id.wav"
    if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
        echo "skip $id (exists)"
        continue
    fi

    tmp_aiff="$(mktemp -t "auris_${id}").aiff"
    say -v "$voice" -r "$rate" -o "$tmp_aiff" "$text"
    ffmpeg -y -loglevel error -i "$tmp_aiff" -ar 16000 -ac 1 -c:a pcm_s16le "$out"
    rm -f "$tmp_aiff"
    echo "made $id ($voice, ${rate}wpm)"
done < "$TSV"

echo
echo "Duration table:"
printf '%-6s %-10s %-10s %s\n' "id" "seconds" "voice" "rate"
total=0
i=0
while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    voice="${VOICES[$((i % ${#VOICES[@]}))]}"
    rate="${RATES[$((i % ${#RATES[@]}))]}"
    i=$((i + 1))
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/$id.wav")
    printf '%-6s %-10.2f %-10s %s\n' "$id" "$dur" "$voice" "$rate"
    total=$(echo "$total + $dur" | bc)
done < "$TSV"
printf 'TOTAL  %-10.2f\n' "$total"
