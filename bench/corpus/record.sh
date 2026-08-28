#!/usr/bin/env bash
# Interactive recorder: have the owner speak the same fixture sentences into
# bench/corpus/wav/<id>.wav, replacing the synthetic `say` versions.
#
# The existing synthetic wavs are backed up to bench/corpus/wav-synthetic/
# once, on first run, so re-running this script never clobbers that backup.
#
# With no arguments, records all utterances. Pass one or more ids
# (bench/corpus/record.sh u03 u17) to re-record just those — 30 in one
# sitting is a lot.
#
# For each utterance: prints the sentence, waits for Enter to start
# recording, records in the background via ffmpeg + avfoundation, then waits
# for Enter again to stop and move to the next one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSV="$SCRIPT_DIR/utterances.tsv"
OUT_DIR="$SCRIPT_DIR/wav"
BACKUP_DIR="$SCRIPT_DIR/wav-synthetic"
mkdir -p "$OUT_DIR"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "Backing up synthetic wavs to $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp "$OUT_DIR"/*.wav "$BACKUP_DIR"/ 2>/dev/null || true
fi

echo "Recording device: \":default\" (change AUDIO_DEVICE env var if wrong)"
echo "List devices with: ffmpeg -f avfoundation -list_devices true -i \"\""
AUDIO_DEVICE="${AUDIO_DEVICE:-:default}"

# Only re-record the ids named on the command line, if any were given.
declare -A WANT
for id in "$@"; do
    WANT["$id"]=1
done

while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    if [[ $# -gt 0 && -z "${WANT[$id]:-}" ]]; then
        continue
    fi
    out="$OUT_DIR/$id.wav"

    echo
    echo "=== $id ==="
    echo "Say this sentence:"
    echo "  $text"
    read -r -p "Press Enter to start recording..."

    ffmpeg -y -loglevel error -f avfoundation -i "$AUDIO_DEVICE" \
        -ar 16000 -ac 1 -c:a pcm_s16le "$out" &
    ffmpeg_pid=$!

    read -r -p "Recording... press Enter to stop."
    kill -INT "$ffmpeg_pid" 2>/dev/null || true
    wait "$ffmpeg_pid" 2>/dev/null || true

    echo "saved $out"
done < "$TSV"

echo
echo "Done. Recorded wavs are in $OUT_DIR, synthetic originals backed up in $BACKUP_DIR."
