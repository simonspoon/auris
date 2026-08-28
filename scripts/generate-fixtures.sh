#!/usr/bin/env bash
# Regenerates tests/fixtures/ from tests/fixtures/manifest.tsv using
# kokoro-rs, plus a handful of non-spoken fixtures generated directly with
# ffmpeg. Idempotent: skips any fixture that already exists. Re-run with
# --force to regenerate all.
#
# Honesty check on what these fixtures actually prove: every fixture here is
# synthesised, not dictated — kokoro-rs pronounces a name like "khora" far
# more clearly and consistently than a person reading it aloud does. These
# fixtures exercise the plumbing (decode, resample, downmix, vocabulary
# biasing reaching the recognizer, the daemon round-trip, the error paths).
# They do NOT prove transcription accuracy on real human speech; the
# benchmark task is what proves that.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/../tests/fixtures"
TSV="$FIXTURES_DIR/manifest.tsv"

# Pinned so regeneration is reproducible: kokoro-rs's default voice could
# change out from under this script otherwise, and a different voice is a
# different fixture.
VOICE=af_heart

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

make() {
    local out="$1"
    shift
    if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
        echo "skip $(basename "$out") (exists)"
        return
    fi
    "$@"
    echo "made $(basename "$out")"
}

# --- spoken fixtures, from the manifest ---

while IFS=$'\t' read -r id text; do
    [[ -z "$id" ]] && continue
    out="$FIXTURES_DIR/$id.wav"
    if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
        echo "skip $id.wav (exists)"
        continue
    fi

    tmp_wav="$(mktemp -t "auris_fixture_${id}").wav"
    kokoro-rs -q -v "$VOICE" -o "$tmp_wav" "$text"
    # kokoro-rs emits at its own sample rate; auris fixtures are 16 kHz mono.
    ffmpeg -y -loglevel error -i "$tmp_wav" -ar 16000 -ac 1 -c:a pcm_s16le "$out"
    rm -f "$tmp_wav"
    echo "made $id.wav"
done < "$TSV"

# --- non-spoken fixtures ---

# 1 second of digital silence at 16 kHz mono — the "reaches the recognizer
# but there's no speech" case.
make "$FIXTURES_DIR/silence.wav" \
    ffmpeg -y -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono \
    -t 1 -c:a pcm_s16le "$FIXTURES_DIR/silence.wav"

# The same words as `plain`, but 44.1 kHz stereo, to exercise the
# resample + downmix path. Regenerated from the same kokoro-rs output as
# `plain.wav` rather than re-synthesised, so both fixtures start from
# identical speech.
if [[ ! -f "$FIXTURES_DIR/stereo-44100.wav" || "$FORCE" -eq 1 ]]; then
    plain_text=$(awk -F'\t' '$1 == "plain" { print $2 }' "$TSV")
    tmp_wav="$(mktemp -t auris_fixture_stereo).wav"
    kokoro-rs -q -v "$VOICE" -o "$tmp_wav" "$plain_text"
    ffmpeg -y -loglevel error -i "$tmp_wav" -ar 44100 -ac 2 -c:a pcm_s16le \
        "$FIXTURES_DIR/stereo-44100.wav"
    rm -f "$tmp_wav"
    echo "made stereo-44100.wav"
else
    echo "skip stereo-44100.wav (exists)"
fi

# A few hundred bytes that are plainly not audio, generated deterministically.
if [[ ! -f "$FIXTURES_DIR/not-audio.bin" || "$FORCE" -eq 1 ]]; then
    cat > "$FIXTURES_DIR/not-audio.bin" <<'EOF'
This is not an audio file. It is a plain-text fixture used to exercise
auris's "input is not a wav stream" error path. Repeated to pad the file
out past a trivially small size.
This is not an audio file. It is a plain-text fixture used to exercise
auris's "input is not a wav stream" error path.
EOF
    echo "made not-audio.bin"
else
    echo "skip not-audio.bin (exists)"
fi

echo
echo "Duration table:"
printf '%-18s %-10s %s\n' "file" "seconds" "notes"
for f in "$FIXTURES_DIR"/*.wav; do
    name="$(basename "$f")"
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
    printf '%-18s %-10.2f\n' "$name" "$dur"
done
size=$(wc -c < "$FIXTURES_DIR/not-audio.bin" | tr -d ' ')
printf '%-18s %-10s %s bytes, not audio\n' "not-audio.bin" "-" "$size"
