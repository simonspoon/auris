#!/usr/bin/env bash
# Synthesizes non-speech WAV fixtures used to exercise the Silero VAD gate
# (mesa task 968): audio that Parakeet hallucinates filler words on (a fan,
# a chair scrape, background hiss) but that is comfortably above auris's
# existing energy gate (`src/audio.rs::is_silent`, max 30 ms-window RMS
# below 1e-3) — so a clip landing here proves the VAD gate is doing new
# work, not just re-proving the energy gate.
#
# ALL THREE CLIPS ARE SYNTHETIC — none of this is a real room recording.
# Every one is generated from ffmpeg's `lavfi` noise sources (white/pink
# noise, filtered and enveloped). They are proxies for a fan, an HVAC
# rumble, and a chair scrape / knock, not recordings of one. Label them as
# synthetic wherever they're cited; for a genuine acoustic (speaker-
# playback-and-mic-recapture) recording, see bench/results/acoustic-wav/
# instead (bench/fixtures/README.md explains what that directory is).
#
# Target level: max 30 ms-window RMS (the same statistic `is_silent`
# computes) in 0.01-0.1, matching what mesa's browser-side onset gate
# (0.02) actually passes through. Amplitudes below are tuned to land
# around 0.04 and were verified against that exact algorithm, not asserted
# blind. Measured RMS per clip (also in tests/fixtures/README.md):
#
#   nonspeech-white.wav       0.040417  (40x the 1e-3 gate)
#   nonspeech-rumble.wav      0.034720  (35x the 1e-3 gate)
#   nonspeech-transient.wav   0.040754  (41x the 1e-3 gate)
#
# 16 kHz mono 16-bit PCM WAV throughout, matching this repo's other
# synthesize scripts (bench/corpus/synthesize.sh, scripts/generate-fixtures.sh).
# Idempotent: skips any wav that already exists. Re-run with --force to
# regenerate all.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$REPO/tests/fixtures"
mkdir -p "$OUT_DIR"

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

# Broadband white noise, 4 s — a room-hiss / air-handling proxy at
# speech-like level (amplitude 0.065 -> measured max-window RMS ~0.040).
make "$OUT_DIR/nonspeech-white.wav" \
    ffmpeg -y -loglevel error \
    -f lavfi -i "anoisesrc=color=white:sample_rate=16000:amplitude=0.065:duration=4" \
    -ac 1 -c:a pcm_s16le "$OUT_DIR/nonspeech-white.wav"

# Low-frequency rumble, 4 s — a fan / HVAC proxy: pink noise, lowpassed to
# emphasize the low end (amplitude 0.11 -> measured max-window RMS ~0.038).
make "$OUT_DIR/nonspeech-rumble.wav" \
    ffmpeg -y -loglevel error \
    -f lavfi -i "anoisesrc=color=pink:sample_rate=16000:amplitude=0.11:duration=4" \
    -af "lowpass=f=250" \
    -ac 1 -c:a pcm_s16le "$OUT_DIR/nonspeech-rumble.wav"

# Impulsive transient, 4 s total — a chair-scrape / knock / cough proxy: a
# 150 ms filtered noise burst (fast attack, slower decay) against an
# otherwise-silent floor, so the loudest 30 ms window sits inside the
# burst, not smeared across the whole clip (amplitude 0.07 -> measured
# max-window RMS ~0.041).
if [[ ! -f "$OUT_DIR/nonspeech-transient.wav" || "$FORCE" -eq 1 ]]; then
    ffmpeg -y -loglevel error \
        -f lavfi -i "anullsrc=r=16000:cl=mono:d=1.5" \
        -f lavfi -i "anoisesrc=color=white:sample_rate=16000:amplitude=0.07:duration=0.15" \
        -f lavfi -i "anullsrc=r=16000:cl=mono:d=2.35" \
        -filter_complex "[1:a]afade=t=in:st=0:d=0.005,afade=t=out:st=0.05:d=0.1[burst];[0:a][burst][2:a]concat=n=3:v=0:a=1[out]" \
        -map "[out]" -ac 1 -c:a pcm_s16le "$OUT_DIR/nonspeech-transient.wav"
    echo "made nonspeech-transient.wav"
else
    echo "skip nonspeech-transient.wav (exists)"
fi

echo
echo "Duration table:"
printf '%-24s %-10s\n' "file" "seconds"
for f in "$OUT_DIR"/nonspeech-*.wav; do
    [[ -f "$f" ]] || continue
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f")
    printf '%-24s %-10.2f\n' "$(basename "$f")" "$dur"
done
