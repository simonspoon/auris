#!/usr/bin/env bash
# Decode + score auris against an arbitrary directory of WAVs, keyed by id
# against bench/corpus/utterances.tsv. This is the "equivalent auris
# --no-daemon loop" bench/harness/webspeech/README.md points at but never
# wrote down: bench/harness/run_auris.sh only ever decodes the clean
# bench/corpus/wav/ corpus, so the acoustic half of the Web Speech benchmark
# had no reproducible way to get a matching auris number.
#
# THE FAIRNESS RULE THIS SCRIPT EXISTS FOR (bench/harness/webspeech/README.md
# "Why a virtual audio loopback is required"): when Web Speech is captured in
# `--mode acoustic`, it is scored on room audio -- whatever the microphone
# actually picked up after a trip through the speakers, degraded in ways
# that don't reproduce. If auris is then scored against the original clean
# bench/corpus/wav/*.wav, the comparison is not "auris vs. Web Speech", it's
# "auris on a clean signal vs. Web Speech on a degraded one" -- a rigged
# benchmark that makes Web Speech look worse for a reason that has nothing
# to do with either engine. auris MUST be rescored against the same
# `bench/results/acoustic-wav/` recordings before that comparison is drawn.
# Never compare an acoustic-mode Web Speech score against the committed
# clean-corpus auris-*.score.json files.
#
# Runs the same three configs as run_auris.sh, SEQUENTIALLY (never in
# parallel -- parallel decodes would contend for CPU and perturb the very
# latency-sensitive capture this script's output gets compared against, and
# run_auris.sh's own sequential discipline is what the committed numbers
# were measured under; a parallel rerun here would no longer be comparable
# to them):
#   1. auris-plain-<suffix>            -- no vocabulary file
#   2. auris-vocab-<suffix>             -- --vocabulary-file bench/fixtures/hotwords.txt
#   3. auris-vocab-corrected-<suffix>   -- config 2 through bench/harness/vocab_correct.py
#
# This script is decode-and-score ONLY -- it does not touch latency. Run
# bench/harness/latency.sh separately if a latency number against this WAV
# directory is ever needed; folding it in here would make a WAV-agnostic
# reuse of this script implicitly re-time whatever hardware happens to be
# under the acoustic capture, which is not this script's job.
#
# Usage: bench/harness/rescore_wavs.sh WAV_DIR SUFFIX
#   e.g.  bench/harness/rescore_wavs.sh bench/results/acoustic-wav acoustic
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 WAV_DIR SUFFIX" >&2
  exit 2
fi
WAV_DIR="$1"
SUFFIX="$2"

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HARNESS/.." && pwd)"
REPO="$(cd "$BENCH/.." && pwd)"
REF_TSV="$BENCH/corpus/utterances.tsv"
HOTWORDS_FILE="$REPO/bench/fixtures/hotwords.txt"
MODEL_DIR="$REPO/spike/models/parakeet"
RESULTS="$BENCH/results"
mkdir -p "$RESULTS"

BIN="$REPO/target/release/auris"

# ids present in WAV_DIR, restricted to those also in the reference corpus,
# in REF_TSV's own order. A WAV_DIR that only covers a subset of the 40 (a
# partial re-recording) is scored against just that subset's reference rows
# below -- scoring a subset against the full 40-line reference would count
# every unrecorded id as a 100%-deletion miss (score.py always iterates the
# full reference it's given), silently wrecking the WER of a deliberate
# partial run.
#
# THE COMPARABILITY RULE THIS FILTERING MUST NOT VIOLATE: both engines in a
# comparison have to be scored against the IDENTICAL reference set, or the
# comparison is rigged by exactly the same mechanism as the clean-vs-room-
# audio trap above, just subtler. If this script's WAV_DIR ends up covering
# fewer than the full 40 ids (an aborted utterance, a failed recording,
# whatever the reason), filtering the reference *here* only fixes auris's
# side -- the Web Speech transcript being compared against MUST be filtered
# to that exact same id set before scoring, not left scored against all 40.
# Never let one engine's score quietly cover more or fewer utterances than
# the other's. When WAV_DIR covers all 40, this filtering is a no-op and
# there is nothing to reconcile -- that is the expected, common case; a
# short WAV_DIR is the exception that needs a human to explicitly state the
# excluded ids before the two scores are compared, not something this
# script papers over on its own.
IDS=()
while IFS=$'\t' read -r id _text; do
  if [ -f "$WAV_DIR/$id.wav" ]; then
    IDS+=("$id")
  fi
done < "$REF_TSV"

if [ "${#IDS[@]}" -eq 0 ]; then
  echo "no .wav files in $WAV_DIR match any id in $REF_TSV" >&2
  exit 1
fi
echo ">>> rescoring ${#IDS[@]} id(s) from $WAV_DIR: ${IDS[*]}"

TOTAL_REF_IDS=$(wc -l < "$REF_TSV" | tr -d ' ')
if [ "${#IDS[@]}" -ne "$TOTAL_REF_IDS" ]; then
  echo ">>> WARNING: $WAV_DIR covers ${#IDS[@]}/$TOTAL_REF_IDS reference ids." \
    "auris is being scored against a FILTERED reference (see comment above)." \
    "Whatever this is being compared against (e.g. a Web Speech score) MUST" \
    "be rescored against this same ${#IDS[@]}-id set, or the comparison is" \
    "rigged. State the excluded ids explicitly wherever these numbers are" \
    "reported." >&2
fi

FILTERED_REF="$RESULTS/.ref-$SUFFIX.tsv"
: > "$FILTERED_REF"
while IFS=$'\t' read -r id text; do
  for wanted in "${IDS[@]}"; do
    if [ "$id" = "$wanted" ]; then
      printf '%s\t%s\n' "$id" "$text" >> "$FILTERED_REF"
      break
    fi
  done
done < "$REF_TSV"
REF_TSV="$FILTERED_REF"

clean_transcript() {
  # identical to run_auris.sh's clean_transcript: strip trailing newline,
  # collapse embedded whitespace runs to a single space, punctuation
  # untouched -- see that script's header for why a whisper-shaped cleaner
  # is not reused here.
  python3 -c '
import sys
text = sys.stdin.read()
print(" ".join(text.split()))
'
}

decode_config() {
  local name="$1"
  shift
  local extra_args=("$@")
  local out_tsv="$RESULTS/$name-$SUFFIX.tsv"
  echo ">>> decoding $name-$SUFFIX"
  : > "$out_tsv"
  for id in "${IDS[@]}"; do
    local wav="$WAV_DIR/$id.wav"
    local raw exit_code
    # Room audio can legitimately trip auris's own silence gate
    # (audio::is_silent) on a quiet or noisy recording that the clean
    # corpus never does -- README "Exit codes": exit 1 means
    # NOTHING_TRANSCRIBED, with empty stdout, and that is a real "" hyp for
    # this utterance (score.py scores it as a full deletion, not a
    # skipped row -- see bench/harness/score.py's read_tsv). Under
    # `set -e`, letting that exit code propagate from a command
    # substitution would abort the whole run instead of recording the
    # empty transcript, so it's caught here explicitly. Any OTHER nonzero
    # exit (2 usage, 130 interrupted, a crash) is a real failure and must
    # still abort loudly.
    set +e
    raw=$("$BIN" --no-daemon -q -m "$MODEL_DIR" ${extra_args[@]+"${extra_args[@]}"} "$wav")
    exit_code=$?
    set -e
    if [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne 1 ]; then
      echo "auris exited $exit_code (not 0 or 1/NOTHING_TRANSCRIBED) on $wav -- aborting" >&2
      exit "$exit_code"
    fi
    local cleaned
    cleaned=$(printf '%s' "$raw" | clean_transcript)
    printf '%s\t%s\n' "$id" "$cleaned" >> "$out_tsv"
  done
}

score_config() {
  local name="$1"
  local tsv="$RESULTS/$name-$SUFFIX.tsv"
  local csv="$RESULTS/$name-$SUFFIX.csv"
  local score_json="$RESULTS/$name-$SUFFIX.score.json"
  echo ">>> scoring $name-$SUFFIX"
  python3 "$BENCH/harness/score.py" "$REF_TSV" "$tsv" --csv "$csv" > "$score_json"
}

# --- config 1: plain, no vocabulary file -----------------------------------
decode_config "auris-plain"
score_config "auris-plain"

# --- config 2: with vocabulary file -----------------------------------------
decode_config "auris-vocab" --vocabulary-file "$HOTWORDS_FILE"
score_config "auris-vocab"

# --- config 3: config 2's transcripts through vocab_correct.py -------------
echo ">>> correcting auris-vocab-$SUFFIX through bench/harness/vocab_correct.py"
python3 "$REPO/bench/harness/vocab_correct.py" --hotwords "$HOTWORDS_FILE" \
  "$RESULTS/auris-vocab-$SUFFIX.tsv" > "$RESULTS/auris-vocab-corrected-$SUFFIX.tsv"
score_config "auris-vocab-corrected"

rm -f "$FILTERED_REF"

echo
echo ">>> rescore_wavs.sh done ($SUFFIX, ${#IDS[@]} id(s), results in $RESULTS)"
