#!/usr/bin/env bash
# auris decode + score benchmark against the 40-utterance bench/corpus set.
#
# Runs three configs SEQUENTIALLY (never in parallel -- parallelism poisons
# timings), each via `auris --no-daemon` against spike/models/parakeet:
#   1. auris-plain            -- no vocabulary file (unbiased baseline)
#   2. auris-vocab             -- --vocabulary-file bench/fixtures/hotwords.txt
#                                  (what the crate ships today)
#   3. auris-vocab-corrected   -- config 2's transcripts passed through
#                                  bench/harness/vocab_correct.py. This
#                                  correction pass is NOT in the Rust crate --
#                                  it lives only in the spike harness (see the
#                                  "correction pass" grep note this script
#                                  prints at the top of its run).
#
# Writes id<TAB>text transcript TSVs and score.py output under bench/results/,
# then calls bench/harness/latency.sh for the client-latency measurement.
#
# Transcript capture: a single line per utterance with embedded tabs/
# newlines collapsed to spaces, but punctuation PRESERVED (needed by
# score.py's punctuation-boundary metric). spike/harness/_clean_transcript.py
# is NOT reused here: it strips whisper-cli's bracketed timestamp lines and
# joins whisper's multi-line stdout, a shape auris's stdout never has (README
# "stdout": auris writes exactly one transcript line, no brackets, no
# decoration). Reusing a whisper-shaped cleaner for auris output would be
# solving a problem auris doesn't have, so this script instead only strips
# the trailing newline and collapses any embedded whitespace runs to a
# single space -- nothing that touches punctuation.
#
# Usage: bench/harness/run_auris.sh
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HARNESS/.." && pwd)"
REPO="$(cd "$BENCH/.." && pwd)"
WAV_DIR="$BENCH/corpus/wav"
REF_TSV="$BENCH/corpus/utterances.tsv"
HOTWORDS_FILE="$REPO/bench/fixtures/hotwords.txt"
MODEL_DIR="$REPO/spike/models/parakeet"
RESULTS="$BENCH/results"
mkdir -p "$RESULTS"

BIN="$REPO/target/release/auris"

echo ">>> verifying the correction pass is not in the Rust crate (src/)"
# grep for an actual implementation (a fn definition), not doc-comment
# mentions of the external script -- src/vocabulary.rs's own doc comment
# references "vocab_correct.py" by name while explaining that the pass lives
# outside the crate, which would otherwise false-positive this check.
if grep -rl "fn correct_text\|fn sound_key\|fn soundKey" "$REPO/src" >/dev/null 2>&1; then
  echo "!!! found a correction-pass implementation under src/ -- claim in this script's header is now false" >&2
else
  echo "    confirmed: no correct_text/sound_key implementation in src/ -- the correction pass" \
       "lives only in bench/harness/vocab_correct.py (src/vocabulary.rs says so in its own" \
       "doc comment: \"The post-ASR correction pass (docs/correction.md) today lives outside" \
       "the crate, as the spike vocab_correct.py\")"
fi

IDS=()
while IFS=$'\t' read -r id _text; do IDS+=("$id"); done < "$REF_TSV"

clean_transcript() {
  # stdin: raw auris stdout for one utterance. Strip trailing newline,
  # collapse embedded whitespace runs to a single space. Punctuation is
  # untouched.
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
  local out_tsv="$RESULTS/$name.tsv"
  echo ">>> decoding $name"
  : > "$out_tsv"
  for id in "${IDS[@]}"; do
    local wav="$WAV_DIR/$id.wav"
    local raw
    raw=$("$BIN" --no-daemon -q -m "$MODEL_DIR" ${extra_args[@]+"${extra_args[@]}"} "$wav")
    local cleaned
    cleaned=$(printf '%s' "$raw" | clean_transcript)
    printf '%s\t%s\n' "$id" "$cleaned" >> "$out_tsv"
  done
}

score_config() {
  local name="$1"
  local tsv="$RESULTS/$name.tsv"
  local csv="$RESULTS/$name.csv"
  local score_json="$RESULTS/$name.score.json"
  echo ">>> scoring $name"
  python3 "$BENCH/harness/score.py" "$REF_TSV" "$tsv" --csv "$csv" > "$score_json"
}

# --- config 1: plain, no vocabulary file -----------------------------------
decode_config "auris-plain"
score_config "auris-plain"

# --- config 2: with vocabulary file -----------------------------------------
decode_config "auris-vocab" --vocabulary-file "$HOTWORDS_FILE"
score_config "auris-vocab"

# --- config 3: config 2's transcripts through vocab_correct.py -------------
echo ">>> correcting auris-vocab through bench/harness/vocab_correct.py"
python3 "$REPO/bench/harness/vocab_correct.py" --hotwords "$HOTWORDS_FILE" \
  "$RESULTS/auris-vocab.tsv" > "$RESULTS/auris-vocab-corrected.tsv"
score_config "auris-vocab-corrected"

# --- headline numbers --------------------------------------------------------
echo
echo ">>> headline numbers"
python3 - "$RESULTS" <<'PYEOF'
import json
import sys

results = sys.argv[1]
configs = ["auris-plain", "auris-vocab", "auris-vocab-corrected"]

print(f"{'config':<24}{'WER':>8}{'name_F1':>10}{'name_P':>9}{'name_R':>9}{'punct_F1':>10}{'any_term_punct':>16}")
for config in configs:
    with open(f"{results}/{config}.score.json") as f:
        d = json.load(f)
    print(
        f"{config:<24}{d['wer']*100:>7.1f}%{d['name_accuracy']*100:>9.1f}%"
        f"{d['name_precision']*100:>8.1f}%{d['name_recall']*100:>8.1f}%"
        f"{d['punct_boundary_f1']*100:>9.1f}%{d['any_terminal_punct_rate']*100:>15.1f}%"
    )
PYEOF

# --- latency -----------------------------------------------------------------
echo
echo ">>> running latency.sh"
"$HARNESS/latency.sh"

echo
echo ">>> run_auris.sh done"
