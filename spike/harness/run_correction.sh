#!/usr/bin/env bash
# Vocabulary-correction measurement harness (mesa task 922 port, auris side).
#
# For every existing raw ASR transcript (spike/results/raw/*.tsv and
# spike/results/hotwords/*.tsv), applies the post-ASR vocabulary correction
# pass (spike/harness/vocab_correct.py) and rescores the result, so the
# before/after effect of the correction pass on WER and name accuracy can be
# read off directly.
#
# Usage:
#   spike/harness/run_correction.sh
#
# Writes spike/results/corrected/<config>.tsv and
# spike/results/corrected/<config>.score.json for each config, and prints a
# before/after table to stdout.
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE="$(cd "$HARNESS/.." && pwd)"
REF_TSV="$SPIKE/fixtures/utterances.tsv"
HOTWORDS="$SPIKE/fixtures/hotwords.txt"
CORRECTED="$SPIKE/results/corrected"
mkdir -p "$CORRECTED"

PY="${PYTHON:-python3}"

configs=()
raw_tsvs=()
for f in "$SPIKE"/results/raw/*.tsv "$SPIKE"/results/hotwords/*.tsv; do
  [[ -e "$f" ]] || continue
  config="$(basename "$f" .tsv)"
  configs+=("$config")
  raw_tsvs+=("$f")
done

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "no raw transcripts found under $SPIKE/results/{raw,hotwords}" >&2
  exit 1
fi

for i in "${!configs[@]}"; do
  config="${configs[$i]}"
  raw="${raw_tsvs[$i]}"
  echo ">>> $config" >&2

  corrected_tsv="$CORRECTED/$config.tsv"
  "$PY" "$HARNESS/vocab_correct.py" --hotwords "$HOTWORDS" "$raw" > "$corrected_tsv"

  "$PY" "$HARNESS/score.py" "$REF_TSV" "$corrected_tsv" > "$CORRECTED/$config.score.json"
done

# --- before/after table ------------------------------------------------------

"$PY" - "$SPIKE" "${configs[@]}" <<'PYEOF'
import json
import sys

spike = sys.argv[1]
configs = sys.argv[2:]

def find_before_score(spike, config):
    for sub in ("raw", "hotwords"):
        path = f"{spike}/results/{sub}/{config}.score.json"
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return None

rows = []
for config in configs:
    before = find_before_score(spike, config)
    with open(f"{spike}/results/corrected/{config}.score.json") as f:
        after = json.load(f)
    if before is None:
        continue
    rows.append((config, before, after))

def fmt_pct(x):
    return f"{x*100:.1f}%"

print()
print(f"{'config':<32}{'WER before':>12}{'WER after':>12}{'dWER':>10}{'F1 before':>12}{'F1 after':>11}{'dF1':>10}")
print("-" * 99)
for config, before, after in rows:
    wb, wa = before["wer"], after["wer"]
    fb, fa = before["name_accuracy"], after["name_accuracy"]
    dwer = wa - wb
    df1 = fa - fb
    print(f"{config:<32}{fmt_pct(wb):>12}{fmt_pct(wa):>12}{dwer*100:>+9.1f}p{fmt_pct(fb):>12}{fmt_pct(fa):>11}{df1*100:>+9.1f}p")
PYEOF
