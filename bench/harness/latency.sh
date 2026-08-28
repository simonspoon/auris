#!/usr/bin/env bash
# Warm-daemon and cold (--no-daemon) latency measurement for auris, against
# the 40-utterance bench/corpus fixture set.
#
# Measures the CLIENT process wall-clock (spawn, WAV read, socket round
# trip, decode, teardown) via python3 time.perf_counter() around
# subprocess.run, not the shell `time` builtin. Every call passes
# --vocabulary-file spike/fixtures/hotwords.txt.
#
# Warm pass: start `auris serve`, wait for `auris status` to report ready,
# sleep 2s so model load is definitely not counted, do 3 discarded warm-up
# calls, then run each of the 40 utterances 3 times (120 measurements) against
# the warm daemon. Writes every raw measurement to
# bench/results/latency-raw.tsv (id, rep, ms, audio_seconds) and summary
# stats to bench/results/latency.json.
#
# Cold pass: the same 40 utterances with --no-daemon, once each (reloads the
# model every call), into bench/results/latency-cold.json.
#
# Usage: bench/harness/latency.sh
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HARNESS/.." && pwd)"
REPO="$(cd "$BENCH/.." && pwd)"
WAV_DIR="$BENCH/corpus/wav"
REF_TSV="$BENCH/corpus/utterances.tsv"
HOTWORDS_FILE="$REPO/spike/fixtures/hotwords.txt"
MODEL_DIR="$REPO/spike/models/parakeet"
RESULTS="$BENCH/results"
mkdir -p "$RESULTS"

BIN="$REPO/target/release/auris"

TMP_ROOT="${CLAUDE_JOB_DIR:-/tmp}/tmp/latency"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"
SOCKET="$TMP_ROOT/auris.sock"

DAEMON_PID=""
cleanup() {
  if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    "$BIN" stop --socket "$SOCKET" >/dev/null 2>&1 || kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

IDS=()
while IFS=$'\t' read -r id _text; do IDS+=("$id"); done < "$REF_TSV"

echo ">>> computing per-utterance audio durations"
# macOS ships bash 3.2, which has no associative arrays -- write id->duration
# to a lookup file instead of declare -A.
DURATIONS_TSV="$TMP_ROOT/durations.tsv"
: > "$DURATIONS_TSV"
for id in "${IDS[@]}"; do
  d=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WAV_DIR/$id.wav")
  printf '%s\t%s\n' "$id" "$d" >> "$DURATIONS_TSV"
done
audio_seconds_for() {
  awk -F'\t' -v id="$1" '$1==id{print $2; exit}' "$DURATIONS_TSV"
}

echo ">>> starting auris serve"
"$BIN" serve -m "$MODEL_DIR" --socket "$SOCKET" &
DAEMON_PID=$!

deadline=$(( $(date +%s) + 30 ))
until "$BIN" status --socket "$SOCKET" >/dev/null 2>&1; do
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "auris serve did not become ready within 30s" >&2
    exit 1
  fi
  sleep 0.2
done
echo "daemon ready: $("$BIN" status --socket "$SOCKET")"

echo ">>> sleeping 2s so model load is definitely finished and not counted"
sleep 2

echo ">>> 3 discarded warm-up calls"
for i in 1 2 3; do
  "$BIN" -q -m "$MODEL_DIR" --socket "$SOCKET" --vocabulary-file "$HOTWORDS_FILE" "$WAV_DIR/${IDS[0]}.wav" >/dev/null
done

echo ">>> warm pass: 40 utterances x 3 reps = 120 client invocations against the warm daemon"
RAW_TSV="$RESULTS/latency-raw.tsv"
: > "$RAW_TSV"

FAILURES="$TMP_ROOT/failures.tsv"
: > "$FAILURES"

for id in "${IDS[@]}"; do
  wav="$WAV_DIR/$id.wav"
  audio_s="$(audio_seconds_for "$id")"
  for rep in 1 2 3; do
    out_file="$TMP_ROOT/${id}_${rep}.out"
    read -r ms status < <(python3 - "$BIN" -m "$MODEL_DIR" --socket "$SOCKET" --vocabulary-file "$HOTWORDS_FILE" -q "$wav" "$out_file" <<'PYEOF'
import subprocess
import sys
import time

bin_path, *rest = sys.argv[1:]
out_file = rest[-1]
args = rest[:-1]
t0 = time.perf_counter()
with open(out_file, "wb") as f:
    proc = subprocess.run([bin_path, *args], stdout=f, stderr=subprocess.DEVNULL)
t1 = time.perf_counter()
ms = (t1 - t0) * 1000.0
print(f"{ms:.3f} {proc.returncode}")
PYEOF
)
    printf '%s\t%s\t%s\t%s\n' "$id" "$rep" "$ms" "$audio_s" >> "$RAW_TSV"
    if [[ "$status" != "0" ]] || [[ ! -s "$out_file" ]]; then
      printf '%s\trep%s\texit=%s\tempty=%s\n' "$id" "$rep" "$status" "$([[ -s "$out_file" ]] && echo no || echo yes)" >> "$FAILURES"
    fi
  done
done
echo "warm pass done: $(wc -l < "$RAW_TSV" | tr -d ' ') measurements"

if [[ -s "$FAILURES" ]]; then
  echo ">>> WARM PASS FAILURES:"
  cat "$FAILURES"
fi

python3 - "$RAW_TSV" "$RESULTS/latency.json" <<'PYEOF'
import csv
import json
import statistics
import sys

raw_path, out_path = sys.argv[1:]
rows = []
with open(raw_path) as f:
    for line in f:
        id_, rep, ms, audio_s = line.rstrip("\n").split("\t")
        rows.append((id_, int(rep), float(ms), float(audio_s)))

ms_values = [r[2] for r in rows]
ms_per_audio_s = [r[2] / r[3] for r in rows if r[3] > 0]
total_ms = sum(ms_values)
total_audio_s = sum(r[3] for r in rows)

def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def stats_block(values):
    return {
        "n": len(values),
        "min": min(values),
        "p50": pct(values, 50),
        "p75": pct(values, 75),
        "p90": pct(values, 90),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

out = {
    "warm_client_ms": stats_block(ms_values),
    "warm_ms_per_audio_second": stats_block(ms_per_audio_s),
    "daemon_rtf_total": total_ms / 1000.0 / total_audio_s,
    "total_client_ms": total_ms,
    "total_audio_seconds": total_audio_s,
    "fraction_over_300ms": sum(1 for v in ms_values if v > 300) / len(ms_values),
    "count_over_300ms": sum(1 for v in ms_values if v > 300),
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PYEOF

echo ">>> stopping daemon"
"$BIN" stop --socket "$SOCKET" >/dev/null 2>&1 || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

echo ">>> cold pass: 40 utterances, --no-daemon, once each (reloads model every call)"
COLD_RAW="$TMP_ROOT/latency-cold-raw.tsv"
: > "$COLD_RAW"
COLD_FAILURES="$TMP_ROOT/cold_failures.tsv"
: > "$COLD_FAILURES"

for id in "${IDS[@]}"; do
  wav="$WAV_DIR/$id.wav"
  audio_s="$(audio_seconds_for "$id")"
  out_file="$TMP_ROOT/cold_${id}.out"
  read -r ms status < <(python3 - "$BIN" --no-daemon -m "$MODEL_DIR" --vocabulary-file "$HOTWORDS_FILE" -q "$wav" "$out_file" <<'PYEOF'
import subprocess
import sys
import time

bin_path, *rest = sys.argv[1:]
out_file = rest[-1]
args = rest[:-1]
t0 = time.perf_counter()
with open(out_file, "wb") as f:
    proc = subprocess.run([bin_path, *args], stdout=f, stderr=subprocess.DEVNULL)
t1 = time.perf_counter()
ms = (t1 - t0) * 1000.0
print(f"{ms:.3f} {proc.returncode}")
PYEOF
)
  printf '%s\t%s\t%s\n' "$id" "$ms" "$audio_s" >> "$COLD_RAW"
  if [[ "$status" != "0" ]] || [[ ! -s "$out_file" ]]; then
    printf '%s\texit=%s\tempty=%s\n' "$id" "$status" "$([[ -s "$out_file" ]] && echo no || echo yes)" >> "$COLD_FAILURES"
  fi
done
echo "cold pass done: $(wc -l < "$COLD_RAW" | tr -d ' ') measurements"

if [[ -s "$COLD_FAILURES" ]]; then
  echo ">>> COLD PASS FAILURES:"
  cat "$COLD_FAILURES"
fi

python3 - "$COLD_RAW" "$RESULTS/latency-cold.json" <<'PYEOF'
import json
import statistics
import sys

raw_path, out_path = sys.argv[1:]
rows = []
with open(raw_path) as f:
    for line in f:
        id_, ms, audio_s = line.rstrip("\n").split("\t")
        rows.append((id_, float(ms), float(audio_s)))

ms_values = [r[1] for r in rows]
ms_per_audio_s = [r[1] / r[2] for r in rows if r[2] > 0]
total_ms = sum(ms_values)
total_audio_s = sum(r[2] for r in rows)

def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def stats_block(values):
    return {
        "n": len(values),
        "min": min(values),
        "p50": pct(values, 50),
        "p75": pct(values, 75),
        "p90": pct(values, 90),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

out = {
    "cold_client_ms": stats_block(ms_values),
    "cold_ms_per_audio_second": stats_block(ms_per_audio_s),
    "cold_rtf_total": total_ms / 1000.0 / total_audio_s,
    "total_client_ms": total_ms,
    "total_audio_seconds": total_audio_s,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PYEOF

echo ">>> latency.sh done. bench/results/latency.json, bench/results/latency-cold.json, bench/results/latency-raw.tsv written."
