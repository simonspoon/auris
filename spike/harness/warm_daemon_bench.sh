#!/usr/bin/env bash
# Measures whether task 951's acceptance property actually holds: once
# `auris serve` is up, a second (and third, ... eighth) utterance decodes at
# warm speed -- the ~4 s model load is paid once per daemon lifetime, not
# once per call.
#
# This is the END-TO-END client number: it necessarily includes process
# spawn, WAV read/resample, the socket round-trip, and process teardown on
# top of the decode itself, so it can never equal the spike's decode-only
# 0.082 RTF and is not trying to. The number directly comparable to 0.082 is
# `decode_only_warm_rtf_matches_spike_methodology` in src/engine.rs -- one
# `Recognizer::load`, then decode all 8 fixtures in-process with no spawn/IPC
# overhead at all (run it with `cargo test --release
# decode_only_warm_rtf_matches_spike_methodology -- --nocapture`).
#
# What THIS script measures: wall-clock time of the CLIENT process
# (`target/release/auris`, talking over the Unix socket) for each of the 8
# spike fixtures in spike/fixtures/wav/ (u01..u08), with --vocabulary-file
# spike/fixtures/hotwords.txt on every call, run sequentially against an
# already-warm `auris serve`. It reports:
#   - per-utterance wall time (ms)
#   - the first-utterance-vs-rest split (first call may still show connection
#     setup / OS warmup, though the daemon itself was already loaded before
#     any of these calls ran)
#   - total wall time / total fixture audio seconds, as the "daemon RTF" --
#     what mesa actually pays per call, spawn and all
#
# What it is compared against: the same 8 fixtures run with `--no-daemon`,
# which pays the ~4 s in-process model load on every single invocation (the
# behaviour auris would have with no daemon at all) -- same binary, same
# call, same per-call overhead, with vs without the warm recognizer. This is
# the honest "cold RTF" contrast for the daemon RTF above: any gap between
# the daemon RTF and 0.082 is per-call process/IPC overhead, not a cold
# recognizer, and the --no-daemon number is what shows that.
#
# Baseline this is checked against (spike/RESULTS.md SS6, SS4): warm in-process
# RTF 0.082 with per-word hotwords; spawned-fresh-per-utterance RTF 0.634;
# total fixture audio 45.43 s across the 8 files. Those numbers came from a
# Python spike calling sherpa-onnx directly, not from this binary -- this
# script produces auris's own numbers for the same shape of comparison, it
# does not reproduce the spike's numbers verbatim.
#
# Usage: spike/harness/warm_daemon_bench.sh
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE="$(cd "$HARNESS/.." && pwd)"
REPO="$(cd "$SPIKE/.." && pwd)"
WAV_DIR="$SPIKE/fixtures/wav"
HOTWORDS_FILE="$SPIKE/fixtures/hotwords.txt"
MODEL_SRC="$SPIKE/models/parakeet"
MODEL_NAME="parakeet-tdt-0.6b-v2-int8"

TMP_ROOT="${CLAUDE_JOB_DIR:-/tmp}/tmp/warm_daemon_bench"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

AURIS_HOME="$TMP_ROOT/home"
SOCKET="$AURIS_HOME/auris.sock"
MODEL_DIR="$AURIS_HOME/models/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

now_s() { python3 -c 'import time; print(time.time())'; }
elapsed_ms() { python3 -c "print(round(($2 - $1) * 1000, 1))"; }

BIN="$REPO/target/release/auris"

DAEMON_PID=""
cleanup() {
  if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    "$BIN" stop --socket "$SOCKET" >/dev/null 2>&1 || kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo ">>> cargo build --release"
(cd "$REPO" && cargo build --release)

echo ">>> setting up scratch AURIS_HOME at $AURIS_HOME (symlinks only, no copies)"
for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
  ln -sf "$MODEL_SRC/$f" "$MODEL_DIR/$f"
done
ln -sf "$MODEL_SRC/bpe_synth.vocab" "$MODEL_DIR/bpe.vocab"

export AURIS_HOME

IDS=(u01 u02 u03 u04 u05 u06 u07 u08)

echo ">>> computing fixture audio durations"
AUDIO_SECONDS_TOTAL=0
for id in "${IDS[@]}"; do
  d=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WAV_DIR/$id.wav")
  AUDIO_SECONDS_TOTAL=$(python3 -c "print($AUDIO_SECONDS_TOTAL + $d)")
done
echo "total fixture audio: ${AUDIO_SECONDS_TOTAL}s (documented baseline: 45.43s)"

# ---------------------------------------------------------------------------
# Phase 1: warm daemon. Start `auris serve` explicitly and wait for it to be
# ready via `auris status`, so none of the ~4 s load is folded into any of
# the timed client calls below.
# ---------------------------------------------------------------------------
echo
echo ">>> starting auris serve"
"$BIN" serve -m "$MODEL_NAME" --socket "$SOCKET" &
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

echo
echo ">>> daemon-warm pass: 8 client invocations against the already-warm daemon"
DAEMON_TOTAL_MS=0
DAEMON_FIRST_MS=0
for i in "${!IDS[@]}"; do
  id="${IDS[$i]}"
  t0=$(now_s)
  "$BIN" -q --socket "$SOCKET" --vocabulary-file "$HOTWORDS_FILE" "$WAV_DIR/$id.wav" >/dev/null
  t1=$(now_s)
  ms=$(elapsed_ms "$t0" "$t1")
  DAEMON_TOTAL_MS=$(python3 -c "print($DAEMON_TOTAL_MS + $ms)")
  if [[ $i -eq 0 ]]; then DAEMON_FIRST_MS="$ms"; fi
  printf '  %s  %8s ms\n' "$id" "$ms"
done

DAEMON_REST_MS=$(python3 -c "print($DAEMON_TOTAL_MS - $DAEMON_FIRST_MS)")
DAEMON_REST_COUNT=$(( ${#IDS[@]} - 1 ))
DAEMON_REST_AVG_MS=$(python3 -c "print(round($DAEMON_REST_MS / $DAEMON_REST_COUNT, 1))")
DAEMON_TOTAL_S=$(python3 -c "print($DAEMON_TOTAL_MS / 1000.0)")
DAEMON_RTF=$(python3 -c "print(round($DAEMON_TOTAL_S / $AUDIO_SECONDS_TOTAL, 4))")

echo
echo "daemon: first utterance ${DAEMON_FIRST_MS}ms; remaining $DAEMON_REST_COUNT avg ${DAEMON_REST_AVG_MS}ms"
echo "daemon: total wall ${DAEMON_TOTAL_MS}ms / ${AUDIO_SECONDS_TOTAL}s audio = RTF ${DAEMON_RTF}"

"$BIN" stop --socket "$SOCKET" >/dev/null 2>&1 || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

# ---------------------------------------------------------------------------
# Phase 2: --no-daemon, cold-every-time comparison. Each invocation loads its
# own in-process recognizer and pays the ~4 s load, matching the pre-daemon
# behaviour this task is measuring against.
# ---------------------------------------------------------------------------
echo
echo ">>> --no-daemon pass: 8 client invocations, each paying the in-process load"
NODAEMON_TOTAL_MS=0
for id in "${IDS[@]}"; do
  t0=$(now_s)
  "$BIN" -q --no-daemon --vocabulary-file "$HOTWORDS_FILE" -m "$MODEL_NAME" "$WAV_DIR/$id.wav" >/dev/null
  t1=$(now_s)
  ms=$(elapsed_ms "$t0" "$t1")
  NODAEMON_TOTAL_MS=$(python3 -c "print($NODAEMON_TOTAL_MS + $ms)")
  printf '  %s  %8s ms\n' "$id" "$ms"
done

NODAEMON_TOTAL_S=$(python3 -c "print($NODAEMON_TOTAL_MS / 1000.0)")
NODAEMON_RTF=$(python3 -c "print(round($NODAEMON_TOTAL_S / $AUDIO_SECONDS_TOTAL, 4))")

echo
echo "--no-daemon: total wall ${NODAEMON_TOTAL_MS}ms / ${AUDIO_SECONDS_TOTAL}s audio = RTF ${NODAEMON_RTF}"

echo
echo "=== summary ==="
echo "total fixture audio:        ${AUDIO_SECONDS_TOTAL}s (documented: 45.43s)"
echo "daemon (warm) RTF:          ${DAEMON_RTF}   (spike warm baseline: 0.082)"
echo "--no-daemon (cold) RTF:     ${NODAEMON_RTF}   (spike per-invocation baseline: 0.634)"
echo "daemon first utterance:     ${DAEMON_FIRST_MS}ms"
echo "daemon remaining avg:       ${DAEMON_REST_AVG_MS}ms"
