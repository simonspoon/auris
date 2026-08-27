#!/usr/bin/env bash
# ASR engine benchmark: whisper.cpp (whisper-cli) vs sherpa-onnx (parakeet).
#
# Runs 9 configs SEQUENTIALLY (never in parallel -- parallelism would poison
# timings), each measuring warm decode wall-clock/RTF, a cold-first-decode
# number, peak RSS, and WER/name-accuracy via score.py. Writes per-config
# transcripts + scores under spike/results/raw/ and a merged summary at
# spike/results/summary.json.
#
# Usage:
#   spike/harness/run_bench.sh              # full 9-config benchmark
#   spike/harness/run_bench.sh --smoke      # 1 config (whisper-tiny.en-noprompt),
#                                            # 1 fixture (u01) -- plumbing check only
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE="$(cd "$HARNESS/.." && pwd)"
WAV_DIR="$SPIKE/fixtures/wav"
REF_TSV="$SPIKE/fixtures/utterances.tsv"
WHISPER_MODELS="$SPIKE/models/whisper"
PARAKEET_MODELS="$SPIKE/models/parakeet"
RESULTS="$SPIKE/results"
RAW="$RESULTS/raw"
mkdir -p "$RAW"

TMP="${CLAUDE_JOB_DIR:-/tmp}/tmp/run_bench"
rm -rf "$TMP"
mkdir -p "$TMP"

WHISPER_BIN="${WHISPER_BIN:-whisper-cli}"
PARAKEET_PY="$HARNESS/.venv/bin/python"
THREADS=8
PROMPT_TEXT="mesa, auris, khora, qorvex, helios, kokoro"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

now() { python3 -c 'import time; print(time.time())'; }

# --- audio duration (seconds), summed over the fixture set ---------------
compute_audio_seconds() {
  local dir="$1"
  for f in "$dir"/*.wav; do
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f"
  done | python3 -c "import sys; print(sum(float(x) for x in sys.stdin))"
}

# --- parse whisper-cli's stderr timing block ------------------------------
# echoes "LOAD_MS TOTAL_MS" (empty strings if a line was missing, e.g. -np)
parse_whisper_timing() {
  local errfile="$1"
  local load_ms total_ms
  load_ms=$(grep -oE 'load time *= *[0-9.]+ ms' "$errfile" | grep -oE '[0-9.]+' | head -1 || true)
  total_ms=$(grep -oE '^whisper_print_timings: *total time *= *[0-9.]+ ms' "$errfile" | grep -oE '[0-9.]+' | head -1 || true)
  echo "${load_ms:-} ${total_ms:-}"
}

clean_transcript() { python3 "$HARNESS/_clean_transcript.py"; }

write_fragment() { python3 "$HARNESS/_write_fragment.py" "$@"; }

file_size_mb() { python3 -c "import os; print(round(os.path.getsize('$1')/1e6, 2))"; }

# ---------------------------------------------------------------------------
# whisper config: NAME MODEL_PATH USE_PROMPT(0/1)
# ---------------------------------------------------------------------------
bench_whisper_config() {
  local config="$1" model_path="$2" use_prompt="$3"
  echo ">>> $config"

  local hyp_tsv="$RAW/$config.tsv"
  : > "$hyp_tsv"

  local -a base_args=(-m "$model_path" -t "$THREADS" -nt)
  local -a prompt_args=()
  if [[ "$use_prompt" == "1" ]]; then
    prompt_args=(--prompt "$PROMPT_TEXT")
  fi

  local ids=()
  while IFS=$'\t' read -r id _ref; do ids+=("$id"); done < "$REF_TSV"
  if [[ "$SMOKE" == "1" ]]; then ids=("${ids[0]}"); fi

  # --- COLD: very first invocation, before any warmup. Also captures peak
  # RSS via /usr/bin/time -l for this same invocation. ---
  local first_wav="$WAV_DIR/${ids[0]}.wav"
  local cold_time_file="$TMP/${config}_cold.time"
  local cold_txt="$TMP/${config}_cold.txt"
  /usr/bin/time -l "$WHISPER_BIN" "${base_args[@]}" ${prompt_args[@]+"${prompt_args[@]}"} -f "$first_wav" \
    >"$cold_txt" 2>"$cold_time_file"

  local cold_seconds
  cold_seconds=$(grep -oE '^ *[0-9.]+ real' "$cold_time_file" | grep -oE '[0-9.]+' | head -1)
  local peak_rss_bytes
  peak_rss_bytes=$(grep 'maximum resident set size' "$cold_time_file" | awk '{print $1}')
  local peak_rss_mb
  peak_rss_mb=$(python3 -c "print(round($peak_rss_bytes/1e6, 2))")

  # --- WARMUP: discarded pass over the fixture set (file cache goes hot) ---
  for id in "${ids[@]}"; do
    "$WHISPER_BIN" "${base_args[@]}" ${prompt_args[@]+"${prompt_args[@]}"} -f "$WAV_DIR/$id.wav" \
      >/dev/null 2>/dev/null || true
  done

  # --- MEASURED WARM PASS ---
  local total_decode_ms=0
  local have_decode_only=1
  local t_start t_end
  t_start=$(now)
  for id in "${ids[@]}"; do
    local txt_file="$TMP/${config}_${id}.txt"
    local err_file="$TMP/${config}_${id}.err"
    "$WHISPER_BIN" "${base_args[@]}" ${prompt_args[@]+"${prompt_args[@]}"} -f "$WAV_DIR/$id.wav" \
      >"$txt_file" 2>"$err_file"

    local timing tot_ms
    timing=$(parse_whisper_timing "$err_file")
    tot_ms=$(echo "$timing" | awk '{print $2}')
    if [[ -n "$tot_ms" ]]; then
      total_decode_ms=$(python3 -c "print($total_decode_ms + $tot_ms)")
    else
      have_decode_only=0
    fi

    local hyp_text
    hyp_text=$(clean_transcript < "$txt_file")
    printf '%s\t%s\n' "$id" "$hyp_text" >> "$hyp_tsv"
  done
  t_end=$(now)

  local warm_decode_seconds
  warm_decode_seconds=$(python3 -c "print($t_end - $t_start)")

  local audio_seconds
  if [[ "$SMOKE" == "1" ]]; then
    audio_seconds=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$first_wav")
  else
    audio_seconds="$AUDIO_SECONDS_TOTAL"
  fi

  local rtf_warm
  rtf_warm=$(python3 -c "print($warm_decode_seconds / $audio_seconds)")

  local decode_only_frag_args=()
  if [[ "$have_decode_only" == "1" ]]; then
    local decode_only_seconds rtf_decode_only
    decode_only_seconds=$(python3 -c "print($total_decode_ms / 1000.0)")
    rtf_decode_only=$(python3 -c "print($decode_only_seconds / $audio_seconds)")
    decode_only_frag_args=(rtf_decode_only="$rtf_decode_only" decode_only_seconds="$decode_only_seconds")
  fi

  local model_size_mb
  model_size_mb=$(file_size_mb "$model_path")

  write_fragment \
    _path="$RAW/$config.fragment.json" \
    config="$config" \
    engine="whisper" \
    model="$(basename "$model_path")" \
    prompt="$([[ "$use_prompt" == "1" ]] && echo true || echo false)" \
    audio_seconds="$audio_seconds" \
    warm_decode_seconds="$warm_decode_seconds" \
    rtf_warm="$rtf_warm" \
    cold_first_decode_seconds="$cold_seconds" \
    peak_rss_mb="$peak_rss_mb" \
    model_size_mb="$model_size_mb" \
    status="ok" \
    ${decode_only_frag_args[@]+"${decode_only_frag_args[@]}"}
}

# ---------------------------------------------------------------------------
# parakeet config (single config: parakeet-tdt-0.6b-int8)
# ---------------------------------------------------------------------------
bench_parakeet_config() {
  local config="parakeet-tdt-0.6b-int8"
  echo ">>> $config"

  local hyp_tsv="$RAW/$config.tsv"
  local ref_tsv="$REF_TSV"
  local wav_dir="$WAV_DIR"
  if [[ "$SMOKE" == "1" ]]; then
    ref_tsv="$TMP/parakeet_smoke_ref.tsv"
    head -n 1 "$REF_TSV" > "$ref_tsv"
    wav_dir="$WAV_DIR"
  fi

  # In-process cold+warmup+warm pass, wrapped in /usr/bin/time -l so peak RSS
  # covers recognizer construction + all decodes for this config.
  local out_json="$TMP/${config}_inprocess.json"
  local time_file="$TMP/${config}.time"
  /usr/bin/time -l "$PARAKEET_PY" "$HARNESS/parakeet_bench.py" \
    "$ref_tsv" "$wav_dir" "$out_json" "$hyp_tsv" \
    2>"$time_file"

  local peak_rss_bytes peak_rss_mb
  peak_rss_bytes=$(grep 'maximum resident set size' "$time_file" | awk '{print $1}')
  peak_rss_mb=$(python3 -c "print(round($peak_rss_bytes/1e6, 2))")

  local audio_seconds cold_seconds warm_seconds
  audio_seconds=$(python3 -c "import json; print(json.load(open('$out_json'))['audio_seconds'])")
  cold_seconds=$(python3 -c "import json; print(json.load(open('$out_json'))['cold_first_decode_seconds'])")
  warm_seconds=$(python3 -c "import json; print(json.load(open('$out_json'))['warm_decode_seconds'])")

  local rtf_warm
  rtf_warm=$(python3 -c "print($warm_seconds / $audio_seconds)")

  local model_size_mb
  model_size_mb=$(python3 -c "
import os
paths = ['$PARAKEET_MODELS/encoder.int8.onnx', '$PARAKEET_MODELS/decoder.int8.onnx', '$PARAKEET_MODELS/joiner.int8.onnx']
print(round(sum(os.path.getsize(p) for p in paths) / 1e6, 2))
")

  local per_invoc_args=()
  if [[ "$SMOKE" != "1" ]]; then
    # Per-invocation cost: spawn a fresh process per utterance (the cost auris
    # would pay if it could not keep a long-lived recognizer around). This is
    # a DIFFERENT number from the in-process warm pass above -- it repeats
    # model construction every time, so it is expected to be much slower.
    local per_invoc_total=0
    for id in $(cut -f1 "$REF_TSV"); do
      local wav="$WAV_DIR/$id.wav"
      local t0 t1
      t0=$(now)
      "$PARAKEET_PY" "$HARNESS/parakeet_decode.py" "$wav" --threads "$THREADS" >/dev/null
      t1=$(now)
      per_invoc_total=$(python3 -c "print($per_invoc_total + ($t1 - $t0))")
    done
    local rtf_per_invocation
    rtf_per_invocation=$(python3 -c "print($per_invoc_total / $AUDIO_SECONDS_TOTAL)")
    per_invoc_args=(per_invocation_total_seconds="$per_invoc_total" rtf_per_invocation="$rtf_per_invocation")
  fi

  write_fragment \
    _path="$RAW/$config.fragment.json" \
    config="$config" \
    engine="parakeet" \
    model="parakeet-tdt-0.6b-v2-int8" \
    prompt=false \
    audio_seconds="$audio_seconds" \
    warm_decode_seconds="$warm_seconds" \
    rtf_warm="$rtf_warm" \
    cold_first_decode_seconds="$cold_seconds" \
    peak_rss_mb="$peak_rss_mb" \
    model_size_mb="$model_size_mb" \
    status="ok" \
    ${per_invoc_args[@]+"${per_invoc_args[@]}"}
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if [[ "$SMOKE" != "1" ]]; then
  echo "Computing total fixture audio duration..."
  AUDIO_SECONDS_TOTAL=$(compute_audio_seconds "$WAV_DIR")
  echo "Total audio: ${AUDIO_SECONDS_TOTAL}s across $(ls "$WAV_DIR"/*.wav | wc -l | tr -d ' ') fixtures"
fi

if [[ "$SMOKE" == "1" ]]; then
  CONFIGS=("whisper-tiny.en-noprompt")
else
  CONFIGS=(
    "whisper-tiny.en-noprompt"
    "whisper-tiny.en-prompt"
    "whisper-base.en-noprompt"
    "whisper-base.en-prompt"
    "whisper-small.en-noprompt"
    "whisper-small.en-prompt"
    "whisper-medium.en-noprompt"
    "whisper-medium.en-prompt"
    "parakeet-tdt-0.6b-int8"
  )
fi

model_path_for() {
  case "$1" in
    whisper-tiny.en-*)   echo "$WHISPER_MODELS/ggml-tiny.en.bin" ;;
    whisper-base.en-*)   echo "$WHISPER_MODELS/ggml-base.en.bin" ;;
    whisper-small.en-*)  echo "$WHISPER_MODELS/ggml-small.en.bin" ;;
    whisper-medium.en-*) echo "$WHISPER_MODELS/ggml-medium.en.bin" ;;
  esac
}

for config in "${CONFIGS[@]}"; do
  status=0
  case "$config" in
    parakeet-*)
      if ! bench_parakeet_config; then status=1; fi
      ;;
    *-noprompt)
      if ! bench_whisper_config "$config" "$(model_path_for "$config")" 0; then status=1; fi
      ;;
    *-prompt)
      if ! bench_whisper_config "$config" "$(model_path_for "$config")" 1; then status=1; fi
      ;;
    *)
      echo "unknown config: $config" >&2
      status=1
      ;;
  esac

  if [[ "$status" != "0" ]]; then
    echo "!!! $config FAILED -- recording failure and continuing" >&2
    write_fragment _path="$RAW/$config.fragment.json" config="$config" status="failed"
  else
    ref_for_score="$REF_TSV"
    if [[ "$SMOKE" == "1" ]]; then
      ref_for_score="$TMP/smoke_ref.tsv"
      head -n 1 "$REF_TSV" > "$ref_for_score"
    fi
    if python3 "$HARNESS/score.py" "$ref_for_score" "$RAW/$config.tsv" > "$RAW/$config.score.json"; then
      echo "    scored: $(python3 -c "import json; d=json.load(open('$RAW/$config.score.json')); print(f\"wer={d['wer']:.3f} name_accuracy={d['name_accuracy']:.3f}\")")"
    else
      echo "!!! $config: scoring FAILED" >&2
      rm -f "$RAW/$config.score.json"
    fi
  fi
done

python3 "$HARNESS/_merge_summary.py" "$RESULTS" "${CONFIGS[@]}"
echo "Done. Summary: $RESULTS/summary.json"
