#!/usr/bin/env python3
"""In-process cold+warm timing driver for the parakeet-tdt-0.6b-int8 config.

Unlike whisper-cli (fresh process per file), sherpa-onnx recognizers are
built once and reused, so "cold" and "warm" mean something different here:
cold = process start + recognizer construction + first decode; warm =
subsequent decodes in that same already-built recognizer. This script
measures exactly that, in one process (so it can also be wrapped by
`/usr/bin/time -l` from the caller to get one peak-RSS number covering
build + all decodes).

Usage:
    .venv/bin/python parakeet_bench.py UTTERANCES_TSV WAV_DIR OUT_JSON HYP_TSV

Writes HYP_TSV (id<TAB>text, from the warmup+measured passes) and OUT_JSON
(cold_first_decode_seconds, warm_decode_seconds, audio_seconds) to disk.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from parakeet_decode import build_recognizer  # noqa: E402
import soundfile as sf  # noqa: E402


def read_utterances(tsv_path):
    ids = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, _ref = line.split("\t", 1)
            ids.append(uid)
    return ids


def decode_one(recognizer, wav_path):
    samples, sample_rate = sf.read(wav_path, dtype="float32")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text


def main():
    utterances_tsv, wav_dir, out_json, hyp_tsv = sys.argv[1:5]
    ids = read_utterances(utterances_tsv)
    wavs = {uid: os.path.join(wav_dir, f"{uid}.wav") for uid in ids}

    audio_seconds = 0.0
    for uid in ids:
        info = sf.info(wavs[uid])
        audio_seconds += info.frames / info.samplerate

    # COLD: recognizer construction + first decode, before any warmup.
    t0 = time.time()
    recognizer = build_recognizer(threads=8)
    first_id = ids[0]
    cold_text = decode_one(recognizer, wavs[first_id])
    cold_seconds = time.time() - t0

    # WARMUP: discarded pass over all 8 fixtures (same in-process recognizer).
    for uid in ids:
        decode_one(recognizer, wavs[uid])

    # MEASURED WARM PASS: recognizer already built, cache hot.
    hyps = {}
    t_start = time.time()
    for uid in ids:
        hyps[uid] = decode_one(recognizer, wavs[uid])
    warm_decode_seconds = time.time() - t_start

    with open(hyp_tsv, "w", encoding="utf-8") as f:
        for uid in ids:
            text = " ".join(hyps[uid].split())
            f.write(f"{uid}\t{text}\n")

    result = {
        "audio_seconds": audio_seconds,
        "cold_first_decode_seconds": cold_seconds,
        "warm_decode_seconds": warm_decode_seconds,
        "cold_first_id": first_id,
        "cold_first_text": cold_text,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    main()
