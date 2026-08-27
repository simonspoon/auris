#!/usr/bin/env python3
"""Minimal CLI: decode 16kHz mono wav files with sherpa-onnx + NeMo Parakeet TDT.

Usage:
    .venv/bin/python parakeet_decode.py file1.wav [file2.wav ...]

Thread count defaults to 8; override with --threads N or env PARAKEET_THREADS.
"""
import argparse
import os
import sys

import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "parakeet")


def build_recognizer(threads: int) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(MODEL_DIR, "encoder.int8.onnx"),
        decoder=os.path.join(MODEL_DIR, "decoder.int8.onnx"),
        joiner=os.path.join(MODEL_DIR, "joiner.int8.onnx"),
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", nargs="+", help="16kHz mono wav file(s)")
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("PARAKEET_THREADS", 8)),
        help="number of threads (default 8, or env PARAKEET_THREADS)",
    )
    args = parser.parse_args()

    recognizer = build_recognizer(args.threads)

    for wav_path in args.wavs:
        samples, sample_rate = sf.read(wav_path, dtype="float32")
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        print(stream.result.text)


if __name__ == "__main__":
    main()
