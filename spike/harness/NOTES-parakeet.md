# Parakeet via sherpa-onnx — spike notes

## Environment
- Host: macOS Darwin 25.5.0, Intel i9-9880H, 16 cores, 32GB RAM, CPU only (no Metal/CUDA acceleration used by sherpa-onnx here).
- venv: `spike/harness/.venv`, created with `uv venv --python 3.11` (Python 3.11.13). The system python3 (3.14.5) had no `sherpa-onnx` wheel available on PyPI, so an older interpreter was pinned via `uv`.
- Installed with: `uv pip install --python .venv/bin/python sherpa-onnx soundfile`
- `sherpa-onnx` version installed: **1.13.6** (`sherpa-onnx==1.13.6`, `sherpa-onnx-core==1.13.6`).

## Model
- Model: **csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8** (NVIDIA Parakeet TDT 0.6B v2, int8-quantized ONNX export packaged for sherpa-onnx).
- Source: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
- Unlike some sherpa-onnx model repos, this one is **not** a single `.tar.bz2` archive — it's individual files under `resolve/main/`, downloaded directly with curl:
  - `decoder.int8.onnx` (7.3 MB)
  - `encoder.int8.onnx` (652 MB)
  - `joiner.int8.onnx` (1.7 MB)
  - `tokens.txt` (9.4 KB)
- On-disk location: `spike/models/parakeet/`, total size ~633 MB (encoder dominates).
- A non-int8 fp32/fp16 variant also exists (`sherpa-onnx-nemo-parakeet-tdt-0.6b-v2`, `-fp16`) if higher accuracy is wanted at the cost of size/speed; not downloaded here since int8 was requested as preferred.
- Newer v3 variants also exist (`sherpa-onnx-nemo-parakeet-tdt-0.6b-v3[-int8|-fp16]`) — not tried, v2-int8 was sufficient and matched the task's suggested naming pattern.

## Decoder CLI
`spike/harness/parakeet_decode.py` — minimal script, builds the `sherpa_onnx.OfflineRecognizer` once via `OfflineRecognizer.from_transducer(...)` with `model_type="nemo_transducer"`, then decodes each wav argv and prints one transcript per line.

Thread count: `--threads N` flag or `PARAKEET_THREADS` env var, default 8.

### Working command line
```
cd spike/harness
.venv/bin/python parakeet_decode.py /tmp/p.wav
```

## End-to-end verification
Test audio generated with macOS `say` + ffmpeg resample to 16kHz mono PCM16:
```
say -o /tmp/p.aiff "testing khora and qorvex one two three"
ffmpeg -y -i /tmp/p.aiff -ar 16000 -ac 1 -c:a pcm_s16le /tmp/p.wav
.venv/bin/python parakeet_decode.py /tmp/p.wav
```

**Transcript output:**
```
Testing Cora and Corvex 123.
```

(Input was "testing khora and qorvex one two three" — the model reasonably mis-transcribed the invented product names "khora"/"qorvex" as "Cora"/"Corvex" and wrote digits "123" for "one two three", which is expected/normal ASR behavior for out-of-vocabulary proper nouns.)

Decode wall time for the ~2s clip: ~5.1s total process time (includes recognizer build/model load, which dominates for a single short clip), 108% CPU with `num_threads=8` default — model loading, not decoding, is the bulk of that time for a single short file.

## Surprises / gotchas
- Python 3.14 (the system default `python3`) has no published `sherpa-onnx` wheel; had to pin the venv to 3.11 via `uv venv --python 3.11` (uv auto-fetched the interpreter).
- The HF model repo for this Parakeet export is plain files, not a `.tar.bz2` archive like some other sherpa-onnx model repos (e.g. some Whisper/Zipformer repos) — no `tar` extraction step needed, just curl each file from `resolve/main/`.
- `OfflineRecognizer.from_transducer` requires `model_type="nemo_transducer"` for NeMo/Parakeet-style transducer exports (the default `model_type` is plain `"transducer"`, which is for k2/icefall-style exports and will not work correctly with this model).
