# whisper.cpp setup notes

## Binary
- `whisper-cli` v1.9.1 (homebrew `whisper-cpp` 1.9.1), at `/usr/local/bin/whisper-cli`
- Backends loaded: BLAS + CPU (haswell), no GPU/Metal (Intel i9-9880H)
- Every invocation prints two `load_backend:` lines to stderr on startup regardless of `-np` (these come from the ggml backend loader, before whisper-cli's own arg parsing takes effect) — harmless, redirect stderr if it matters.

## Models
Downloaded to `spike/models/whisper/` from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-<name>.bin` (f16, no quantization):

| model | file | size |
|---|---|---|
| tiny.en | `spike/models/whisper/ggml-tiny.en.bin` | 78M (~80MB) |
| base.en | `spike/models/whisper/ggml-base.en.bin` | 148M (~145MB) |
| small.en | `spike/models/whisper/ggml-small.en.bin` | 488M (~480MB) |
| medium.en | `spike/models/whisper/ggml-medium.en.bin` | 1.5G (~1.4GB) |

All four verified to decode correctly on a synthetic 16kHz mono test clip (`say -o /tmp/t.aiff "testing one two three"` piped through ffmpeg to 16kHz/mono/pcm_s16le wav). Sample outputs:
- tiny.en: "Testing one, two, three."
- base.en: "Testing 123"
- small.en: "Testing one, two, three."
- medium.en: "Testing 1, 2, 3."

(Numeral vs. spelled-out formatting differs by model size — expected whisper behavior, not a decode failure.)

## Benchmark invocation

Base command (no timestamps, no extra prints, 8 threads, plain text to stdout):

```
whisper-cli -m <model.bin> -f <input.wav> -nt -np -t 8
```

With an initial prompt (biases decoding style/vocabulary, does not require `--carry-initial-prompt` for a single-segment clip):

```
whisper-cli -m <model.bin> -f <input.wav> -nt -np -t 8 --prompt "<prompt text>"
```

Confirmed the prompt flag has real effect: on the test clip, base.en without a prompt output "Testing 123"; with `--prompt "Numbers should be written as digits: 1, 2, 3."` it output "Testing 1, 2, 3." — same audio, different formatting driven by the prompt.

Notes:
- `--prompt` truncates to `n_text_ctx/2` tokens (per `--help`).
- `--carry-initial-prompt` (default false) re-prepends the initial prompt to every internal segment/window; irrelevant for short single-segment benchmark clips but worth knowing for longer audio.
- Input must be 16kHz mono; whisper-cli decodes via miniaudio and accepts wav/mp3/flac/ogg directly, but 16kHz mono wav is the standard/expected format.
- Non-`-nt -np` stdout still starts with a leading blank line before the transcript text (from the two backend-load stderr lines interleaving) — worth trimming when parsing output in the harness.
