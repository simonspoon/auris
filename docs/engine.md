# Engine decision

Date: 2026-08-27. Host: Intel Core i9-9880H, x86_64, 16 logical cores, 32 GB,
macOS Darwin 25.5.0. CPU only — no Metal, no CUDA.

**Decision: NVIDIA Parakeet TDT 0.6B v2, int8 ONNX, run through sherpa-onnx with
per-word hotword biasing, in a persistent process.**

This reverses the original plan (whisper.cpp via `whisper-rs`, with Metal). Two
things broke that plan: the host is not Apple Silicon, so there is no Metal; and
the premise that only whisper can be steered toward mesa's vocabulary turned out
to be false.

## The measurement

Full data: `spike/RESULTS.md` (task 947, §6 addendum from task 948), rescored
under the corrected name metric from task 949. Eight dictation fixtures,
45.43 s, containing mesa's own names. "Name F1" is alignment-based precision and
recall over {mesa, auris, khora, qorvex, helios, kokoro} — a name only counts
when it lands in the right slot, so hallucinated names cost.

| Config | RTF (warm) | WER | Name F1 | Peak RSS | Cold start |
|---|---|---|---|---|---|
| **parakeet + per-word hotwords** | **0.082** | 4.0% | **83.3%** | 1557 MB | ~4.0 s |
| whisper small.en + prompt | 0.657 | **3.3%** | **83.3%** | 834 MB | 3.4 s |
| whisper base.en + prompt | 0.244 | 4.7% | 75.0% | 338 MB | 1.25 s |
| parakeet, greedy, no hotwords | 0.087 | 5.3% | 66.7% | 1557 MB | ~4.0 s |
| whisper medium.en + prompt | 2.173 | 3.3% | 83.3% | 2232 MB | 11.2 s |

Parakeet with hotwords ties the best name F1 in the benchmark and gets there
**8x faster than small.en+prompt**, which is the only whisper config that
matches it on names. medium.en is ~2x slower than realtime on this CPU and is
out. base.en+prompt is cheap on RAM but strictly behind on names.

## Why the original reasoning no longer holds

The plan said whisper wins because `initial_prompt` is the one lever that makes
mesa's names come back spelled right. sherpa-onnx has the same lever:
`hotwords_file`, `hotwords_score`, `modeling_unit` and `bpe_vocab` on
`OfflineRecognizer::from_transducer`, working with `model_type="nemo_transducer"`.
They are inert under greedy decoding — the original spike's mistake — and require
`decoding_method="modified_beam_search"`, which costs nothing measurable here
(0.076 vs 0.087 RTF).

The second trap: `bpe_vocab` needs a sentencepiece-style `token score` file.
Passing the model's `tokens.txt` (`token id`) silently miscalibrates the
tokenizer — hotwords then do nothing at low scores and shred the transcript at
high ones. For a BPE model the piece score is the negative merge rank, so the
right file is one line of awk over `tokens.txt`; see
`spike/models/parakeet/bpe_synth.vocab`.

Biasing is also **better behaved than whisper's prompt**. Whisper's prompt pulls
toward *any* prompt term, including into slots where none was spoken —
tiny.en+prompt inserted "auris" into "the iOS simulator", and base.en+prompt
heard `khora` as `qorvex`. No parakeet run at a usable boost inserted a single
name that was not spoken.

Per-word boosts are what make it usable, since the strength that fixes the hard
terms wrecks the easy ones if applied uniformly (`spike/fixtures/hotwords.txt`):

```
mesa :3.0
auris :3.0
khora :6.5
qorvex :5.0
helios :3.0
kokoro :3.0
```

## What this costs

- **A persistent process is now mandatory, not an optimization.** Loading the
  652 MB int8 encoder takes ~4 s. Spawned per utterance, parakeet's effective RTF
  is 0.634 and the whole speed argument evaporates. auris must hold a loaded
  recognizer across utterances; a one-shot `audio in, text out` binary that exits
  each time is not compatible with this decision. See tasks 925 and 926.
- **~1.5 GB resident**, against 338 MB for base.en+prompt. Acceptable on a 32 GB
  box, and it is a fixed cost paid once by a daemon rather than per utterance.
- **English only** (plus 24 European languages in this model), which mesa is.

## What it does not fix

`khora` is never produced by any engine or configuration tested, whisper or
parakeet, prompted or biased. Every parakeet run hears "qora"; raising its boost
alone changes nothing. A post-ASR correction pass over mesa's own vocabulary is
required regardless of engine — the misses are all near-homophones ("qora",
"corvex", "aurus"), so a small edit-distance pass over a known term list handles
them. That is now a first-class part of the design, not a fallback.

**Implemented, task 950: see `docs/correction.md` and
`spike/harness/vocab_correct.py`.** The chosen config (`tuned`, parakeet +
per-word hotwords) goes from 83.3% to 96.3% name F1 with the pass applied,
while WER also improves, 4.0% → 2.0%.

## Versions

- `sherpa-onnx` 1.13.6 (Python wheel used for the spike; the Rust crate
  `sherpa-onnx` / `sherpa-onnx-sys` is at the same 1.13.6, published 2026-08-24).
- Model: `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` —
  `encoder.int8.onnx` (652 MB), `decoder.int8.onnx`, `joiner.int8.onnx`,
  `tokens.txt`, ~633 MB total. Note the repo ships **no** sentencepiece vocab;
  derive it from `tokens.txt` as above.
- `OfflineRecognizer::from_transducer` with `model_type="nemo_transducer"` — the
  default `"transducer"` is for k2/icefall exports and does not work with this
  model.
- Rejected: whisper.cpp 1.9.1 / `whisper-rs` (slower here at equal name
  accuracy, and the Metal argument does not apply on this host); Moonshine (not
  benchmarked; no vocabulary lever); MLX whisper (Apple Silicon and a Python
  runtime, both disqualifying for a binary mesa shells out to).

## Rust binding: verified, no fallback

The spike ran sherpa-onnx through Python, so the plan was to confirm the Rust
side before committing. Confirmed 2026-08-27 by reading the published crate
source (`sherpa-onnx` 1.13.6 from crates.io): `OfflineModelConfig` carries
`model_type`, `modeling_unit` and `bpe_vocab`; `OfflineRecognizerConfig` carries
`decoding_method`, `hotwords_file` and `hotwords_score`; and the crate ships
`rust-api-examples/examples/nemo_parakeet.rs`. Every knob the spike used exists
in Rust. No fork, no patched crate, no direct `sherpa-onnx-sys` work needed.

**There is no whisper fallback.** If a problem shows up in the Rust path, the
answer is to fix the binding — patch or vendor the crate, or bind the C API
through `sherpa-onnx-sys` — not to switch engines. The first implementation task
(933) still reproduces the spike's 83.3% name F1 / 4.0% WER from Rust as an
end-to-end check, but that is verification, not a decision point.

Second caveat, inherited from the spike: the fixtures are macOS `say` TTS, not a
real dictated voice. Absolute numbers are a floor; the ordering is what this
decision rests on. `spike/fixtures/record.sh` re-records in a real voice if the
ordering is ever in doubt.
