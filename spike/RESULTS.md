# ASR Engine Benchmark Results

Date: 2026-08-27
Host: Intel Core i9-9880H @ 2.30GHz, x86_64, 16 logical cores, 32 GB RAM, macOS Darwin 25.5.0. CPU only — no Metal, no CUDA. All engines run at 8 threads.

Engines under test:
- **whisper.cpp** — homebrew `whisper-cli` v1.9.1, ggml f16 models (`tiny.en`, `base.en`, `small.en`, `medium.en`)
- **parakeet** — NVIDIA Parakeet TDT 0.6B v2, int8-quantized ONNX, via `sherpa-onnx` 1.13.6 (Python 3.11 venv)

Fixture: 8 synthetic utterances, `spike/fixtures/utterances.tsv`, total audio 45.43 s.

## ⚠️ Fixture caveat — read before trusting absolute numbers

The 8 reference utterances were generated with macOS `say` (text-to-speech), **not the owner's real dictated voice** — this spike ran headless with no microphone access. Synthetic TTS speech is cleaner, more evenly paced, and more consistently articulated than real human dictation. That means:

- **Absolute WER and name-accuracy numbers below are optimistic — treat them as a floor**, not a prediction of real-world performance.
- **Relative comparisons are the trustworthy part of this report**: engine vs. engine, prompt vs. no-prompt, model size vs. model size. Those orderings should hold under real speech even if the absolute numbers shift.

`spike/fixtures/record.sh` re-records the same 8 sentences in the owner's real voice. Re-running the benchmark against real speech is two commands:

```
spike/fixtures/record.sh
spike/harness/run_bench.sh
```

This regenerates `spike/results/summary.json` and this identical table structure with real-voice numbers.

## 1. Benchmark results (9 configs)

RTF = real-time factor (decode time ÷ 45.43 s audio); lower is better; RTF < 1.0 = faster than realtime.

| Config | RTF (warm, end-to-end) | RTF (decode-only) | Cold first decode (s) | Peak RSS (MB) | WER | Name accuracy |
|---|---|---|---|---|---|---|
| whisper tiny.en, no prompt | 0.153 | 0.105 | 0.76 | 215 | 12.0% | 30.8% |
| whisper tiny.en, prompt | 0.163 | 0.112 | 0.72 | 224 | 13.3% | 61.5% |
| whisper base.en, no prompt | 0.240 | 0.188 | 1.13 | 329 | 9.3% | 30.8% |
| whisper base.en, prompt | 0.244 | 0.195 | 1.25 | 338 | 4.7% | 76.9% |
| whisper small.en, no prompt | 0.640 | 0.570 | 3.15 | 825 | 7.3% | 46.2% |
| whisper small.en, prompt | 0.657 | 0.587 | 3.40 | 834 | 3.3% | 76.9% |
| whisper medium.en, no prompt | 1.955 | 1.862 | 10.19 | 2223 | 6.7% | 38.5% |
| whisper medium.en, prompt | 2.173 | 2.072 | 11.24 | 2232 | 3.3% | 76.9% |
| parakeet tdt-0.6b int8 | 0.087 (warm) / **0.634 (per-invocation)** | — | 4.00 | 1557 | 5.3% | 53.8% |

Parakeet has two RTF numbers because it has no separate "decode-only" phase reported by the harness the way whisper does:
- **0.087** — decode RTF inside a warm, already-loaded process (recognizer built once, all 8 utterances decoded in-process). This is the number that matters for a persistent/daemonized service.
- **0.634** — RTF when the process is spawned fresh per utterance (`per_invocation_total_seconds` / 45.43s), i.e. model load + decode each time. Model load on this host takes ~4s regardless of clip length, dominated by the 652 MB encoder. A cold-start-per-call architecture pays this every time; a warm/persistent process pays it once.

WER and name accuracy are computed against reference text (`spike/fixtures/utterances.tsv`); whisper's "prompt" runs used an initial prompt seeded with the six product/task names.

## 2. Per-term name accuracy

Accuracy per config for each of the 6 product/task names (source: `per_term` in each `*.score.json`; ref counts: mesa=3, auris=3, khora=2, qorvex=2, helios=2, kokoro=1, total=13):

| Config | mesa | auris | khora | qorvex | helios | kokoro |
|---|---|---|---|---|---|---|
| tiny, no prompt | 66.7% | 0% | 0% | 0% | 100% | 0% |
| tiny, prompt | 100% | 100% | 0% | 0% | 100% | 0% |
| base, no prompt | 66.7% | 0% | 0% | 0% | 100% | 0% |
| base, prompt | 100% | 100% | 0% | 100% | 100% | 0% |
| small, no prompt | 100% | 0% | 0% | 0% | 100% | 100% |
| small, prompt | 100% | 100% | 50% | 0% | 100% | 100% |
| medium, no prompt | 66.7% | 0% | 0% | 0% | 100% | 100% |
| medium, prompt | 100% | 100% | 0% | 50% | 100% | 100% |
| parakeet | 66.7% | 66.7% | 0% | 0% | 100% | 100% |

Observations:
- **helios** is recognized correctly in every single config — it's already a common word/name in every model's vocabulary.
- **khora** is the hardest term by far — correct in exactly one config (small.en+prompt, 50%), wrong everywhere else including every whisper prompt config.
- **qorvex** is nearly as hard — correct only in base.en+prompt (100%) and medium.en+prompt (50%); the prompt doesn't reliably fix it.
- **kokoro** flips from always-wrong (tiny/base) to always-right (small/medium/parakeet) — it's a real word/brand already in the larger models' vocabulary, so it's not really testing the prompt mechanism, it's testing model size.
- **auris** and **mesa** are the terms the prompt actually fixes cleanly: auris goes from 0% to 100% in every whisper config once prompted; mesa goes from ~67% to 100%.

**Sample sizes are small — read the per-term rows with their `ref` counts.** There are 13 name occurrences total across the 8 utterances: mesa 3, auris 3, khora 2, qorvex 2, helios 2, **kokoro 1**. A per-term percentage is therefore a count, not a rate: "kokoro 100%" means one occurrence was right, and "khora 50%" means one of two. The kokoro finding in particular (wrong on tiny/base, right on small/medium/parakeet) rests on a single occurrence per config and should be treated as a hint to test further, not an established result. The aggregate name-accuracy column is over all 13 and is the more robust number. If this fixture set is reused, repeat kokoro and khora two or three more times.

## 3. What the engines actually heard

Quotes are raw transcript lines from `spike/results/raw/<config>.tsv`.

**auris** — never once transcribed correctly without a prompt:
- tiny.en (no prompt): *"under **Aorus** for the Parakeet spike"*
- base.en (no prompt): *"transcribed Helios as Helios again... is **Oris**"*
- small.en (no prompt): *"under **Oris** for the parakeet spike"*
- medium.en (no prompt): *"open a follow up under **auras**"* / *"what happened is **AORUS** transcribed Helios"* — literally spells it like the Gigabyte AORUS gaming-hardware brand.
- parakeet: *"log it under **Aurus** and link it"*

**khora** — the worst term across the board, prompt or not:
- base.en (no prompt): *"add a note to **core of** that headless mode"*, later *"blocked on **Cora**"*
- small.en (no prompt): *"add a note to **core of**"*, later *"blocked on **Quora**"*
- medium.en (prompt): *"add a note to **quora**"*, later *"blocked on **quora**"* — consistently hears the well-known brand "Quora" instead.
- parakeet: *"add a note to **Korra**"*, later *"blocked on **Korra**"* — consistently hears the Nickelodeon character/brand name.

**qorvex** — consistently heard as the near-homophone "corvex", Q→C substitution survives the prompt in most configs:
- base.en (no prompt): *"wire up **Corex** to the iOS simulator"*
- tiny.en (prompt): *"wire-up **korvex**"*
- small.en (prompt): *"wire-up-**korvex**"*
- parakeet: *"wire up **Corvex** to the iOS simulator"*, later *"**Corvex** tap failed twice"*

**helios** — the control case, never mangled by anything:
- Every config, every occurrence: *"Helios"*, verbatim, correct case even.

**kokoro** — flips to correct once the model is big enough to know the word as itself:
- tiny.en: *"**Kakaro's** latency numbers"* / *"**Kakora's**"*
- base.en: *"**Kakoro's**"* / *"**kakoro's**"*
- small.en / medium.en / parakeet: *"**Kokoro's** latency numbers"* — correct, unprompted.

This is the concrete case for auris existing: whisper and parakeet, run stock, turn "khora" into "Quora" or "Korra," turn "qorvex" into "corvex," and turn "auris" into a gaming-hardware brand name — silently, with no indication anything went wrong. A dictation pipeline built on these engines needs a vocabulary-biasing layer for this project's own terms or it will misfile tasks under the wrong project every time.

## 4. Findings

**medium.en is unusable for live dictation on this host.** RTF is 1.955 (no prompt) / 2.173 (prompt) — roughly 2x realtime, i.e. it takes twice as long to decode as the audio takes to speak. It also doesn't win on accuracy: medium.en+prompt gets 76.9% name accuracy and 3.3% WER, identical name accuracy to base.en+prompt (76.9%) and small.en+prompt (76.9%), and the same WER as small.en+prompt (3.3% vs 3.3%). Its 8x higher RTF and 6.6x higher peak RSS (2232 MB vs 338 MB for base.en+prompt) buy nothing here.

**The initial prompt is the single highest-leverage lever available, at essentially zero latency cost.** Per-model name accuracy, no-prompt → prompt:
- tiny.en: 30.8% → 61.5% (2.0x)
- base.en: 30.8% → 76.9% (2.5x)
- small.en: 46.2% → 76.9% (1.67x)
- medium.en: 38.5% → 76.9% (2.0x)

RTF cost of the prompt is negligible: base.en goes from 0.240 → 0.244 RTF (+1.7%), small.en 0.640 → 0.657 (+2.7%), medium.en 1.955 → 2.173 (+11%, still within noise given medium's already-unusable RTF). WER also improves with the prompt in every model (e.g. base.en 9.3% → 4.7%; small.en 7.3% → 3.3%).

**The prompt has a failure mode of its own: it injects vocabulary into text that never contained it.** tiny.en is the only config whose WER got *worse* with the prompt (12.0% → 13.3%), driven by insertions rising 1 → 6. The mechanism is visible in the transcripts — tiny.en+prompt turned "wire up qorvex to the **iOS** simulator" into *"wire-up korvex to the **auris** simulator"*, substituting a prompt term into an unrelated phrase. base.en+prompt shows a milder version, hearing `khora` as `qorvex` in u02 — one project noun swapped for another rather than corrected. So the prompt does not only pull toward correct names; it pulls toward *any* prompt name, including into slots where none was spoken. base.en is the floor at which the prompt is a net win, and any downstream correction layer should expect confusions **among** mesa's own terms, not just between a mesa term and an English word.

**parakeet is the fastest engine by a wide margin — but only in a warm/persistent process.** In-process RTF is 0.087, ~1.8x faster than whisper tiny.en (0.153) and ~2.8x faster than base.en (0.244). But spawned fresh per utterance, effective RTF is 0.634 because each invocation pays ~4s to load the 652 MB int8 encoder. A persistent parakeet process (loaded once, decoding many utterances) is required to realize the fast number; a cold-process-per-utterance architecture would put parakeet roughly on par with whisper small.en for latency.

**The real tension for task 924: parakeet is fast but can't be steered; whisper+prompt is steerable but costs more RTF/RAM.** This setup's sherpa-onnx parakeet integration has no prompt or vocabulary-biasing mechanism, so its name accuracy (53.8%) is stuck wherever the model's native vocabulary lands it — better than unprompted whisper of any size, but well below any prompted whisper config (all 76.9%). Whisper's `--prompt` flag is cheap and effective, but even base.en+prompt at 0.244 RTF is ~2.8x slower than parakeet's warm in-process RTF. There is no config in this benchmark that is both the fastest and the most name-accurate — pick one axis to prioritize.

**Recommended default for task 924: whisper base.en with the initial prompt.**
- RTF 0.244 — comfortably realtime (4x headroom) on this 2019-era CPU-only host, unlike medium.en.
- Name accuracy 76.9% — tied for the best in this benchmark, matching small.en+prompt and medium.en+prompt.
- WER 4.7% — worse than small.en+prompt's 3.3%, but the difference is on non-name words; on the terms that actually matter for filing tasks under the right project, base.en+prompt performs identically to small.en+prompt.
- Peak RSS 338 MB and cold-start 1.25s — both a fraction of small.en+prompt (834 MB / 3.4s) or medium.en+prompt (2232 MB / 11.24s), which matters for running alongside other apps on a 32 GB box with no GPU offload.
- Rejects parakeet as the default specifically because this domain's vocabulary (task/product names) is exactly what prompt-based biasing exists to fix, and parakeet's setup here has no equivalent mechanism — its 53.8% name accuracy is meaningfully behind every prompted whisper config.

If peak general-purpose WER matters more than resource footprint (e.g. running on a beefier machine, or batch/offline transcription rather than live dictation), small.en+prompt is the next reasonable step up: same 76.9% name accuracy, better 3.3% WER, at 2.7x the RTF and 2.5x the RAM of base.en+prompt.

If/when a warm, persistent parakeet process becomes part of the architecture (avoiding the ~4s per-call load penalty) and a vocabulary-biasing mechanism is added on top (sherpa-onnx supports contextual biasing word lists in some configurations — not exercised in this spike), parakeet's raw 0.087 RTF makes it worth revisiting for a throughput-sensitive path. That's follow-up work, not something this spike settles.

## 5. Reproduction

**whisper.cpp**, per utterance/config:
```
whisper-cli -m spike/models/whisper/ggml-<size>.en.bin -f <input.wav> -nt -np -t 8
whisper-cli -m spike/models/whisper/ggml-<size>.en.bin -f <input.wav> -nt -np -t 8 --prompt "<prompt text>"
```
- Binary: homebrew `whisper-cpp` 1.9.1, `whisper-cli` at `/usr/local/bin/whisper-cli`
- Models: ggml f16, downloaded from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-<name>.bin` to `spike/models/whisper/`
  - tiny.en — `ggml-tiny.en.bin`, 77.7 MB
  - base.en — `ggml-base.en.bin`, 147.96 MB
  - small.en — `ggml-small.en.bin`, 487.61 MB
  - medium.en — `ggml-medium.en.bin`, 1533.77 MB
- Backends: BLAS + CPU (haswell), no GPU/Metal. Input: 16kHz mono WAV.

**parakeet**, via `spike/harness/parakeet_decode.py`:
```
cd spike/harness
.venv/bin/python parakeet_decode.py <input.wav> [...]
```
- venv: Python 3.11.13, created with `uv venv --python 3.11` (system python3 is 3.14.5, no `sherpa-onnx` wheel available for it)
- `sherpa-onnx==1.13.6` / `sherpa-onnx-core==1.13.6`
- Model: `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8` from `https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`, individual files (not a tar archive) in `spike/models/parakeet/`: `encoder.int8.onnx` (652 MB), `decoder.int8.onnx` (7.3 MB), `joiner.int8.onnx` (1.7 MB), `tokens.txt` (9.4 KB). Total ~633 MB.
- `OfflineRecognizer.from_transducer(..., model_type="nemo_transducer")` — required for NeMo/Parakeet exports; the default `model_type="transducer"` is for k2/icefall exports and does not work with this model.
- `--threads` / `PARAKEET_THREADS` env var, default 8.

Full setup notes: `spike/harness/NOTES-whisper.md`, `spike/harness/NOTES-parakeet.md`. Raw data: `spike/results/summary.json`, `spike/results/raw/*.tsv`, `spike/results/raw/*.score.json`.
