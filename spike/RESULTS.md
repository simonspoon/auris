# ASR Engine Benchmark Results

Date: 2026-08-27
Host: Intel Core i9-9880H @ 2.30GHz, x86_64, 16 logical cores, 32 GB RAM, macOS Darwin 25.5.0. CPU only — no Metal, no CUDA. All engines run at 8 threads.

Engines under test:
- **whisper.cpp** — homebrew `whisper-cli` v1.9.1, ggml f16 models (`tiny.en`, `base.en`, `small.en`, `medium.en`)
- **parakeet** — NVIDIA Parakeet TDT 0.6B v2, int8-quantized ONNX, via `sherpa-onnx` 1.13.6 (Python 3.11 venv)

Fixture: 8 synthetic utterances, `spike/fixtures/utterances.tsv`, total audio 45.43 s.

## ⚠️ Metric change, 2026-08-27 (task 949) — every number below is the corrected metric

Name accuracy used to be recall-only and position-blind: `min(ref_count, hyp_count)`
per term over the whole utterance. It never penalised a name the engine emitted
that wasn't actually spoken, so a config that babbled a mesa name into every
slot could score as well as one that got it right. It is now alignment-based:
the hypothesis is Levenshtein-aligned to the reference, and a vocab term only
counts as a hit when it lands on a matching slot; anything else (an inserted
name, or a name substituted into a slot where a different word — or a
different vocab term — was spoken) counts as a false positive. **`name_accuracy`
in the JSON, and every "name accuracy" figure in this document, is now the F1
of that precision/recall pair.** All configs have been rescored; nothing here
is comparable to an older copy of this file that predates task 949.

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

| Config | RTF (warm, end-to-end) | RTF (decode-only) | Cold first decode (s) | Peak RSS (MB) | WER | Name F1 | Name precision | Name recall |
|---|---|---|---|---|---|---|---|---|
| whisper tiny.en, no prompt | 0.153 | 0.105 | 0.76 | 215 | 12.0% | 35.3% | 75.0% | 23.1% |
| whisper tiny.en, prompt | 0.163 | 0.112 | 0.72 | 224 | 13.3% | 69.6% | 80.0% | 61.5% |
| whisper base.en, no prompt | 0.240 | 0.188 | 1.13 | 329 | 9.3% | 44.4% | 80.0% | 30.8% |
| whisper base.en, prompt | 0.244 | 0.195 | 1.25 | 338 | 4.7% | 75.0% | 81.8% | 69.2% |
| whisper small.en, no prompt | 0.640 | 0.570 | 3.15 | 825 | 7.3% | 60.0% | 85.7% | 46.2% |
| whisper small.en, prompt | 0.657 | 0.587 | 3.40 | 834 | 3.3% | 83.3% | 90.9% | 76.9% |
| whisper medium.en, no prompt | 1.955 | 1.862 | 10.19 | 2223 | 6.7% | 52.6% | 83.3% | 38.5% |
| whisper medium.en, prompt | 2.173 | 2.072 | 11.24 | 2232 | 3.3% | 83.3% | 90.9% | 76.9% |
| parakeet tdt-0.6b int8 | 0.087 (warm) / **0.634 (per-invocation)** | — | 4.00 | 1557 | 5.3% | 66.7% | 87.5% | 53.8% |

Parakeet has two RTF numbers because it has no separate "decode-only" phase reported by the harness the way whisper does:
- **0.087** — decode RTF inside a warm, already-loaded process (recognizer built once, all 8 utterances decoded in-process). This is the number that matters for a persistent/daemonized service.
- **0.634** — RTF when the process is spawned fresh per utterance (`per_invocation_total_seconds` / 45.43s), i.e. model load + decode each time. Model load on this host takes ~4s regardless of clip length, dominated by the 652 MB encoder. A cold-start-per-call architecture pays this every time; a warm/persistent process pays it once.

WER and name accuracy are computed against reference text (`spike/fixtures/utterances.tsv`); whisper's "prompt" runs used an initial prompt seeded with the six product/task names.

## 2. Per-term name accuracy

F1 per config for each of the 6 product/task names (source: `per_term` in each `*.score.json`; ref counts: mesa=3, auris=3, khora=2, qorvex=2, helios=2, kokoro=1, total=13). Cell is `F1 (hits/ref, N emitted)` so both sides — what the engine missed and what it hallucinated — are visible in one number:

| Config | mesa | auris | khora | qorvex | helios | kokoro |
|---|---|---|---|---|---|---|
| tiny, no prompt | 80% (2/3, 2) | 0% (0/3, 0) | 0% (0/2, 0) | 0% (0/2, 0) | 50% (1/2, 2) | 0% (0/1, 0) |
| tiny, prompt | 100% (3/3, 3) | 86% (3/3, 4) | 0% (0/2, 0) | 0% (0/2, 0) | 80% (2/2, 3) | 0% (0/1, 0) |
| base, no prompt | 80% (2/3, 2) | 0% (0/3, 0) | 0% (0/2, 0) | 0% (0/2, 0) | 80% (2/2, 3) | 0% (0/1, 0) |
| base, prompt | 100% (3/3, 3) | 100% (3/3, 3) | 0% (0/2, 0) | 80% (2/2, 3) | 50% (1/2, 2) | 0% (0/1, 0) |
| small, no prompt | 100% (3/3, 3) | 0% (0/3, 0) | 0% (0/2, 0) | 0% (0/2, 0) | 80% (2/2, 3) | 100% (1/1, 1) |
| small, prompt | 100% (3/3, 3) | 100% (3/3, 3) | 67% (1/2, 1) | 0% (0/2, 0) | 80% (2/2, 3) | 100% (1/1, 1) |
| medium, no prompt | 80% (2/3, 2) | 0% (0/3, 0) | 0% (0/2, 0) | 0% (0/2, 0) | 80% (2/2, 3) | 100% (1/1, 1) |
| medium, prompt | 100% (3/3, 3) | 100% (3/3, 3) | 0% (0/2, 0) | 67% (1/2, 1) | 80% (2/2, 3) | 100% (1/1, 1) |
| parakeet | 80% (2/3, 2) | 80% (2/3, 2) | 0% (0/2, 0) | 0% (0/2, 0) | 80% (2/2, 3) | 100% (1/1, 1) |

Observations:
- **helios is *not* the clean control case the old metric implied.** It scores 80% F1 in 7 of 9 configs, not 100% — every engine we checked, prompted or not, emits a **third** "helios" that has no match in the reference. This is a fixture artifact, not an engine defect: u06's reference deliberately contains the pun *"transcribed helios **as hell EOS** again"*, and every engine hears "hell EOS" as "helios" too (verified against `spike/results/raw/*.tsv`: base.en-noprompt, medium.en-noprompt/prompt, small.en-noprompt/prompt, tiny.en-prompt, and parakeet all render u06 as "...transcribed Helios as Helios again..."). The corrected metric is right to flag that second "Helios" as a false positive — no one said the word twice — but it means helios was never actually 100% clean; the old recall-only metric just couldn't see the extra one. The two configs at 50% get there differently, but both by missing the *real* helios in u06 while still catching the "hell EOS" pun as an extra "Helios": tiny.en-noprompt renders u06 as *"...always transcribed **Helius** as **Helios** again..."* (real helios misspelled "Helius", a miss; the pun word is the false positive), and base.en-prompt renders it *"...transcribed **heliosus** helios again..."* (real helios misspelled "heliosus", same pattern).
- **khora** is the hardest term by far — correct in exactly one config (small.en+prompt, 67%), wrong everywhere else including every whisper prompt config.
- **qorvex** is nearly as hard — correct only in base.en+prompt (80%) and medium.en+prompt (67%); the prompt doesn't reliably fix it.
- **kokoro** flips from always-wrong (tiny/base) to always-right (small/medium/parakeet) — it's a real word/brand already in the larger models' vocabulary, so it's not really testing the prompt mechanism, it's testing model size.
- **auris** and **mesa** are the terms the prompt actually fixes cleanly: auris goes from 0% to 100% in every whisper config once prompted; mesa goes from ~80% to 100%.

**Sample sizes are small — read the per-term rows with their `ref` counts.** There are 13 name occurrences total across the 8 utterances: mesa 3, auris 3, khora 2, qorvex 2, helios 2, **kokoro 1**. A per-term F1 is therefore built from a small count, not a large-sample rate: "kokoro 100%" means one occurrence was right, and "khora 67%" means one of two, matched, with no extra emissions. The kokoro finding in particular (wrong on tiny/base, right on small/medium/parakeet) rests on a single occurrence per config and should be treated as a hint to test further, not an established result. The aggregate name-accuracy column is over all 13 and is the more robust number. If this fixture set is reused, repeat kokoro and khora two or three more times.

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

**helios** — the closest thing to a control case, but not clean (see §2): the word itself is never misspelled, but u06's reference contains a deliberate pun, *"transcribed helios **as hell EOS** again"*, and almost every engine hears "hell EOS" as a second "Helios":
- parakeet, base.en, medium.en, small.en (all no-prompt and prompt variants except two): *"...transcribed Helios as Helios again..."* — verbatim "Helios" twice, but only one was actually spoken.
- tiny.en-noprompt: *"...always transcribed **Helius** as **Helios** again..."* — misspells the real one and still catches the pun.
- base.en-prompt: *"...transcribed **heliosus** helios again..."* — same pattern.

**kokoro** — flips to correct once the model is big enough to know the word as itself:
- tiny.en: *"**Kakaro's** latency numbers"* / *"**Kakora's**"*
- base.en: *"**Kakoro's**"* / *"**kakoro's**"*
- small.en / medium.en / parakeet: *"**Kokoro's** latency numbers"* — correct, unprompted.

This is the concrete case for auris existing: whisper and parakeet, run stock, turn "khora" into "Quora" or "Korra," turn "qorvex" into "corvex," and turn "auris" into a gaming-hardware brand name — silently, with no indication anything went wrong. A dictation pipeline built on these engines needs a vocabulary-biasing layer for this project's own terms or it will misfile tasks under the wrong project every time.

### Ordering changes from the metric fix

The corrected metric doesn't just relabel numbers — it changes which configs
rank where:
- **The old four-way tie at 76.9% breaks.** base.en+prompt, small.en+prompt,
  medium.en+prompt, and tuned parakeet were previously tied for the best name
  accuracy in the benchmark. Under F1: small.en+prompt = medium.en+prompt =
  tuned parakeet at **83.3%**; base.en+prompt drops to **75.0%**, strictly
  behind all three (see §4 and §6 for what this means for the task 924
  recommendation).
- **tiny.en-noprompt's recall (23.1%) is lower than its old name-accuracy
  number (30.8%).** The old whole-utterance count credited it with a name it
  hadn't actually produced in the right place; positional alignment catches
  that. This is direct evidence that position-blindness, not just
  insertion-blindness, mattered to the old metric's error.
- **No-prompt configs' F1 rises relative to their old number** (e.g.
  base.en-noprompt 30.8% → 44.4%) purely because they are precise-but-silent —
  they rarely say a mesa name at all, so they rarely get one wrong either, and
  the precision term rewards that. Recall for base.en-noprompt is unchanged at
  30.8%. Don't read the F1 rise as the engine getting more accurate at names;
  it didn't.

## 4. Findings

**medium.en is unusable for live dictation on this host.** RTF is 1.955 (no prompt) / 2.173 (prompt) — roughly 2x realtime, i.e. it takes twice as long to decode as the audio takes to speak. It also doesn't win on accuracy: medium.en+prompt gets 83.3% name F1 and 3.3% WER, matching small.en+prompt on both (83.3% F1, 3.3% WER) and actually *ahead* of base.en+prompt's 75.0% F1. Its 8x higher RTF and 6.6x higher peak RSS (2232 MB vs 338 MB for base.en+prompt) buy nothing over small.en+prompt here.

**The initial prompt is the single highest-leverage lever available, at essentially zero latency cost.** Per-model name F1, no-prompt → prompt:
- tiny.en: 35.3% → 69.6% (2.0x)
- base.en: 44.4% → 75.0% (1.7x)
- small.en: 60.0% → 83.3% (1.4x)
- medium.en: 52.6% → 83.3% (1.6x)

The no-prompt side of this comparison is higher than it used to look (e.g. base.en-noprompt was 30.8% under the old recall-only metric, now 44.4%) — that is not the no-prompt engines getting more accurate, it's the corrected metric giving them credit for staying precise. No-prompt configs rarely say a mesa name at all, so their recall is unchanged (still 30.8% for base.en) but they get almost no false positives either, which the F1 rewards. The prompt is still the thing that actually recovers the missing names — recall is what moves.

RTF cost of the prompt is negligible: base.en goes from 0.240 → 0.244 RTF (+1.7%), small.en 0.640 → 0.657 (+2.7%), medium.en 1.955 → 2.173 (+11%, still within noise given medium's already-unusable RTF). WER also improves with the prompt in every model (e.g. base.en 9.3% → 4.7%; small.en 7.3% → 3.3%).

**The prompt has a failure mode of its own: it injects vocabulary into text that never contained it.** tiny.en is the only config whose WER got *worse* with the prompt (12.0% → 13.3%), driven by insertions rising 1 → 6. The mechanism is visible in the transcripts — tiny.en+prompt turned "wire up qorvex to the **iOS** simulator" into *"wire-up korvex to the **auris** simulator"*, substituting a prompt term into an unrelated phrase. base.en+prompt shows a milder version, hearing `khora` as `qorvex` in u02 — one project noun swapped for another rather than corrected. So the prompt does not only pull toward correct names; it pulls toward *any* prompt name, including into slots where none was spoken. base.en is the floor at which the prompt is a net win, and any downstream correction layer should expect confusions **among** mesa's own terms, not just between a mesa term and an English word.

**parakeet is the fastest engine by a wide margin — but only in a warm/persistent process.** In-process RTF is 0.087, ~1.8x faster than whisper tiny.en (0.153) and ~2.8x faster than base.en (0.244). But spawned fresh per utterance, effective RTF is 0.634 because each invocation pays ~4s to load the 652 MB int8 encoder. A persistent parakeet process (loaded once, decoding many utterances) is required to realize the fast number; a cold-process-per-utterance architecture would put parakeet roughly on par with whisper small.en for latency.

**The real tension for task 924: parakeet is fast but can't be steered; whisper+prompt is steerable but costs more RTF/RAM.** This setup's sherpa-onnx parakeet integration has no prompt or vocabulary-biasing mechanism, so its name F1 (66.7%) is stuck wherever the model's native vocabulary lands it — better than unprompted whisper of any size, but behind small.en+prompt and medium.en+prompt (83.3% each) and roughly level with base.en+prompt (75.0%). Whisper's `--prompt` flag is cheap and effective, but even base.en+prompt at 0.244 RTF is ~2.8x slower than parakeet's warm in-process RTF. There is no config in this benchmark that is both the fastest and the most name-accurate — pick one axis to prioritize. (Section 6 revisits this once parakeet's own hotword biasing is on the table.)

**Recommended default for task 924: whisper base.en with the initial prompt.** This recommendation was written under the old recall-only metric, where base.en+prompt showed 76.9% name accuracy "tied for the best in this benchmark, matching small.en+prompt and medium.en+prompt." **That basis no longer holds.** Under the corrected metric (§0, task 949), base.en+prompt scores 75.0% name F1 while small.en+prompt and medium.en+prompt both score 83.3% — base.en+prompt is not tied for best, it is strictly behind both. The RTF/RSS case for base.en (below) is unchanged and still real, but the accuracy justification for picking it over small.en+prompt is gone. **This finding should not be treated as settled — task 924 should re-read §6, where tuned parakeet now matches the best name F1 in the benchmark (83.3%) at a lower WER and roughly 3x the speed of base.en+prompt.** The rest of this paragraph is preserved as originally written, for the RTF/RSS case only:
- RTF 0.244 — comfortably realtime (4x headroom) on this 2019-era CPU-only host, unlike medium.en.
- WER 4.7% — worse than small.en+prompt's 3.3%.
- Peak RSS 338 MB and cold-start 1.25s — both a fraction of small.en+prompt (834 MB / 3.4s) or medium.en+prompt (2232 MB / 11.24s), which matters for running alongside other apps on a 32 GB box with no GPU offload.

If peak name accuracy matters more than resource footprint, small.en+prompt is the stronger pick now: 83.3% name F1 vs. base.en+prompt's 75.0%, better WER (3.3% vs 4.7%), at 2.7x the RTF and 2.5x the RAM of base.en+prompt.

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

## 6. Addendum, 2026-08-27: parakeet contextual biasing (task 948)

Section 4 above claims "this setup's sherpa-onnx parakeet integration has no
prompt or vocabulary-biasing mechanism". That is wrong. `sherpa_onnx 1.13.6`'s
`OfflineRecognizer.from_transducer` accepts `hotwords_file`, `hotwords_score`,
`modeling_unit` and `bpe_vocab`, and they work with `model_type="nemo_transducer"`
— they are simply ignored unless `decoding_method="modified_beam_search"`, and
the original benchmark ran greedy search. Same host, same 8 fixtures, same
scorer; harness at `spike/harness/parakeet_hotwords_bench.py`, raw output in
`spike/results/hotwords/`.

**`bpe_vocab` must be a sentencepiece-style `token score` file, not `tokens.txt`.**
`tokens.txt` is `token id`, and passing it produces a silently miscalibrated
tokenizer: hotwords then do nothing at scores 2-4 and destroy the transcript at
10 (name accuracy 84.6% on output reading *"qorvex qo' qo' auris auris koko"*,
WER 91.3%). For a BPE model the piece score is the negative merge rank, so the
correct file is derivable from `tokens.txt` directly and is committed as
`spike/models/parakeet/bpe_synth.vocab`:

```
awk '{printf "%s %d\n", $1, -NR+1}' tokens.txt > bpe_synth.vocab
```

### Uniform hotword score sweep (all 6 names at the same boost)

| Config | RTF (warm, in-process) | WER | Name F1 |
|---|---|---|---|
| greedy, no hotwords (section 1 baseline) | 0.087 | 5.3% | 66.7% |
| modified_beam_search, no hotwords | 0.076 | 5.3% | 66.7% |
| + hotwords, score 1 / 2 / 3 / 3.5 | 0.074-0.077 | **4.0%** | 78.3% |
| + hotwords, score 4 | 0.077 | 6.0% | 80.0% |
| + hotwords, score 4.5 | 0.078 | 9.3% | 80.0% |
| + hotwords, score 5 | 0.078 | 9.3% | 84.6% |
| + hotwords, score 6 | 0.079 | 22.0% | 71.0% |
| + hotwords, score 7 | 0.079 | 69.3% | 42.9% |

The corrected metric separates configs the old one couldn't. Score 5.0 and
6.0 used to tie at 84.6% "name accuracy"; under F1, 5.0 holds 84.6% while 6.0
collapses to **71.0%** (precision 61.1% — 7 name false positives, up from 2 at
score 5.0). Score 7.0 falls from 69.2% to **42.9%** (precision 31.0%, 20 name
false positives): this is the babble case — the beam starts inserting and
substituting mesa vocabulary everywhere as the boost overwhelms the acoustic
model — being punished, which is exactly the acceptance criterion task 949
set out to fix.

### Best config: per-word boosts

sherpa accepts a per-line boost (`khora :6.5`), so the hard terms can be pushed
without paying the WER cost on the easy ones. `spike/fixtures/hotwords.txt`:

```
mesa :3.0
auris :3.0
khora :6.5
qorvex :5.0
helios :3.0
kokoro :3.0
```

| Config | RTF (warm) | WER | Name F1 |
|---|---|---|---|
| **parakeet + per-word hotwords** | **0.082** | **4.0%** | **83.3%** |
| whisper small.en + prompt | 0.657 | 3.3% | 83.3% |
| whisper base.en + prompt (section 1 recommendation) | 0.244 | 4.7% | 75.0% |

Parakeet with per-word biasing now matches the best name F1 in the whole
benchmark — small.en+prompt, not base.en+prompt (base.en+prompt no longer ties
for best; see §4) — at a lower WER than either, at roughly 3x base.en+prompt's
speed warm. Modified beam search costs nothing measurable here (0.076 vs
0.087 RTF greedy).

**Correction:** this section previously claimed the biasing showed "no
cross-term contamination" at usable strengths, citing `ins = 0` in every
score file. That claim was wrong — it relied on the old metric's blind spot.
`ins = 0` only means no name was *inserted as an extra word*; it says nothing
about a name *substituted* into a slot where a different word (or a different
vocab term) was spoken, which the corrected alignment-based metric does
catch. Every hotwords config in the sweep above has at least 1 name false
positive, and the score 4.0/4.5/5.0 configs each have 2. One of those two is
the helios/"hell EOS" fixture pun that affects every engine in this benchmark
(§2) — not specific to biasing. The other is real: at score ≥ 4.0, u03's
hypothesis reads *"Create a task in mesa **khora** qorvex to the iOS
simulator..."* where the reference says *"create a task in mesa **called
wire up** qorvex..."* (verified in `spike/results/hotwords/hot_bpe_4.0.tsv`
against `spike/fixtures/utterances.tsv`) — the khora hotword gets pulled into
a slot where "called wire up" was actually said, while the two *real* khora
occurrences (u02, u07) are still both missed as "qora." Biasing at usable
strengths is therefore roughly *comparable* to whisper's prompt on precision,
not cleaner than it: hot_bpe 1.0-3.5 reaches 90.0% and 4.0-5.0 sits at
83.3-84.6%, against 80.0% (tiny), 81.8% (base) and 90.9% (small, medium) for
whisper+prompt — overlapping ranges, with the best whisper+prompt configs
slightly ahead of the best biasing ones. What both have in common is that
they stay far clear of the precision collapse at scores 6.0/7.0 (61.1% /
31.0%). "No contamination" was an artifact of a metric that could not see a
substitution, not a property of the biasing mechanism.

### What this does not fix

The two real occurrences of `khora` (u02, u07) are still never produced —
both are still heard as *"qora"* at every boost tested. Raising its boost to
8 or 10 with the others left at 3 changes nothing about that; the correct
token sequence for those two spots is not reachable from this audio, so the
post-ASR correction pass over mesa vocabulary is still required regardless of
engine. `qora` → `khora` is a trivial edit distance, so that layer is cheap.
(The literal string `khora` does appear, spuriously, at score ≥ 4.0 in u03 —
see the correction above — but that is the beam hallucinating the word into
an unrelated slot, not the model correctly hearing either real occurrence.)

### Consequences

- Task 924's engine decision is genuinely open again; the "parakeet cannot be
  steered" premise no longer holds. Parakeet's remaining real costs are the ~4s
  model load (persistent process required) and 1557 MB peak RSS vs 338 MB.
- Caveat from section 1 still applies in full: these are `say` TTS fixtures, so
  the absolute numbers are a floor and only the ordering is trustworthy.
- The name-accuracy metric used to be recall-only and rewarded babble; that was
  fixed in task 949 (§0). All F1/precision/recall numbers in this document,
  including this addendum's tables, are the corrected metric.

## 7. Post-ASR vocabulary correction (task 950)

Section 6 established that `khora` is never produced correctly by any engine
or configuration tested here, prompted or biased — the gap left after biasing
is a spelling problem, not an acoustic one. `spike/harness/vocab_correct.py`
closes it with a lexical correction pass over the six hotword terms, run on
decoded transcript text, ported from mesa's task-922 vocabulary correction
(`frontend/src/liveRecognition.ts`), with one addition on top of the port: a
correction requires an edit-distance match on top of mesa's sound-key match,
because the sound-key rule alone rewrote 321 ordinary words from
`/usr/share/dict/web2` at this project's vocabulary scale. Design rationale,
the term-to-term, capitalization and edit-distance decisions, and full "what
this does not fix" detail are in `docs/correction.md`; this section records
the measurement only.

Reproduction: `bash spike/harness/run_correction.sh`. Corrected transcripts
and scores are written to `spike/results/corrected/<config>.tsv` and
`<config>.score.json`, one pair per existing `spike/results/raw/*.tsv` and
`spike/results/hotwords/*.tsv`.

| Config | WER before | WER after | ΔWER | F1 before | F1 after | ΔF1 |
|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-int8 | 5.3% | 1.3% | −4.0p | 66.7% | 96.3% | +29.6p |
| whisper-base.en-noprompt | 9.3% | 6.0% | −3.3p | 44.4% | 78.3% | +33.8p |
| whisper-base.en-prompt | 4.7% | 3.3% | −1.3p | 75.0% | 84.6% | +9.6p |
| whisper-medium.en-noprompt | 6.7% | 1.3% | −5.3p | 52.6% | 96.3% | +43.7p |
| whisper-medium.en-prompt | 3.3% | 1.3% | −2.0p | 83.3% | 96.3% | +13.0p |
| whisper-small.en-noprompt | 7.3% | 4.7% | −2.7p | 60.0% | 83.3% | +23.3p |
| whisper-small.en-prompt | 3.3% | 1.3% | −2.0p | 83.3% | 96.3% | +13.0p |
| whisper-tiny.en-noprompt | 12.0% | 8.0% | −4.0p | 35.3% | 78.3% | +43.0p |
| whisper-tiny.en-prompt | 13.3% | 10.0% | −3.3p | 69.6% | 92.9% | +23.3p |
| hot_bpe_1.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_2.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_3.0 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_3.5 | 4.0% | 1.3% | −2.7p | 78.3% | 96.3% | +18.0p |
| hot_bpe_4.0 | 6.0% | 4.0% | −2.0p | 80.0% | 92.9% | +12.9p |
| hot_bpe_4.5 | 9.3% | 8.0% | −1.3p | 80.0% | 85.7% | +5.7p |
| hot_bpe_5.0 | 9.3% | 8.7% | −0.7p | 84.6% | 85.7% | +1.1p |
| hot_bpe_6.0 | 22.0% | 20.7% | −1.3p | 71.0% | 78.8% | +7.8p |
| hot_bpe_7.0 | 69.3% | 69.3% | +0.0p | 42.9% | 41.9% | −1.0p |
| mbs | 5.3% | 1.3% | −4.0p | 66.7% | 96.3% | +29.6p |
| **tuned** (chosen config) | **4.0%** | **2.0%** | **−2.0p** | **83.3%** | **96.3%** | **+13.0p** |

WER never regresses in any config. Name F1 improves everywhere except
`hot_bpe_7.0`, where the pass folds a babble fragment onto a real term
spelling and adds one false positive on an already-failed decode — see
`docs/correction.md`, "What this does not fix," for the mechanism.

The maximum name F1 reached by any config after correction is 96.3%, not
100% — the one residual error (u06's fixture pun producing a second, already
correctly-spelled "Helios") is out of reach for a lexical pass by
construction. `docs/correction.md` covers what the pass does and does not
fix in full, including the acceptance-criterion shortfall.
