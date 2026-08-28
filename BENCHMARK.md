# Benchmark: auris against the Web Speech baseline

Date: 2026-08-28. Task 945.
Host: Intel Core i9-9880H @ 2.30 GHz, 16 logical cores, macOS 26.5.1, CPU only.
Binary: `auris 0.1.0`, release build, model `spike/models/parakeet`
(NVIDIA Parakeet TDT 0.6B v2, int8 ONNX, sherpa-onnx 1.13.6).

**Verdict, up front: the latency budget is met for every utterance of ten
seconds or less, and breached only by the two longest clips in the corpus.
The vocabulary claim holds — but it is the correction pass, not the biasing,
that carries it. The Web Speech columns are NOT MEASURED: the method is
decided, written down and committed, and it is blocked on a piece of
hardware configuration this run could not perform. That gap is stated in
full below rather than papered over.**

## What was measured, and what was not

| | Status |
|---|---|
| auris, no vocabulary | measured, 40 utterances |
| auris, vocabulary biasing (what the crate ships) | measured, 40 utterances |
| auris, biasing + post-ASR correction | measured, 40 utterances |
| Web Speech, raw | **NOT MEASURED** — see "The Web Speech baseline" |
| Web Speech, after task 922 `correctVocabulary` | **NOT MEASURED** — same reason |

## ⚠️ Fixture caveat — read before trusting absolute numbers

The 40 reference utterances were synthesised with macOS `say` across five
voices (Daniel, Samantha, Karen, Moira, Tessa) at three speaking rates
(175, 195, 160 wpm) — **not the owner's real dictated voice.** This run had no microphone
access. Synthetic speech is cleaner, more evenly paced and more
consistently articulated than real dictation, so:

- **Absolute WER and name-accuracy numbers below are optimistic. Treat them
  as a floor**, not a prediction of real-world performance.
- **Relative comparisons are the trustworthy part** — config vs config,
  biasing vs correction, warm vs cold. Those orderings should survive real
  speech even as the absolute numbers move.

This is the same caveat `spike/RESULTS.md` carries, for the same reason, and
it is the single largest threat to this document's conclusions.
`bench/corpus/record.sh` re-records the same 40 sentences in a real voice —
it accepts individual ids (`record.sh u09 u10`) because 40 in one sitting is
a lot. After recording, `bench/harness/run_auris.sh` regenerates every
number here unchanged.

## The corpus

40 utterances, 298.03 s (4:58) of audio, `bench/corpus/utterances.tsv`, with
16 kHz mono WAVs in `bench/corpus/wav/`. u01–u08 are the spike's original
eight, copied byte-identical so the two documents' numbers stay comparable;
u09–u40 were written for this task.

They are written as dictation into mesa's live conversation view, not as
read passages: task ids spoken aloud, mesa's own tool names, half-thoughts
and mid-sentence corrections, lengths from four words to 22 seconds of
unbroken rambling. Two constraints were deliberate:

- **Some utterances contain no vocabulary term at all**, so that name
  *precision* is measurable. A config that babbles mesa names into every
  slot must be punished, not rewarded.
- **Utterances u31–u40 were added specifically to give the punctuation
  metric a base.** The first 30 contained only 7 internal sentence
  boundaries between them, which is too thin for an F1 to mean anything —
  one hit moved it by 20 points. The corpus now has 29. The ten new
  utterances were *written* as multi-sentence thoughts; no existing
  reference was repunctuated to manufacture boundaries, and u01–u30 are
  byte-identical to what they were before.

## The metrics

Scorer: `bench/harness/score.py`, extended from `spike/harness/score.py`.
The WER and name-accuracy definitions are unchanged from the spike and
reproduce its numbers exactly on its data.

- **WER** — Levenshtein over normalised words (lowercased, punctuation
  stripped, digit and spelled-out numbers folded to the same digit
  sequence), concatenated across utterances.
- **Vocabulary accuracy** — alignment-based **F1** over the six mesa terms.
  A term counts only when it lands on a matching slot in the
  hypothesis-to-reference alignment; an inserted or substituted name is a
  false positive. Recall alone would reward babble, which is why task 949
  replaced it.
- **Punctuation** — sentence-boundary F1. A boundary is a position after a
  token ending in `.`, `!` or `?`; **the trailing boundary at the end of
  each utterance is excluded from both sides**, because every utterance has
  one trivially and counting it inflates the score. Hypothesis boundaries
  are mapped onto reference positions through the word alignment. Reported
  alongside is a blunter companion, `any_terminal_punct_rate` — the
  fraction of utterances whose hypothesis contains *any* terminal mark at
  all — because the failure this metric was written to catch (an engine
  that emits no punctuation whatsoever) shows up more honestly there.
- **Latency** — wall-clock of the whole client process, measured with
  `time.perf_counter()` around `subprocess.run`: spawn, WAV read, socket
  round trip, decode, teardown. Reported as a distribution.

## 1. Vocabulary accuracy — the headline table

The three columns the brief asks for, plus the two auris configs that
bracket them:

| Config | Name F1 | Precision | Recall |
|---|---|---|---|
| Web Speech, raw | **NOT MEASURED** | — | — |
| Web Speech + task 922 `correctVocabulary` | **NOT MEASURED** | — | — |
| auris, no vocabulary | 62.2% | 95.8% | 46.0% |
| **auris + vocabulary biasing** (ships today) | **69.2%** | 96.4% | 54.0% |
| auris + biasing + correction pass | **89.1%** | 97.6% | 82.0% |

Per term, which is where the real story is:

| Term | plain | + biasing | + correction |
|---|---|---|---|
| mesa | 93.3 | 100.0 | 100.0 |
| auris | 80.0 | 87.5 | 87.5 |
| **khora** | **0.0** | **0.0** | **76.9** |
| **qorvex** | **0.0** | 50.0 | **90.9** |
| helios | 83.3 | 83.3 | 92.3 |
| kokoro | 100.0 | 100.0 | 100.0 |

**Biasing alone does not recover `khora`.** At 40 utterances and eight
occurrences of the term, the number is the same 0.0 the spike found at
eight utterances: `spike/RESULTS.md` §6 and §8 predicted exactly this, and
a corpus five times larger did not soften it. What biasing does is decide
*what the model produces instead* — "qora" and "qor" rather than something
unrecoverable — and the correction pass then repairs it to 76.9. That is
the two-layer argument in `docs/correction.md`, measured at scale:
`khora :6.5` is a threshold that keeps the miss long enough to be
repairable, not an escalation.

**The correction pass is not in the crate.** It lives only as
`spike/harness/vocab_correct.py`; `src/vocabulary.rs` says so in its module
docs, and grepping `src/` confirms it. So the row auris actually ships
today is 69.2%, and the 89.1% row is a promise about the port, not a
current capability. Stated plainly because the difference is 20 points.

## 2. Word error rate

| Config | WER |
|---|---|
| auris, no vocabulary | 6.52% |
| auris + vocabulary biasing | 6.32% |
| auris + biasing + correction | 4.89% |

Biasing barely moves WER (−0.20p) and the correction pass moves it properly
(−1.43p), which is the expected shape: both act on six words out of 981,
but the correction pass fixes every occurrence rather than nudging a beam.
WER never regressed in any config, matching `spike/RESULTS.md` §7.

## 3. Punctuation

| Config | Boundary F1 | Precision | Recall | Any terminal punct | Commas (ref 39) |
|---|---|---|---|---|---|
| auris, no vocabulary | 27.8% | 71.4% | 17.2% | 100.0% | 40 |
| auris + vocabulary biasing | 17.6% | 60.0% | 10.3% | 85.0% | 37 |
| auris + biasing + correction | 17.6% | 60.0% | 10.3% | 85.0% | 37 |

29 reference boundaries; auris produced 5–7.

**This is the weakest result in the document, and it should not be read as
a win.** auris punctuates the *end* of an utterance almost always, and
places commas at very nearly the reference rate (37–40 against 39). What it
does not do is end a sentence in the middle of a turn. Recall of 10–17%
means roughly one internal boundary in seven is found; the rest are
swallowed. u33 is typical — reference: *"khora dropped every websocket
frame during the recording. i checked the console and there's nothing
useful in it. we'll need to reproduce it live…"*; auris: *"Khora dropped
every web socket frame during the recording I checked the console and
there's nothing useful in it we'll…"* — three sentences merged into one
run-on, with both boundaries lost.

The brief names punctuation as a specific Web Speech failing, and the
honest position is that **this benchmark cannot yet say auris is better at
it**, because the Web Speech column is missing. What it can say is that
auris's own internal-boundary recall is poor in absolute terms. The
`any_terminal_punct_rate` column is the one to watch when the baseline
lands: an engine that emits no terminal punctuation at all scores 0% there,
and that single number will settle the comparison more bluntly than F1.

One caveat on precision: with only 5–7 hypothesis boundaries, precision is
computed over a handful of events and is noisy. The recall figure, over 29
reference boundaries, is the sturdier half.

Note also that biasing *costs* punctuation here (27.8 → 17.6, and terminal
punctuation drops from 100% to 85% of utterances). The hotword beam is
buying name recall with sentence structure. That trade was not previously
measured and is worth a follow-up.

## 4. Latency

120 measurements: 40 utterances × 3 repetitions, against an already-warm
`auris serve`, three warm-up calls discarded, `--vocabulary-file` on every
call. Client process wall-clock, in milliseconds.

| | min | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| **Warm (daemon)** | 211 | **590** | 940 | 1231 | **1504** | 1885 | **1993** |
| Cold (`--no-daemon`, 40 calls) | 2757 | 3186 | 3563 | 4290 | 4460 | 7268 | 9061 |

Daemon RTF over the whole corpus: 0.097. Cold: 0.460.

Normalised per second of audio, warm: p50 100 ms/s, p95 166 ms/s.

### Verdict against the budget

`docs/latency.md` sets **300 ms of added latency**, with a hard ceiling of
**2300 ms** from mouth-close to the turn existing, and argues the cost is
zero *by structure*: auris's work overlaps mesa's existing 2000 ms silence
wait rather than following it. Taking that argument's own timeline — t=0 is
the mouth closing, the VAD closes the segment at t≈500 ms — auris has
**1500 ms of slack** before mesa's auto-send timer fires, and added latency
is `max(0, client_ms − 1500)`.

| | Result |
|---|---|
| Calls finishing inside the 1500 ms slack (0 ms added) | **113 / 120 = 94.2%** |
| Calls exceeding the 300 ms budget (>1800 ms) | **2 / 120 = 1.7%** |
| Calls exceeding the 2300 ms hard ceiling (>1800 ms → turn >2300 ms) | 2 / 120 |
| Worst case | u14, 1992.7 ms → 493 ms added, turn at 2493 ms |

**The budget is met, with a stated exception.** Every one of the 93
measurements on utterances of ten seconds or less finished in **1111 ms or
better** — p50 533 ms — comfortably inside the slack, costing nothing at
all. The entire breach is three clips: u14 (22.45 s of continuous speech),
u27 (19.97 s) and u37 (15.14 s), and only u14 crosses the 300 ms budget, on
two of its three repetitions.

Two things temper that exception, and one sharpens it:

- **Tempering:** a 22-second unbroken utterance is not what mesa's VAD would
  hand auris. `docs/streaming.md` segments on silence, so the real per-call
  payload is a segment, not a whole monologue. And the measured wall-clock
  includes process spawn and reading a whole WAV off disk, neither of which
  is on mesa's streaming path.
- **Sharpening:** 95.8% of calls exceed 300 ms in absolute terms. The budget
  is met only because of the overlap argument. If mesa's timer is ever
  retuned downward — `live.auto-send-ms` clamps as low as 250 ms — the slack
  vanishes and essentially every call becomes visible. The budget is met by
  structure, exactly as `docs/latency.md` claimed, and it is that structure
  and not auris's speed that is load-bearing.

### The cold process is the real risk

`docs/latency.md` names it: *"the one thing that destroys it is a cold
process."* Measured, that is not an exaggeration. **Every single one of the
40 `--no-daemon` calls breached the 2300 ms ceiling** — the fastest was
2757 ms, before the VAD's 500 ms is even added, and the worst was 9061 ms.
There is no tail to argue about here; the distribution does not overlap the
budget at any percentile. A daemon that is not running is not a slow auris,
it is a broken one.

## The Web Speech baseline — the method, and why it is unmeasured

The brief asks for the method to be stated, and warns that an
unreproducible baseline is worth less than a stated one. The method is
decided and committed at `bench/harness/webspeech/` — capture page, CDP
driver, and a README. It was not run, for a reason worth recording.

`SpeechRecognition` listens to a microphone, not to a file. The obvious
workaround is Chrome's fake audio device
(`--use-file-for-fake-audio-capture`). **It does not work, and the way it
fails is a trap.** Two findings, both measured on this host with Google
Chrome 151.0.7922.174:

1. The fake device *can* be made to feed a WAV into `getUserMedia`, but only
   with `--disable-features=AudioServiceSandbox` added to the usual flags.
   Without it Chrome logs `Failed to read <path> as input to the fake
   device` and the microphone reads digital silence — measured WebAudio RMS
   exactly 0. With it, peak RMS 0.3334.
2. **`SpeechRecognition` does not read from that device at all.** Measured
   in a single page and session running an RMS meter on `getUserMedia`
   alongside `webkitSpeechRecognition`, against the same fake device:

   ```json
   {"micRMSmax":0.3334,"micSamples":75,"speechStarted":true,"speechError":"no-speech","speechResults":[]}
   ```

   The page's own microphone hears the file loudly and continuously while
   the recognizer, in that same session, hears nothing and ends in
   `no-speech`. Chrome's recognizer opens the system default audio input
   directly and bypasses the media-stream pipeline.

So a file-driven Web Speech baseline needs an **OS-level loopback** set as
the macOS default input. Three HAL plugins are already installed on this
host; the two that expose inputs were tested and neither loops back
(playing into "Microsoft Teams Audio" and recording from it captured pure
silence, peak 0). A dedicated driver — `brew install blackhole-2ch` — needs
admin rights, which this run did not have and would not have taken
unattended.

`capture.py` therefore ships two modes:

- `--mode loopback` (preferred): clean, bit-exact, reproducible on any
  machine with the driver installed.
- `--mode acoustic`: plays the corpus through the speakers and lets the
  built-in microphone hear it — nothing to install, but it degrades the
  signal in ways that are not reproducible. It records what the microphone
  heard into `bench/results/acoustic-wav/`, **and auris must then be
  rescored against those recordings rather than the clean corpus.**
  Comparing Web Speech on room audio against auris on clean audio would be
  a rigged benchmark, and the harness is built to make the fair comparison
  the easy one.

Either mode also needs its raw transcripts run through
`spike/harness/vocab_correct.py` to produce the second column, so that the
comparison is against task 922 and not against an unpatched browser.

**What this means for the acceptance criterion:** the table above has three
vocabulary columns as asked, but two of them are empty, and the benchmark's
central claim — that auris beats the browser — is therefore **not yet
demonstrated by measurement.** The apparatus to demonstrate it is written,
committed and one `brew install` away from running. That is the honest
state, and it is tracked as **task 964**, which carries the operator steps
an agent cannot perform.

## Reproduction

```bash
# 1. corpus (synthetic; skip if bench/corpus/wav/ is populated)
bash bench/corpus/synthesize.sh

# 1b. or re-record it in a real voice — this is the run that matters
bash bench/corpus/record.sh            # all 40, or: record.sh u09 u10

# 2. auris: three configs, scores, and the latency distribution
bash bench/harness/run_auris.sh

# 3. Web Speech baseline (needs a loopback as default in+out)
python3 bench/harness/webspeech/capture.py \
    --corpus bench/corpus/utterances.tsv --wav-dir bench/corpus/wav \
    --out bench/results/webspeech-raw.tsv --device "BlackHole 2ch"
python3 spike/harness/vocab_correct.py \
    bench/results/webspeech-raw.tsv > bench/results/webspeech-corrected.tsv
python3 bench/harness/score.py bench/corpus/utterances.tsv \
    bench/results/webspeech-raw.tsv
```

Raw data: `bench/results/*.tsv`, `*.score.json` (per-utterance included),
`*.csv`, `latency-raw.tsv`, `latency.json`, `latency-cold.json`.

## Open questions this run raised

- **Biasing costs punctuation** (§3): boundary F1 27.8 → 17.6 and terminal
  punctuation 100% → 85% when the hotwords file is passed. Unexplained.
- **Internal sentence boundaries are mostly lost** (§3), recall 10–17%. If
  auris is to beat the browser on punctuation specifically, this is the
  number that has to move.
- **The corpus is still synthetic.** Every conclusion above inherits that.
