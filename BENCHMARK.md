# Benchmark: auris against the Web Speech baseline

Date: 2026-08-28. Task 945, 964.
Host: Intel Core i9-9880H @ 2.30 GHz, 16 logical cores, macOS 26.5.1, CPU only.
Binary: `auris 0.1.0`, release build, model `spike/models/parakeet`
(NVIDIA Parakeet TDT 0.6B v2, int8 ONNX, sherpa-onnx 1.13.6).

**Verdict, up front: the Web Speech columns are now measured, and auris beats
the browser decisively. On the same acoustic recordings — both engines
scored on exactly what a MacBook microphone heard, per the fairness rule
below — auris's full configuration (vocabulary + correction) scores 10.8%
WER and 79.1% name F1 against Web Speech's best configuration (best-effort
finalization + `correctVocabulary`) at 52.9% WER and 21.4% name F1. auris's
WORST configuration (plain, no vocabulary: 13.4% WER, 52.2% name F1) still
beats Web Speech's BEST on both axes. On punctuation the gap is total: Web
Speech emits zero terminal punctuation marks across all 40 utterances in
every configuration; auris emits at least one in 87.5–100% of utterances.
The vocabulary claim still holds, and it is still the correction pass, not
the biasing, that carries it. The latency verdict is unchanged: the budget
is met for every utterance of ten seconds or less, breached only by the two
longest clips in the corpus.**

## What was measured

| | Status |
|---|---|
| auris, no vocabulary (clean corpus) | measured, 40 utterances |
| auris, vocabulary biasing (clean corpus) | measured, 40 utterances |
| auris, biasing + post-ASR correction (clean corpus) | measured, 40 utterances |
| auris, all three configs, rescored on acoustic recordings | measured, 40 utterances |
| Web Speech, strict (finals only) | measured, 40 utterances, acoustic mode |
| Web Speech, strict + task 922 `correctVocabulary` | measured, 40 utterances, acoustic mode |
| Web Speech, best-effort (final, else last interim) | measured, 40 utterances, acoustic mode |
| Web Speech, best-effort + `correctVocabulary` | measured, 40 utterances, acoustic mode |

## ⚠️ Fixture caveat — read before trusting absolute numbers

The 40 reference utterances were synthesised with macOS `say` across five
voices (Daniel, Samantha, Karen, Moira, Tessa) at three speaking rates
(175, 195, 160 wpm) — **not the owner's real dictated voice.** The original
task-945 run had no microphone access; task 964 used the built-in microphone
only to capture Web Speech's baseline acoustically (see "The Web Speech
baseline — method used" below), not to re-record the reference corpus
itself. So the corpus remains synthetic throughout this document, and:

- **Absolute WER and name-accuracy numbers below are optimistic. Treat them
  as a floor**, not a prediction of real-world performance.
- **Relative comparisons are the trustworthy part** — config vs config,
  engine vs engine, biasing vs correction, warm vs cold. Those orderings
  should survive real speech even as the absolute numbers move.

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

**Primary comparison: both engines scored on the same acoustic recordings**
(`bench/results/acoustic-wav/`, what the microphone actually heard — see
"The Web Speech baseline — method used" below for why this is the fair
comparison):

| Config | WER | Name F1 | Precision | Recall |
|---|---|---|---|---|
| Web Speech, strict (finals only) | 62.2% | 7.7% | 100.0% | 4.0% |
| Web Speech, strict + `correctVocabulary` | 62.1% | 11.3% | 100.0% | 6.0% |
| **Web Speech, best-effort** [HEADLINE] | 53.1% | 14.8% | 100.0% | 8.0% |
| Web Speech, best-effort + `correctVocabulary` | 52.9% | 21.4% | 100.0% | 12.0% |
| auris, no vocabulary | 13.4% | 52.2% | 94.7% | 36.0% |
| auris + vocabulary biasing (ships today) | 11.9% | 61.3% | 92.0% | 46.0% |
| **auris + biasing + correction pass** | **10.8%** | **79.1%** | 94.4% | 68.0% |

auris's *worst* acoustic-mode configuration (plain, 13.4% WER / 52.2% name
F1) beats Web Speech's *best* (best-effort + `correctVocabulary`, 52.9% WER
/ 21.4% name F1) on both axes. Web Speech's name precision is 100% in every
one of its four configurations — **it does not babble the terms into wrong
slots, it simply almost never produces them**; recall tops out at 12%.

Per term, acoustic mode, the strongest configuration each engine has:

| Term | Web Speech, best-effort + correctVocab | auris + vocabulary + correction |
|---|---|---|
| mesa | 54.5 (100.0/37.5) | 94.1 (88.9/100.0) |
| **auris** | **0.0 (0/0)** | 80.0 (100.0/66.7) |
| khora | 22.2 (100.0/12.5) | 72.0 (100.0/56.2) |
| **qorvex** | **0.0 (0/0)** | 80.0 (100.0/66.7) |
| helios | 28.6 (100.0/16.7) | 72.7 (80.0/66.7) |
| **kokoro** | **0.0 (0/0)** | 75.0 (100.0/60.0) |

**Web Speech never once produces `auris`, `qorvex` or `kokoro` correctly,
across all 40 utterances, in its best configuration.** These are exactly
the terms with no language-model prior and no English-word shape; Chrome's
recognizer has no biasing mechanism to compensate for them, and Web Speech's
own `correctVocabulary` pass cannot correct a term that never appears in the
transcript to be corrected. This is the sharpest single finding in this
section.

### Reference only — clean corpus, not comparable to the rows above

These are the original numbers from this document's first run, decoded from
the clean `bench/corpus/wav/` rather than the acoustic recordings. Web
Speech has never been run against this audio — loopback capture was never
available (see "method used" below) — so there is no Web Speech row here to
compare against. Keep these as a ceiling on auris's own best-case
performance, not as evidence about the browser:

| Config | WER | Name F1 | Precision | Recall |
|---|---|---|---|---|
| auris, no vocabulary | 6.5% | 62.2% | 95.8% | 46.0% |
| auris + vocabulary biasing (ships today) | 6.3% | 69.2% | 96.4% | 54.0% |
| auris + biasing + correction pass | 4.9% | 89.1% | 97.6% | 82.0% |

Per term, clean corpus, which is where the two-layer correction argument was
first made:

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
today is 69.2% on the clean corpus (61.3% acoustic), and the 89.1%/79.1%
rows are a promise about the port, not a current capability. Stated plainly
because the difference is 20 points either way.

## 2. Word error rate

**Acoustic mode, both engines on the same recordings** — the comparison
that matters:

| Config | WER |
|---|---|
| Web Speech, strict | 62.2% |
| Web Speech, strict + `correctVocabulary` | 62.1% |
| Web Speech, best-effort | 53.1% |
| Web Speech, best-effort + `correctVocabulary` | 52.9% |
| auris, no vocabulary | 13.4% |
| auris + vocabulary biasing | 11.9% |
| auris + biasing + correction | 10.8% |

auris's full configuration finishes at under a fifth of Web Speech's best
WER (10.8% vs 52.9%).

**The acoustic path costs auris real accuracy, and that cost is not
hidden.** WER on the correction-pass configuration rises from 4.9% (clean
corpus) to 10.8% (acoustic recordings) — roughly +6 percentage points. That
degradation is real: playing the corpus through speakers and re-recording it
through a laptop microphone adds room noise, a second lossy analog step a
clean WAV never passes through, and mild clipping on a handful of utterances
(see "method used" below). But it lands on both engines identically, since
both are scored against the same recordings — that is the entire point of
the fairness rule in `bench/harness/webspeech/README.md`, and it is why the
acoustic-mode comparison above is a legitimate head-to-head despite neither
engine hearing clean audio.

Within auris, biasing and correction still move WER in the same direction
as on the clean corpus: biasing −1.5pp (13.4 → 11.9), correction a further
−1.1pp (11.9 → 10.8). WER never regressed in any config, on either corpus,
matching `spike/RESULTS.md` §7.

## 3. Punctuation

**Web Speech emits zero terminal punctuation.** Verified directly —
`grep -o '[.!?]' | wc -l` across all four Web Speech transcript files
(strict, strict+correctVocabulary, best-effort, best-effort+correctVocabulary)
returns **zero** in every one. Not "close to zero" — the browser's
recognizer never once outputs a `.`, `!` or `?`, across 40 utterances, in
any of its four configurations.

| Config | Boundary F1 | Precision | Recall | Any terminal punct |
|---|---|---|---|---|
| Web Speech, all four configs | 0.0% | 0.0% | 0.0% | 0.0% |
| auris, no vocabulary (acoustic) | 54.5% | 80.0% | 41.4% | 100.0% |
| auris + vocabulary biasing (acoustic) | 54.5% | 80.0% | 41.4% | 87.5% |
| auris + biasing + correction (acoustic) | 54.5% | 80.0% | 41.4% | 87.5% |

**This document's first run predicted this exactly, before Web Speech was
measured**: *"an engine that emits no terminal punctuation at all scores 0%
[on `any_terminal_punct_rate`], and that single number will settle the
comparison more bluntly than F1."* It has. auris's `any_terminal_punct_rate`
of 87.5–100% against Web Speech's 0.0% is the sharpest, least arguable
finding in this document.

That said, **beating an engine that produces no punctuation at all is a low
bar, and this document should say so rather than claim more than it has
shown.** auris's own internal-boundary recall, computed over the 29
reference boundaries, is 41.4% on the acoustic run — better than half the
boundaries are still missed. u33 is a clean-corpus example of the failure
mode: reference *"khora dropped every websocket frame during the recording.
i checked the console and there's nothing useful in it. we'll need to
reproduce it live…"*; auris (clean corpus, plain config): *"Khora dropped
every web socket frame during the recording I checked the console and
there's nothing useful in it we'll…"* — three sentences merged into one
run-on.

Punctuation raw counts (tp/fp/fn, ref boundaries = 29):

| Config | tp | fp | fn | hyp boundaries | Precision |
|---|---|---|---|---|---|
| auris-plain (clean) | 5 | 2 | 24 | 7 | 0.714 |
| auris-vocab / +correction (clean) | 3 | 2 | 26 | 5 | 0.600 |
| auris, all three configs (acoustic) | 12 | 3 | 17 | 15 | 0.800 |

**Honest anomaly: auris punctuates better on the degraded acoustic audio
than on the clean corpus, and it is unexplained.** Boundary F1 rises from
17.6–27.8% (clean) to 54.5% (acoustic) — identical across all three auris
configurations — and precision rises too, from 0.600–0.714 to 0.800, even
though the number of hypothesis boundaries roughly doubled (5–7 → 15). That
is the opposite of what degraded audio should do to a punctuation model:
more boundaries emitted, and a *higher* fraction of them correct, not the
noise-hallucination pattern a worse signal would ordinarily produce.
Reproduced across all three auris configurations on the acoustic run. Filed
below under "open questions" rather than explained away.

On the clean corpus, biasing still costs punctuation relative to the plain
config (boundary F1 27.8 → 17.6, terminal punctuation 100% → 85%) — the
hotword beam appears to buy name recall with sentence structure. That
pattern does not reproduce on the acoustic run, where all three
configurations tie exactly; see open questions.

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

## The Web Speech baseline — method used

The loopback driver (`brew install blackhole-2ch`) needs an admin password
that was not available on this host, so this run used `--mode acoustic`
instead: the corpus was played out loud through the MacBook's speakers and
re-recorded through the built-in microphone. Chrome 151.0.7922.174, macOS,
`ffmpeg -f avfoundation` mic index 1 on this machine (index 0 was a virtual
device, "Immersed" — these indices are machine-specific and must be
re-read on any other machine, never copied from here), system output volume
45.

**The fairness rule from `bench/harness/webspeech/README.md`: Web Speech on
room audio must be compared against auris on the same room audio, never
against the clean corpus.** The 40 recordings the microphone actually heard
are committed at `bench/results/acoustic-wav/`, and **auris was rescored
against those same recordings** via `bench/harness/rescore_wavs.sh` — every
acoustic-mode auris number in this document, including §1's headline table,
is decoded from that directory, not from `bench/corpus/wav/`. This is why
the comparison above is legitimate despite neither engine hearing clean
audio.

A throwaway warm-up utterance — a `say`-generated control sentence — was
played first and discarded, per the README's finding that utterance #1 of a
fresh browser session is systematically worse than the ones that follow it:
observed in every session tested. This is consistent with
`run_auris.sh`'s own practice of discarding warm-up calls before measuring
latency, not special treatment for the browser.

**Chrome's fake audio device was retested today, with the timing bug fixed,
and finding 2 still holds — re-confirmed, not overturned:** the recognizer
fires `no-speech` while `getUserMedia`'s own RMS meter hears the same file,
in the same page and session, at peak RMS 1.0. `SpeechRecognition` still
opens the system default audio input directly and bypasses the media-stream
pipeline; there remains no way to feed it a file except an OS-level
loopback, which is why acoustic mode — not the fake device — is what
actually produced the numbers in this document.

**Two numbers, not one, because Chrome's finalization is nondeterministic.**
Chrome promotes a result from interim to final only when it chooses to: a
control sentence produced a word-perfect interim transcript while the final
stayed empty through a 10-second poll after `stop()`. So this document
reports:

- **best-effort** (final if present, else the last interim result) — the
  headline number, because when testing auris's own claim to beat the
  browser, the baseline should get its strongest case;
- **strict** (finals only) — the conservative bound, what `--out`'s
  documented contract actually records, and what a real product would
  render, since interim results are provisional by spec and no product
  commits them.

Over the 40 utterances: 17 finalized, 20 fell back to interim, 3 produced
nothing under either measure (u02, u17, u37).

**A harness bug was found and fixed, and it matters to this document's
credibility.** An earlier run showed cross-utterance bleed: a late
finalization from utterance N landing in utterance N+1's row (u30's
leftover text opened u31's transcript), corrupting both. The harness now
constructs a fresh `SpeechRecognition` object per utterance with a
generation counter, so a discarded recognizer's late result has nowhere to
land. The delivered run reported `0 poll timeouts, 0 id mismatches, 0 stale
results discarded`, and `bench/harness/detect_bleed.py` — new, with a
`--selftest` that plants a known synthetic bleed and asserts it is caught —
reports `no bleed suspects found (40 ids checked)` on both the strict and
best-effort transcripts. That check is substantiated by a self-test proving
it can fire, not merely an absence of evidence from a detector that has
never fired on real data.

**What acoustic mode does not give you.** It is not reproducible across
machines the way loopback would be — a different room, speaker, or
microphone would produce a different signal. Five utterances showed
microphone RMS at or near 1.0 (mild clipping, scattered across the run);
these were accepted rather than re-recorded, and the degradation they
represent hits both engines identically, since auris was rescored on the
same recordings Web Speech was scored against. The corpus itself is still
synthetic `say` speech — the fixture caveat at the top of this document
already covers that, and it applies here too. And the best-effort column
credits Web Speech with text Chrome itself never committed to a final
result — a real integration built on `SpeechRecognition` would never render
it, so best-effort is a ceiling on the browser's capability, not a
description of what a shipped product would show a user.

## Reproduction

```bash
# 1. corpus (synthetic; skip if bench/corpus/wav/ is populated)
bash bench/corpus/synthesize.sh

# 1b. or re-record it in a real voice -- this is the run that matters
bash bench/corpus/record.sh            # all 40, or: record.sh u09 u10

# 2. auris: three configs, scores, and the latency distribution, clean corpus
bash bench/harness/run_auris.sh

# 3. Web Speech baseline, acoustic mode (no driver install required;
#    --mic-device is whatever `ffmpeg -f avfoundation -list_devices true -i ""`
#    reports for YOUR built-in mic -- do not copy the index used on this host)
python3 bench/harness/webspeech/capture.py \
    --corpus bench/corpus/utterances.tsv --wav-dir bench/corpus/wav \
    --out bench/results/webspeech-acoustic-raw.tsv \
    --diagnostics-out bench/results/webspeech-acoustic-diagnostics.tsv \
    --mode acoustic --mic-device <your built-in mic's index>

# 4. derive the headline best-effort column from capture.py's own two
#    outputs (final transcript, else last interim, else empty)
python3 bench/harness/best_effort.py bench/results/webspeech-acoustic-raw.tsv \
    bench/results/webspeech-acoustic-diagnostics.tsv \
    bench/results/webspeech-acoustic-best-effort.tsv

# 5. rescore auris against the same recordings the microphone captured,
#    so both engines are compared on identical audio
bash bench/harness/rescore_wavs.sh bench/results/acoustic-wav acoustic

# 6. mechanical check for cross-utterance bleed (run --selftest first)
python3 bench/harness/detect_bleed.py --selftest
python3 bench/harness/detect_bleed.py bench/corpus/utterances.tsv \
    bench/results/webspeech-acoustic-raw.tsv

# 7. vocabulary correction pass, and scoring, over the Web Speech transcripts
python3 spike/harness/vocab_correct.py \
    bench/results/webspeech-acoustic-raw.tsv > bench/results/webspeech-acoustic-corrected.tsv
python3 bench/harness/score.py bench/corpus/utterances.tsv \
    bench/results/webspeech-acoustic-raw.tsv
```

Raw data: `bench/results/*.tsv`, `*.score.json` (per-utterance included),
`*.csv`, `bench/results/acoustic-wav/*.wav`, `latency-raw.tsv`,
`latency.json`, `latency-cold.json`.

## Open questions this run raised

- **auris punctuates better on degraded acoustic audio than on the clean
  corpus** (§3): boundary F1 rises from 17.6–27.8% (clean) to 54.5%
  (acoustic), and precision rises too (0.600–0.714 → 0.800) rather than
  collapsing, despite hypothesis boundary count roughly doubling. Measured,
  reproduced across all three auris configurations, and currently
  unexplained.
- **Biasing costs punctuation on the clean corpus** (§3): boundary F1
  27.8 → 17.6 and terminal punctuation 100% → 85% when the hotwords file is
  passed. Unexplained, and does not reproduce on the acoustic run (all three
  configs tie at 54.5% / 87.5–100%) — possibly related to the anomaly above.
- **Internal sentence boundaries are still often lost.** Acoustic-mode
  recall is 41.4%, better than the clean corpus's 10.3–17.2% but still
  under half. If auris is to beat the browser on internal punctuation and
  not just terminal punctuation, this is the number that has to move.
- **The corpus is still synthetic.** Every conclusion above inherits that.
- **`audio::is_silent` gave a flag-dependent verdict on the same file**,
  observed during this run and filed as mesa task 965.
