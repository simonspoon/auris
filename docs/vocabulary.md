# How the vocabulary reaches the decoder

Date: 2026-08-27. Task 929.

**Decision: one flag, `--vocabulary-file FILE`, holding one `term :boost` per
line — the same file `docs/correction.md` already reads. It is passed to the
decoder per request as a per-stream hotwords string, never as recognizer
construction state, so a vocabulary change costs nothing and never reloads the
model. The global `hotwords_score` is fixed at 3.0 at construction and is not
exposed. There is no `--prompt`.**

This is the feature auris exists for, so the interface is decided here rather
than being whatever flag the transcribe path happens to grow.

## What this replaces

mesa task 922 shipped a soundex-style repair pass in
`frontend/src/liveRecognition.ts` (`soundKey`, `MESA_VOCABULARY`,
`buildVocabulary`, `correctVocabulary`) that runs *after* a settled result,
because Chrome's `SpeechGrammarList` is a no-op and the browser engine cannot
be told mesa's words up front. auris changes the timing: the vocabulary
reaches the decoder *before* the audio is decoded, so a mesa name is biased
into existence rather than swapped in afterwards.

It does not subsume 922. Two reasons, and both are load-bearing:

- 922 remains correct for as long as the browser is doing the listening. It
  stays as the fallback path's repair pass and is not touched by this
  decision.
- Biasing alone does not close the gap even in auris. `khora` is never
  produced by any engine or boost tested (`docs/engine.md`,
  `spike/RESULTS.md` §6), so auris runs its own port of 922's algorithm after
  decoding — `docs/correction.md`. Biasing and correction are two halves of
  one feature, which is why they read one file.

## The premise this task was written on is wrong

Task 929 was re-scoped by the engine decision (task 924) with a note saying
the hotwords file is *"constructor state, not a per-utterance argument,"* and
that the recognizer must be rebuilt to pick up a changed term list. README.md
hedged the same way, calling a rebuild on changed per-word boosts *"specified
behaviour here, not a caveat to resolve later."*

Both were checked against sherpa-onnx 1.13.6 source and against the running
model, and both are wrong in the way that matters. **Per-word `:score` syntax
works in the per-stream path.** `CreateStream(const std::string &hotwords)`
replaces `/` with newline and hands the result to `EncodeHotwords` — the same
parser the `hotwords_file` path uses, including the `:score` suffix
(`utils.cc:51-55`, `offline-recognizer-transducer-nemo-impl.h`). Verified
empirically: a per-stream string and a `hotwords_file` carrying the same terms
and scores produce byte-identical transcripts.

That removes the ~4 s rebuild from this design entirely, which matters because
`docs/latency.md` budgets 300 ms total and measures a recognizer build at
2.85 s on its own — 9.5x the whole budget. A vocabulary mechanism that reloads
is not affordable, and now none is needed.

**README.md's daemon section is superseded on this point** and should be
corrected when task 935 lands.

## The one thing that really is construction state

The correction above has a limit, and skipping it would produce a design that
silently does nothing. Boosting is **two stages**, and only the second is
per-word:

1. **The gate**, before top-k selection. Every token that continues a hotword
   has the *global* `hotwords_score` added to its logit
   (`offline-transducer-modified-beam-search-nemo-decoder.cc:266-271`). This
   value comes from the decoder's constructor and nothing per-request can
   change it. It decides whether a hotword token survives into the beam at
   all.
2. **The amplifier**, after selection. The per-word score is carried out of the
   context graph by `ContextGraph::ForwardOneStep`
   (`context-graph.cc:70-76`) and added to the hypothesis log-prob
   (`…nemo-decoder.cc:357`). A term written with no `:score` falls back to the
   global here as well — `ContextGraph::Build` stores the global whenever a
   term's score is left at zero (`context-graph.cc:34-35`).

So a generous per-word score under a near-zero global is useless — the token
is pruned before its boost is ever applied. Measured, on `u02.wav` with
`khora :12.0` held fixed:

| global `hotwords_score` | output |
| --- | --- |
| 0.5 | "Korra" — unaffected, pruned at the gate |
| 1.0 | "kora" — the per-word score now reaches the beam |
| 3.0 | "kora" |
| 6.5 | "khora is the default" — a second slot starts corrupting |
| 10.0+ | "khora khora khora khora…" — runaway repetition |

**auris fixes the global at 3.0 and does not expose it.** 3.0 is not a fresh
number: it is the value that reproduces `spike/results/hotwords/tuned.tsv`
byte-for-byte, i.e. the config behind the 83.3% name-F1 / 4.0% WER headline in
`docs/engine.md`. (sherpa-onnx's own default is 1.5, which is below the useful
gate on this model.) Because the global is the only construction-time part and
auris never varies it, **every vocabulary change is per-request and free.**

The cost of holding the global fixed is that auris cannot offer an unbounded
per-word range — see "Bounds" below. That is the right trade: a flag nobody can
use correctly is worth less than a reload nobody has to pay.

## The flag

**One flag, `--vocabulary-file FILE`.** Already committed to in README.md's
flag table; this task ratifies it and rules out the alternatives.

- **Not a repeatable `--vocabulary NAME`.** The list this is built to carry is
  a project's term list, assembled programmatically by mesa, not typed. A
  repeatable flag makes the common case an argv of hundreds of entries, and it
  has nowhere natural to put a per-term boost.
- **Not an environment variable.** It cannot legibly carry `term :boost` pairs,
  and it hides the vocabulary from `ps`, which is the one place an operator
  looks to see what a daemon was asked to do.
- **No `--prompt`, and the "one flag or two" question is dissolved rather than
  answered.** A raw prompt was a whisper concept: `initial_prompt` took prose
  and the caller might reasonably have wanted to write that prose themselves.
  Parakeet is a transducer. There is no prompt for raw text to be, so there is
  nothing for a second flag to accept. The file *is* the raw form — a caller
  wanting full control writes the boosts.

Passing a vocabulary is always optional. With no `--vocabulary-file`, biasing
and correction are both off and auris decodes plainly.

## The file format

Ratified from `docs/correction.md`, not re-invented — that pass already reads
this file and parses the term while discarding the boost, so that "the
correction vocabulary and the biasing vocabulary cannot drift apart."

UTF-8, one term per line:

```
# mesa's own names
mesa :3.0
auris :3.0
khora :6.5
qorvex :5.0
helios :3.0
kokoro :3.0
```

- `#` to end of line is a comment; blank lines are ignored.
- `TERM :BOOST` — the boost is optional. `mesa` alone is legal and is scored at
  the global, 3.0.
- The colon must be its own whitespace-separated token. This is not cosmetic:
  `khora:6.5` is parsed by sherpa-onnx as a single out-of-vocabulary token and
  the hotword is **silently dropped**. auris rejects that spelling rather than
  passing it through (see "Validation").
- A term may be a multi-word phrase: `wire up :3.0` biases the phrase.

## What becomes of the names

Nothing. They are passed through as terms.

Under whisper this was a real question — `initial_prompt` is prose, and
"khora, qorvex, helios" is a weaker hint than a sentence those words plausibly
occur in, so auris would have had to decide a wrapping phrasing. The engine
change deletes the question. Hotwords are a token-level bias on the decoder's
beam, not text the model reads; there is no sentence to wrap them in and
wrapping them would be actively wrong. auris joins the terms with `/` (the
per-stream separator sherpa-onnx expects) and passes them down.

## Bounds

The `:score` value is accepted in **(0.0, 8.0]** and defaults to the global,
3.0. The measured-good range is 3.0 for ordinary terms and up to 6.5 for a hard
one.

The upper guard rail is not arbitrary. `spike/RESULTS.md`'s uniform sweep shows
the transcript coming apart as the boost rises — 4.0% WER at scores 1 through
3.5, 9.3% at 5, **22.0% at 6, and 69.3% at 7** — and the gate table above shows
runaway repetition once the *global* gets large. Note those are uniform sweeps,
where every term moves together; the production file's `khora :6.5` is a single
term held high against a global of 3.0, which is why it is safe and a global of
6.5 is not. 8.0 is a stop, not a recommendation.

## The cap

**A vocabulary is capped at 128 terms. Terms past the cap are dropped in file
order — the file's order is the caller's priority signal, and auris does not
reorder it.** Exceeding the cap is a warning on stderr, not an error: a
transcript from a truncated vocabulary is far more useful to mesa than no
transcript.

mesa has no usage data to send and auris has none to infer, so "most recently
used first" is not available to either side; order-and-truncate leaves the
choice with the caller, who is the only party that knows which terms matter.

The cap exists because a large list degrades the transcript, and it degrades
it in an unobvious place — not on the boosted names, which stay correct, but
on **ordinary unrelated words elsewhere in the sentence**. Measured on the
eight dictation fixtures, real terms always present at production boosts and
padded with plausible synthetic terms:

| terms | WER | name F1 | F1 spread | decode, 8 fixtures |
| --- | --- | --- | --- | --- |
| 6 | 4.0% | 83.3% | 0.0 | 3.63 s |
| 32 | 4.2% | 83.3% | 0.0 | 4.06 s |
| 64 | 5.1% | 83.3% | 0.0 | 3.80 s |
| **128** | **4.4%** | **83.3%** | **0.0** | **3.78 s** |
| 256 | 7.1% | 81.6% | 5.1 pp | 3.73 s |
| 512 | 7.6% | 81.6% | 5.1 pp | 5.40 s |
| 1024 | 12.2% | 80.0% | 5.1 pp | 7.38 s |
| 2048 | 15.6% | 78.1% | 10.6 pp | 8.58 s |

Mean of three different random padding sets per row, scored with
`spike/harness/score.py` under the task-949 name metric; the `N=6` row
reproduces `docs/engine.md`'s tuned config exactly (4.00% / 83.33%), which is
what makes the rest of the column comparable to it.

Three signals agree on where the curve turns. Name F1 is *exactly* flat through
128, with zero spread across all three padding sets — the boosted names simply
do not care about list size up to there. From 256 it both drops and starts
varying by padding set. WER breaks trend at the same point (4.4% → 7.1%) and
doubles again by 1024. Decode time is flat through 256 and then grows, more
than doubling by 2048.

128 is the last row where all three are indistinguishable from the six-term
baseline, so 128 is the cap. 256 is not a disaster — it is simply the first
point that is measurably worse, and a cap should sit on the clean side of that
line rather than on it.

An earlier, single-padding-set pass over three fixtures showed no coherent
trend at all. That was padding-set noise, not a property of the list size; it
disappeared once each row averaged three seeds. Recorded because the noisy
version is the measurement a re-check is likely to reproduce first.

## Validation

Task 935 asks for a sanity bound in the shape of mesa's `speech::is_voice_name`.
A hotword string is model input, not a shell string, so this is not an
injection defence — but three of these are silent-corruption or crash bugs in
sherpa-onnx, not fastidiousness. auris rejects a vocabulary file, with a
`auris: ` diagnostic naming the offending line, when:

| Rule | Why |
| --- | --- |
| A term contains `/` | `/` is the per-stream phrase separator with no escape. sherpa-onnx silently splits the term into two unrelated hotwords — no error, no warning. |
| A boost is not a number | Parsed with `std::stof` and no guard; a malformed value is a hard crash, not a skip. |
| A term is written `term:boost` (no space) | Silently parsed as one OOV token and the hotword is dropped. Rejecting it beats a vocabulary that quietly does nothing. |
| A boost is outside `(0.0, 8.0]` | See "Bounds". |
| A term is empty, longer than 64 bytes, or contains control characters | A sanity bound. A 4 MB "name" is not a name. |

Leading, trailing and doubled `/` in the *assembled* string are safe —
sherpa-onnx drops empty segments silently (`utils.cc:137-138`) — so the join
logic needs no guard for an empty term list.

## Ordinary English words are not safe to include

**They are not, and this is the sharpest edge in the design.**

Task 935's acceptance asks auris to promise that "a vocabulary of ordinary
English words does not make auris hallucinate them into a recording that does
not contain them." That promise cannot be kept, and it is better to say so here
than to have it discovered as a failing test.

Measured: 200 common English function and content words at boost 3.0, no mesa
names, over the eight fixtures. WER goes from 5.3% to 15.3%, and **five of the
eight utterances gain a boosted word that is in neither the reference nor the
unbiased decode of the same audio.** The damage concentrates on spoken digits
and short function words, where the acoustic model is least certain and the
beam is easiest to push:

Each row quotes the clause the boosted word damaged, not the whole utterance —
"unbiased" means that clause decoded correctly without a vocabulary, not that
the whole sentence was clean (u07's and u08's unbiased decodes still miss
`khora` and `qorvex` elsewhere, which is the ordinary problem auris exists to
solve):

| reference clause | unbiased | with 200 ordinary words |
| --- | --- | --- |
| "close task 924" | correct | "Close to ask nine hundred and twenty-four and old a follow up" |
| "link it to task 902" | correct | "link it to task nine house or two" |
| "mark task 730 done and task 731 blocked" | correct | "mark task 730 done in task 730 one blocked" |

`docs/engine.md` says no parakeet run at a usable boost inserted a name that
was not spoken, and that stands — it was measured over six distinctive,
low-frequency names. It does not generalise to an arbitrary list. Hotword
biasing is safe *because* the terms are rare and acoustically distinctive, not
because the mechanism refuses to hallucinate.

The design consequence is a rule on what a vocabulary is, and it belongs to the
caller: **a vocabulary is a curated list of terms the engine gets wrong, not a
dictionary.** Ordinary words already decode correctly and have nothing to gain
from a boost; adding them costs accuracy on everything nearby. Where mesa's
terms come from is task 935's problem, but the constraint on that list is set
here — and note that it points the same way as the cap. Both say: send the
names that need help, not everything you have.

Task 935's acceptance criterion should be restated as the property that is
true and worth testing: *a curated vocabulary of mesa's own names does not
hallucinate those names into recordings that do not contain them.* Unmeasured,
and flagged rather than guessed: whether a large ordinary-word list is safe at
a lower boost.

## Before and after

`u03`, decoded with the same recognizer and `modified_beam_search` in both
runs, so the only variable is the vocabulary.

Reference:

> create a task in mesa called wire up qorvex to the ios simulator and assign
> it to me

**Without `--vocabulary-file`:**

> Create a task in **Mesa** called wire up **Corvex** to the iOS simulator and
> assign it to me.

**With `--vocabulary-file` (the six terms above):**

> Create a task in **mesa** called wire **qorvex** to the iOS simulator and
> assign it to me.

Both names land. Two honest notes, because this doc is the record:

- The biased run drops "up". Boosting reshapes the beam and occasionally costs
  a function word; it is visible in the WER column of the cap table and is the
  price of the name accuracy.
- `khora` is the one term this does not fix, at any boost, in any run. It
  decodes as "qora" biased and "Korra" unbiased. That is not a gap in this
  interface — it is why `docs/correction.md` exists, and the two together take
  name F1 from 83.3% to 96.3%.

## What task 935 must build

- `--vocabulary-file FILE`, parsed and validated per the tables above.
- The terms joined with `/` into a per-stream hotwords string, passed to
  `create_stream_with_hotwords`, **not** to `hotwords_file`.
- `hotwords_score` fixed at 3.0 at recognizer construction.
- The same parsed term list handed to the correction pass, boosts discarded.
- Correcting README.md's daemon section, which still describes a rebuild that
  this decision removes.
