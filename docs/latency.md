# Latency decision

Date: 2026-08-27. Task 927.

**Decision: the budget is 300 ms of added latency against today's Web
Speech experience, a hard ceiling of 2300 ms from mouth-close to the `user`
turn existing. On this host, warm, the measured cost is 0 ms — the whole of
auris's work fits inside the 2000 ms silence wait mesa already performs.
The budget is met by structure, not by speed, and the one thing that
destroys it is a cold process.**

## What the baseline actually is

`liveCapture.ts:46` defines `export const AUTO_SEND_IDLE_MS = 2000`, with
`MIN_AUTO_SEND_IDLE_MS = 250` (`liveCapture.ts:53`) and
`MAX_AUTO_SEND_IDLE_MS = 60_000` (`liveCapture.ts:54`); `autoSendIdleMs()`
at `liveCapture.ts:63` clamps the configured `live.auto-send-ms` between
them.  Rust side, the same default lives at
`core::config::DEFAULT_LIVE_AUTO_SEND_MS`.

The path, traced: `LiveHub.tsx:347-352` — `heardAt` is a ref bumped by
`markHeard()` on **every result, interim included**, so a mid-sentence
pause the person fills back in restarts the wait. `LiveHub.tsx:851-867` —
an effect arms `window.setTimeout(..., autoSendMs)`, re-armed on every
`heardTick`; on fire it calls `shouldFlushSilence({ listening, recording,
interim, idleMs: Date.now() - heardAt.current, idleThresholdMs:
autoSendMs })` (`liveRecognition.ts:199-209`), and if true,
`flushRecording()` (`LiveHub.tsx:831-840`), which joins held sentences with
the unsettled interim via `heldFlush` and calls `post(text)`
(`postRef.current` at `LiveHub.tsx:837`, resolving to `post` defined at
`LiveHub.tsx:1439`) → `sendLiveUtterance` (`frontend/src/api.ts:948-949`)
→ `POST /api/live/utterance` (`src/api.rs:813`, handler `src/api.rs:2637`).

So today, mouth-close to turn is 2000 ms of timer plus one localhost POST.
The text costs nothing because Chrome recognised **while the person
spoke**.

One honest caveat: the timer's zero point is not the mouth closing, it is
Chrome's last *result event*, which lags the audio by an unmeasured
amount. Today's true mouth-close-to-turn is therefore somewhat **more**
than 2000 ms. **NOT MEASURED.** This makes the budget below conservative
rather than optimistic — it is being compared against a floor, not the
real number.

Also worth stating plainly: mesa ships no speech-to-text today, and **no
route accepts an audio body** (`liveRecognition.ts:49-51`). Audio
transport is genuinely new surface, not a rewire of something that exists.

## The number, and why that number

300 ms, derived, not asserted. The person is not waiting for a 0 ms
response — they are already waiting 2000 ms, and are calibrated to it.
What has to stay invisible is a *change* to an interval they already
know. Duration discrimination over supra-second intervals follows a Weber
fraction: a difference below roughly 10-15% of the base interval is not
reliably noticed. 15% of 2000 ms is 300 ms. That is the budget.

The Weber-fraction argument is a rule of thumb used to turn an adjective
into a number, not a measurement of this user, and it is offered as
exactly that — but it beats picking a round number by feel, and it scales
correctly if `live.auto-send-ms` is retuned. The budget is 15% of whatever
that setting is; at the 250 ms floor the budget is 38 ms, and what saves
it at that setting is auris's structure, not its speed.

## Where it goes — the per-stage allocation

This is the heart of the decision. The crucial structural point: **auris's
stages overlap mesa's existing 2000 ms wait rather than following it**,
provided the timer restarts on auris's `speech` lines as `docs/streaming.md`
already instructs. Everything that finishes before t=2000 ms costs zero.
t=0 is the mouth closing:

| Stage | When | Cost against budget |
|---|---|---|
| Capture + stream a 16 kHz mono PCM16 frame to mesa | continuous; last frame lands t≈20-50 ms | 0 — inside the wait |
| Transport mesa → auris stdin (a pipe) | continuous | 0 — inside the wait |
| VAD closes the segment (`min_silence_duration`, assumed stock Silero 0.5 s — see below) | t = 500 ms | 0 — see the backdating rule |
| Model load | never on this path | 0 warm / **4000 ms cold** |
| Decode the final segment (RTF 0.082) | starts t = 500 ms | 0 for any segment under ~18 s |
| Correction pass (`docs/correction.md`) | after decode | <1 ms, a lexical fold over six terms |
| auris writes the `segment` line, mesa reads it, POSTs | t ≈ decode end + 10 ms | 0 — inside the wait |
| **Total added** | | **0 ms** |

Two rules make those zeroes real, not hopeful.

**1. The backdating rule — a concrete instruction for the mesa port.**
`docs/streaming.md` says mesa's timer must restart on auris's `speech`
lines. If mesa sets `heardAt` from the moment the
`{"type":"speech","active":false}` line *arrives*, it starts the 2000 ms
wait 500 ms late and total latency becomes 2500 ms — 200 ms over budget,
a regression caused entirely by bookkeeping. The protocol carries the fix,
but only implicitly, and that gap has to be closed here rather than
assumed away: `docs/streaming.md` shows `at` in its examples and never
defines it in prose, and in the one worked example the `speech`/`active:
false` line carries `at: 2.46` against the `segment` line's `end: 2.46` —
the same instant. **This document makes that binding rather than
incidental: `at` is the stream-time second at which speech stopped, which
for the closing line is exactly the segment's `end` — when speech
stopped, not when the VAD noticed it had.** It follows that **mesa must
set `heardAt` from `at`, not from the moment the line arrives.** Both
halves are requirements on the mesa route task, not notes: without the
first, `at` is free to drift to arrival time and the second is
unimplementable.

**2. The decode window, and the segment cap it forces.** Decode starts at
t=500 ms and must finish by t=2000 ms, so it has **1500 ms**.

That 500 ms was an assumption when this document was written, flagged
rather than buried because every number in this section is derived from it.
**Task 968 (the VAD gate) resolved it: `MIN_SILENCE_SECONDS` is 0.5 s**
(`src/vad.rs`), Silero's own stock default, configured verbatim — auris
does not override it. It was a private constant rather than a CLI flag at
that point: it was swept from 0.05 s to 0.5 s and found to have zero effect
on the gate's accept/reject decision at every value, because the gate only
asks whether a speech span ever opens, never when one closes — so there was
nothing there for a flag to tune. **Task 936 (segmentation) changed that,
and it is now `--vad-min-silence`**, because it is exactly the rule for
where one `segment` line stops and the next begins; the default is
unchanged at 0.5 s, so the arithmetic here still holds for a default run,
and a caller who moves the flag moves the window with it exactly as the
counterfactual below describes. The arithmetic in this section therefore stands as
written; nothing here needs to be redone. (Task 968 shipped the gate, not
the segmentation this document's `min_silence_duration`-vs-`live.auto-send-ms`
reconciliation is about — see "What this does not decide" below, which is
still open.) For the counterfactual the rest of this section is worth
keeping: the window is `live.auto-send-ms` minus this number and the cap
moves with it — at a 0.25 s close the window is 1750 ms and the cap could
rise, at 1.0 s it is 1000 ms and `max_speech_duration = 8.0` **fails** at
the 2x safety factor (8 s x 0.164 = 1312 ms > 1000 ms). The structural
argument — that auris's work fits inside a wait mesa already performs —
survives any of those values. The specific cap does not. At the
measured warm RTF of 0.082 (`spike/RESULTS.md` §6, parakeet + per-word
hotwords) that decodes a final segment of up to **18.3 s** at no cost. But
`spike/RESULTS.md` and `docs/engine.md` both warn the fixtures are macOS
`say` TTS and the absolute numbers are a floor, so apply a 2x safety
factor: at RTF 0.164 the 1500 ms window covers **9.1 s**. Therefore set
`max_speech_duration = 8.0` on the Silero VAD — 8 s x 0.164 = 1312 ms,
inside the window with margin, and inside it at the measured 0.082 (8 s x
0.082 = 656 ms) by a factor of 2.3. That is a concrete value handed to the
segmentation task,
which `docs/streaming.md` left open. A *final* segment is bounded by the
last pause of 500 ms or more, which in ordinary speech arrives every few
seconds, so the cap is a backstop rather than something that fires often.

The worked figure that shows how much slack this actually leaves: the
spike's eight fixtures are 45.43 s of audio, so the average utterance is
5.68 s; decoded whole at 0.082 that is 466 ms, finishing at t≈966 ms with
1034 ms of slack before the wait even expires.

## Model size — a cross-check, not a re-decision

The engine is settled (`docs/engine.md`, task 924) — parakeet TDT 0.6B v2
**int8**, the only variant downloaded (`spike/models/parakeet/`,
encoder.int8.onnx 652 MB, decoder 7.3 MB, joiner 1.7 MB, ~633 MB total,
per `spike/harness/NOTES-parakeet.md`), 8 decode threads by default. The
useful thing the budget can add is a **cross-check**: does the budget
independently confirm that choice?

Work it: whisper small.en+prompt at RTF 0.657 needs 3.73 s for the average
5.68 s utterance — it blows the 1500 ms window by more than double, and is
out on latency alone, quite apart from speed. base.en+prompt at 0.244
needs 1386 ms, inside the window by 114 ms and outside it at any safety
factor at all. medium.en at 2.173 is not worth the arithmetic. Parakeet
int8 at 0.082 is the only engine in the benchmark with real headroom.

The budget and the accuracy argument reach the same answer by different
routes, which is worth stating: `docs/engine.md` picked parakeet on name
F1 and RTF; this document picks it again on nothing but the decode window
above. fp16/fp32 parakeet variants exist and were not downloaded; they
would raise RTF and RSS, and there is no accuracy problem the budget is
being asked to buy off, so there is no reason to want them.

## Process warmth — the stated position

**Warm, mandatory, and it is a property of the contract rather than an
optimisation.** Cold load is ~4000 ms (`docs/engine.md`; `spike/RESULTS.md`
line 60 — model load ~4 s regardless of clip length, dominated by the 652
MB encoder) — **13x the entire 300 ms budget on its own** — and it would
land on every single utterance. The ~4 s the docs commit to is load plus
the first decode; isolating the recognizer build alone on this host, 2026-08-27,
measured 2.854 s of pure load. That is corroboration, not a correction — it
confirms load rather than decode is the bulk of it, and ~4 s remains the
figure to budget against, since it is what an utterance arriving at a cold
process actually waits.

Even at the lower figure the conclusion does not move: 2.854 s is still
9.5x the budget. `spike/RESULTS.md` already puts the
number on it: spawned per utterance, parakeet's effective RTF is 0.634
against 0.087 warm, and the whole speed argument evaporates.
`docs/engine.md` reached the same conclusion from the accuracy side; the
budget reaches it again from latency, which means a one-shot binary is
now excluded twice over.

The lifecycle position, concretely: the recognizer is loaded when
mesa's microphone opens — the `shouldListen` false→true transition
(`liveRecognition.ts:165`) — not lazily on the first utterance, because
that would put the 4 s squarely on the first thing the person says.
Mic-open buys the whole interval between the person reaching for the
switch and finishing their first sentence, several seconds, to absorb a
4 s load. If the first utterance does arrive before loading finishes it
queues, the budget is missed exactly once, visibly, at the start of a
session — and that is the right trade against holding ~1.5 GB resident
(`docs/engine.md`) for a mesa nobody is talking to.

This confirms rather than changes the direction of task 925 (the CLI
contract, already a thin client over a persistent daemon) and task 933
(bind sherpa-onnx and hold the recognizer warm); the contribution here is
the *number* that makes it non-negotiable, and the mic-open trigger. See
`docs/daemon.md` (task 951) for how that persistent daemon is actually
built and its measured RTF.

## Decoding before silence — asked and answered

**The overlap already exists and no further mechanism is warranted.** VAD
segmentation *is* the overlap: every segment except the last is decoded
while the person is still talking, so the only decode ever on the
critical path is the tail after the last pause. That is precisely why the
tail-segment arithmetic above is the whole analysis. A speculative
sliding-window decode over the still-open utterance would spend real
complexity to reduce a number that is already zero, and worse, it would
re-introduce revisable hypotheses — which `docs/streaming.md` rejected
outright, along with any retraction verb.

So the answer is no, and the reason is not "too hard," it is "there is
nothing left to buy." The one thing that would change this is the 250 ms
floor of `live.auto-send-ms`, where the wait is shorter than the VAD's
own 500 ms close and the whole overlap argument collapses — and even then
the fix is to reconcile `min_silence_duration` with the setting, as
`docs/streaming.md` already says, not to add speculative decoding.

## What this does not decide

- The remaining Silero knobs — `threshold`, `min_speech_duration`,
  `window_size` — stay with the segmentation task; only
  `max_speech_duration = 8.0` and the reconciliation of
  `min_silence_duration` with `live.auto-send-ms` are fixed here, and
  only because the budget forces them. The cap is fixed *conditionally*:
  it is correct for a 0.5 s VAD close, and the segmentation task owes it a
  recomputation once the real `min_silence_duration` is known.
- How audio actually reaches mesa from the page. The budget requires it
  be **streamed during speech, not uploaded after silence** — a post-hoc
  blob upload puts a transfer proportional to the whole recording after
  the wait instead of under it, which is the one shape that cannot fit —
  but the mechanism (websocket, chunked POST) is the route task's.
- Re-measuring RTF against a real dictated voice rather than `say`
  fixtures (`spike/fixtures/record.sh`). The 2x safety factor stands in
  for that measurement and should be replaced by it.
- The daemon's idle retirement, which is a memory question, not a latency
  one — see `docs/daemon.md`'s `IDLE_TIMEOUT` for the number and its own
  "not decided" note on whether it is the right one.
