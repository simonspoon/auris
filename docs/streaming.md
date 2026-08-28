# Streaming decision

Date: 2026-08-27. Task 926.

**Decision: auris ships one-shot transcription now and VAD-segmented
transcription next; partial hypotheses are rejected outright. Output is JSON
Lines — one self-describing JSON object per line, discriminated by a `type`
field — from day one, and no line auris writes is ever retracted or
revised.**

A crude fixed-threshold energy gate (`audio::is_silent`, task 958) already
exists ahead of this work, as a precondition for the README "Exit codes"
contract: the real model hallucinates words on digital silence instead of
returning nothing, so auris now catches that before decoding rather than
trusting the recognizer's output. It is not the VAD segmentation described
below — it answers one yes/no question about a whole utterance and detects
no speech/silence boundary at all — and the Silero VAD this document settles
on supersedes it once segmentation is built.

## Why "streaming" had to be redefined for STT

kokoro-rs streams because its *output* is elastic: it synthesises sentence by
sentence and mesa plays the first chunk while the rest renders
(`speech.rs`). STT is the mirror image of that, but not in the convenient
sense. Here the *input* is elastic — it arrives in real time and cannot be
transcribed until enough of it has been heard. "auris is kokoro-rs backwards"
does not carry kokoro-rs's streaming design across for free; the word
"streaming" has to be given a meaning on this side of the pipe, or it gets
built twice, once by assumption and once for real.

Three candidate meanings were on the table:

1. **One-shot** — stdin to EOF, one transcript, exit.
2. **Partials** — a revisable best guess emitted on a cadence while an
   utterance is still open.
3. **Segmented** — a VAD cuts the stream into utterances; each one is decoded
   once it closes and emitted final.

## Correcting the premise this task was written under

This has to be stated plainly, because it is load-bearing for everything
that follows. The task carried a late note saying the engine change to a
transducer meant "Parakeet emits as soon as it has heard enough rather than
decoding a fixed window." **That is not correct, and this decision does not
rest on it.** Parakeet TDT 0.6B v2 is run in sherpa-onnx as an *offline*
transducer: `OfflineRecognizer`, which decodes a complete utterance and
emits nothing until the utterance it was handed is over. "Transducer"
describes the model's loss and architecture, not incremental emission at the
API — those are independent properties, and this model has the first without
the second. `README.md` already had this right ("Parakeet is an *offline*
recognizer"); this document is not correcting auris, only the note the task
arrived with.

Verified facts:

- There is no streaming/online `parakeet-tdt-0.6b-v2` in sherpa-onnx. The
  parakeet family is asked about upstream in exactly these terms:
  k2-fsa/sherpa-onnx #3573 requests "true online/streaming RNNT inference"
  for a parakeet model and states in its own body that current Parakeet TDT
  exports support offline inference only, not stateful streaming RNNT
  sessions; #2918 asks for "real streaming" for the v3 sibling and its
  author reports falling back to pseudo-streaming. Neither issue is about v2
  specifically and neither is a maintainer statement — they are cited as
  evidence of what the exports do, not as a roadmap. What settles it for v2
  is the positive fact in the next bullet: the shipped way to run this model
  live is a VAD around the offline recognizer.
- sherpa-onnx's own answer to "run parakeet on a live microphone" is
  `cxx-api-examples/parakeet-tdt-simulate-streaming-microphone-cxx-api.cc`,
  and the operative word in that filename is *simulate*. It runs a Silero
  VAD over the input and decodes each cut with the *offline* recognizer.
  That is mode 3 above, sanctioned by the engine's own vendor.

## Mode 2 (partials) is rejected, not deferred

Three independent reasons.

1. **The mechanism is a whole-window re-decode, so its cost is quadratic in
   utterance length.** The upstream example above produces its partials by,
   every 0.2 s while speech is active, creating a fresh `OfflineStream`,
   feeding it *the entire accumulated buffer*, and decoding again from
   scratch. Each tick redoes all the work of every tick before it. At the
   measured warm RTF of 0.082 (`docs/engine.md`), re-decoding ten seconds of
   held audio costs ~0.8 s — already four times the 0.2 s cadence it is
   supposed to keep, on a CPU-only host. The cadence a demo can hold for a
   short phrase collapses on the long sentences mesa's held-recording model
   is specifically built for.
2. **It reintroduces exactly the retraction problem the engine change was
   supposed to have removed.** A hypothesis produced by re-decoding a longer
   window can differ from the one before it anywhere in the sentence, not
   just at the end. The task's own framing of mode 2 said this about
   whisper; switching to parakeet does not change it, because the
   revisability comes from re-decoding a growing window, not from the
   model that sits inside it.
3. **mesa discards earliness anyway.** `liveRecognition.ts` holds every
   settled result and posts the whole recording as one `user` turn (mesa
   task 917) — the design there is that a turn is a whole thought, not one
   breath at a time. Words arriving early are thrown away by the consumer
   that would receive them.

## Why segmentation survives the engine change

The task carried a second challenge, and it deserves a direct answer rather
than a passing footnote: if warm RTF is 0.082, maybe one-shot per utterance
is simply fast enough on its own, and segmentation is not needed at all.

Concede the half of that which is right. The reason segmentation was
recorded as expected on 2026-08-27 was latency: on a CPU-only Intel Mac a
decode took roughly as long as the audio, so one-shot meant the person
finished a twenty-second sentence and then waited another ten or twenty in
silence. That argument is genuinely dead. At the warm RTF of 0.082 in
`docs/engine.md`, twenty seconds of speech decodes in about 1.6 s, and
`README.md`'s daemon is what keeps that RTF warm across calls. Segmentation
is no longer needed to hide a decode cost, because there is no longer a
decode cost worth hiding.

But the half of the original argument that was always stronger was never
about speed, and it is still standing. One-shot is defined as "read stdin
to EOF, then decode." A live capture — `arecord -f S16_LE -r 16000 | auris`,
which is already in `README.md`'s own synopsis — has no EOF until the
microphone closes. One-shot on that stream does not mean a slow transcript;
it means buffering the entire session in memory and emitting nothing until
the capture ends, which is an absence of output altogether, not a latency
problem a faster decode could shrink. Segmentation is not an optimisation
laid over one-shot on such a stream; it is the only thing that makes the
stream transcribable at all, because it is what defines where one utterance
ends and the next begins. `README.md` already commits to flushing each
utterance's text as soon as that utterance ends — segmentation is what
gives that guarantee a referent. Without it, "per utterance" names nothing.

A third, independent claim holds even if the first two vanished. The
speech-activity heartbeat mesa's silence timer needs, argued in the next
section, is computed by the same VAD that segmentation runs. Segmentation
and the heartbeat come from one component, so the cost of running a VAD is
paid once for two reasons, not two costs for two features. That cost is also
small in the terms this project measures things in: `silero_vad.onnx` is
about 630 KB, against the 652 MB encoder the daemon exists to keep loaded.

So the engine change removed one of segmentation's two justifications and
left the other standing, and the one it left is the one that was never
about speed. The ordering recorded on the task is unchanged — one-shot
first, to prove the pipe; segmented next — but the reason for the second
step is different from the one written down, and worth having recorded
correctly.

## What mesa actually wanted from partials — and the cheap thing that supplies it

This is the crux, and reason 3 above conceals it rather than closing it.
mesa's interim results are doing a job that is not "showing text early."

`liveRecognition.ts`'s own doc comment says the silence timer "restarts on
every result, interim included, so a mid-sentence pause the person fills
back in does not get cut off by a clock that only heard the settled words"
(around line 36), and `shouldFlushSilence` fires on `idleMs >=
idleThresholdMs`. Put together: **interim results are a
speech-is-still-happening heartbeat**, not a preview of the transcript.
Swap the Web Speech API for an engine that emits only completed
utterances, and a person
speaking one long unbroken sentence produces no results at all for the
sentence's whole duration — the silence timer expires mid-thought and posts
a half-sentence. Removing partials without replacing that heartbeat is a
regression, and it is not a transcript problem, so no transcript-shaped fix
addresses it.

The resolution: the heartbeat does not need a transcript. sherpa-onnx's VAD
answers it directly and for free — `VoiceActivityDetector::detected() ->
bool` in the Rust crate (`SherpaOnnxVoiceActivityDetectorDetected` in the C
API), "true if speech is currently being detected," computed by the same
VAD that segmentation needs anyway, with no decode of any kind. So auris
supplies a speech-activity event rather than a partial transcript: the
thing mesa's timer actually consumes, at none of the cost of the thing it
was consuming it from.

## The line protocol

Output is **JSON Lines**: one JSON object per line, `\n`-terminated, UTF-8,
never pretty-printed (so a line is never split across writes), flushed as
soon as it is written.

Every object carries a `type` discriminator. Line types:

```
{"type":"speech","active":true,"at":1.20}
{"type":"segment","index":0,"text":"book a call with khora for tomorrow","start":0.32,"end":2.46}
{"type":"speech","active":false,"at":2.46}
{"type":"transcript","text":"book a call with khora for tomorrow"}
```

- `segment` — one completed utterance, already corrected by the pass in
  `docs/correction.md`. `index` counts from 0 and is monotonic.
  `start`/`end` are seconds from the start of the stream. **A `segment` is
  final and immutable.**
- `transcript` — the last line before exit on any run that produced a
  transcript: the whole corrected text, the segment texts joined in order.
  It exists so that the simplest possible reader — read to EOF, parse the
  last line — is a *correct* reader, which is exactly what mesa's
  held-recording model wants today.
- `speech` — the activity heartbeat described above. Emitted only on a
  segmented stream; a single-file one-shot run emits none.

The rule that makes this extensible has to be stated as an obligation on
readers, not a note: **a reader MUST ignore any line whose `type` it does
not recognise.** That is the entire extension mechanism. It is why `speech`
can appear when segmentation ships without anything that reads `segment`
and `transcript` today needing a change, and why there is no version
negotiation.

A run that produces no transcript writes nothing at all to stdout and exits
1, unchanged from `README.md`. `--format text` is unchanged: the plain
transcript, one line per completed utterance.

Day one — one WAV, one utterance — this emits exactly one `segment` line
and one `transcript` line. The general shape is what ships from the first
commit; segmentation adds line *volume*, not line *kinds*, except for
`speech`.

## The divergence from the recorded decision — `type`, not `is_final`

The decision recorded on the task on 2026-08-27 was JSON Lines "with an
`is_final` flag from day one," and its intent — a self-describing line
protocol that modes 2 and 3 fit into without rewriting mesa's reader — is
honoured exactly. Only the spelling changed, for a reason that emerged
after that decision: a boolean can only qualify lines that are all the same
kind, and the design now needs at least two kinds that are not transcripts
at all — the `speech` heartbeat and the terminal `transcript` summary.
`is_final: true` on every line forever would also have been a field that
never varies, which is a placeholder, not a protocol. `type` carries
everything `is_final` would have and admits the lines a boolean cannot
describe, at the same cost.

## Retraction — answered

Flatly: **the protocol has no retraction verb, and will not grow one.** No
line auris writes is ever unsaid. Since partials are rejected, no line is
ever revised either.

If a preview line is ever introduced despite this, it must be expressed as
*supersession by utterance index* — a hypothetical
`{"type":"partial","index":N,...}` is superseded by the next line bearing
index N — and never as a line that retracts an earlier one. A reader that
ignores `partial` entirely, which is what the ignore-unknown-types rule
already makes every reader written today, stays correct with no special
handling. A retraction verb, by contrast, is not ignorable: a reader that
skipped it would have already acted on text that was withdrawn. That
asymmetry is the reason for the rule.

## mesa's two send boundaries, afterwards

The acceptance criterion, answered for each of the two boundaries in
`liveRecognition.ts`.

- **The listen chord stays in the browser, permanently.** It is a
  keystroke — an intent signal about when the person wants their turn
  sent. auris hears audio; it can never know the chord was pressed.
  Nothing about auris moves it.
- **The silence boundary splits.** *Detecting* speech and silence becomes
  auris's, because the VAD computes it anyway and the browser would
  otherwise be doing it twice. *Deciding that silence means send the turn*
  stays in mesa, because that is conversation policy resting on things
  auris cannot see: `shouldListen` (which excludes the seconds mesa is
  herself speaking, so the person listening to her reply is never read as
  having fallen silent), the `live.auto-send-ms` setting shared with the
  typed capture box, and the `HELD_MAX` split in `heldWith`/`heldFlush`.
  The concrete port instruction: the timer that today restarts on every
  result including interims must instead restart on auris's `speech`
  lines, or the regression described above lands.
- **One thing has no equivalent, and the replacement is better.**
  `heldFlush` deliberately includes the unsettled interim tail at the
  moment the chord is pressed, because "the person finished speaking and
  then reached for the switch, so the sentence the engine has not settled
  yet is the last thing they said" (the function's own doc comment). auris
  has no unsettled tail to hand over. With segmentation the chord instead
  tells auris to flush — `VoiceActivityDetector::flush()`, which forces
  final segmentation of the buffered tail samples — and what comes back is
  a properly decoded final segment rather than the engine's best mid-guess.
  The boundary survives the port and the text at it gets better.

## Why this needs nothing new from mesa

`README.md` used to claim mesa "reads stdout to EOF the same way it already
reads WAV bytes from kokoro-rs." That undersold what is already there — the
real situation is stronger. `speech.rs::start` does not drain to EOF: it
reads kokoro-rs's stdout *incrementally* in a loop
(`stdout.read(&mut buf)`, roughly lines 200-222) until it has enough bytes
to patch the WAV header, then hands the rest to `stream()` (roughly lines
241-270), which pumps the remainder into an `mpsc::channel` chunk by chunk
and reaps the child at the end. mesa therefore already runs a progressive,
incremental reader over a child's stdout. Reading `\n`-delimited JSON lines
is strictly *simpler* than the byte-oriented header scan it already
performs. Among children spawned with piped stdio this is the only
incremental reader in mesa — `scripts.rs` and `hooks.rs` use
`wait_with_output()` and `agents.rs` uses `Command::output()`, all of which
block until the child is done. (`api.rs::pump_pty` also pumps a child's
output chunk by chunk into a channel, but it reads a PTY master fd and
relays raw bytes to a websocket rather than parsing a piped stdout.) So the
precedent for reading a piped child incrementally and acting on what comes
out is `speech.rs`, and it is the speech one.

## What this does not decide

- The VAD's own parameters are a tuning problem for the task that builds
  segmentation, not settled here. `SileroVadModelConfig` in the Rust crate
  exposes `threshold`, `min_silence_duration`, `min_speech_duration`,
  `max_speech_duration` and `window_size`; `min_silence_duration` is the one
  that decides where an utterance ends, and it is the knob mesa's
  `live.auto-send-ms` has to be reconciled with rather than duplicated.
- Whether the `speech` heartbeat cadence needs a minimum interval, so a
  stuttering VAD cannot flood a reader, is left to that task too.
- The task that binds sherpa-onnx and holds the recognizer warm is 933.
