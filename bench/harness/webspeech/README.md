# Web Speech baseline capture

`bench/harness/run_auris.sh` scores auris against the 40-utterance
`bench/corpus` set. This harness exists to put a Web Speech
(`webkitSpeechRecognition`, Chrome's built-in recognizer) number on the same
corpus, so auris has a baseline other than itself to be measured against.

That should be simple -- feed each corpus WAV to Chrome, read back the
transcript -- and it is not, for the measured reasons below. Read them
before touching the flags or the session handling in `capture.py`; each one
cost real time to find on this machine (Google Chrome 151.0.7922.174, macOS
Darwin 25.5.0) and is easy to silently re-break.

## Two modes, and the trade-off between them

`capture.py --mode loopback` (default) and `capture.py --mode acoustic` get
audio into the recognizer two different ways, and neither is free:

- **`loopback`** plays each corpus WAV into an OS-level virtual audio
  device set as both the macOS default output and default input (see "Why a
  virtual audio loopback is required at all" below for why this is the only
  way to reach `webkitSpeechRecognition` at all). It's clean and
  reproducible -- the recognizer hears exactly the corpus WAV, bit for bit,
  every run, on every machine -- but it needs a driver installed with admin
  rights (`brew install blackhole-2ch`) before it will work.
- **`acoustic`** plays each corpus WAV out of the default speakers and lets
  the built-in microphone hear it, the way a person talking to their laptop
  actually would. Nothing to install. The cost is that it degrades
  *whatever engine is being measured* -- room noise, speaker volume, mic
  placement, and the room's own acoustics all become part of the signal,
  and none of that is controlled or reproducible across machines or even
  across two runs on the same machine.

**Acoustic mode's degradation is not specific to Web Speech, and the
comparison must account for that.** Whatever the mic actually picked up is
what Web Speech is scored against -- but if auris is then scored against
the original clean `bench/corpus/wav/*.wav`, that is not a fair comparison
of the two engines; it is a comparison of "auris on a clean signal" against
"Web Speech on a degraded one," which would make Web Speech look worse than
it is for a reason that has nothing to do with either engine. This is why
`--mode acoustic` re-records the microphone into
`bench/results/acoustic-wav/<id>.wav` for every utterance (16 kHz mono
`pcm_s16le`, recording starts ~0.5s before playback and stops ~0.5s after,
via `ffmpeg -f avfoundation`) instead of just discarding what the mic
heard. **auris must be rescored against `bench/results/acoustic-wav/`, not
`bench/corpus/wav/`, when comparing to an acoustic-mode Web Speech run** --
run `bench/harness/run_auris.sh`'s decode step (or an equivalent
`auris --no-daemon` loop) against `bench/results/acoustic-wav/` to get a
matching auris number.

If you can install a driver, prefer `loopback` -- it's the only mode that
produces a number you can trust to reproduce on a different machine.
`acoustic` exists for when that install isn't an option.

## Why a virtual audio loopback is required at all (loopback mode)

**Finding 2 (the one that actually forces this design): `SpeechRecognition`
does not read from `getUserMedia`'s fake audio device.** Measured in a
single page/session running a WebAudio RMS meter on `getUserMedia` and
`webkitSpeechRecognition` at the same time, against the same fake device,
with the flags from finding 1 below in effect:

```json
{"micRMSmax":0.3334,"micSamples":75,"speechStarted":true,"speechError":"no-speech","speechResults":[]}
```

The page's own microphone metering hears the fake device loudly and
continuously (peak RMS 0.3334, non-silent) for the whole session, while the
speech recognizer, running in that same session against that same device,
hears nothing and ends in `no-speech`. Chrome's speech recognizer opens the
*system default audio input* directly -- it does not go through
`getUserMedia`'s media-stream pipeline, fake or real. There is no
command-line flag that changes this; it was not for lack of trying the
right flags.

**The consequence:** the only way to get a corpus WAV into
`webkitSpeechRecognition` at all is an OS-level virtual audio device, set as
the macOS default *input*, with the corpus played into the same device set
as the macOS default *output* -- a loopback. `capture.py` and
`capture.html` are built around that requirement; there is no way to make
this harness work without one installed.

Two HAL plugins were already installed on this machine
(`/Library/Audio/Plug-Ins/HAL/`: `ImmersedAudio.driver`,
`MSTeamsAudioDevice.driver`, `ParrotAudioPlugin.driver`) and both "Immersed"
and "Microsoft Teams Audio" show up as `avfoundation` audio inputs. Neither
is a loopback: recording from "Microsoft Teams Audio" (`avfoundation` index
3) while playing a WAV into it with
`ffmpeg -f audiotoolbox -audio_device_index 3` captured pure silence (peak
0). A dedicated loopback driver is needed -- see Operator setup below.

## Finding 1: what it takes to feed a file into `getUserMedia` at all

This isn't the path `webkitSpeechRecognition` uses (see finding 2), but
`capture.html` still runs a `getUserMedia` + WebAudio RMS meter as a
diagnostic alongside the recognizer (see "The diagnostic meter" below), and
this is the flag set that makes `getUserMedia`'s fake device work at all --
worth recording exactly, so nobody re-derives it the hard way if this fake
device is reused elsewhere:

```
--use-fake-device-for-media-stream
--use-fake-ui-for-media-stream
--use-file-for-fake-audio-capture=<absolute path to wav>
--disable-features=AudioServiceSandbox
```

Without `--disable-features=AudioServiceSandbox`, Chrome logs, verbatim:

```
ERROR:media/audio/simple_sources.cc:35] Failed to read <path> as input to the fake device. Try disabling the sandbox with --no-sandbox.
```

and the fake device reads pure digital silence (measured WebAudio RMS
exactly 0). With the flag, measured peak RMS was 0.3334.

**`capture.py` does not actually pass this flag set to Chrome.** It uses
only `--use-fake-ui-for-media-stream` (to auto-accept the permission prompt
headlessly), deliberately without `--use-fake-device-for-media-stream` /
`--use-file-for-fake-audio-capture`. Turning the fake device on would point
`getUserMedia` at one file baked in at Chrome launch, which would sever the
mic-RMS diagnostic from the real loopback device it needs to be watching --
see "The diagnostic meter" below. This flag set is recorded here for
reference (and is what was used to *produce* the JSON above), not because
`capture.py` uses it.

One more gotcha, unrelated to the fake device but hit while chasing this:
in zsh, quote `--remote-allow-origins=*` -- unquoted, the shell glob-expands
the `*` before Chrome ever sees it, and CDP then rejects the websocket
connection with `Rejected an incoming WebSocket connection from the ...
origin`. `capture.py` launches Chrome via `subprocess.Popen` with an argv
list, which never goes through a shell, so this can't bite there -- but it
will bite if you ever copy this flag into a shell script or a manual
terminal invocation.

## Finding 3: `/json/new` requires PUT, not GET (Chrome 111+)

`capture.py` opens its tab by hitting Chrome's CDP HTTP endpoint,
`/json/new`. On Chrome 151.0.7922.174 a plain GET to that endpoint fails
outright:

```
HTTP Error 405: Method Not Allowed
Using unsafe HTTP verb GET to invoke /json/new. This action supports only PUT verb.
```

confirmed by hand with `curl` against a manually-launched Chrome on a
scratch port (GET -> 405 with that exact body; PUT -> 200 with the tab's
`webSocketDebuggerUrl`). Chrome tightened this starting with 111 as a CSRF
hardening measure; GET used to work. `capture.py`'s `open_tab()` sends a
`urllib.request.Request(..., method="PUT")`, not the bare `urlopen()` GET
that older references (including earlier versions of this script) use --
recorded here so nobody burns time rediscovering it against a fresh Chrome
install, the way finding 1's flag set and finding 2's fake-device result
were recorded for the same reason.

## The diagnostic meter

Because of finding 2, "Web Speech got every word wrong" and "Web Speech
never heard anything because the loopback isn't wired up" produce the exact
same symptom from outside the page: no final results, `no-speech`. Scoring
the second case as the first would be a silent, misleading zero.

`capture.html` runs a `getUserMedia` + WebAudio RMS meter in the *same*
page and session as `webkitSpeechRecognition`, reading the real system
default input (not the fake device -- see above), exposed as
`window.__micRMS`. `capture.py` reads it after every utterance and writes it
to `bench/results/webspeech-diagnostics.tsv` alongside the transcript, and
**aborts loudly if the very first utterance's peak RMS is 0** -- that means
the loopback isn't wired up as both default input and output, and every
number after it would be garbage, not evidence about Web Speech.

## Operator setup for loopback mode (one-time, needs admin)

1. Install a loopback driver:
   ```
   brew install blackhole-2ch
   ```
2. In System Settings -> Sound, set **BlackHole 2ch** as both the default
   **output** and the default **input** device. (Both -- the corpus is
   played into the output side and the recognizer listens on the input
   side; on a loopback they're the same device.)
3. If you want to hear the corpus yourself while it plays, or want the
   normal speaker to also get sound, create a Multi-Output Device in Audio
   MIDI Setup combining BlackHole and your speakers, and use that as the
   default output instead of BlackHole alone -- optional, not needed for
   the capture to work.

## Acoustic mode: what to expect

No driver install, but read this before running it:

- It plays every selected utterance **out loud** through your speakers and
  **records your microphone** at the same time. `capture.py` prints a
  loud warning with an estimated total runtime before it starts, and waits
  for an Enter keypress (or `--yes` to skip that) -- it will not start
  silently.
- Stay quiet and don't make noise near the mic while it runs; any sound
  other than the played-back utterance becomes noise in the recording.
- `--mic-device` is the `ffmpeg -f avfoundation` index for the built-in
  mic. **These indices are machine-specific and not stable across
  machines, or even across reboots/reconnects on the same one** -- do not
  copy a number from this doc as if it were a constant. Always run
  `ffmpeg -f avfoundation -list_devices true -i ""` yourself and read the
  built-in mic's own index off that output. On the machine this harness was
  developed on, the list looked like:
  ```
  [0] Immersed
  [1] MacBook Pro Microphone
  [2] Simon's iPhone Microphone
  [3] Microsoft Teams Audio
  [4] Hillbilly Microphone
  ```
  i.e. index 0 was a *virtual* device (a screen-share/remote-collab tool),
  not the built-in mic, which was at index 1 -- given as an example of why
  guessing "0 is probably the built-in mic" is wrong often enough to be
  dangerous, not as a rule to hard-code instead.
- The recorded room audio for each utterance lands in
  `bench/results/acoustic-wav/<id>.wav` (override with
  `--acoustic-out-dir`). Rescore auris against that directory, not
  `bench/corpus/wav/`, for a fair comparison -- see "Two modes" above.

## Running it

```bash
python3 -c "import websocket"   # sanity check: websocket-client is installed

# loopback mode (default) -- needs the operator setup above
python3 bench/harness/webspeech/capture.py \
  --corpus bench/corpus/utterances.tsv \
  --wav-dir bench/corpus/wav \
  --out bench/results/webspeech-raw.tsv \
  --device "BlackHole 2ch"

# acoustic mode -- no install, plays audio aloud and records the mic.
# --mic-device is whatever YOUR `ffmpeg -f avfoundation -list_devices true
# -i ""` reports for the built-in mic -- see "Acoustic mode: what to
# expect" above for why 0 is not a safe assumption.
python3 bench/harness/webspeech/capture.py \
  --corpus bench/corpus/utterances.tsv \
  --wav-dir bench/corpus/wav \
  --out bench/results/webspeech-acoustic-raw.tsv \
  --mode acoustic --mic-device <your built-in mic's index>

# spot-check two utterances before committing to the full 40-utterance corpus,
# either mode:
python3 bench/harness/webspeech/capture.py \
  --corpus bench/corpus/utterances.tsv \
  --wav-dir bench/corpus/wav \
  --out /tmp/webspeech-spotcheck.tsv \
  --device "BlackHole 2ch" --ids u09,u10
```

`--device` (loopback mode) is whatever
`ffmpeg -f avfoundation -list_devices true -i ""` lists for the loopback's
output side (an index or a name works with recent ffmpeg). Output:

- `--out` (`id<TAB>transcript`) -- one row per utterance, final results
  only, joined.
- `--diagnostics-out` (`id<TAB>micRMSpeak<TAB>wsError<TAB>wsInterim`,
  default `bench/results/webspeech-diagnostics.tsv`) -- so a silent or empty
  row is visible as data, not just inferred from a bad score. `wsError` is
  `webkitSpeechRecognition`'s own `onerror` event (`no-speech`, `network`,
  `aborted`, `audio-capture`, ...) if one fired for that utterance, empty
  otherwise -- an empty transcript *with a recorded reason* is evidence; an
  empty transcript with no reason is not. `wsInterim` is the last
  non-final result the recognizer produced for that utterance -- see
  "Finalization is nondeterministic" below for why this column exists and
  what it's for.
- Acoustic mode only: `--acoustic-out-dir` (default
  `bench/results/acoustic-wav/`) -- the re-recorded room audio, one wav per
  utterance, for rescoring auris against.

Score a transcript the same way `run_auris.sh` scores auris's own
transcripts, with
`bench/harness/score.py bench/corpus/utterances.tsv <transcript.tsv>`.

## A throwaway warm-up utterance, played first and discarded

Utterance #1 of a fresh browser session is systematically worse than the
utterances that follow it -- observed in every single capture session run
against this corpus, acoustic and otherwise: the first utterance either
came back completely empty or visibly more garbled than its neighbors,
regardless of which specific sentence happened to be first. This isn't
specific to any one corpus utterance -- it reproduced with a `say`-generated
control sentence ("the quick brown fox jumps over the lazy dog") standing
in as utterance #1 too.

`bench/harness/run_auris.sh` already discards warm-up calls before
measuring latency for the same kind of reason (the first inference in a
process eats a one-time cost that isn't representative of steady-state
behavior); treating a browser session's first recognition the same way is
consistent with that established method, not special pleading for Web
Speech. Play one throwaway utterance before the real corpus, in the same
browser session, and discard its result -- don't count it in the delivered
transcript or diagnostics files, and record what was used as the warm-up
in whatever report cites the numbers.

## Finalization is nondeterministic

`webkitSpeechRecognition` promotes a result from interim (`isFinal: false`)
to final only when it decides to, and calling `recognition.stop()` --
which forces whatever audio has been captured so far to be returned -- does
not reliably make that happen promptly, or at all. Measured directly: a
control sentence ("please open the settings window and turn on dark mode")
produced `window.__wsInterim` equal to the sentence *word for word*, while
`window.__wsFinal` stayed empty through the normal settle margin and
through an extended 10-second poll on `window.__wsEnded` -- `onend` never
fired, and the correct text simply sat in `wsInterim` forever, still
correct, never promoted. On a different run of the identical clip, the same
text finalized promptly. Nothing about the audio, the mic signal, or the
recognizer's own judgment of what was said changed between these two
outcomes -- only whether Chrome chose to commit the result.

This means `--out`'s documented contract (final results only, one row per
utterance) will under-report Web Speech's actual transcription quality on
some fraction of utterances: the recognizer heard and transcribed correctly,
but the harness, faithfully recording only what Chrome committed, shows
nothing. `--out` stays final-only regardless -- that is the correct
behavior for a harness whose job is to record what actually happened, and
it also happens to match what an end user of Web Speech would ever see
rendered, since interim results are provisional by spec and no real product
treats them as committed output. The `wsInterim` diagnostics column exists
specifically so that a *second*, best-effort number can be derived after
the fact (final if present, else the last interim reading, else empty)
without changing `--out`'s semantics -- run something like:

```python
# best-effort transcript = final if capture.py recorded one, else the last
# interim reading from the diagnostics file, else empty. Built from --out
# and --diagnostics-out; does not touch either file's own contract.
```

Report both numbers when comparing to auris: best-effort as the headline
(the strongest case for the baseline -- if auris still wins against it, the
win isn't an artifact of Chrome's finalization luck) and strict finals-only
alongside as the conservative bound (what a real Web Speech integration
would actually render). The gap between the two is itself a finding about
this harness, not noise to average away.

## Cross-utterance bleed, and the fresh-recognizer-per-utterance fix

A single long-lived `SpeechRecognition` object, reused across utterances by
resetting it in place, has a failure mode worse than an empty result: a
late finalization from utterance N can land in the results array utterance
N+1 is reading, because nothing stops it from arriving after N+1 has
already started. Observed directly in a run of this harness: utterance
u30's own final result never arrived by the time its row was read (empty),
and utterance u31's recorded transcript opened with `"Maya food"` --
u30's leftover, mis-heard interim text -- immediately followed by u31's own
correctly-transcribed content. u30's real words were never lost; they
arrived late and were filed under the wrong id.

This corrupts the attribution of *two* rows, not one: u30 looks like it
under-transcribed (its real trailing words are missing from its own row),
and u31 is now scored against a reference it doesn't match at the start,
which name-accuracy and WER will both charge to the wrong utterance. Given
finding 2 above (finalization timing cannot be trusted), no settle margin
or polling delay can be sized to reliably prevent this -- the fix has to be
structural, not timing-based.

`capture.html`'s `window.__wsStartUtterance(id)` builds a brand-new
`SpeechRecognition` object for every utterance and discards the previous
one (`recognition.abort()`) rather than resetting it in place. Every
handler on a given recognizer closes over the generation it was built
under and refuses to touch `window.__wsFinal`/`__wsInterim`/`__wsError` if
a newer generation now exists (tracked in `window.__wsStaleResultCount`, so
a discarded instance's late arrival is counted rather than silently
dropped or silently accepted). A stale instance's late final then has
nowhere to land, no matter how late it arrives or whether it arrives at
all -- the bleed is structurally impossible, not merely unlikely.
`capture.py` also bounded-polls `window.__wsEnded === true` (capped at 5s,
falling back to the fixed settle margin on timeout, with every timeout
logged) so a fast finalization doesn't waste the rest of the margin, and
asserts `window.__wsCurrentId` matches the utterance it just started before
accepting a row, logging any mismatch rather than accepting it silently.

`bench/harness/detect_bleed.py` is a mechanical, after-the-fact check for
this specific pathology: for each utterance it compares how well the
recorded transcript's opening words match its *own* reference against how
well they match the *tail* of the previous utterance's reference, and flags
ids that look more like they're continuing the previous utterance than
starting their own. Run it against any transcript this harness produces --
`python3 bench/harness/detect_bleed.py bench/corpus/utterances.tsv
<transcript.tsv>` -- and run `--selftest` first if it's been a while, since
a detector that has never fired on real data is indistinguishable from one
that cannot fire; `--selftest` plants a known synthetic bleed and asserts
the script actually catches it.

## The three-column comparison: Web Speech, Web Speech corrected, auris

A vocabulary benchmark needs three columns, not one: Web Speech raw, Web
Speech after the same post-ASR vocabulary correction auris uses, and auris
with its vocabulary file. Without the middle column, auris's win would
partly just be "auris gets a correction pass and Web Speech doesn't" rather
than a real difference between the two engines on the hotword vocabulary.

The correction pass is `bench/harness/vocab_correct.py` -- mesa task 922's
`correctVocabulary`, ported (see `docs/correction.md`). It's the exact same
script `bench/harness/run_auris.sh` runs over auris's own vocabulary-biased
transcripts to produce `auris-vocab-corrected.tsv`, so running it over a
`capture.py` transcript keeps the correction pass identical across both
engines -- neither gets a pass tuned to its own failure modes.

```bash
# webspeech-raw.tsv -> corrected, using the same hotword term list auris uses
python3 bench/harness/vocab_correct.py \
  --hotwords bench/fixtures/hotwords.txt \
  bench/results/webspeech-raw.tsv > bench/results/webspeech-corrected.tsv

# score both, alongside auris's own bench/results/auris-vocab*.score.json
python3 bench/harness/score.py bench/corpus/utterances.tsv \
  bench/results/webspeech-raw.tsv > bench/results/webspeech-raw.score.json
python3 bench/harness/score.py bench/corpus/utterances.tsv \
  bench/results/webspeech-corrected.tsv > bench/results/webspeech-corrected.score.json
```

That gives the three columns to compare: `webspeech-raw.score.json`,
`webspeech-corrected.score.json`, and `bench/results/auris-vocab.score.json`
/ `auris-vocab-corrected.score.json` from `run_auris.sh`. In acoustic mode,
run the same two commands against `webspeech-acoustic-raw.tsv` instead, and
compare against an auris run scored on `bench/results/acoustic-wav/` (see
"Two modes" above) -- not against `auris-vocab.score.json`, which was
decoded from the clean corpus.
