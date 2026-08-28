# Web Speech baseline capture

`bench/harness/run_auris.sh` scores auris against the 40-utterance
`bench/corpus` set. This harness exists to put a Web Speech
(`webkitSpeechRecognition`, Chrome's built-in recognizer) number on the same
corpus, so auris has a baseline other than itself to be measured against.

That should be simple -- feed each corpus WAV to Chrome, read back the
transcript -- and it is not, for two measured reasons below. Read them
before touching the flags in `capture.py`; both cost real time to find on
this machine (Google Chrome 151.0.7922.174, macOS Darwin 25.5.0) and are
easy to silently re-break.

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
  mic -- list devices with
  `ffmpeg -f avfoundation -list_devices true -i ""`.
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

# acoustic mode -- no install, plays audio aloud and records the mic
python3 bench/harness/webspeech/capture.py \
  --corpus bench/corpus/utterances.tsv \
  --wav-dir bench/corpus/wav \
  --out bench/results/webspeech-acoustic-raw.tsv \
  --mode acoustic --mic-device 0

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
- `--diagnostics-out` (`id<TAB>micRMSpeak`, default
  `bench/results/webspeech-diagnostics.tsv`) -- so a silent run is visible
  as data, not just inferred from a bad score.
- Acoustic mode only: `--acoustic-out-dir` (default
  `bench/results/acoustic-wav/`) -- the re-recorded room audio, one wav per
  utterance, for rescoring auris against.

Score a transcript the same way `run_auris.sh` scores auris's own
transcripts, with
`bench/harness/score.py bench/corpus/utterances.tsv <transcript.tsv>`.

## The three-column comparison: Web Speech, Web Speech corrected, auris

A vocabulary benchmark needs three columns, not one: Web Speech raw, Web
Speech after the same post-ASR vocabulary correction auris uses, and auris
with its vocabulary file. Without the middle column, auris's win would
partly just be "auris gets a correction pass and Web Speech doesn't" rather
than a real difference between the two engines on the hotword vocabulary.

The correction pass is `spike/harness/vocab_correct.py` -- mesa task 922's
`correctVocabulary`, ported (see `docs/correction.md`). It's the exact same
script `bench/harness/run_auris.sh` runs over auris's own vocabulary-biased
transcripts to produce `auris-vocab-corrected.tsv`, so running it over a
`capture.py` transcript keeps the correction pass identical across both
engines -- neither gets a pass tuned to its own failure modes.

```bash
# webspeech-raw.tsv -> corrected, using the same hotword term list auris uses
python3 spike/harness/vocab_correct.py \
  --hotwords spike/fixtures/hotwords.txt \
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
