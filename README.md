# auris

Command-line speech-to-text using [NVIDIA Parakeet TDT
0.6B v2](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8),
run through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Takes audio
on stdin or as a file argument and writes the transcript to stdout. The
mirror of kokoro-rs: where kokoro turns text into audio, auris turns audio
into text, and the two are meant to sit on either side of the same pipe.

**Decision: auris is one binary with two roles — a thin per-call client, and
a persistent daemon that holds the loaded recognizer. A bare invocation is
the client; `auris serve` is the daemon.**

That split exists because two constraints point in opposite directions and
neither one moves. `docs/engine.md` makes a persistent process mandatory:
the 652 MB int8 encoder takes ~4 s to load, and spawned fresh per utterance
parakeet's effective RTF is 0.634 against 0.082 warm — the whole reason
parakeet was chosen over whisper evaporates if the process exits after every
utterance. But the consumer this binary exists for, mesa's speech driver
(`mesa/src/core/speech.rs`), is strictly spawn-per-call: it runs
`Command::new(bin)` with all three stdio piped, writes the payload to
stdin from a thread, drains stderr on its own thread for the child's whole
life, drains stdout to EOF even after the listener has gone, carries no
wall-clock timeout (`speech.rs:14-17`), and kills the child only on a stdout
read error before anything useful has landed (`speech.rs:200-213`). That
pattern is the *only* one anywhere in
mesa — kokoro-rs, the `bash` runner in `scripts.rs`, `claude` in
`agents.rs` — there is no daemon, no Unix-socket client, no persistent-child
case to build against.

The alternative — a long-lived auris process reading framed utterances off
stdin, with mesa holding the child across requests — was rejected because it
moves the cost to the wrong side. It would force mesa to invent a framing
protocol and rewrite the three-thread stdin/stdout/stderr drain that
`speech.rs` documents as load-bearing, just to save auris a socket. Putting
the persistence inside auris instead means mesa needs no changes at all: the
same spawn-per-call driver that already renders speech with kokoro-rs reads
speech back with auris, `-q` and piped stdio and all. The ~4 s load is paid
once, by `auris serve`; every other invocation is a socket write and a
socket read.

## Synopsis

```sh
auris < utterance.wav                        # transcribe a WAV from stdin
arecord -f S16_LE -r 16000 | auris            # transcribe a live capture
auris path/to.wav                             # transcribe a file argument
auris --vocabulary-file names.txt             # bias + correct toward these terms
auris -m parakeet-tdt-0.6b-v2-int8            # pick a model explicitly
auris --format json < utterance.wav           # segments + timing, not plain text
auris --no-download --list-models             # what's on disk, no network
auris -q < utterance.wav                      # suppress the progress line
auris serve                                   # start (or become) the daemon
auris status                                  # is a daemon running, with what model
auris stop                                    # ask the daemon to exit
```

## The contract

This section states what a caller — mesa's driver above all — may rely on
without reading auris's source. Each rule is an obligation on auris, not a
description of what it happens to do today.

### stdin

If `PATH` is given, auris reads that file and does not touch stdin at all.
If `PATH` is omitted, auris reads stdin — unless stdin is a terminal, which
is a usage error, since there is no audio and none is coming. This is the
same split kokoro-rs makes for text versus its `text` arguments
(`cli.rs:193-199`: the usage error fires iff `text.is_empty() &&
stdin.is_terminal()`, never merely because an argument was given). It is
also what makes a payload beginning with `-o` or any other flag-shaped byte
harmless: audio bytes are never parsed as arguments, because they are never
handed to the argument parser at all.

Audio is read to EOF before decoding, because Parakeet is an *offline*
recognizer — it decodes a whole utterance and cannot emit a hypothesis for
audio it has not yet seen the end of. What auris promises instead is that
**stdout's first byte does not wait for stdin's EOF on a multi-utterance
stream**: incoming audio is segmented into utterances as they complete, and
each utterance's text is flushed to stdout as soon as that utterance ends.
For a single short recording (the common case: one WAV, one utterance) this
collapses to "auris reads it all, then writes the transcript" — there is
nothing to flush early. The segmentation itself — where an utterance boundary
falls — is not this task's decision; see task 926 and Notes below.

### stdout

Only the transcript goes to stdout — no progress, no logging, no framing
around it beyond what `--format` specifies. Text mode writes the decoded (and
corrected) text; nothing is written to stdout at all on a run that produces
no transcript. `--list-models` writes one model name per line, plain text,
no JSON, no decoration — the same shape kokoro-rs's `--list-voices` uses, and
for the same reason: mesa's `voices()` splits stdout on lines and filters
each one through a bounded-identifier check, so the format is load-bearing,
not cosmetic.

### stderr

Progress goes to stderr, and only when stderr is a terminal:
`verbose = !quiet && stderr.is_terminal()`, mirroring kokoro-rs
(`cli.rs:253`) exactly. This means `-q` is belt-and-braces for a caller that
already pipes stderr — the auto-suppression alone would be enough for mesa's
driver, which pipes all three streams, but the flag is passed anyway because
that is what the existing driver already does for kokoro-rs and there is no
reason for auris to diverge. Errors are always written to stderr regardless
of `-q` or terminal-ness, each line prefixed `auris: `.

### Exit codes

Copied from kokoro-rs (`cli.rs:91-95`), not reinvented:

```
const OK: i32 = 0;
const NOTHING_TRANSCRIBED: i32 = 1;
const USAGE: i32 = 2;
const INTERRUPTED: i32 = 130;
```

`0` means a transcript was produced, or a `--list-*` / `status` call
succeeded. `1` means no transcript came out — unreadable or silent audio, a
model missing under `--no-download`, or a daemon that could not be reached
or started. `2` means a bad flag value, an unknown model name, or (mirroring
kokoro-rs's stdin rule above) no `PATH` given while stdin is a terminal.
`130` is Ctrl-C.

The consequence for mesa: a nonzero exit means "there is no transcript," the
same rule `speech.rs` already applies to kokoro-rs — a failed render is not
data, it is an `Err` the API answers `unavailable` with. auris's driver
reads the exit status the same way: consulted only when nothing usable
landed on stdout, never as the primary signal.

## Flags

| Flag | Meaning |
| --- | --- |
| `PATH` | Audio file to transcribe (positional). Given, stdin is simply not read; omitted, stdin is read unless it is a terminal. |
| `-m, --model NAME` | Model to use (default `parakeet-tdt-0.6b-v2-int8`). See `--list-models` for what's on disk. |
| `--vocabulary-file FILE` | Terms for hotword biasing and post-ASR correction, one `term :boost` per line (default: none — biasing and correction both off). Served per-request against the warm recognizer; a change in per-word boosts costs a reload — see "The daemon". |
| `--format text\|json` | Output shape (default `text`). See `--format json` below. |
| `--no-download` | Fail rather than fetch a missing model; also never starts the daemon or loads an encoder for `--list-models`. |
| `--list-models` | Print installed model names, one per line, and exit. |
| `--no-daemon` | Load the recognizer in-process for this call instead of talking to a daemon; pays the ~4 s load every time. |
| `--socket PATH` | Daemon socket path (default `$AURIS_HOME/auris.sock`). |
| `-q, --quiet` | Suppress the progress line (stderr is already silent when not a terminal). |

## The daemon

`auris serve` loads the recognizer once and listens on a Unix socket, default
`$AURIS_HOME/auris.sock` (`$AURIS_HOME` defaults to `~/.cache/auris`,
mirroring kokoro-rs's `KOKORO_HOME` convention), overridable with
`--socket`. A plain `auris` invocation is the client: it connects to that
socket, and if nothing is listening, starts a daemon itself before
proceeding — a caller never has to run `auris serve` by hand for auris to
behave as a fast per-call filter. An idle daemon exits after a period with no
requests, so a warm ~1.5 GB process is not held forever on a machine that
transcribes once and then goes quiet.

A vocabulary term list is a per-request property, not a daemon-startup one.
`OfflineRecognizer` in the Rust crate (`sherpa-onnx` 1.13.6) offers
`create_stream_with_hotwords(&self, hotwords: &str)` alongside the plain
`create_stream`, wrapping the C API's
`SherpaOnnxCreateOfflineStreamWithHotwords` — so a client's `--vocabulary-file`
is normally honoured by creating a fresh, cheap `OfflineStream` per request
against the already-loaded recognizer, with no rebuild and none of the ~4 s
cost. `-m`, by contrast, remains a daemon-startup property: switching models
always means loading a different recognizer.

Three things stay fixed at recognizer construction regardless of per-request
hotwords: `hotwords_score` — a single *global* boost, not per-word and not
settable per stream — plus `bpe_vocab` and `modeling_unit`, the synthesised
sentencepiece vocab `docs/engine.md` requires for this model. This matters
because the spike's headline result (`docs/engine.md`) came from
*differentiated per-word* boosts (`khora :6.5`, `qorvex :5.0`, the rest
`3.0`), built through the construction-time `hotwords_file` path — and
whether the inline per-stream string in `create_stream_with_hotwords`
accepts that same `word :score` syntax, or only a flat term list scored
uniformly by the recognizer's one global `hotwords_score`, is not verified.
So the defined behaviour is: a request whose vocabulary needs only the
daemon's already-loaded per-word boosts is served per-stream, cheaply; a
request needing *different* per-word boosts than what the daemon loaded
falls back to a full recognizer rebuild (~4 s, once). That fallback is
specified behaviour here, not a caveat to resolve later.

`--no-daemon` is the escape hatch: it loads the recognizer in-process and
pays the ~4 s cold start on every call, with no socket, no background
process, and no dependency on a daemon having been started. It exists for
one-off use and for debugging the daemon path itself, not for mesa's call
pattern.

## What mesa will spawn

auris does not exist in mesa's codebase yet, so this is not a quotation of
running code — it is the call mesa's driver will make, written in the exact
shape of the kokoro-rs call already there (`speech.rs:164-174`). The point it
demonstrates is the acceptance requirement itself: this needs no machinery
mesa does not already have.

```rust
let mut child = Command::new(&bin)
    .args(["-q", "--format", "json"])
    .args(vocabulary.map_or_else(Vec::new, |f| vec!["--vocabulary-file", f]))
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()?;
```

Sequence: mesa spawns auris with all three pipes open, writes the audio
payload to the child's stdin from a dedicated thread (never a shell string,
never an argument — the same load-bearing property `speech.rs`'s own
comment states for kokoro-rs), and drains stderr on its own thread for the
child's whole life so a verbose failure cannot block the write auris is
waiting on. auris's client half connects to the daemon (starting it if
absent), forwards the audio, and streams back JSON as it decodes. mesa reads
stdout to EOF the same way it already reads WAV bytes from kokoro-rs; a
nonzero exit with nothing committed to stdout is the one signal mesa treats
as failure, exactly as it does today when kokoro-rs produces no audio.

## `--format json`

```json
{
  "text": "book a call with khora for tomorrow",
  "segments": [
    {
      "text": "book a call with khora for tomorrow",
      "start": 0.0,
      "end": 2.14
    }
  ]
}
```

`text` is the full corrected transcript; `segments` is one entry per
completed utterance, each with its own `start`/`end` in seconds and its own
corrected `text`. A single-utterance recording produces one segment whose
text equals the top-level `text`.

## Notes

- **Partial hypotheses do not exist.** Parakeet TDT is an offline
  recognizer: it decodes a whole utterance at once, so there is no
  in-progress transcript to emit for audio still arriving within an
  utterance. What auris ships instead is per-utterance flushing on a
  multi-utterance stream (see "stdin" above) — progressive output at
  utterance granularity, not word-by-word streaming.
- **Where an utterance boundary falls is out of scope here.** This contract
  fixes the flush-per-utterance guarantee; the endpointing/VAD design that
  decides where one utterance ends and the next begins is task 926's.
- **Per-stream hotwords are confirmed, not an open question.**
  `create_stream_with_hotwords` exists in the Rust crate at v1.13.6 and is
  what lets a per-request vocabulary avoid the daemon's ~4 s reload in the
  common case. What remains unconfirmed is whether that inline path accepts
  the same per-word `word :score` syntax the construction-time
  `hotwords_file` does, or only a flat term list under one global
  `hotwords_score` — that should be settled empirically in task 933 before
  the daemon relies on the cheap path for anything but a uniform boost.
- **A warm daemon holds ~1.5 GB resident** (`docs/engine.md`'s measured peak
  RSS for parakeet int8), for as long as it stays warm. That is the cost of
  making per-call latency match the 0.082 warm RTF instead of the 0.634
  cold one — accepted in `docs/engine.md` as "a fixed cost paid once by a
  daemon rather than per utterance."
