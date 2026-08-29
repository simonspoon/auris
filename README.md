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

auris is deliberately one crate, not a workspace: one binary, one job, and
nobody should split it into a workspace on reflex.

## Install

One binary, nothing bundled inside it: everything auris needs at runtime is
either linked into the executable or downloaded once into `~/.cache/auris`.

**auris is not published anywhere yet — no remote, no release, no formula.**
`simonspoon/tap` (`Formula/khora.rb`, `loki.rb`, `mesa.rb`, `qorvex.rb`, and
a dozen more) is where it will live, in the shape those formulas already
have — a prebuilt binary per platform off a GitHub release — but neither the
release nor the formula exists today. Building from source, from the repo
directly, is the only install path:

```sh
scripts/install.sh
```

Run from the repo root, that builds in release mode and copies the binary to
`~/.local/bin`, the same `PREFIX`/`BINDIR` overrides kokoro-rs's installer
takes:

```sh
PREFIX=/usr/local scripts/install.sh
BINDIR=/opt/bin scripts/install.sh
```

To build without installing, use `cargo build --release`; the binary lands in
`target/release/auris`.

### Build prerequisites

**Rust and a linker. That is all.**

`sherpa-onnx-sys` 1.13.6's build.rs does not compile C++ (build.rs:117-211, fn
`download_prebuilt_libs`): it downloads a prebuilt archive from
`https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-osx-x64-static-lib.tar.bz2`
(the name is chosen per-platform at build.rs:225-279) and caches it under
`$CARGO_TARGET_DIR/sherpa-onnx-prebuilt/`. Its own build-dependencies are
`ureq`, `tar`, `bzip2` — to fetch and unpack, not to compile. So there is **no
cmake, no C++ compiler, no libclang** — the three things kokoro-rs's install
section requires, because kokoro-rs builds espeak-ng from source and auris
builds nothing from source. On macOS the Xcode command line tools alone are
enough: a linker, `libc++`, and the `Foundation` framework (build.rs:298-301).

Verified today: a probe crate depending on `sherpa-onnx = "=1.13.6"` with
default features built in 32 s on this host, invoking only rustc/cargo plus
one HTTPS GET to github.com. The build needs network once, for that archive —
`SHERPA_ONNX_ARCHIVE_DIR` points at a locally cached copy of it
(build.rs:144-154), and `SHERPA_ONNX_LIB_DIR` skips the download entirely and
links an existing lib directory (build.rs:102-112); either makes an offline
build possible.

The dependency is pinned exact, `sherpa-onnx = "=1.13.6"`: the crate version
selects which prebuilt native archive is downloaded, so it pins the C++ and
ONNX Runtime builds too, not just the Rust API.

### ONNX Runtime

**There is nothing to do. ONNX Runtime is statically linked into the auris
binary, so auris has no `ORT_DYLIB_PATH`, no runtime dylib, and no download
for it.**

The crate's default `static` feature emits
`cargo:rustc-link-lib=static=onnxruntime` among 13 static libs
(build.rs:14-28, :286-304); the downloaded archive carries
`lib/libonnxruntime.a`. Verified empirically: `otool -L` on the built probe
binary lists only `/usr/lib/libc++.1.dylib`, `Foundation`, `libSystem.B.dylib`
and `CoreFoundation` — no `libonnxruntime.dylib`.

This is where auris diverges from kokoro-rs, and why it is simpler: kokoro-rs
downloads a 40 MB `libonnxruntime.1.23.2.dylib` into its cache on first run
and sets `ORT_DYLIB_PATH` to point at it (`models.rs:145-165`), because the
`ort` crate uses its `load-dynamic` feature. auris links the runtime instead,
so the only thing that ever lands in the cache is the model itself. The
non-default `shared` feature would reintroduce a dylib; auris does not use it.

### The cache

**`$AURIS_HOME`, defaulting to `$HOME/.cache/auris`, computed exactly as
kokoro-rs computes its own** (`models.rs:65-71`: read the env var, else
`$HOME/.cache/<name>`, falling back to `.` when `$HOME` is unset — no `dirs`
crate).

```
~/.cache/auris/
  auris.sock                       # the daemon's socket (see "The daemon")
  models/
    parakeet-tdt-0.6b-v2-int8/
      encoder.int8.onnx
      decoder.int8.onnx
      joiner.int8.onnx
      tokens.txt
      bpe.vocab                    # generated, not downloaded
```

kokoro-rs's model is one file, so `KOKORO_MODEL` is a path to a file. A
parakeet model is four downloaded files plus one generated one that must stay
together and stay consistent with each other, so auris's unit is a
**directory**, and every model gets its own under `models/`. More than one
model may be installed at once; a model directory is only considered installed
when all five files are present, so a half-finished download is invisible to
`--list-models` rather than being offered and then failing.

### Environment

| Variable | Meaning |
| --- | --- |
| `AURIS_HOME` | The cache directory. Default `$HOME/.cache/auris`. |
| `AURIS_MODEL` | A path to a model *directory*, overriding the name-based lookup entirely. Tilde-expanded. If it does not exist or is missing one of the four required files, that is a hard error — never a silent fall back to downloading. |

There is no `AURIS_VOICES` and no `ORT_DYLIB_PATH`: auris has no voices, and
the runtime is linked in. A bad override fails the way kokoro-rs's own does —
it bails with `KOKORO_MODEL=<path> does not exist` (models.rs:74-102); auris
does the same with its own name.

### Which model, and `-m`

**`parakeet-tdt-0.6b-v2-int8` is the default and, today, the only model auris
knows how to fetch.**

`-m` accepts either a name from `--list-models` or a path to a model
directory. The discriminator is a path separator: an argument containing `/`
(or beginning with `~`) is a path, anything else is a name looked up under
`$AURIS_HOME/models/`. An unknown name is a usage error, exit 2. Precedence:
`-m` beats `AURIS_MODEL` beats the default name. The model is a daemon-startup
property, not a per-request one — see "The daemon".

### What is downloaded

**The four files, individually, from HuggingFace at a pinned commit — not the
single `.tar.bz2` that sherpa-onnx publishes.**

Repo: `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`, commit
`1ab9323565ddb038682214b292f588070a538ce2`. URL form:
`https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/1ab9323565ddb038682214b292f588070a538ce2/<file>`

| File | Bytes | sha256 |
| --- | --- | --- |
| `encoder.int8.onnx` | 652,184,296 | `a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab` |
| `decoder.int8.onnx` | 7,257,753 | `b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e` |
| `joiner.int8.onnx` | 1,739,080 | `7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2` |
| `tokens.txt` | 9,384 | `ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d` |

Total 661,190,513 bytes — call it ~661 MB.

Why not the archive: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2`
exists as a GitHub release asset under the `asr-models` tag at 482,468,385
bytes — 27% smaller, and one request instead of four — but **no checksum is
published for it anywhere**, and the whole point of verification below is that
a truncated model must be an error rather than a crash. The HuggingFace files
carry their sha256 as the LFS `oid` in the repo's own API, so each one can be
verified against a digest the upstream publishes rather than one auris made
up. That trade — 180 MB of extra transfer, once, for a verifiable download —
is the one this section makes.

The commit sha rather than `main` is deliberate too: both currently resolve to
the same bytes, but pinning the commit is what makes "the model does not
change under people" true rather than hoped for. And all four URLs advertise
`accept-ranges: bytes`, so a resumed download is possible per file — which
four files also buy over one archive.

`tokens.txt` is small enough that HuggingFace stores it as a plain git blob
rather than LFS, so its sha256 is not published by the API — unlike the other
three, this one is not checked against a digest upstream publishes. The digest
above was computed from the bytes actually served at that commit, and is
auris's own pin.

### Getting the model

```sh
cargo build --release        # the binary; no cmake, no system libraries
auris serve                  # fetches the model if it is missing, then loads it
```

There is no separate download flag, and there deliberately is not one:
`auris serve` is already the process that loads the recognizer, so it is
already the process that fetches it, and running it once after install is the
whole install-time fetch step. The daemon it starts idles out on its own (see
"The daemon"), so this leaves nothing behind — no process to remember to stop.
`--no-download` is what turns that same command into a check instead of a
fetch: with it, `auris serve` refuses to fetch a missing model and exits 1.

### Verification

**Every downloaded file is checked against its sha256 before it is put in
place.**

Downloads stream to `<name>.part` and are renamed into position only after the
digest matches, mirroring kokoro-rs's atomic write (`models.rs:243-252`) but
adding the check kokoro-rs does not have — kokoro-rs verifies nothing at all,
and tests only that the file exists. A mismatch deletes the `.part` and fails
the run; nothing that failed verification is ever left where a later run would
trust it.

On load, auris checks byte sizes only, not digests: a stat is free, rehashing
661 MB is about a second, and a file at the final path can only have got there
by passing its digest already. The size check is what catches a file truncated
by something other than auris — a full disk, a manual copy.

The reason for having any of this: an ONNX file that is short or wrong does
not produce a clean error, it faults somewhere inside the runtime, and a stack
trace from inside a static ONNX Runtime is not a bug report anyone can act on.

### `bpe.vocab`

**Generated on first install, beside the weights. Never downloaded, never
committed.**

The model repo ships no sentencepiece vocab, and `docs/engine.md` requires
one: `bpe_vocab` plus `modeling_unit` are what make per-word hotword boosts
work at all, and passing `tokens.txt` in its place silently miscalibrates the
tokenizer. It is a pure function of `tokens.txt`: for a BPE model the piece
score is the negative merge rank, so line *n* (1-based) of `tokens.txt`
becomes `<piece> <-(n-1)>`. The spike's file was produced by `awk '{printf "%s
%d\n", $1, -NR+1}' tokens.txt`, and generating it in auris reproduces that
file byte for byte — verified today against
`spike/models/parakeet/bpe_synth.vocab`, 1025 lines, `<unk> 0` first and
`<blk> -1024` last.

Committing it was rejected: it is derived data, it is only correct for the
`tokens.txt` it came from, and a committed copy would survive a model swap
that invalidated it. Generating it from the file that was just checksummed
makes that failure impossible instead of merely unlikely.

Generation is local, so `--no-download` does not prevent it — the same
exemption kokoro-rs makes for its espeak data, which `ensure_espeak_data()`
unpacks regardless of the flag (`cli.rs:156`).

### `--no-download`

**`--list-models` never touches the network, with or without the flag.** It
lists the complete model directories under `$AURIS_HOME/models/` and exits 0 —
including exiting 0 with empty stdout when nothing is installed. It does not
resolve a model, start a daemon, load a recognizer, or fetch anything. The
flag is accepted and changes nothing, because there is nothing for it to
refuse.

That is a deliberate divergence from kokoro-rs, and this is why: mesa's
`voices()` (`mesa/src/core/speech.rs:87-110`) runs `kokoro-rs --no-download
--list-voices` with `.output()`, no timeout at all, and memoizes the answer in
a `OnceLock` for the life of the process — its own comment says "there is no
cheap timeout here, so the fix is not to start anything that can hang."
kokoro-rs then answers that call by loading a full 326 MB ONNX session before
it even looks at the `--list-voices` flag (`cli.rs:156-169`), and exits 1
without listing anything if the model is not downloaded yet. mesa reads that
failure as an empty list. auris will not repeat it: the equivalent call must
be a directory read that cannot hang and cannot fail merely because no model
is installed yet.

**A real run with the model missing and `--no-download` given exits 1** —
`NOTHING_TRANSCRIBED`, the same code the exit-code table gives for a run that
produced no transcript — with one stderr line naming the model, the path it
was expected at, and `auris serve` as the way to fetch it, the same shape
kokoro-rs's own message takes (`models.rs:93-99`). No daemon is started and no
encoder is loaded.

**Without `--no-download`, a real run fetches what is missing first**, with
progress on that process's stderr only when it is a terminal. `auris serve`
and a `--no-daemon` run fetch in the process that is about to load the
recognizer; a client that would auto-start a daemon fetches in itself first,
before spawning one — the daemon-spawn reachability timeout is too short for
a ~661 MB download, and the client's stderr is the terminal the caller is
actually watching.

mesa's synthesis path spawns kokoro-rs *without* `--no-download`
(`speech.rs:163-169`) and never times the child out, so the first render on a
cold cache blocks on a 394 MB download. The auris equivalent would block on
661 MB. The model should therefore be fetched once at install time, not on a
user's first utterance.

### A note on the model name

`parakeet-tdt-0.6b-v2-int8` contains a `.`, and mesa's `is_voice_name`
(`speech.rs:117-124`) accepts only ASCII alphanumerics, `_` and `-`, so as
written today mesa would filter that name straight out of a `--list-models`
answer. The name is kept, because it is the upstream model's name and renaming
it to dodge a validator would be the wrong repo paying. mesa's model-name
check must allow `.` — one character, in a check that does not exist yet.

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
falls — is decided in `docs/streaming.md`; see Notes below.

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

Silent audio needs its own gate to actually get to exit 1: the real
Parakeet model doesn't reliably return an empty transcript on silence, it
hallucinates a word or two ("Okay.", observed on 1 s of digital silence)
instead. Trusting the recognizer's output alone would let that hallucinated
text out at exit 0, which is worse than an empty transcript at exit 1 —
mesa's driver would treat it as something the user actually said. So auris
runs a cheap energy gate over the decoded samples before the recognizer is
reached at all (`audio::is_silent`), and exits `NOTHING_TRANSCRIBED`
directly when nothing but near-zero signal arrived, model or no model.

The consequence for mesa: a nonzero exit means "there is no transcript," the
same rule `speech.rs` already applies to kokoro-rs — a failed render is not
data, it is an `Err` the API answers `unavailable` with. auris's driver
reads the exit status the same way: consulted only when nothing usable
landed on stdout, never as the primary signal.

## Flags

| Flag | Meaning |
| --- | --- |
| `PATH` | Audio file to transcribe (positional). Given, stdin is simply not read; omitted, stdin is read unless it is a terminal. |
| `-m, --model NAME` | Model to use: a name from `--list-models`, or a path to a model directory (default `parakeet-tdt-0.6b-v2-int8`). |
| `--vocabulary-file FILE` | Terms for hotword biasing and post-ASR correction, one `term :boost` per line (default: none — biasing and correction both off). Served per-request against the warm recognizer; a vocabulary change is always free and never reloads it — see "The daemon". |
| `--format text\|json` | Output shape (default `text`). See `--format json` below. |
| `--no-download` | Fail rather than fetch a missing model on a real run; `--list-models` is always offline, flag or not. |
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
behave as a fast per-call filter. An idle daemon exits after 300 s with no
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
sentencepiece vocab `docs/engine.md` requires for this model. The global is
fixed at 3.0 and is not exposed as a CLI choice (`docs/vocabulary.md`): it is
the value that reproduces the spike's headline 83.3% name-F1 / 4.0% WER
result, and per-word `:score` syntax (`khora :6.5`, `qorvex :5.0`, the rest
`3.0`) works identically through `create_stream_with_hotwords` as it does
through the construction-time `hotwords_file` path — verified empirically, a
per-stream string and a `hotwords_file` carrying the same terms and scores
produce byte-identical transcripts (`docs/vocabulary.md`). So **every
vocabulary change, including a changed per-word boost, is served per-stream
against the already-loaded recognizer and never rebuilds it** — there is no
fallback path and no reload cost to pay.

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
absent), forwards the audio, and streams back JSON Lines as it decodes. mesa
reads stdout incrementally — the same pattern `speech.rs::start` already
runs over kokoro-rs's stdout, reading in a loop rather than draining to EOF
(`docs/streaming.md`) — so parsing `\n`-delimited JSON lines is no new
machinery. A nonzero exit with nothing committed to stdout is the one signal
mesa treats as failure, exactly as it does today when kokoro-rs produces no
audio.

## `--format json`

Output is JSON Lines, not a single document: one `\n`-terminated JSON object
per line, flushed as written, each carrying a `type` discriminator. A reader
must ignore any `type` it does not recognise — that is the whole extension
mechanism. Full protocol and reasoning: `docs/streaming.md`.

```
{"type":"segment","index":0,"text":"book a call with khora for tomorrow","start":0.0,"end":2.14}
{"type":"transcript","text":"book a call with khora for tomorrow"}
```

`segment` is one completed, corrected utterance and is never revised.
`transcript` is always the last line on a run that produced one — the whole
corrected text — so reading to EOF and parsing the last line is a correct
reader on its own. A single-utterance recording produces one `segment` line
whose text equals the `transcript` line's.

## Verification

**auris's central claim is the vocabulary — biasing, plus a correction pass
not yet in the crate, make mesa's own names legible where a browser's speech
recognizer garbles them. The receipts are `BENCHMARK.md`**, measured against
40 utterances of mesa dictation (`bench/corpus/`) — task ids, mesa's own tool
names, half-thoughts, four words to 22 seconds of unbroken rambling — not
read passages:

| Config | Name F1 | WER |
| --- | --- | --- |
| auris, no vocabulary | 62.2% | 6.52% |
| **auris + vocabulary biasing (what ships today)** | **69.2%** | **6.32%** |
| auris + biasing + correction pass (not in the crate — `spike/harness/vocab_correct.py`) | 89.1% | 4.89% |

Per-term is where the real story is. `khora` never recovers from biasing
alone — 0.0% F1 across eight occurrences, the same zero the original 8-fixture
spike found, unmoved by a corpus five times larger. What biasing does instead
is decide what the model produces in `khora`'s place ("qora", not something
unrecoverable), and the correction pass repairs that to 76.9%. `qorvex` moves
0.0% → 50.0% → 90.9% the same way. Full per-term table, the punctuation
numbers, and the latency distribution measured against `docs/latency.md`'s
300 ms budget (94.2% of warm calls cost nothing at all; every `--no-daemon`
call breached the 2300 ms ceiling): `BENCHMARK.md`.

Two things that table deliberately does not paper over:

- **The Web Speech baseline is NOT MEASURED.** `bench/harness/webspeech/` has
  a written, committed method, but `SpeechRecognition` opens the system
  microphone directly and bypasses `getUserMedia` — Chrome's fake audio device
  feeds a WAV in (measured, peak RMS 0.3334) and the recognizer still ends in
  `no-speech`. Closing that gap needs an OS-level loopback (`blackhole-2ch`)
  that this measurement run had no admin rights to install; tracked as task
  964. Until it lands, "auris beats the browser" is an unmeasured claim.
- **The corpus is synthetic** — macOS `say`, five voices, three rates, not
  the owner's own dictated voice. `BENCHMARK.md` treats the absolute numbers
  as a floor, not a prediction; relative orderings (biasing vs. correction,
  warm vs. cold) are the trustworthy part. `bench/corpus/record.sh`
  re-records the same 40 sentences in a real voice.

The 83.3% name-F1 / 4.0% WER figure in "The daemon" above is the original
8-fixture spike measurement that fixed `hotwords_score` at 3.0; that config is
unchanged. `BENCHMARK.md` is the same claim re-measured at five times the
corpus size, and is the fuller record.

## Notes

- **Partial hypotheses do not exist.** Parakeet TDT is an offline
  recognizer: it decodes a whole utterance at once, so there is no
  in-progress transcript to emit for audio still arriving within an
  utterance. What auris ships instead is per-utterance flushing on a
  multi-utterance stream (see "stdin" above) — progressive output at
  utterance granularity, not word-by-word streaming.
- **Where an utterance boundary falls is decided in `docs/streaming.md`.**
  This contract fixes the flush-per-utterance guarantee; the endpointing/VAD
  design that decides where one utterance ends and the next begins, and the
  `speech`/`segment`/`transcript` line protocol that carries it, is there.
- **Per-stream hotwords are confirmed, not an open question.**
  `create_stream_with_hotwords` exists in the Rust crate at v1.13.6 and is
  what lets a per-request vocabulary avoid ever reloading the recognizer.
  Whether that inline path accepts the same per-word `word :score` syntax
  the construction-time `hotwords_file` does was the open question — it was
  settled empirically (`docs/vocabulary.md`): it does, byte-identically, so
  the daemon serves every vocabulary change per-stream, not only a uniform
  boost.
- **A warm daemon holds ~1.5 GB resident** (`docs/engine.md`'s measured peak
  RSS for parakeet int8), for as long as it stays warm. That is the cost of
  making per-call latency match the 0.082 warm RTF instead of the 0.634
  cold one — accepted in `docs/engine.md` as "a fixed cost paid once by a
  daemon rather than per utterance."

## License

GPL-3.0-or-later. This reverses the earlier expectation, from before the
engine decision, that auris could be MIT the way kokoro-rs's sibling tools
are — `sherpa-onnx`'s prebuilt static archive links `espeak-ng` and
`piper_phonemize` unconditionally, both GPL, and neither has a feature
flag that excludes it. auris never phonemises anything — that machinery
belongs to text-to-speech, not ASR — but GPL's copyleft triggers on
linking a covered work into the binary, not on whether the linked code
runs. Full reasoning and the exact `build.rs` lines: `THIRD-PARTY.md`.

Not covered: the Parakeet TDT 0.6B v2 weights fetched into
`~/.cache/auris` at runtime are CC-BY-4.0, NVIDIA's, and never
redistributed by this repository — see `THIRD-PARTY.md` for the
attribution and source links.
