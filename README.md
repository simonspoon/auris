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

**Homebrew** (macOS and Linux, Apple Silicon and Intel):

```sh
brew install simonspoon/tap/auris
```

That installs a prebuilt binary from the matching GitHub release; pushing a
`v*` tag builds the four binaries and updates `Formula/auris.rb` in
`simonspoon/homebrew-tap` automatically. To build from source instead, from
the repo root:

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
  silero_vad.onnx                  # the VAD gate's model — see below
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

`silero_vad.onnx` lives at `$AURIS_HOME` directly, a sibling of `auris.sock`
and `models/`, not inside a model directory. That is deliberate, not an
oversight: "a model is a directory of five files" is the invariant
`--list-models`/`dir_is_complete` depend on to tell an installed model from a
half-downloaded one, and folding a sixth, unrelated file into that directory
would break it. The VAD is also model-independent — one VAD gates the input
ahead of whichever recognizer `-m` selects — so it belongs at the level that
outlives a model swap, not inside the thing that gets swapped.

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

#### The Silero VAD model

Fetched from `csukuangfj/vad`, commit
`fba88cd2e921609e7675c3aaf51e0b9b295da4bc` — sherpa-onnx's own author's repo,
chosen for the identical reason as the parakeet repo above: HuggingFace
publishes the sha256 as the LFS `oid`, so the digest comes from upstream
rather than being one auris made up.

| File | Bytes | sha256 |
| --- | --- | --- |
| `silero_vad.onnx` | 1,807,522 | `a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28` |

It goes through the identical verified-write path the parakeet files do:
stream to `<name>.part`, verify size and sha256, atomic rename into place —
see "Verification" below. `auris serve` fetches it alongside the parakeet
model, so "run `auris serve` once after install" (see "Getting the model")
remains the complete install-time fetch step; a transcribe run fetches it too
if `auris serve` was never run first.

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

Two auris processes can end up racing to fetch the same missing model — two
terminals cold-starting `auris serve` at once, say, or a transcribe run
launched while `auris serve` is still fetching. Both would otherwise stage
into the same deterministic staging directory and clobber each other's
download mid-stream, so each fetch first takes a blocking advisory lock
(`flock`) scoped to that model (or, for the VAD file, to the VAD download).
Whichever process gets there first proceeds as above; the other blocks until
the lock is free, then checks again whether the model is already installed
before doing anything else — finding that it is, since the winner just
finished, it simply returns rather than fetching a second copy or reporting a
failure. The wait is silent unless progress output is already enabled —
`verbose`, the same `!quiet && stderr.is_terminal()` condition that gates
download progress lines (see "stderr" below) — in which case the waiting
process prints one line saying it is waiting on another auris before it
blocks.

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

An *utterance* is read to EOF before decoding, because Parakeet is an
*offline* recognizer — it decodes a whole utterance and cannot emit a
hypothesis for audio it has not yet seen the end of. The stream is not:
**stdout's first byte does not wait for stdin's EOF on a multi-utterance
stream**. Incoming audio is segmented into utterances as they complete
(Silero VAD, `--vad-min-silence` of trailing quiet ends one), and each
utterance's text is flushed to stdout as soon as the *next* utterance
begins — a closed utterance's right edge is where the following one starts,
which is what lets the recognizer keep the acoustic context on both sides
of it that `--vad-*` below explains is load-bearing. For a single short
recording (the common case: one WAV, one utterance) this collapses to
"auris reads it all, then writes the transcript" — there is nothing to
flush early, and the recognizer is handed the whole buffer, byte for byte,
exactly as it was before segmentation existed. The segmentation itself —
where an utterance boundary falls — is decided in `docs/streaming.md`; see
Notes below.

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

Two ordering notes, because they are the kind of thing a caller pins a test
to. Everything that can be judged without looking at the audio — a bad flag
value, a `--vocabulary-file` that fails "Validation" in
`docs/vocabulary.md`, a model directory missing a file — is judged *before*
the audio is read, so a bad vocabulary file exits `2` even when the audio
behind it was silent (segmentation moved this ahead of the gates; it used
to exit `1`). Only the WAV *header* is parsed ahead of that, so "input was
empty" and "not a wav file" still precede everything. The same reordering
means `--no-daemon` loads the recognizer before reading the audio, so a
silent clip now pays that ~4 s load even though the recognizer will never
be run on it — the price of not putting "this model directory is broken"
behind the audio.

Silent, or merely non-speech, audio needs its own gates to actually get to
exit 1: the real Parakeet model doesn't reliably return an empty transcript
on silence or noise, it hallucinates a word or two ("Okay.", "Yeah.",
observed on digital silence and on a fan, a chair, a cough, TTS playback)
instead. Trusting the recognizer's output alone would let that hallucinated
text out at exit 0, which is worse than an empty transcript at exit 1 —
mesa's driver would treat it as something the user actually said (mesa
session 37: three consecutive turns transcribed as nothing but "Yeah.",
none of them spoken by the person). So there are now three gates around the
recognizer, each capable of exiting `NOTHING_TRANSCRIBED` on its own. The
first two run ahead of the recognizer, in sequence; the third runs after it
and answers a different question entirely — see below.

1. A cheap energy gate over the decoded samples (`audio::is_silent`) —
   pure arithmetic, no model to load, and it catches digital silence for
   free. It answers "is there any signal at all," nothing more refined.
2. The Silero VAD gate (`--vad-*` below) — a small model that answers the
   different question the energy gate can't: is this signal *speech*, at
   all, anywhere in it. It is a **decision, not a filter**: if Silero finds
   no speech span whatsoever, the recognizer never runs. If it finds even
   one, the recognizer decodes **slices of the original samples cut at
   *neighbouring* utterance boundaries** — never the spans Silero found. On
   a single-utterance recording that slice is the whole buffer, exactly as
   `is_silent` saw it; on a stream of several it is a partition of that
   buffer, so every sample is decoded exactly once and no utterance is
   cropped to Silero's idea of where it starts.

That second sentence is not the obvious design, and the obvious one was
tried first and rejected on evidence, not preference: an earlier version of
this gate concatenated only the speech spans Silero found and handed the
recognizer that trimmed buffer. It made things worse in both directions.
Silero's span boundaries are tuned for segmentation, where whatever consumes
a span adds its own padding and the exact edge doesn't matter much; Parakeet
is an *offline* recognizer decoding a single buffer, where acoustic context
right at the edge of the audio is load-bearing. Trimming 166 ms off the
front of one real-speech fixture turned "mesa" into "Nessa" and "khora" into
"Cora"; across a 40-clip benchmark corpus, 35 of 40 transcripts changed
merely from being trimmed to Silero's idea of where speech starts and ends.
Worse, trimming *created* the exact failure mode this gate exists to
prevent: a 4 s clip of pure background noise correctly decodes to nothing
when the recognizer sees the whole clip, but Silero false-positives a
0.55 s span inside it, and narrowing the audio down to just that span is
what makes Parakeet hallucinate "Uh" — the isolated-fragment problem
`audio::is_silent`'s own doc comment already describes for digital silence,
reproduced by the very gate meant to guard against it.

So the gate never decides where audio gets cut; it decides only whether the
recognizer runs, and where one utterance ends. The payoff is a guarantee
stronger than a benchmark number: on any **single-utterance** audio Silero
finds speech in, auris's transcript is byte-identical to a `--no-vad` run —
not "measured close," but true by construction, because one utterance means
one slice, and one slice means the whole buffer.

Segmentation is where that stops being a byte-for-byte identity and starts
being a measurement, because a stream cut into several utterances is
genuinely decoded as several buffers. Measured on the same 40-clip corpus,
biased, `--no-vad` against the shipped segmenting default: 33 of 40
transcripts are byte-identical and 7 differ, all of them long dictations
with a pause inside them, and the differences are overwhelmingly where a
sentence break lands rather than which words were heard. Scored rather than
eyeballed, segmentation is a wash to slightly better — **WER 6.32% →
6.12%**, name F1 **69.2% → 69.2%** (unchanged, term for term), sentence
boundary F1 **0.176 → 0.186**. The accuracy claim this design has to make
is "segmenting costs nothing," and that is what those numbers say.

Byte-identical output also means Silero's precision no longer has to be
very good: a false positive costs one wasted decode that comes back empty
(still exit `NOTHING_TRANSCRIBED`, just paid for at the recognizer instead
of at Silero), never a wrong transcript. That is what lets
`--vad-threshold`'s default sit at **0.2, not Silero's own stock 0.5**: a
sweep from 0.5 down to 0.05 against 43 real-speech files (the 40-clip
corpus plus the marginal, `mesa-names.wav`, and `plain.wav` fixtures) found
that 0.5 rejects `u17.wav` outright — a genuinely quiet but real spoken
clip — while every value from 0.10 to 0.38 accepts all 43 and still rejects
white noise, rumble, and silence cleanly. 0.2 is the middle of that band,
not an edge of it. `--vad-min-silence` was swept the same way, from 0.05 s
to 0.5 s, and under the accept/reject gate alone it was deliberately *not*
a flag: it controls only *when* an already-open speech span closes, and a
gate that only asks whether one ever opens could not be changed by it at
any value tested. Segmentation makes it load-bearing for the first time —
it is now exactly the rule for where one `segment` line stops and the next
begins — so it **is** a flag now, defaulting to the same 0.5 s (Silero's
stock value, and the number `docs/latency.md`'s backdating arithmetic is
written against). Zero or negative is a usage error: no trailing silence
does not mean "no minimum," it means a span that never closes, and so a
stream that never produces a `segment` line at all. Reconciling this
default with mesa's own `live.auto-send-ms` is explicitly still open
(`docs/latency.md`, "What this does not decide"). `--no-vad` skips the VAD
entirely — for audio that is already known to be one utterance of speech,
or to compare against no VAD at all — and yields exactly one segment
covering the whole buffer.

One gap this sweep did not close, and could not: `nonspeech-transient.wav`,
a synthetic noise burst, is not separable from real speech at any threshold
tried, down to 0.05 — Silero reports a confident ~0.55 s speech span in it
regardless. So of the four non-speech fixtures this task tested, **three
are rejected by the gate (white noise, rumble, digital silence) and one is
not.** That is not a regression: unbiased, the full clip still decodes to
empty and exits 1 at the third `NOTHING_TRANSCRIBED` site below, exactly as
it did before this gate existed, because one detected span in one clip
still hands the recognizer the untouched original samples — a VAD-on and a
`--no-vad` run on this clip are byte-identical, same as every other
single-utterance file. The
threshold that separates this clip from real speech does not exist in the
range tested, and the alternative — an invented heuristic distinct from
Silero's own decision — would risk eating genuinely short real utterances
like "yes," so that heuristic was not attempted; this remains a known,
pre-existing gap task 968 does not close, not one it introduces.

This is a deliberate, and considered, departure from mesa task 968's
original wording, which asked auris to "drop non-speech spans, and
concatenate or transcribe the speech spans only." That clause was a means;
the acceptance criterion — no measurable accuracy loss on real speech — was
the end, and the means defeated the end, for the reasons measured above.
auris implements the end, not the clause.

But with a vocabulary file loaded — mesa's actual production configuration
— the VAD gap above used to matter far more than "one unbiased clip decodes
to empty": the same untouched clip decoded to a long hallucinated string
built out of boosted vocabulary terms instead, and that was a real,
user-visible leak, not a hypothetical one. Closing it did not mean closing
the VAD gap above — the false positive on `nonspeech-transient.wav` is
still there, and still cannot be tuned away for the reasons just given.
Instead, mesa task 970 adds a third gate, downstream of everything above,
that catches the *consequence* of the false positive rather than the false
positive itself.

The manufactured-vocabulary guard runs after the recognizer returns a
non-empty transcript and only when `--vocabulary-file` is in effect.
Hotword biasing is meant to nudge an existing hypothesis
toward the vocabulary, not to manufacture a transcript out of nothing, but
on non-speech audio it can do exactly that — `nonspeech-transient.wav`
biased decoded to 286 characters of "The vegetable mesa mesa khora khora
khora ... mesa khan mesa q" at exit 0, where the same clip unbiased
correctly exits 1 with empty stdout. Lowering the boost cannot fix this:
the hallucination is non-monotonic in boost — the same clip hallucinates
at 2.0 and 3.0, is clean at 4.0 and 6.5, and produces different (digit)
garbage at 8.0, and other non-speech fixtures hallucinate at their own
different boosts, so no boost is universally safe. Worse, a vocabulary
file with no `:boost` values is not "unboosted" — bare terms inherit the
global `hotwords_score`, fixed at 3.0 (`docs/vocabulary.md`), which is
already inside the affected range, and 3.0 is also exactly where real
speech first *gains* a feature (`mesa-names.wav` only decodes "qorvex"
correctly from boost 3.0 up) — so lowering the boost trades away the
vocabulary feature's own purpose rather than fixing the leak.

Trusting a cheap heuristic as the decision was rejected too: instead,
the guard is a prefilter plus a confirming decode, the same
"decision, not a filter" shape the VAD gate above already uses. A cheap
prefilter, `vocabulary::looks_manufactured`, fires when the transcript
contains a run of 2 or more consecutive words that are each a vocabulary
word (case-insensitive, punctuation stripped; a run counts across
different terms, since the measured hallucinations mix terms freely).
Firing decides nothing on its own — it only spends a second, unbiased
decode of the exact same audio, on the already-loaded recognizer or via
a second daemon call with empty hotwords, no second model load. If that
unbiased decode also comes back empty, the vocabulary manufactured the
whole transcript, so it is discarded and auris exits
`NOTHING_TRANSCRIBED` instead. If the unbiased decode comes back with
anything at all, the biased transcript is emitted completely unchanged —
biasing nudged a real hypothesis rather than inventing one — and a
confirming decode that *errors* is treated the same way, since an error
is not evidence of hallucination. This split is what makes a blunt
heuristic safe to run unconditionally: **the guard cannot change the
output for any audio that decodes to something unbiased**, so a false
trigger costs one wasted decode (measured at roughly 400 ms for a 4 s
clip on a warm daemon, `bench/results/latency.json`) and never a wrong
or altered transcript. There is deliberately no `--no-...` escape hatch
for this gate — unlike VAD, it provably cannot reject audio that decodes
to something unbiased, so there is nothing to escape from, and a caller
wanting no vocabulary behaviour already has one: omit
`--vocabulary-file`.

The threshold of 2 consecutive vocabulary words came from measuring, not
guessing: across 88 real-speech transcripts at shipped production boosts
(40 real room recordings, 40 synthesized corpus clips, the 8 dictation
fixtures, plus `plain.wav` and `mesa-names.wav`), the longest run of
consecutive vocabulary words is **1** — no real transcript ever puts two
vocabulary terms side by side, even a 40-word dictation fixture
containing four vocabulary words in one sentence, because ordinary words
separate them. All four confirmed non-speech hallucinations reach at
least 3. So 2 sits in an empty dead zone with no observation on either
side, and the prefilter fired on 0 of those 88 real transcripts — it is
not on the normal path. The one honestly-stated residual risk: no corpus
here contains real speech saying two vocabulary terms back to back
("auris and khora" with nothing between), so that case is untested
rather than ruled out. It is harmless regardless, because it triggers
the confirming decode, which returns non-empty, and the transcript is
emitted unchanged — the cost is one decode, not a lost transcript, which
is what justified picking the more sensitive threshold over a
safer-looking 3. Out of scope, and not caught by this guard: at boost
8.0 the non-speech fixtures produce short runs of digit/letter garbage
containing no vocabulary terms at all — a different failure mode.

All four sites that can produce `NOTHING_TRANSCRIBED` — `is_silent`, the
VAD gate, the recognizer returning an empty decode, and the
manufactured-vocabulary guard — print the same wording, so there is one
thing for mesa to match on; each adds its own verbose-only stderr line
naming which one fired.

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
| `--vad-threshold F` | Silero VAD speech-probability threshold (default `0.2` — not Silero's own stock 0.5; see below). Out of `0.0..=1.0` is a usage error. |
| `--vad-min-speech SECS` | Speech spans shorter than this are dropped (default `0.25`, Silero's stock value). Negative or non-finite is a usage error. |
| `--vad-min-silence SECS` | Trailing silence that ends an utterance — where one `segment` line stops and the next begins (default `0.5`, Silero's stock value, and the value `docs/latency.md`'s arithmetic assumes). Zero, negative, or non-finite is a usage error. |
| `--no-vad` | Skip the VAD entirely — for audio that is already segmented, or to measure accuracy against no VAD at all. The whole buffer becomes one segment. Transcribe-path only; `auris serve` has no VAD flags because the VAD is client-side. |
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
{"type":"speech","active":true,"at":0.51}
{"type":"speech","active":false,"at":2.14}
{"type":"segment","index":0,"text":"book a call with khora for tomorrow","start":0.17,"end":2.14}
{"type":"transcript","text":"book a call with khora for tomorrow"}
```

`segment` is one completed, corrected utterance and is never revised;
`index` counts them from 0, and `start`/`end` are that utterance's own
speech boundaries in seconds from the start of the stream — not the bounds
of the slice the recognizer was handed, which reaches further on both
sides. `transcript` is always the last line on a run that produced one —
every `segment` text joined in order — so reading to EOF and parsing the
last line is a correct reader on its own. A single-utterance recording
produces one `segment` line whose text equals the `transcript` line's.

`speech` is the activity heartbeat `docs/streaming.md` supplies in place of
the partial hypotheses it rejects, emitted whenever the VAD is running
(i.e. unless `--no-vad`). `at` on the closing line is exactly the segment's
`end` — when speech *stopped*, not when auris noticed — which is what lets
mesa backdate its silence timer instead of starting it a whole
`--vad-min-silence` late (`docs/latency.md`). Note the ordering: because a
`segment` line waits for the following utterance to open, both of an
utterance's `speech` lines are written *before* its `segment` line rather
than around it, as the worked example in `docs/streaming.md` originally
drew them. Nothing in the protocol was ever ordered — every line is
self-describing, `index` orders the segments, and a reader must ignore any
`type` it does not recognise.

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
| auris + biasing + correction pass (not in the crate — `bench/harness/vocab_correct.py`) | 89.1% | 4.89% |

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
