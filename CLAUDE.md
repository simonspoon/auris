# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

auris is a command-line speech-to-text binary — NVIDIA Parakeet TDT 0.6B v2
(int8 ONNX) run through sherpa-onnx — that reads audio from stdin or a file
argument and writes a transcript to stdout. It is one binary with two roles: a
thin per-call client, and `auris serve`, a persistent daemon holding the
loaded recognizer. It is the mirror of kokoro-rs (text → audio); auris does
audio → text, and both are meant to sit on either side of the same pipe, with
mesa's speech driver (`mesa/src/core/speech.rs`) as the primary consumer.

## Status: partial

Built and tested today: audio input (`src/audio.rs`), the recognizer
(`src/engine.rs`), the vocabulary term list (`src/vocabulary.rs`), the
one-shot transcribe path (`src/cli.rs`), and the daemon and its client
(`src/daemon.rs`, task 951 — `auris serve` / `status` / `stop`, documented in
`docs/daemon.md`). Not built: model downloading and sha256 verification (so
`auris serve` does not yet fetch a missing model), VAD segmentation and
streaming (`docs/streaming.md`), and post-ASR correction
(`docs/correction.md`). The invariants below describe what auris **must** do,
read from `README.md` — the contract written before the code — not what it
currently does. Check the source before relying on any of it.

## Commands

```bash
cargo build --release                            # release binary; no cmake, no C++ toolchain needed
cargo test                                        # unit + integration tests
cargo clippy --all-targets -- -D warnings         # lint (warnings are errors)
cargo fmt --check                                 # format check
scripts/install.sh                                # build --release and install to ~/.local/bin
PREFIX=/usr/local scripts/install.sh              # install to /usr/local/bin
BINDIR=/opt/bin scripts/install.sh                # install to an exact directory
```

CI (`.github/workflows/ci.yml`) runs check, test, clippy `-D warnings`, and
fmt on Linux and macOS.

## Architecture

`README.md` is the source of truth for the CLI contract; `docs/engine.md`
explains the parakeet/sherpa-onnx/persistent-process decision and its
measurements. `docs/streaming.md`, `docs/vocabulary.md`, `docs/correction.md`,
`docs/latency.md`, and `docs/posture.md` cover streaming/endpointing,
hotwords, post-ASR correction, the latency budget, and the client/daemon
split respectively. **The repo is contract-first**: README and docs/ predate
the implementation. When code and README disagree, that is a bug in one of
them — decide which, don't let them silently drift.

## Conventions

- **One crate, never a workspace.** auris is one binary with one job; the
  client and daemon are two roles of that one binary, not two crates.
- **stdout carries only the transcript.** No progress, no logging, no
  decoration beyond `--format`. `--list-models` prints one bare model name per
  line — mesa's `voices()` line-splits and filters that output, so the format
  is load-bearing, not cosmetic (README "stdout").
- **Progress goes to stderr, only when stderr is a terminal**:
  `verbose = !quiet && stderr.is_terminal()`. Errors always go to stderr,
  each line prefixed `auris: ` (README "stderr").
- **Exit codes are copied from kokoro-rs, not reinvented**: `0` OK, `1`
  NOTHING_TRANSCRIBED, `2` USAGE, `130` INTERRUPTED. Nonzero means "there is
  no transcript" — the sole signal mesa's driver treats as failure (README
  "Exit codes").
- **Audio never reaches the argument parser.** It arrives on stdin or as a
  file path, never as a parsed flag value — this is what makes a payload
  starting with `-o` harmless (README "stdin").
- **`--list-models` must never touch the network and must never hang.** It is
  a directory read that exits 0 even with nothing installed. mesa's
  `voices()` calls the equivalent with no timeout and memoizes the result for
  the process lifetime; kokoro-rs's `--list-voices` gets this wrong by
  loading a full ONNX session first — auris must not repeat that (README
  "`--no-download`").
- **Every downloaded file is sha256-verified before being renamed into
  place**; downloads stream to `<name>.part` first. Nothing that fails
  verification is ever left where a later run would trust it — a short or
  truncated ONNX file faults inside the runtime instead of erroring cleanly
  (README "Verification").
- **A model is a directory of five files** (`encoder.int8.onnx`,
  `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`, `bpe.vocab`), and is
  "installed" only when all five are present — a half-finished download must
  be invisible to `--list-models`, never offered and then broken (README
  "The cache").
- **`bpe.vocab` is generated from `tokens.txt`, never downloaded, never
  committed.** It is derived data, correct only for the `tokens.txt` it came
  from; a committed copy would survive a model swap that invalidated it
  (README "`bpe.vocab`").
- **`sherpa-onnx` is pinned exact (`=1.13.6`).** The crate version selects
  which prebuilt native archive is downloaded, which pins the C++ and ONNX
  Runtime builds too, not just the Rust API — do not relax it to a caret
  range (README "Build prerequisites").
- **The licence is GPL-3.0-or-later, and it is not a choice.** sherpa-onnx's
  prebuilt static archive links espeak-ng (GPL-3.0) and piper_phonemize
  unconditionally; see THIRD-PARTY.md. Do not "simplify" this to MIT.
- **No cmake, no C++ compiler, no libclang.** auris builds nothing from
  source; `sherpa-onnx-sys`'s build.rs downloads a prebuilt archive instead
  of compiling one. A change that starts requiring a C++ toolchain is a
  regression worth arguing about (README "Build prerequisites").
- **The model is a daemon-startup property, not a per-request one.**
  Switching `-m` always means loading a different recognizer (~4 s). A
  vocabulary term list, by contrast, is normally a per-request property served
  cheaply against the already-loaded recognizer — see README "The daemon" for
  when a vocabulary change instead forces a full reload.
