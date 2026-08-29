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

Built and tested today: audio input (`src/audio.rs`), an energy gate ahead
of the recognizer for silent audio (`audio::is_silent`, task 958), a Silero
VAD gate immediately after it that filters non-speech audio before the
recognizer ever sees it (`src/vad.rs`, task 968 — `--vad-*` / `--no-vad`),
the recognizer (`src/engine.rs`), the vocabulary term list
(`src/vocabulary.rs`), the one-shot transcribe path (`src/cli.rs`), and the
daemon and its client (`src/daemon.rs`, task 951 — `auris serve` / `status`
/ `stop`, documented in `docs/daemon.md`), and model downloading with sha256
verification (`src/model.rs`, task 967 — `auris serve` and the transcribe
path both fetch a missing default model, and now the VAD model too, unless
`--no-download` is given). Not built: VAD *segmentation* and streaming — the
`speech` heartbeat, multiple `segment` lines, endpointing (`docs/streaming.md`)
— and post-ASR correction (`docs/correction.md`); task 968 shipped a gate,
not segmentation, and output is still one transcript, exactly as before. The
invariants below describe what auris **must** do, read from `README.md` —
the contract written before the code — not what it currently does. Check the
source before relying on any of it.

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
- **Silent audio never reaches the recognizer.** The real model hallucinates
  words on digital silence instead of returning nothing, so
  `audio::is_silent` gates on the decoded samples before the recognizer is
  invoked — client-side, so it also covers the daemon path (README "Exit
  codes", `docs/daemon.md`). It is a fixed-threshold energy check, not VAD,
  and it stays: the Silero VAD gate below is a heavier, model-based check
  that answers a different question, not a replacement for this free one.
- **Audio with no speech at all never reaches the recognizer, but audio
  that does is decoded untouched.** Immediately after `is_silent`, the
  Silero VAD gate (`src/vad.rs`, task 968) is a **decision, not a filter**:
  no speech span found anywhere in the buffer means the recognizer never
  runs (a fan, a cough, or TTS playback that passes the energy check gets
  caught here); any speech span found means the recognizer decodes the
  original samples, byte-for-byte, exactly as `is_silent` saw them — never
  the trimmed spans Silero found. An earlier version fed the recognizer
  only the concatenated speech spans and was reverted: Silero's span edges
  are tuned for segmentation, not for an offline recognizer's acoustic
  context, and trimming both corrupted real speech (35 of 40 benchmark
  transcripts changed) and manufactured a hallucination on a noise clip
  that decoded correctly when left whole. This is a deliberate departure
  from mesa task 968's literal wording ("drop non-speech spans... transcribe
  the speech spans only") in favor of its acceptance criterion (no accuracy
  loss on real speech) — see README "Exit codes". `--vad-threshold`
  defaults to 0.2, not Silero's own stock 0.5 (measured, not copied — see
  README); `--vad-min-silence` does not exist as a flag, because it cannot
  affect this gate's accept/reject decision at any value (it only affects
  when a span closes, and the gate only asks whether one ever opened) — it
  is a private constant in `src/vad.rs` instead. The gate is client-side,
  like `is_silent`, so it covers the daemon path too; `--no-vad` is the
  escape hatch. This is a single-utterance gate, not the segmentation
  `docs/streaming.md` describes — no `speech` heartbeat, no multiple
  `segment` lines — that remains unbuilt.
