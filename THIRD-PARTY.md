# Third-party components

## espeak-ng and piper_phonemize — GPL-3.0(-or-later)

auris never phonemises anything — Parakeet is a NeMo transducer over raw
audio, with no text-to-phoneme step anywhere in its path. Even so, both
libraries travel into the auris binary as unconditional passengers of
`sherpa-onnx`.

`sherpa-onnx-sys` 1.13.6 downloads a prebuilt static archive rather than
compiling from source (`build.rs`, fn `download_prebuilt_libs`), and its
`emit_static_link_directives` (`build.rs:286-288`) unconditionally links a
fixed list of 13 static libraries out of that archive (`build.rs:14-28`),
among them:

- `piper_phonemize` — GPL-3.0-or-later in its own right, because it wraps
  espeak-ng
- `espeak-ng` — GPL-3.0

The crate's `[features]` are only `default = ["static"]`, `shared = []`,
`static = []` (`Cargo.toml`) — nothing excludes either library, and there is
no feature flag that would. Every build of `sherpa-onnx = "=1.13.6"` links
them, whether or not the program using the crate ever calls the phonemiser.

That linkage is why **auris's licence is GPL-3.0-or-later**, and it is the
same reason kokoro-rs is GPL: both link espeak-ng into the final binary.
The irony is that kokoro-rs links it because it *needs* a phonemiser, and
auris links the same GPL code while never calling it — GPL's copyleft
triggers on linking a covered work into your binary, not on whether the
linked code path ever executes. Being unused doesn't launder the
obligation.

espeak-ng is:

> Copyright (C) 2005 to 2013 by Jonathan Duddington
> Copyright (C) 2013-2017 Reece H. Dunn
> Copyright (C) 2015-2024 by the espeak-ng contributors

Upstream: <https://github.com/espeak-ng/espeak-ng>. piper_phonemize:
<https://github.com/rhasspy/piper-phonemize>. Full licence text is in
[LICENSE](LICENSE).

**Written offer of corresponding source.** The corresponding source for
both libraries as linked into auris is whatever espeak-ng and
piper_phonemize revisions the `sherpa-onnx` v1.13.6 prebuilt static archive
was built from — obtainable from the upstream repositories above and from
the `sherpa-onnx` v1.13.6 release itself:
<https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6>.

## sherpa-onnx and the k2-fsa libraries — Apache-2.0

`sherpa-onnx`, `sherpa-onnx-sys`, and the remaining static libraries in the
same prebuilt archive — `sherpa-onnx-c-api`, `sherpa-onnx-core`,
`kaldi-decoder-core`, `sherpa-onnx-kaldifst-core`, `sherpa-onnx-fstfar`,
`sherpa-onnx-fst`, `kaldi-native-fbank-core`, `kissfft-float`, `ucd`,
`ssentencepiece_core` — are Apache-2.0, from upstream
[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Apache-2.0 §4
requires that a NOTICE file, if the original carries one, be reproduced in
any redistribution; auris carries the upstream NOTICE forward if and when
one is shipped with the v1.13.6 release.

## ONNX Runtime — MIT

Also one of the 13 static libraries in the same archive
(`libonnxruntime.a`), linked statically via
`cargo:rustc-link-lib=static=onnxruntime` (`build.rs:14-28`, `:286-304`).
This is where auris diverges from kokoro-rs: kokoro-rs downloads a dylib
into `~/.cache/kokoro-rs` at runtime and loads it dynamically (`ort`'s
`load-dynamic` feature); auris links the runtime straight into the binary,
so nothing ONNX-Runtime-shaped is ever fetched or shipped separately.
MIT imposes no obligation beyond preserving its copyright notice, which is
satisfied by this file.

Upstream: <https://github.com/microsoft/onnxruntime>.

## Components downloaded at runtime

Not bundled or redistributed here — fetched from the publisher on first
`auris serve`, into `~/.cache/auris`.

| Component | Licence | Source |
| --- | --- | --- |
| Parakeet TDT 0.6B v2 weights (int8) | CC-BY-4.0 | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (original), [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8) (int8 re-upload auris fetches from) |
| Silero VAD model (`silero_vad.onnx`) | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (original), [csukuangfj/vad](https://huggingface.co/csukuangfj/vad) (re-upload auris fetches from, sherpa-onnx's own author's repo) |

CC-BY-4.0 requires attribution to NVIDIA when the model (or a derivative)
is redistributed; this table is that attribution. MIT requires only that
its copyright notice travel with the software, which this table satisfies
for the same reason it does for ONNX Runtime above — nothing here
redistributes either the Parakeet weights or the Silero VAD model
themselves, auris only ever downloads them onto the machine that runs it.

## A note on Cargo.toml

The `sherpa-onnx` dependency described above is not yet in this project's
`Cargo.toml` — this initial skeleton predates it, and the dependency lands
in a later task. The licence is fixed as GPL-3.0-or-later now regardless,
because the engine decision that requires it is already made and recorded
in `docs/engine.md` and the README. Choosing the licence when the decision
is made, rather than after the dependency lands and contributions have
accrued under a different one, is the whole point of doing it here.

## Rust dependencies

The rest of the crate graph auris builds against is expected to be
permissively licensed (MIT, Apache-2.0, or MIT/Apache-2.0 dual), with
`sherpa-onnx`/`sherpa-onnx-sys` the exception noted above: the binding
crate is Apache-2.0, but the espeak-ng and piper_phonemize code its build
script links is not. `cargo tree` will list the full set once
`sherpa-onnx` is a real dependency.
