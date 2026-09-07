//! Runs Silero VAD over already-16kHz-mono-f32 samples and reports where
//! each utterance ends (README's `--vad-*` flags, `docs/streaming.md`).
//! This module knows nothing about the CLI or the daemon;
//! [`Vad::segmenter`] is the one entry point `cli.rs::run_transcribe`
//! calls, driven chunk by chunk as audio arrives in the same spot
//! [`crate::audio::is_silent`] already runs and immediately after it —
//! `is_silent` stays as a free, model-free short-circuit ahead of this
//! heavier gate (CLAUDE.md).
//!
//! **The VAD is a decision, not a filter.** [`Segmenter::next_segment`]
//! returns a [`Segment`] of *offsets*; it never trims, crops, copies, or
//! otherwise hands back audio, and in particular it never returns
//! `SpeechSegment::samples()`. The caller keeps the original buffer and
//! decodes a slice of it whose edges are the *neighbouring utterances'*
//! boundaries, not this utterance's — so a single-utterance recording is
//! decoded as the whole buffer, byte for byte, exactly as it was before
//! segmentation existed (`cli.rs`, mesa task 936). That is not a stylistic
//! choice. It was not the original design (task 968), and it was not a
//! preference — two independent measurements ruled out trimming in both
//! directions, and re-introducing it as an "optimization" without
//! re-running both sweeps would silently reopen both bugs:
//!
//! 1. **Trimming corrupts real speech.** The first implementation had
//!    `speech_only` return only the detected speech spans, concatenated.
//!    Measured against `tests/fixtures/mesa-names.wav` and all 40 real room
//!    recordings in `bench/results/acoustic-wav/`, it broke real speech:
//!    even a single contiguous span with **95.5%** of the marginal fixture
//!    (`bench/fixtures/u02-16h38-marginal.wav`) surviving as one unbroken
//!    slice — no splicing anywhere — was enough to turn "mesa at a note to
//!    khora that headless mode is the default now." into "Nessa had a note
//!    to Cora that headless mode is the default now." Trimming 166 ms off
//!    the front did that, and on `mesa-names.wav` the same trimming dropped
//!    "qorvex" from the hotword-biased transcript entirely. Across the
//!    40-file sweep, 35 of 40 transcripts changed from what `--no-vad`
//!    produces. Padding the span edges before slicing (tried as two more
//!    candidate strategies, envelope and pad-merge) only shrank the damage,
//!    never removed it — the edge itself is where Parakeet's acoustic
//!    context is load-bearing, and any strategy that decides where that
//!    edge falls pays for it.
//! 2. **Trimming turns a false positive into a *different* false transcript
//!    instead of leaving it alone.** `tests/fixtures/nonspeech-transient.wav`
//!    (a synthetic noise burst) makes Silero false-positive a confident
//!    ~0.55 s "speech" span at essentially any threshold, so the segmenter
//!    finds a segment in it and the recognizer runs. That part is a genuine,
//!    unclosed gap: unbiased, the full clip happens to decode to nothing,
//!    but with hotword biasing — mesa's actual production configuration —
//!    the same untouched clip decodes to a ~40-word hallucinated string of
//!    boosted vocabulary terms. That second half is no longer a leak,
//!    though: a post-decode manufactured-vocabulary guard (mesa task 970,
//!    `vocabulary::looks_manufactured`, README "Exit codes",
//!    `docs/vocabulary.md`) now re-decodes this clip unbiased whenever the
//!    biased transcript looks like a run of vocabulary terms, finds
//!    nothing, and discards the hallucination before it reaches stdout —
//!    so the leak this comment used to describe as open is closed, even
//!    though the Silero false positive that lets the recognizer run on
//!    this clip at all is not. Neither of those is new: `--no-vad`
//!    produces the identical output either way (confirmed by running both
//!    through the real binary, task 968), because this fixture was never
//!    speech-gated before the VAD existed either — `audio::is_silent`
//!    passes it (well above `SILENCE_RMS_THRESHOLD`), so it always reached
//!    the recognizer. So of the four non-speech fixtures, three
//!    (`nonspeech-white.wav`, `nonspeech-rumble.wav`, `silence.wav`) are
//!    correctly rejected; this one is not, and no threshold from 0.5 down
//!    to 0.05 separates it from real speech (task 968's sweep). What
//!    trimming would have added on top is a *second*, different bug: an
//!    earlier design that narrowed the audio down to just the misdetected
//!    span turned the same clip's decode into a hallucinated filler word
//!    ("Uh.") even in the unbiased case where the untouched full clip
//!    decodes to nothing — i.e. trimming can manufacture a false positive
//!    transcript where none existed before, on top of the gap the VAD gate
//!    itself does not close.
//!
//! So the recognizer is never handed a Silero-cut span. On a
//! single-utterance recording it sees the *original*, complete `samples`;
//! on a stream of several it sees a partition of that same buffer cut at
//! utterance *starts*, so every sample is decoded exactly once and no
//! utterance loses the acoustic context on either side of it that the
//! measurements above showed is load-bearing. The only thing the gate can
//! do is refuse to run the recognizer on audio with no speech in it at
//! all — exactly the
//! fan/cough/TTS-playback case task 968 exists for, and exactly what it
//! delivers for the three noise fixtures it does close. One consequence
//! worth keeping in mind: since the gate cannot make a clip's transcript
//! any worse than `--no-vad` already produces (it either blocks the
//! recognizer outright or hands it the same bytes), a false positive here
//! is at worst a wasted decode of a pre-existing problem, never a new one —
//! which is why [`VadConfig::default`]'s threshold can afford to be
//! permissive.
//!
//! Construction is separated from the decision the same way
//! [`crate::engine::Recognizer`] separates `load` from `decode`:
//! [`Vad::load`] pays the one-time model-load cost, and [`Vad::segmenter`]
//! is the cheap, repeatable, `&self` call that starts one fresh pass over
//! one stream.

use std::path::PathBuf;

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

/// Samples are fed to the detector in chunks of exactly this size — Silero's
/// own frame size, and the value `docs/latency.md` leaves to this task.
const WINDOW_SIZE: i32 = 512;

/// Ring-buffer capacity `VoiceActivityDetector::create` reserves internally,
/// in seconds of audio. Generous headroom over `max_speech_duration` (8 s,
/// fixed by `docs/latency.md`'s decode-budget arithmetic): the detector can
/// have a full segment queued awaiting drain while still accepting new
/// waveform, so the buffer must outlive one whole segment with margin, not
/// just match it.
const BUFFER_SECONDS: f32 = 30.0;

/// Longest single speech span the detector will emit before force-closing
/// it, in seconds. Fixed by `docs/latency.md`'s decode-budget arithmetic: at
/// the 2x-safety-factor RTF of 0.164, 8 s of audio decodes in 1312 ms,
/// inside the 1500 ms window a segment has to finish in — not a value this
/// task is free to pick. Irrelevant to the accept/reject decision itself
/// (a long utterance still counts as "has speech" however many segments it
/// is force-split into); kept because [`SileroVadModelConfig`] has no
/// "don't force-split" setting, and this is the value the budget already
/// derived.
const MAX_SPEECH_SECONDS: f32 = 8.0;

/// Default seconds of trailing silence the detector waits before closing
/// an already-open speech span — Silero's own stock value, and the one
/// `docs/latency.md` fixes for its backdating arithmetic (the decode window
/// is `live.auto-send-ms` minus this number).
///
/// Under task 968's accept/reject gate this was a private constant and
/// deliberately not a flag: it only affected *when* a span closed, never
/// *whether* one opened, so it could not change the accept/reject outcome
/// at all — swept from 0.05 s to 0.5 s against every file in
/// `bench/results/acoustic-wav/` plus the marginal/noise/silence fixtures
/// and found byte-identical at every value. Segmentation (mesa task 936)
/// makes it load-bearing for the first time: it is now exactly the rule for
/// where one `segment` line ends and the next begins, so it is exposed as
/// `--vad-min-silence`. Reconciling it with mesa's own `live.auto-send-ms`
/// is explicitly deferred (`docs/latency.md`, "What this does not decide");
/// 0.5 s is the interim value and the arithmetic in that document stands as
/// written.
pub const MIN_SILENCE_SECONDS: f32 = 0.5;

/// Construction settings for [`Vad::load`]. `sample_rate` and `num_threads`
/// are not exposed — auris only ever feeds [`crate::audio::TARGET_SAMPLE_RATE`]
/// audio, and one thread is enough for a model this small — matching how
/// [`crate::engine::EngineConfig`] only exposes what varies per run.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Path to `silero_vad.onnx` (`crate::model::vad_model_path`).
    pub model: PathBuf,
    /// Silero's speech-probability threshold.
    pub threshold: f32,
    /// Speech spans shorter than this are dropped.
    pub min_speech: f32,
    /// Trailing silence that closes an open span — see
    /// [`MIN_SILENCE_SECONDS`]. This is where one utterance ends and the
    /// next begins, so it is the one knob that changes how a stream is cut
    /// into `segment` lines.
    pub min_silence: f32,
}

impl Default for VadConfig {
    /// `threshold = 0.2` is not Silero's own stock default (0.5) — it is
    /// this task's own measurement (task 968): a threshold sweep from 0.5
    /// down to 0.05, against every file in `bench/results/acoustic-wav/`
    /// (40 real room recordings) plus `bench/fixtures/u02-16h38-marginal.wav`,
    /// `tests/fixtures/mesa-names.wav`, and `plain.wav` on the accept side,
    /// and `tests/fixtures/nonspeech-{white,rumble,transient}.wav` plus
    /// `silence.wav` on the reject side. 0.5 (and 0.4) missed one real
    /// clip outright (`u17.wav`, a genuinely quiet recording — max 30 ms
    /// window RMS 0.0070, only ~7x above `audio::SILENCE_RMS_THRESHOLD`,
    /// well below every other speech fixture in this repo); every value
    /// from 0.38 down to 0.10 accepted all 43 real clips and rejected
    /// white/rumble/silence cleanly, so 0.2 sits at the middle of that
    /// clean band rather than on an edge. `min_speech = 0.25` stays
    /// Silero's stock default: it did not change the outcome anywhere in
    /// the clean band, only at the 0.4 boundary this default no longer
    /// sits near.
    ///
    /// `nonspeech-transient.wav` (a synthetic 150 ms noise burst) was NOT
    /// separable from real speech at any threshold tested, down to 0.05 —
    /// Silero classifies a confident ~0.55 s speech span in it regardless.
    /// That is a known, accepted gap in this gate, not something a
    /// threshold choice can close.
    fn default() -> Self {
        VadConfig {
            model: PathBuf::new(),
            threshold: 0.2,
            min_speech: 0.25,
            min_silence: MIN_SILENCE_SECONDS,
        }
    }
}

/// Everything that can go wrong loading the VAD model.
#[derive(Debug)]
pub enum VadError {
    /// `model` does not exist.
    MissingModel(PathBuf),
    /// `VoiceActivityDetector::create` returned `None` despite a
    /// valid-looking config — a genuine engine failure, not a bad parameter.
    CreateFailed,
}

impl VadError {
    pub fn exit_code(&self) -> i32 {
        match self {
            VadError::MissingModel(_) => 2,
            VadError::CreateFailed => 1,
        }
    }
}

impl std::fmt::Display for VadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VadError::MissingModel(path) => {
                write!(f, "VAD model not found: {}", path.display())
            }
            VadError::CreateFailed => write!(f, "failed to create the voice activity detector"),
        }
    }
}

impl std::error::Error for VadError {}

/// A loaded Silero VAD detector.
pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    /// Loads the model and constructs the detector.
    pub fn load(cfg: &VadConfig) -> Result<Vad, VadError> {
        if !cfg.model.is_file() {
            return Err(VadError::MissingModel(cfg.model.clone()));
        }

        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(cfg.model.to_string_lossy().into_owned()),
                threshold: cfg.threshold,
                min_silence_duration: cfg.min_silence,
                min_speech_duration: cfg.min_speech,
                window_size: WINDOW_SIZE,
                max_speech_duration: MAX_SPEECH_SECONDS,
            },
            sample_rate: crate::audio::TARGET_SAMPLE_RATE as i32,
            num_threads: 1,
            ..Default::default()
        };

        let inner =
            VoiceActivityDetector::create(&config, BUFFER_SECONDS).ok_or(VadError::CreateFailed)?;
        Ok(Vad { inner })
    }

    /// Starts one fresh pass over one stream. Resets the detector, so
    /// nothing from a previous pass bleeds into this one — that is what
    /// makes a `&self` method mean what it looks like it means, rather than
    /// something every caller has to remember to do first.
    pub fn segmenter(&self) -> Segmenter<'_> {
        self.inner.reset();
        Segmenter {
            vad: self,
            pending: Vec::new(),
        }
    }
}

/// Where one closed utterance sits in the stream, in samples from the first
/// sample ever fed to the [`Segmenter`] that produced it. Offsets only: see
/// the module doc comment for why this deliberately does not carry the
/// audio Silero cut out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// First sample of the detected speech.
    pub start: usize,
    /// Length of the detected speech, in samples.
    pub len: usize,
}

/// One pass of the detector over one stream, fed incrementally.
///
/// Silero wants exactly [`WINDOW_SIZE`] samples at a time and the audio
/// arrives in whatever sizes the source produces, so this holds back the
/// remainder between calls rather than letting a short tail chunk shift the
/// window alignment — which would move every boundary after it.
pub struct Segmenter<'a> {
    vad: &'a Vad,
    pending: Vec<f32>,
}

impl Segmenter<'_> {
    /// Feeds the next arriving samples. Cheap and incremental: only whole
    /// windows are handed to the detector, the rest waits here for the next
    /// call.
    pub fn accept(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        let window = WINDOW_SIZE as usize;
        let whole = self.pending.len() - self.pending.len() % window;
        for chunk in self.pending[..whole].chunks(window) {
            self.vad.inner.accept_waveform(chunk);
        }
        self.pending.drain(..whole);
    }

    /// End of stream: hands over the held-back remainder and forces the
    /// detector to close whatever span is still open, so a recording that
    /// ends mid-utterance still produces that utterance rather than losing
    /// it.
    pub fn finish(&mut self) {
        if !self.pending.is_empty() {
            self.vad.inner.accept_waveform(&self.pending);
            self.pending.clear();
        }
        self.vad.inner.flush();
    }

    /// Pops the next closed utterance, or `None` when none has closed yet.
    /// Draining matters even when a caller does nothing with the result:
    /// the detector's internal queue is finite.
    pub fn next_segment(&mut self) -> Option<Segment> {
        let segment = self.vad.inner.front()?;
        let found = Segment {
            start: segment.start().max(0) as usize,
            len: segment.n().max(0) as usize,
        };
        drop(segment);
        self.vad.inner.pop();
        Some(found)
    }

    /// True while the detector believes speech is happening right now —
    /// sherpa-onnx's own `detected()`, computed by the pass already running
    /// and costing no decode of any kind. This is the `speech` heartbeat
    /// `docs/streaming.md` supplies in place of the partial hypotheses it
    /// rejects.
    pub fn speaking(&self) -> bool {
        self.vad.inner.detected()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Locates the Silero VAD model, honouring `AURIS_TEST_VAD_MODEL` — same
    /// convention as `engine::tests::spike_model_dir`: tests that need it
    /// skip (not fail) when it's absent, since CI has no model checked in.
    fn vad_model_path() -> Option<PathBuf> {
        let path = std::env::var("AURIS_TEST_VAD_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home_env = std::env::var("HOME").ok();
                let home = crate::cli::auris_home(home_env.as_deref());
                crate::model::vad_model_path(&home)
            });
        if path.is_file() {
            Some(path)
        } else {
            eprintln!(
                "skip: no VAD model at {} (set AURIS_TEST_VAD_MODEL to override)",
                path.display()
            );
            None
        }
    }

    fn fixture_wav(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Vec<f32> {
        let path = fixture_wav(name);
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
        crate::audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
    }

    /// The whole-buffer form of [`Vad::segmenter`], for tests that have a
    /// complete fixture in hand rather than a live stream. Deliberately not
    /// on `Vad` itself: the transcribe path never has a complete buffer to
    /// start with, so an API shaped for one would be a second, unexercised
    /// path through the detector.
    fn segments_of(vad: &Vad, samples: &[f32]) -> Vec<Segment> {
        let mut segmenter = vad.segmenter();
        segmenter.accept(samples);
        segmenter.finish();
        let mut found = Vec::new();
        while let Some(segment) = segmenter.next_segment() {
            found.push(segment);
        }
        found
    }

    #[test]
    fn config_defaults_match_the_contract() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.threshold, 0.2);
        assert_eq!(cfg.min_speech, 0.25);
        assert_eq!(cfg.min_silence, 0.5);
    }

    #[test]
    fn missing_model_is_a_usage_error_not_a_panic() {
        let cfg = VadConfig {
            model: PathBuf::from("/nonexistent/auris-vad-test-model.onnx"),
            ..Default::default()
        };
        let err = match Vad::load(&cfg) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert_eq!(err.exit_code(), 2);
        assert!(matches!(err, VadError::MissingModel(_)));
    }

    #[test]
    fn broadband_noise_has_no_speech() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        // A speech-like-RMS but structureless signal: a fixed-seed
        // pseudo-random sequence, not a tone — Silero classifies it as
        // non-speech because it carries none of speech's spectral structure.
        let mut state: u32 = 0x2545F491;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let samples: Vec<f32> = (0..crate::audio::TARGET_SAMPLE_RATE * 3)
            .map(|_| next() * 0.1)
            .collect();

        assert!(
            segments_of(&vad, &samples).is_empty(),
            "expected no speech in broadband noise"
        );
    }

    #[test]
    fn real_speech_is_detected() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        let samples = decode_fixture("mesa-names.wav");
        assert!(
            !segments_of(&vad, &samples).is_empty(),
            "expected speech in mesa-names.wav"
        );
    }

    /// [`Vad::segmenter`] takes `&self` and must behave as a pure,
    /// repeatable check — nothing from one pass may bleed into the next.
    /// Without the `reset()` in `segmenter`, the detector's residual state
    /// (and the final segmentation `finish()` forces) would carry over and
    /// this would fail.
    #[test]
    fn segmentation_is_idempotent_across_repeated_passes() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        let samples = decode_fixture("mesa-names.wav");
        let first = segments_of(&vad, &samples);
        let second = segments_of(&vad, &samples);
        assert_eq!(
            first, second,
            "segmenter produced different boundaries on a second pass over the same input"
        );
        assert!(!first.is_empty());
    }

    /// Locates the spike Parakeet model, same lookup as
    /// `engine::tests::spike_model_dir` — needed here to decode the marginal
    /// fixture below and check the VAD gate against a real, quiet room
    /// recording rather than only synthetic noise/speech.
    fn spike_model_dir() -> Option<PathBuf> {
        let dir = std::env::var("AURIS_TEST_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/models/parakeet")
            });
        if dir.join("encoder.int8.onnx").is_file() {
            Some(dir)
        } else {
            eprintln!(
                "skip: no model at {} (set AURIS_TEST_MODEL_DIR to override)",
                dir.display()
            );
            None
        }
    }

    /// Symlinks the spike model into a temp dir with `bpe.vocab`, same as
    /// `engine::tests::symlinked_model_dir`.
    fn symlinked_model_dir(real_dir: &Path) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        for name in [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ] {
            std::os::unix::fs::symlink(real_dir.join(name), tmp.path().join(name))
                .unwrap_or_else(|e| panic!("symlink {name}: {e}"));
        }
        std::os::unix::fs::symlink(
            real_dir.join("bpe_synth.vocab"),
            tmp.path().join("bpe.vocab"),
        )
        .expect("symlink bpe.vocab");
        tmp
    }

    /// `nonspeech-white.wav` and `nonspeech-rumble.wav` (bench/fixtures/README.md)
    /// are loud enough to clear `audio::is_silent` but must still be turned
    /// away here — this is the actual defect task 968 exists to fix (a fan,
    /// a chair scrape, background hiss producing a hallucinated transcript).
    #[test]
    fn real_noise_fixtures_have_no_speech() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        for name in ["nonspeech-white.wav", "nonspeech-rumble.wav"] {
            let samples = decode_fixture(name);
            assert!(
                segments_of(&vad, &samples).is_empty(),
                "expected no speech in {name}"
            );
        }
    }

    /// The case that beat every trimming strategy tried (module doc
    /// comment, point 2), pinned as a known gap rather than as a pass.
    ///
    /// Silero reports a confident ~0.55 s speech span in
    /// `nonspeech-transient.wav` at every threshold tested, from 0.5 down to
    /// 0.05, so the segmenter finds a segment here and no threshold change
    /// makes it stop. Of the four non-speech fixtures, three are rejected by the gate
    /// and this one is not (README "Exit codes").
    ///
    /// The gate still cannot corrupt this clip: a [`Segment`] carries
    /// offsets rather than a buffer, and one segment in one clip makes the
    /// slice `cli.rs` decodes the whole buffer, so a VAD-on run is
    /// byte-identical to `--no-vad`.
    /// That equality was confirmed against the real binary (task 968: 286
    /// bytes each way, hotword-biased) — it is deliberately NOT asserted
    /// here, because within this process both sides would be the same
    /// `decode_with_hotwords` call on the same slice, which is a tautology
    /// that can only fail if the recognizer is nondeterministic. The claim
    /// lives at the CLI level, so it is checked there, not faked here.
    ///
    /// What the recognizer then does with the whole clip is outside this
    /// gate's reach: unbiased it decodes to nothing and exits 1; **with
    /// hotwords it still produces a long hallucinated string**, which used
    /// to be a real, user-visible leak under mesa's actual production
    /// configuration. That string no longer reaches stdout — the
    /// manufactured-vocabulary guard (mesa task 970,
    /// `vocabulary::looks_manufactured`, README "Exit codes") re-decodes the
    /// clip unbiased, finds nothing, and discards it — but the discarding
    /// happens at the CLI level, downstream of everything this gate does, so
    /// nothing about *this* test's subject changed: a segment is still
    /// found here and the recognizer still runs.
    ///
    /// This asserts only the detection, deliberately. Asserting the decode is
    /// empty would be asserting the CLI guard's behaviour from inside the VAD
    /// module, where the guard does not run; asserting the hallucinated text
    /// would enshrine a defect as expected behaviour and rot the moment the
    /// string drifted. Both claims belong at the CLI level, and
    /// `tests/fixtures.rs`'s `manufactured_vocabulary_transcript_yields_no_transcript`
    /// is where the first one is now actually checked.
    #[test]
    fn transient_noise_is_a_known_silero_false_positive() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        let samples = decode_fixture("nonspeech-transient.wav");
        assert!(
            !segments_of(&vad, &samples).is_empty(),
            "this test documents a known Silero false positive on this fixture; \
             if this now fails, Silero (or the threshold) changed for the better \
             and the module doc comment's point 2 needs re-checking, not deleting"
        );
    }

    /// Regression coverage for the hardest real-speech case in the repo:
    /// `bench/fixtures/u02-16h38-marginal.wav` is a genuine quiet room
    /// recording that decodes correctly today, with vocabulary biasing, to
    /// "mesa at a note to khora that headless mode is the default now." An
    /// earlier VAD design that trimmed the audio broke this (see the module
    /// doc comment); under the decision-not-filter design the recognizer
    /// always sees the untouched original samples once the gate says
    /// "speech present", so this must match the `--no-vad` baseline exactly.
    #[test]
    fn marginal_room_recording_survives_the_vad_gate() {
        let Some(vad_model) = vad_model_path() else {
            return;
        };
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let hotwords_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/fixtures/hotwords.txt");
        assert!(hotwords_path.is_file(), "fixture hotwords file missing");
        let hotwords = std::fs::read_to_string(&hotwords_path)
            .expect("read fixture hotwords")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("/");

        let marginal_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/fixtures/u02-16h38-marginal.wav");
        let file = std::fs::File::open(&marginal_path)
            .unwrap_or_else(|e| panic!("open {}: {e}", marginal_path.display()));
        let samples = crate::audio::decode(file)
            .unwrap_or_else(|e| panic!("decode {}: {e}", marginal_path.display()));

        let vad = Vad::load(&VadConfig {
            model: vad_model,
            ..Default::default()
        })
        .expect("load vad");
        let found = segments_of(&vad, &samples);
        assert!(
            !found.is_empty(),
            "VAD rejected the marginal fixture outright — real speech was lost"
        );
        assert_eq!(
            found.len(),
            1,
            "one utterance must stay one segment, or the slice the recognizer sees below \
             is no longer the whole buffer this assertion is the baseline for"
        );

        let recognizer_cfg = crate::engine::EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = crate::engine::Recognizer::load(&recognizer_cfg).expect("load recognizer");
        // The untouched original samples, exactly as `--no-vad` would decode
        // them — the whole point of the decision-not-filter design.
        let text = recognizer
            .decode_with_hotwords(&samples, &hotwords)
            .expect("decode marginal fixture");
        eprintln!("marginal fixture decoded (post-VAD-decision): {text:?}");

        assert_eq!(
            text.trim(),
            "mesa at a note to khora that headless mode is the default now.",
            "VAD-gated decode of the marginal fixture no longer matches the --no-vad baseline"
        );
    }
}
