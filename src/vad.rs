//! Asks Silero VAD whether already-16kHz-mono-f32 samples contain any
//! speech at all (README's `--vad-*` flags, `docs/streaming.md`). This
//! module knows nothing about the CLI or the daemon; [`Vad::has_speech`] is
//! the one entry point `cli.rs::run_transcribe` calls, in the same spot
//! [`crate::audio::is_silent`] already runs and immediately after it —
//! `is_silent` stays as a free, model-free short-circuit ahead of this
//! heavier gate (CLAUDE.md).
//!
//! **The VAD is a decision, not a filter.** [`Vad::has_speech`] returns a
//! `bool`; it never trims, crops, or otherwise modifies the audio the
//! recognizer sees. This was not the original design (task 968), and it was
//! not a preference — two independent measurements ruled out trimming in
//! both directions, and re-introducing it as an "optimization" without
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
//!    ~0.55 s "speech" span at essentially any threshold, so `has_speech`
//!    is `true` for it and the recognizer runs. That part is a genuine,
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
//! So the recognizer now always sees the *original*, complete `samples`
//! whenever any speech is found at all; the only thing the gate can do is
//! refuse to run the recognizer on audio with no speech in it — exactly the
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
//! [`Vad::load`] pays the one-time model-load cost, and [`Vad::has_speech`]
//! is the cheap, repeatable, `&self` call.

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

/// Seconds of trailing silence the detector waits before closing an
/// already-open speech span. Not exposed on [`VadConfig`]: under the
/// decision-not-filter design ([`Vad::has_speech`]) this only affects
/// *when* a span closes, never *whether* one opens in the first place, so
/// it cannot change the accept/reject decision at all — measured by
/// sweeping it from 0.05 s to 0.5 s against every file in
/// `bench/results/acoustic-wav/` plus the marginal/noise/silence fixtures
/// (task 968) and finding the accept/reject outcome byte-identical at
/// every value tested. `docs/latency.md` fixes this exact value (0.5 s,
/// stock Silero) for its own backdating arithmetic, which is reason enough
/// to keep it fixed here rather than pick a different constant.
const MIN_SILENCE_SECONDS: f32 = 0.5;

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
                min_silence_duration: MIN_SILENCE_SECONDS,
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

    /// True if Silero finds at least one speech span anywhere in `samples`.
    /// This is a decision, not a filter (see the module doc comment for
    /// why): `samples` itself is never modified, trimmed, or returned —
    /// the caller hands the recognizer the same, complete buffer it always
    /// did, and only skips that call entirely when this returns `false`.
    ///
    /// Feeds `samples` to the detector in [`WINDOW_SIZE`] chunks and stops
    /// as soon as a span is found — no need to keep feeding once the
    /// decision is already yes. Draining/`pop`-ing found segments still
    /// matters even though their contents are discarded, because the
    /// detector's internal queue is finite.
    ///
    /// The detector carries state between calls (and `flush()` below forces
    /// a final segmentation), so this resets it first — that is what makes
    /// repeated calls on the same `Vad` independent, which is what a
    /// `&self` method promises rather than something a caller has to
    /// remember.
    pub fn has_speech(&self, samples: &[f32]) -> bool {
        self.inner.reset();

        for chunk in samples.chunks(WINDOW_SIZE as usize) {
            self.inner.accept_waveform(chunk);
            if !self.inner.is_empty() {
                return true;
            }
        }
        self.inner.flush();
        !self.inner.is_empty()
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

    #[test]
    fn config_defaults_match_the_contract() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.threshold, 0.2);
        assert_eq!(cfg.min_speech, 0.25);
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
            !vad.has_speech(&samples),
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
            vad.has_speech(&samples),
            "expected speech in mesa-names.wav"
        );
    }

    /// `has_speech` takes `&self` and must behave as a pure, repeatable
    /// check — nothing from one call may bleed into the next. Without the
    /// `reset()` at the top of `has_speech`, the detector's residual state
    /// (and the final segmentation `flush()` forces) would carry over and
    /// this would fail.
    #[test]
    fn has_speech_is_idempotent_across_repeated_calls() {
        let Some(model) = vad_model_path() else {
            return;
        };
        let vad = Vad::load(&VadConfig {
            model,
            ..Default::default()
        })
        .expect("load");

        let samples = decode_fixture("mesa-names.wav");
        let first = vad.has_speech(&samples);
        let second = vad.has_speech(&samples);
        assert_eq!(
            first, second,
            "has_speech produced different answers on a second call with the same input"
        );
        assert!(first);
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
            assert!(!vad.has_speech(&samples), "expected no speech in {name}");
        }
    }

    /// The case that beat every trimming strategy tried (module doc
    /// comment, point 2), pinned as a known gap rather than as a pass.
    ///
    /// Silero reports a confident ~0.55 s speech span in
    /// `nonspeech-transient.wav` at every threshold tested, from 0.5 down to
    /// 0.05, so `has_speech` is `true` here and no threshold change makes it
    /// false. Of the four non-speech fixtures, three are rejected by the gate
    /// and this one is not (README "Exit codes").
    ///
    /// The gate still cannot corrupt this clip: `has_speech` returns a `bool`
    /// rather than a buffer, so the recognizer is handed the untouched
    /// original samples and a VAD-on run is byte-identical to `--no-vad`.
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
    /// nothing about *this* test's subject changed: `has_speech` is still
    /// `true` here and the recognizer still runs.
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
            vad.has_speech(&samples),
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
        assert!(
            vad.has_speech(&samples),
            "VAD rejected the marginal fixture outright — real speech was lost"
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
