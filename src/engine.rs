//! Binds sherpa-onnx's `OfflineRecognizer` to a Parakeet TDT model and holds
//! it warm for the process lifetime (`docs/engine.md`, README "The daemon").
//! This module knows nothing about audio decoding or the CLI; it only turns
//! already-16kHz-mono-f32 samples ([`crate::audio`]) into a transcript.
//!
//! Construction is deliberately separated from decoding: [`Recognizer::load`]
//! pays the ~4 s model-load cost once, and [`Recognizer::decode`] /
//! [`Recognizer::decode_with_hotwords`] are cheap, repeatable, and take
//! `&self` so one recognizer can serve many requests — the daemon
//! (`src/daemon.rs`, `docs/daemon.md`) shares this one instance across
//! calls.

use std::path::{Path, PathBuf};

use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
};

/// The three ONNX graphs plus the token table that make up a Parakeet
/// transducer model directory (README "The cache"). `bpe.vocab` is checked
/// separately below, but is just as required: `modeling_unit`/`bpe_vocab`
/// are set unconditionally at construction (docs/vocabulary.md — the
/// primary vocabulary path is per-request `create_stream_with_hotwords`,
/// never `hotwords_file`, so the tokenizer must be configured for hotwords
/// regardless of whether this particular `EngineConfig` sets a
/// `hotwords_file`).
const REQUIRED_MODEL_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// `modeling_unit`/`bpe_vocab` require this exact filename inside
/// `model_dir` (docs/engine.md, README "`bpe.vocab`").
const BPE_VOCAB_FILENAME: &str = "bpe.vocab";

/// Construction settings for [`Recognizer::load`]. Everything the model
/// *type* fixes (transducer family, `nemo_transducer`, `modified_beam_search`,
/// 16 kHz/80-dim features) is not exposed here — those are not choices
/// (docs/engine.md), only what varies per daemon start is.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Directory holding `encoder.int8.onnx`, `decoder.int8.onnx`,
    /// `joiner.int8.onnx`, `tokens.txt`, and `bpe.vocab` — all five,
    /// unconditionally (see [`EngineError::MissingBpeVocab`]).
    pub model_dir: PathBuf,
    /// ONNX Runtime intra-op thread count.
    pub threads: i32,
    /// A construction-time hotword list, for tests and the `--no-daemon`
    /// one-shot path. The primary vocabulary path (docs/vocabulary.md) is
    /// per-request, via [`Recognizer::decode_with_hotwords`], which needs
    /// no entry here at all.
    pub hotwords_file: Option<PathBuf>,
    /// The single global boost `OfflineRecognizerConfig` takes, fixed at
    /// construction regardless of per-word terms — not exposed as a CLI
    /// choice. 3.0 is not sherpa-onnx's own default (1.5): docs/vocabulary.md
    /// measured 1.5 as below the gate that lets a per-word boost reach the
    /// beam at all, and 3.0 as the value that reproduces
    /// `spike/results/hotwords/tuned.tsv` byte-for-byte — the 83.3% name-F1
    /// / 4.0% WER headline in docs/engine.md.
    pub hotwords_score: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            model_dir: PathBuf::new(),
            threads: 8,
            hotwords_file: None,
            hotwords_score: 3.0,
        }
    }
}

/// Everything that can go wrong loading or running a recognizer.
/// `exit_code` mirrors [`crate::audio::AudioError`]'s split: a parameter
/// that was never valid on its own terms (a missing model directory or
/// file, a missing `bpe.vocab`, an unreadable hotwords file) is USAGE (2);
/// a failure inside the engine itself, despite valid-looking inputs, means
/// no transcript came out and is NOTHING_TRANSCRIBED (1) — the same
/// "no transcript" signal `AudioError` uses for everything that isn't a bad
/// parameter.
#[derive(Debug)]
pub enum EngineError {
    /// `model_dir` does not exist or is not a directory.
    MissingModelDir(PathBuf),
    /// One of the required model files is missing from `model_dir`.
    MissingModelFile(PathBuf),
    /// `hotwords_file` was given but could not be read.
    HotwordsFileUnreadable(PathBuf, std::io::Error),
    /// `<model_dir>/bpe.vocab` is absent. Required unconditionally, not only
    /// when `hotwords_file` is set: the primary vocabulary path is
    /// per-request hotwords (docs/vocabulary.md), which needs the tokenizer
    /// configured at construction regardless — a model directory without it
    /// is not a complete model (README "The cache").
    MissingBpeVocab(PathBuf),
    /// `OfflineRecognizer::create` returned `None` despite valid-looking
    /// config — a genuine engine failure, not a bad parameter.
    CreateFailed,
    /// A stream produced no decodable result.
    DecodeFailed,
}

impl EngineError {
    pub fn exit_code(&self) -> i32 {
        match self {
            EngineError::MissingModelDir(_)
            | EngineError::MissingModelFile(_)
            | EngineError::HotwordsFileUnreadable(_, _)
            | EngineError::MissingBpeVocab(_) => 2,
            EngineError::CreateFailed | EngineError::DecodeFailed => 1,
        }
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::MissingModelDir(dir) => {
                write!(f, "model directory not found: {}", dir.display())
            }
            EngineError::MissingModelFile(path) => {
                write!(f, "model file missing: {}", path.display())
            }
            EngineError::HotwordsFileUnreadable(path, e) => {
                write!(f, "failed to read hotwords file {}: {e}", path.display())
            }
            EngineError::MissingBpeVocab(path) => write!(
                f,
                "hotwords require a bpe vocab at {} but it is missing",
                path.display()
            ),
            EngineError::CreateFailed => write!(f, "failed to create the offline recognizer"),
            EngineError::DecodeFailed => write!(f, "decode produced no result"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Checks `model_dir`, including `bpe.vocab` (required unconditionally —
/// see [`EngineError::MissingBpeVocab`]), and the hotwords file if one was
/// given. Separated from [`Recognizer::load`] so construction fails fast on
/// a bad parameter before any ONNX Runtime session is touched.
fn validate_model_dir(cfg: &EngineConfig) -> Result<(), EngineError> {
    if !cfg.model_dir.is_dir() {
        return Err(EngineError::MissingModelDir(cfg.model_dir.clone()));
    }
    for file in REQUIRED_MODEL_FILES {
        let path = cfg.model_dir.join(file);
        if !path.is_file() {
            return Err(EngineError::MissingModelFile(path));
        }
    }
    let bpe_vocab = cfg.model_dir.join(BPE_VOCAB_FILENAME);
    if !bpe_vocab.is_file() {
        return Err(EngineError::MissingBpeVocab(bpe_vocab));
    }
    if let Some(hotwords_file) = &cfg.hotwords_file {
        std::fs::metadata(hotwords_file)
            .map_err(|e| EngineError::HotwordsFileUnreadable(hotwords_file.clone(), e))?;
    }
    Ok(())
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// A loaded Parakeet recognizer, held warm for the process lifetime.
/// `Send + Sync` (inherited from `sherpa_onnx::OfflineRecognizer`) so the
/// daemon can share one instance across concurrent requests.
pub struct Recognizer {
    inner: OfflineRecognizer,
}

impl Recognizer {
    /// Loads the model and constructs the recognizer. Pays the ~4 s cold
    /// start (docs/engine.md); everything after this is a cheap decode.
    pub fn load(cfg: &EngineConfig) -> Result<Self, EngineError> {
        validate_model_dir(cfg)?;

        // modeling_unit/bpe_vocab are set unconditionally, not only when
        // `hotwords_file` is set: the primary vocabulary path is per-request
        // hotwords via `decode_with_hotwords` (docs/vocabulary.md), which
        // never touches `hotwords_file` at all, so the tokenizer must
        // already be configured for a term to bias correctly instead of
        // silently miscalibrating (docs/engine.md).
        let model_config = OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: Some(path_str(&cfg.model_dir.join("encoder.int8.onnx"))),
                decoder: Some(path_str(&cfg.model_dir.join("decoder.int8.onnx"))),
                joiner: Some(path_str(&cfg.model_dir.join("joiner.int8.onnx"))),
            },
            tokens: Some(path_str(&cfg.model_dir.join("tokens.txt"))),
            num_threads: cfg.threads,
            debug: false,
            model_type: Some("nemo_transducer".to_string()),
            modeling_unit: Some("bpe".to_string()),
            bpe_vocab: Some(path_str(&cfg.model_dir.join(BPE_VOCAB_FILENAME))),
            ..Default::default()
        };

        let hotwords_file = cfg.hotwords_file.as_ref().map(|p| path_str(p));

        let config = OfflineRecognizerConfig {
            model_config,
            decoding_method: Some("modified_beam_search".to_string()),
            hotwords_file,
            hotwords_score: cfg.hotwords_score,
            ..Default::default()
        };

        let inner = OfflineRecognizer::create(&config).ok_or(EngineError::CreateFailed)?;
        Ok(Recognizer { inner })
    }

    /// Decodes one utterance of 16 kHz mono `f32` samples with whatever
    /// hotwords were baked in at construction (if any).
    pub fn decode(&self, samples: &[f32]) -> Result<String, EngineError> {
        let stream = self.inner.create_stream();
        stream.accept_waveform(crate::audio::TARGET_SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .ok_or(EngineError::DecodeFailed)
    }

    /// Decodes one utterance against a per-request hotword string — the
    /// primary vocabulary path (docs/vocabulary.md). `hotwords` is
    /// `term :boost` entries joined with `/`, the same syntax and parser
    /// `hotwords_file` uses (`/` is replaced with newline internally), so a
    /// changed term list costs nothing and never rebuilds this recognizer:
    /// the `per_stream_hotwords_match_hotwords_file_byte_for_byte` test
    /// below confirms per-word boosts delivered this way are byte-identical
    /// to the same boosts delivered via `hotwords_file`. The one thing
    /// genuinely fixed at construction is the global `hotwords_score`, plus
    /// `modeling_unit`/`bpe_vocab`.
    pub fn decode_with_hotwords(
        &self,
        samples: &[f32],
        hotwords: &str,
    ) -> Result<String, EngineError> {
        let stream = self.inner.create_stream_with_hotwords(hotwords);
        stream.accept_waveform(crate::audio::TARGET_SAMPLE_RATE as i32, samples);
        self.inner.decode(&stream);
        stream
            .get_result()
            .map(|r| r.text)
            .ok_or(EngineError::DecodeFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Locates the spike Parakeet model, honouring `AURIS_TEST_MODEL_DIR`.
    /// Tests that need it skip (not fail) when it's absent — CI has no
    /// model checked in.
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

    /// Builds a temp model dir of symlinks pointing at the real spike
    /// model, with `bpe_synth.vocab` linked in as `bpe.vocab` — the engine
    /// expects that exact filename, and the fixture data must not be
    /// copied (631 MB) or renamed in place (spike/ stays untouched).
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
            tmp.path().join(BPE_VOCAB_FILENAME),
        )
        .expect("symlink bpe.vocab");
        tmp
    }

    fn fixture_wav(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench/fixtures/wav")
            .join(name)
    }

    fn decode_fixture(name: &str) -> Vec<f32> {
        let path = fixture_wav(name);
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
        crate::audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
    }

    #[test]
    fn loads_and_decodes_u01() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };

        let start = Instant::now();
        let recognizer = Recognizer::load(&cfg).expect("load");
        eprintln!("cold load: {:?}", start.elapsed());

        let samples = decode_fixture("u01.wav");
        let audio_secs = samples.len() as f64 / crate::audio::TARGET_SAMPLE_RATE as f64;
        let start = Instant::now();
        let text = recognizer.decode(&samples).expect("decode");
        let elapsed = start.elapsed();
        eprintln!(
            "warm decode: {elapsed:?} for {audio_secs:.2}s audio, RTF={:.3}",
            elapsed.as_secs_f64() / audio_secs
        );

        let lower = text.to_lowercase();
        for word in ["task", "924", "auris", "parakeet"] {
            assert!(lower.contains(word), "expected {word:?} in {text:?}");
        }
    }

    #[test]
    fn same_recognizer_decodes_two_fixtures_in_sequence() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = Recognizer::load(&cfg).expect("load");

        let u01 = recognizer
            .decode(&decode_fixture("u01.wav"))
            .expect("decode u01");
        let u04 = recognizer
            .decode(&decode_fixture("u04.wav"))
            .expect("decode u04");

        assert!(u01.to_lowercase().contains("task"));
        assert!(u04.to_lowercase().contains("index"));
    }

    #[test]
    fn missing_model_dir_is_a_usage_error_not_a_panic() {
        let cfg = EngineConfig {
            model_dir: PathBuf::from("/nonexistent/auris-engine-test-model"),
            ..Default::default()
        };
        let err = match Recognizer::load(&cfg) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert_eq!(err.exit_code(), 2);
        assert!(matches!(err, EngineError::MissingModelDir(_)));
    }

    #[test]
    fn incomplete_model_dir_is_a_usage_error() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        // Link only the encoder — decoder/joiner/tokens are missing.
        std::os::unix::fs::symlink(
            model_dir.join("encoder.int8.onnx"),
            tmp.path().join("encoder.int8.onnx"),
        )
        .expect("symlink");
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let err = match Recognizer::load(&cfg) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert_eq!(err.exit_code(), 2);
        assert!(matches!(err, EngineError::MissingModelFile(_)));
    }

    #[test]
    fn hotwords_change_the_transcript() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let hotwords_file =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/fixtures/hotwords.txt");
        assert!(hotwords_file.is_file(), "fixture hotwords file missing");

        let with_hotwords_cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            hotwords_file: Some(hotwords_file),
            ..Default::default()
        };
        let with_hotwords = Recognizer::load(&with_hotwords_cfg).expect("load with hotwords");

        let without_hotwords_cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let without_hotwords = Recognizer::load(&without_hotwords_cfg).expect("load plain");

        let samples = decode_fixture("u07.wav");
        let biased = with_hotwords.decode(&samples).expect("decode biased");
        let plain = without_hotwords.decode(&samples).expect("decode plain");
        eprintln!("plain:  {plain:?}");
        eprintln!("biased: {biased:?}");

        // docs/engine.md: no config, engine, or boost ever produces the
        // literal spelling "khora" — every parakeet run hears "qora". So the
        // live-biasing check is that the transcript changes, not that it
        // becomes correct.
        assert_ne!(
            biased.to_lowercase(),
            plain.to_lowercase(),
            "hotwords had no effect on the transcript"
        );
    }

    /// docs/vocabulary.md's central claim: a per-stream hotwords string and
    /// a `hotwords_file` carrying the same terms/scores produce
    /// byte-identical transcripts, because `create_stream_with_hotwords`
    /// replaces `/` with newline and feeds the same parser
    /// `hotwords_file` does. This is the primary vocabulary path
    /// (`decode_with_hotwords`, never `hotwords_file`, per that doc), so
    /// it's the one path that must actually work.
    #[test]
    fn per_stream_hotwords_match_hotwords_file_byte_for_byte() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let hotwords_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/fixtures/hotwords.txt");
        let hotwords_contents =
            std::fs::read_to_string(&hotwords_path).expect("read fixture hotwords file");
        // The per-stream separator is `/`; the file's separator is a
        // newline. Same terms, same scores, different delivery mechanism.
        let inline_hotwords = hotwords_contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("/");

        let via_file_cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            hotwords_file: Some(hotwords_path),
            ..Default::default()
        };
        let via_file = Recognizer::load(&via_file_cfg).expect("load via hotwords_file");

        let plain_cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let plain = Recognizer::load(&plain_cfg).expect("load plain");

        let samples = decode_fixture("u07.wav");
        let via_file_text = via_file.decode(&samples).expect("decode via hotwords_file");
        let via_stream_text = plain
            .decode_with_hotwords(&samples, &inline_hotwords)
            .expect("decode via per-stream hotwords");
        let plain_text = plain.decode(&samples).expect("decode plain");

        eprintln!("plain:      {plain_text:?}");
        eprintln!("via file:   {via_file_text:?}");
        eprintln!("via stream: {via_stream_text:?}");

        assert_eq!(
            via_file_text, via_stream_text,
            "per-stream hotwords and hotwords_file produced different transcripts \
             for the same terms — this contradicts docs/vocabulary.md"
        );
        assert_ne!(
            via_stream_text.to_lowercase(),
            plain_text.to_lowercase(),
            "per-stream hotwords had no effect on the transcript"
        );
    }

    /// Task 951's decode-only warm RTF, measured the way `docs/spike-results.md`
    /// SS6/SS4 measured its 0.082 baseline: one `Recognizer::load`, then
    /// decode all 8 fixtures with the production per-word hotwords string,
    /// summing decode time only (no process spawn, no socket round-trip, no
    /// audio decode/resample) and dividing by the 45.429 s total fixture
    /// audio. This is the number directly comparable to that 0.082 — the
    /// end-to-end client/daemon RTF measured by
    /// `spike/harness/warm_daemon_bench.sh` is a different, larger number by
    /// design (it also pays process spawn and IPC), and is not what this
    /// test checks.
    /// `#[ignore]` on purpose: this reports a wall-clock number and asserts
    /// nothing, because a timing assertion here would be measuring the
    /// machine, not auris. Run under the default parallel `cargo test` —
    /// alongside the other tests that each load their own 1.5 GB
    /// recognizer — the same decode reads 0.4523 RTF instead of 0.0688,
    /// purely from CPU contention. The *invariant* (the load is paid once,
    /// later decodes are cheap) is asserted relatively, against the load
    /// time measured in the same process, by
    /// `daemon::tests::second_decode_is_fast_after_the_one_time_load`. This
    /// is the measurement instrument, run alone on a quiet machine:
    ///
    /// ```text
    /// cargo test --release decode_only_warm_rtf -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not an assertion: run alone on a quiet machine"]
    fn decode_only_warm_rtf_matches_spike_methodology() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = Recognizer::load(&cfg).expect("load");

        let hotwords_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/fixtures/hotwords.txt");
        let hotwords_contents =
            std::fs::read_to_string(&hotwords_path).expect("read fixture hotwords file");
        let inline_hotwords = hotwords_contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("/");

        let ids = ["u01", "u02", "u03", "u04", "u05", "u06", "u07", "u08"];
        let mut total_audio_secs = 0.0;
        let mut total_decode_secs = 0.0;
        for id in ids {
            let samples = decode_fixture(&format!("{id}.wav"));
            total_audio_secs += samples.len() as f64 / crate::audio::TARGET_SAMPLE_RATE as f64;
            let start = Instant::now();
            recognizer
                .decode_with_hotwords(&samples, &inline_hotwords)
                .unwrap_or_else(|e| panic!("decode {id}: {e}"));
            total_decode_secs += start.elapsed().as_secs_f64();
        }

        let rtf = total_decode_secs / total_audio_secs;
        eprintln!(
            "decode-only warm RTF: {rtf:.4} ({total_decode_secs:.3}s decode / \
             {total_audio_secs:.3}s audio; spike baseline: 0.082)"
        );
    }

    /// The daemon shares one `Recognizer` across connections, which
    /// requires it to be `Send + Sync` — though today's accept loop is
    /// sequential (`docs/daemon.md`), not actually concurrent. This is
    /// a compile-time check, not a runtime assertion — the crate already
    /// declares `OfflineRecognizer: Send + Sync`, so `Recognizer` inherits
    /// it, but pin it here so a future change to this struct can't silently
    /// lose it.
    #[test]
    fn recognizer_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Recognizer>();
    }
}
