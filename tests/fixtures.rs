//! Integration tests driving the real `auris` binary against
//! `tests/fixtures/` (see `scripts/generate-fixtures.sh`). Unlike
//! `tests/errors.rs`, which builds its wavs in memory to pin the failure
//! surface, these tests exist to prove the audio *plumbing* — decode,
//! resample, downmix, the vocabulary path — against fixtures a human can
//! listen to. Every fixture here is synthesised (kokoro-rs), not dictated:
//! that proves the plumbing, not transcription accuracy on real speech — the
//! benchmark task covers accuracy.
//!
//! Every model-dependent test here skips (eprintln + return) rather than
//! fails when no model is installed, matching `src/engine.rs`'s
//! `spike_model_dir`/skip pattern — CI has no model, so this file must not
//! turn that into a red build.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The model these expectations were baselined against. A model upgrade
/// (different training, different tokenizer) is expected to shift exact
/// wording at the margins — re-baselining this file's expectations against
/// the new model is a deliberate, reviewed change, not a red CI run caused
/// by a regression. If these tests start failing after swapping in a new
/// `-m`, that is this constant telling you why, not a bug report. It is
/// named in the skip message below so the model these expectations assume
/// is visible to whoever is looking at a skipped run, not buried here.
const BASELINE_MODEL: &str = "parakeet-tdt-0.6b-v2-int8";

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Folds case, strips punctuation, and collapses whitespace, so a
/// transcript can be compared against the manifest's reference text without
/// caring about a trailing period or doubled space. Used for every
/// transcript assertion in this file.
fn normalise(s: &str) -> String {
    let folded: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Locates the real Parakeet model, honouring `AURIS_TEST_MODEL_DIR` with
/// the same fallback as `src/engine.rs`'s `spike_model_dir` — the
/// established convention in this crate for "here is a model to test
/// against". Skips (does not fail) when incomplete, since CI has no model
/// checked in.
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
            "skip: no model at {} (expectations baselined against {BASELINE_MODEL}; \
             set AURIS_TEST_MODEL_DIR to override)",
            dir.display()
        );
        None
    }
}

/// Builds a temp model dir of symlinks pointing at the real spike model,
/// with `bpe_synth.vocab` linked in as `bpe.vocab` — same layout
/// `src/engine.rs`'s `symlinked_model_dir` builds, and for the same reason:
/// `--no-daemon -m <dir>` requires all five `REQUIRED_MODEL_FILES`
/// (`src/cli.rs`), and the spike model only ships `bpe_synth.vocab`, not the
/// generated `bpe.vocab`. spike/ itself must not be copied (631 MB) or
/// renamed in place.
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

/// Runs the auris binary against a fresh `AURIS_HOME` (never the developer's
/// real one), `--no-daemon`, matching `tests/errors.rs`'s `run_auris` so no
/// test here can touch or start a real daemon socket either.
fn run_auris(args: &[&str], stdin: &[u8]) -> Output {
    let home = tempfile::tempdir().expect("tempdir");
    let mut child = Command::new(env!("CARGO_BIN_EXE_auris"))
        .args(args)
        .env("AURIS_HOME", home.path())
        .env_remove("AURIS_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn auris");

    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(stdin)
        .expect("write stdin");

    child.wait_with_output().expect("wait for auris")
}

fn transcribe(model_dir: &Path, wav_path: &Path) -> Output {
    let wav = std::fs::read(wav_path).unwrap_or_else(|e| panic!("read {wav_path:?}: {e}"));
    let model = model_dir.to_string_lossy().into_owned();
    run_auris(&["--no-daemon", "--quiet", "-m", &model], &wav)
}

/// The manifest's `plain` reference text (`tests/fixtures/manifest.tsv`),
/// inlined rather than parsed from the file at test time — this file is the
/// baseline these expectations were written against, so a change to the
/// manifest not matched here should show up as a diff, not a silently
/// self-updating test.
const PLAIN_REFERENCE: &str = "open the daily notes and add a line about the meeting";

/// `plain.wav` (synthesised, 16 kHz mono) transcribes to the manifest's
/// reference text. Containment rather than exact equality: kokoro-rs's
/// wording of e.g. "notes" vs "note's" or punctuation choices is not the
/// thing under test — the words landing, in order, is.
#[test]
fn plain_transcribes_to_reference_text() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let out = transcribe(tmp.path(), &fixtures_dir().join("plain.wav"));
    assert_eq!(out.status.code(), Some(0), "stderr: {:?}", out.stderr);
    let text = String::from_utf8_lossy(&out.stdout);
    eprintln!("plain.wav: {text:?}");
    assert_eq!(normalise(&text), normalise(PLAIN_REFERENCE));
}

/// `stereo-44100.wav` carries the exact same speech as `plain.wav`, just at
/// 44.1 kHz stereo — the resample + downmix path in `src/audio.rs` must
/// reach the same transcript as the native 16 kHz mono fixture, not merely
/// "a" transcript.
#[test]
fn stereo_44100_resamples_to_the_same_transcript_as_plain() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);

    let plain_out = transcribe(tmp.path(), &fixtures_dir().join("plain.wav"));
    let stereo_out = transcribe(tmp.path(), &fixtures_dir().join("stereo-44100.wav"));
    assert_eq!(plain_out.status.code(), Some(0));
    assert_eq!(stereo_out.status.code(), Some(0));

    let plain_text = String::from_utf8_lossy(&plain_out.stdout);
    let stereo_text = String::from_utf8_lossy(&stereo_out.stdout);
    eprintln!("plain:  {plain_text:?}");
    eprintln!("stereo: {stereo_text:?}");
    assert_eq!(normalise(&plain_text), normalise(&stereo_text));
}

/// README ("Exit codes") documents silence as exit 1, NOTHING_TRANSCRIBED.
/// This used to fail: the real model hallucinates "Okay." on 1 s of
/// digital silence rather than returning nothing, which slipped past the
/// old post-decode `text.trim().is_empty()` check and exited 0. That's why
/// `src/audio.rs::is_silent` — an energy gate — now runs before the
/// recognizer is ever reached; silent audio never gets a chance to
/// hallucinate anything.
#[test]
fn silence_yields_no_transcript() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let out = transcribe(tmp.path(), &fixtures_dir().join("silence.wav"));
    let text = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("silence.wav: exit={:?} stdout={text:?}", out.status.code());
    assert_eq!(out.status.code(), Some(1));
    assert!(text.is_empty(), "expected empty stdout, got {text:?}");
    assert!(stderr.contains("nothing transcribed"));
    assert!(stderr.contains("no speech"));
}

/// `not-audio.bin` is a clean, single-line error (`src/audio.rs`'s
/// `AudioError::NotWav`), not a panic or a hang. This does not need a model:
/// `audio::decode` rejects the input before model resolution ever runs.
#[test]
fn not_audio_is_a_clean_error() {
    let bytes = std::fs::read(fixtures_dir().join("not-audio.bin")).expect("read not-audio.bin");
    let out = run_auris(&["--no-daemon", "--quiet"], &bytes);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    assert!(!stderr.contains('\n'), "expected one line, got {stderr:?}");
    assert!(stderr.contains("not a wav file"));
    assert!(stderr.contains("expected a wav stream"));
}

/// The vocabulary claim (docs/vocabulary.md): the synthesised
/// `mesa-names.wav` transcribes WITHOUT either mesa name when unbiased, and
/// WITH both when biased via `--vocabulary-file` — the actual claim being
/// tested is that contrast, not just "the names appear somewhere", since a
/// name the model already gets right with no biasing at all would prove
/// nothing about biasing.
///
/// The manifest's `mesa-names` line is "hey qorvex hey helios", chosen after
/// testing pairs from the full production vocabulary
/// (`spike/fixtures/hotwords.txt`: mesa, auris, khora, qorvex, helios,
/// kokoro) for exactly this contrast:
/// - `qorvex` + `helios`: unbiased "Hey Corvex Hey Halios.", biased "Hey
///   qorvex hey helios." — both names absent unbiased, both present biased.
///   Confirmed deterministic across 3 repeated runs. **This is the pair
///   used.**
/// - `qorvex` + `kokoro` also worked (unbiased "Hei Kakaro", biased
///   "kokoro"), kept in reserve, not used.
/// - `qorvex` + `auris`: rejected — biased still decoded "Corvex", not
///   "qorvex", so half the contrast failed.
/// - `qorvex` + `mesa`: rejected — "Mesa" was already correctly capitalised
///   unbiased, so it fails the "absent unbiased" half of the contrast.
///
/// `khora` is deliberately NOT in this fixture, and that is a finding in its
/// own right, not a phrasing failure to paper over. Task 959 characterised
/// it (`spike/RESULTS.md` §8), and the axis is grammatical embedding rather
/// than length: once `khora` sits inside an ordinary English clause — after
/// a preposition, under a determiner, or as the subject of a verb — it is
/// never produced, at any boost the vocabulary bounds allow, on either
/// synthesiser. `spike/fixtures/wav/u02.wav` decodes "Korra" unbiased and
/// "qora" biased. Where it stands alone as an address or label it can land,
/// but only at a boost of 6.5 or above and only for some voices, usually
/// damaging a neighbour ("hey khora" becomes "He khora"). The kokoro-rs
/// phrase "hey khora hey qorvex" that lands `khora` is an instance of that
/// narrow address case, not a counterexample to it. So a `khora` fixture
/// would be asserting the one shape of utterance auris does not receive,
/// which is why the committed pair is `qorvex`/`helios` — they land
/// reliably across the phrasings tried.
/// Recovering `khora` from a real sentence is the post-ASR correction
/// pass's job (`docs/correction.md`), not biasing's.
#[test]
fn mesa_names_lands_only_when_biased() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let vocabulary_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("spike/fixtures/hotwords.txt")
        .to_string_lossy()
        .into_owned();
    let wav = std::fs::read(fixtures_dir().join("mesa-names.wav")).expect("read mesa-names.wav");
    let model = tmp.path().to_string_lossy().into_owned();

    let unbiased_out = run_auris(&["--no-daemon", "--quiet", "-m", &model], &wav);
    assert_eq!(
        unbiased_out.status.code(),
        Some(0),
        "stderr: {:?}",
        unbiased_out.stderr
    );
    let unbiased_text = String::from_utf8_lossy(&unbiased_out.stdout);
    eprintln!("mesa-names.wav (unbiased): {unbiased_text:?}");
    let unbiased_normalised = normalise(&unbiased_text);
    assert!(
        !unbiased_normalised.contains("qorvex"),
        "expected qorvex ABSENT unbiased (else biasing proves nothing), got {unbiased_text:?}"
    );
    assert!(
        !unbiased_normalised.contains("helios"),
        "expected helios ABSENT unbiased (else biasing proves nothing), got {unbiased_text:?}"
    );

    let biased_out = run_auris(
        &[
            "--no-daemon",
            "--quiet",
            "-m",
            &model,
            "--vocabulary-file",
            &vocabulary_file,
        ],
        &wav,
    );
    assert_eq!(
        biased_out.status.code(),
        Some(0),
        "stderr: {:?}",
        biased_out.stderr
    );
    let biased_text = String::from_utf8_lossy(&biased_out.stdout);
    eprintln!("mesa-names.wav (biased): {biased_text:?}");
    let biased_normalised = normalise(&biased_text);
    assert!(
        biased_normalised.contains("qorvex"),
        "expected qorvex PRESENT biased, got {biased_text:?}"
    );
    assert!(
        biased_normalised.contains("helios"),
        "expected helios PRESENT biased, got {biased_text:?}"
    );
}
