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
/// (`bench/fixtures/hotwords.txt`: mesa, auris, khora, qorvex, helios,
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
/// it (`docs/spike-results.md` §8), and the axis is grammatical embedding rather
/// than length: once `khora` sits inside an ordinary English clause — after
/// a preposition, under a determiner, or as the subject of a verb — it is
/// never produced, at any boost the vocabulary bounds allow, on either
/// synthesiser. `bench/fixtures/wav/u02.wav` decodes "Korra" unbiased and
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
        .join("bench/fixtures/hotwords.txt")
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

/// mesa task 970: hotword biasing turned an empty decode of non-speech
/// audio into a confident hallucination ("The vegetable mesa mesa khora
/// khora khora ... mesa khan mesa q", 286 chars, exit 0). Unbiased,
/// `nonspeech-transient.wav` already exits 1 with empty stdout (see
/// `src/audio.rs`'s silence gate and `src/vad.rs`'s VAD gate); biased, it
/// must exit 1 with empty stdout too, now via the manufactured-vocabulary
/// guard (`vocabulary::looks_manufactured` plus its confirming unbiased
/// decode in `src/cli.rs`).
#[test]
fn manufactured_vocabulary_transcript_yields_no_transcript() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let vocabulary_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bench/fixtures/hotwords.txt")
        .to_string_lossy()
        .into_owned();
    let model = tmp.path().to_string_lossy().into_owned();
    let wav = std::fs::read(fixtures_dir().join("nonspeech-transient.wav"))
        .expect("read nonspeech-transient.wav");

    let out = run_auris(
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
    let text = String::from_utf8_lossy(&out.stdout);
    eprintln!(
        "nonspeech-transient.wav (biased): exit={:?} stdout={text:?}",
        out.status.code()
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(text.is_empty(), "expected empty stdout, got {text:?}");
}

/// The other half of mesa task 970's guard: it must not change anything for
/// audio that actually contains the vocabulary's speech. Same fixture and
/// vocabulary as `mesa_names_lands_only_when_biased`, run through the
/// guard's code path this time (that test predates the guard).
#[test]
fn manufactured_vocabulary_guard_does_not_affect_real_speech() {
    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let vocabulary_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bench/fixtures/hotwords.txt")
        .to_string_lossy()
        .into_owned();
    let model = tmp.path().to_string_lossy().into_owned();
    let wav = std::fs::read(fixtures_dir().join("mesa-names.wav")).expect("read mesa-names.wav");

    let out = run_auris(
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
    let text = String::from_utf8_lossy(&out.stdout);
    eprintln!(
        "mesa-names.wav (biased, guard active): exit={:?} stdout={text:?}",
        out.status.code()
    );
    assert_eq!(out.status.code(), Some(0), "stderr: {:?}", out.stderr);
    let normalised = normalise(&text);
    assert!(normalised.contains("qorvex"), "got {text:?}");
    assert!(normalised.contains("helios"), "got {text:?}");
}

/// `utterances` copies of `plain.wav` separated by `gap_seconds` of digital
/// silence: one WAV, several utterances, built in memory so the repo does
/// not carry a second multi-utterance fixture whose only content is the
/// first one repeated. The gap is comfortably longer than
/// `--vad-min-silence` (0.5 s), so the VAD closes an utterance in each one
/// rather than running them together.
fn multi_utterance_wav(utterances: usize, gap_seconds: f32) -> Vec<u8> {
    let mut reader =
        hound::WavReader::open(fixtures_dir().join("plain.wav")).expect("open plain.wav");
    let spec = reader.spec();
    let speech: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .expect("read plain.wav");
    let gap = vec![0i16; (spec.sample_rate as f32 * gap_seconds) as usize];

    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("wav writer");
        for utterance in 0..utterances {
            if utterance > 0 {
                for &sample in &gap {
                    writer.write_sample(sample).expect("write gap");
                }
            }
            for &sample in &speech {
                writer.write_sample(sample).expect("write speech");
            }
        }
        writer.finalize().expect("finalize wav");
    }
    cursor.into_inner()
}

/// The acceptance criterion mesa task 936 exists for, and the one README
/// "stdin" states as a promise: **stdout's first byte does not wait for
/// stdin's EOF on a multi-utterance stream.** A consumer that reads auris's
/// stdout incrementally while still writing audio (mesa's driver does
/// exactly this) deadlocks if auris only ever writes at EOF.
///
/// This asserts on *timing*, not on line count: a test that only counted
/// four `segment` lines would pass just as well if all four were written
/// after the pipe closed, which is the bug. The audio is fed at real time
/// (16 kHz, 16-bit mono is 32 000 bytes per second) so "before EOF" means
/// what it means for a live capture, not for a file handed over slowly on
/// purpose.
///
/// A `segment` line is held back until the *next* utterance opens — its
/// right edge is the next utterance's start (`src/cli.rs`, `src/vad.rs`) —
/// so the first line lands one utterance late by design. Four utterances,
/// not two, so that delay is inside the stream rather than at its end.
#[test]
fn first_segment_line_arrives_before_stdin_eof() {
    use std::io::BufRead;
    use std::time::{Duration, Instant};

    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let home = tempfile::tempdir().expect("tempdir");
    let wav = multi_utterance_wav(4, 1.0);

    let mut child = Command::new(env!("CARGO_BIN_EXE_auris"))
        .args([
            "--no-daemon",
            "--quiet",
            "-m",
            &tmp.path().to_string_lossy(),
        ])
        .env("AURIS_HOME", home.path())
        .env_remove("AURIS_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn auris");

    // Reads stdout in a loop rather than draining to EOF, and stamps the
    // arrival of the first line — the whole subject of this test.
    let stdout = child.stdout.take().expect("child stdout");
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut lines = std::io::BufReader::new(stdout).lines();
        let first = lines.next();
        let _ = tx.send(Instant::now());
        let mut collected: Vec<String> = first.into_iter().filter_map(Result::ok).collect();
        collected.extend(lines.map_while(Result::ok));
        collected
    });

    let mut stdin = child.stdin.take().expect("child stdin");
    for chunk in wav.chunks(16_000) {
        stdin.write_all(chunk).expect("write stdin");
        std::thread::sleep(Duration::from_millis(500));
    }
    let stdin_closed_at = Instant::now();
    drop(stdin);

    let first_line_at = rx
        .recv_timeout(Duration::from_secs(180))
        .expect("auris wrote nothing to stdout at all");
    let lines = reader.join().expect("stdout reader thread");
    let out = child.wait_with_output().expect("wait for auris");

    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        lines.len() >= 2,
        "expected one line per utterance, got {lines:?}"
    );
    assert!(
        first_line_at < stdin_closed_at,
        "the first stdout line arrived {:?} after stdin closed; auris is still \
         buffering the whole stream (lines: {lines:?})",
        first_line_at.duration_since(stdin_closed_at)
    );
}

/// The other half of mesa task 936's acceptance: a consumer that stops
/// reading stdout must not wedge auris. Rust ignores SIGPIPE, so the write
/// comes back `EPIPE` rather than killing the process — which means every
/// write site has to notice, or auris grinds on decoding audio nobody will
/// read (or, worse, blocks forever on a full pipe).
///
/// The close happens before any audio is written, so it lands ahead of the
/// first `segment` line whatever the decode timing is. Exit 0: text was
/// produced and committed as far as auris could commit it, and a nonzero
/// exit is the one thing mesa reads as failure (README "Exit codes").
#[test]
fn a_consumer_that_stops_reading_stdout_does_not_wedge_auris() {
    use std::time::{Duration, Instant};

    let Some(model_dir) = spike_model_dir() else {
        return;
    };
    let tmp = symlinked_model_dir(&model_dir);
    let home = tempfile::tempdir().expect("tempdir");
    let wav = std::fs::read(fixtures_dir().join("plain.wav")).expect("read plain.wav");

    let mut child = Command::new(env!("CARGO_BIN_EXE_auris"))
        .args([
            "--no-daemon",
            "--quiet",
            "-m",
            &tmp.path().to_string_lossy(),
        ])
        .env("AURIS_HOME", home.path())
        .env_remove("AURIS_MODEL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn auris");

    // The consumer goes away.
    drop(child.stdout.take().expect("child stdout"));

    // May itself fail with EPIPE if auris has already exited — that is a
    // pass, not a failure, so the result is deliberately discarded.
    let mut stdin = child.stdin.take().expect("child stdin");
    let _ = stdin.write_all(&wav);
    drop(stdin);

    let deadline = Instant::now() + Duration::from_secs(180);
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status,
            None => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    panic!("auris did not exit after its stdout consumer went away");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    };
    assert_eq!(
        status.code(),
        Some(0),
        "expected a clean exit, not a signal or a failure: {status:?}"
    );
}
