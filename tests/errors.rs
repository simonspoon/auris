//! Integration tests for auris's failure surface — the exit code / single
//! stderr line contract mesa's speech driver (`mesa/src/core/speech.rs`)
//! relies on to tell "no transcript" from "here is a transcript" (README
//! "Exit codes", "stderr"). Each case pins one specific failure to an exact
//! exit code and a readable one-line message, so a future change that turns
//! a clean error into a panic, blurs two distinct causes into one message,
//! or starts exiting 0 with nothing on stdout fails a test here instead of
//! failing silently for mesa.
//!
//! Every invocation runs against a freshly created `AURIS_HOME` (never the
//! developer's real one) with `AURIS_MODEL` scrubbed, and transcribe runs
//! pass `--no-daemon`, so no test ever touches or starts a real daemon
//! socket.

use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

/// Runs the auris binary with `stdin` piped in and a brand-new `AURIS_HOME`,
/// so no test can see another test's (or the developer's own) installed
/// model or daemon socket. `stdin` is written and the pipe closed before the
/// process is waited on, so a run that reads to EOF never blocks forever on
/// an unclosed write end.
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

/// The shape every non-usage failure message must have (CLAUDE.md
/// "stderr"): one line, prefixed `auris: `, with no leaked panic or
/// backtrace machinery. Returns the trimmed text so callers can check its
/// content too.
fn assert_single_line_stderr(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr).trim().to_string();
    assert!(
        !text.contains('\n'),
        "stderr should be exactly one line, got: {text:?}"
    );
    assert!(
        text.starts_with("auris: "),
        "stderr should start with 'auris: ', got: {text:?}"
    );
    for banned in ["panicked", "RUST_BACKTRACE", "stack backtrace", "Error:"] {
        assert!(!text.contains(banned), "stderr leaked {banned:?}: {text:?}");
    }
    text
}

/// Builds an in-memory 16-bit PCM WAV, matching the fixture style already
/// used by `src/audio.rs`'s own unit tests — no fixture files on disk.
fn wav_bytes(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}

/// A small, valid 16 kHz mono WAV — enough to pass audio decoding and reach
/// model resolution, which is all these tests need from it.
fn small_valid_wav() -> Vec<u8> {
    let samples: Vec<i16> = (0..1600).map(|i| ((i % 200) * 100) as i16).collect();
    wav_bytes(16_000, 1, &samples)
}

/// ~1 s of digital silence at 16 kHz mono, for the "reaches the recognizer
/// but there's no speech" case.
fn silence_wav_1s() -> Vec<u8> {
    wav_bytes(16_000, 1, &vec![0i16; 16_000])
}

/// mesa needs to tell a caller "you haven't fetched the model" apart from
/// every other failure, so it can point at `auris serve` instead of a
/// generic decode error. Pins that message for the default (fetch-capable)
/// path, where model download simply isn't implemented yet.
#[test]
fn missing_default_model_without_no_download() {
    let out = run_auris(&["--no-daemon", "--quiet"], &small_valid_wav());
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("parakeet-tdt-0.6b-v2-int8"));
    assert!(stderr.contains("is not installed"));
    assert!(stderr.contains("auris serve"));
}

/// Same missing-model case with `--no-download` set: today there is no
/// fetch for the flag to refuse, so the message and exit code must match
/// the without-`--no-download` case exactly (see `src/cli.rs`
/// `run_transcribe`'s comment on this).
#[test]
fn missing_default_model_with_no_download() {
    let out = run_auris(
        &["--no-daemon", "--quiet", "--no-download"],
        &small_valid_wav(),
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("parakeet-tdt-0.6b-v2-int8"));
    assert!(stderr.contains("is not installed"));
    assert!(stderr.contains("auris serve"));
}

/// A caller piping something that isn't audio at all (a text file, a stray
/// log) must get a message naming the actual problem, not a hound panic or
/// a generic I/O error.
#[test]
fn stdin_is_not_audio() {
    let out = run_auris(&["--no-daemon", "--quiet"], b"this is not audio at all\n");
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("not a wav file"));
    assert!(stderr.contains("expected a wav stream"));
}

/// Zero bytes on stdin (a caller's upstream pipe produced nothing) is a
/// distinct cause from "produced bytes that aren't audio" — mesa should be
/// able to tell "nothing arrived" from "garbage arrived" from the message
/// alone, so this also checks the two messages are different.
#[test]
fn stdin_is_empty() {
    let out = run_auris(&["--no-daemon", "--quiet"], b"");
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("input was empty"));
    assert!(stderr.contains("expected a wav stream"));

    let garbage = run_auris(&["--no-daemon", "--quiet"], b"this is not audio at all\n");
    let garbage_stderr = assert_single_line_stderr(&garbage.stderr);
    assert_ne!(
        stderr, garbage_stderr,
        "empty stdin and non-wav stdin must produce distinguishable messages"
    );
}

/// Locates a real, complete model directory for the one test that needs to
/// actually reach the recognizer, using the exact same env var and fallback
/// as `src/engine.rs`'s `spike_model_dir` — the existing convention in this
/// crate for "here is a model to test against" — rather than inventing a
/// second one. Unlike `spike_model_dir` (which only checks for the encoder,
/// because its callers build their own temp dir of symlinks), this checks
/// every file README "The cache" requires, since this test runs the real
/// binary against the directory as given, with no symlinking step of its
/// own.
fn model_dir_for_silence_test() -> Option<PathBuf> {
    let dir = std::env::var("AURIS_TEST_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/models/parakeet")
        });
    let required = [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
        "bpe.vocab",
    ];
    if required.iter().all(|f| dir.join(f).is_file()) {
        Some(dir)
    } else {
        eprintln!(
            "skip: no complete model at {} (set AURIS_TEST_MODEL_DIR to override)",
            dir.display()
        );
        None
    }
}

/// Silence reaching a real recognizer must exit "no transcript" rather than
/// print an empty line. Needs a real installed model, which is not present
/// on CI or this host, so this is gated on `AURIS_TEST_MODEL_DIR` and skips
/// (does not fail) when that isn't set — matching the skip pattern already
/// used by `src/engine.rs`'s own model-gated tests.
#[test]
fn silence_is_nothing_transcribed() {
    let Some(model) = model_dir_for_silence_test() else {
        return;
    };
    let model = model.to_string_lossy().into_owned();
    let out = run_auris(&["--no-daemon", "--quiet", "-m", &model], &silence_wav_1s());
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("nothing transcribed"));
    assert!(stderr.contains("no speech"));
}

/// `install_interrupt_handler` (`src/cli.rs`) only sets a flag on the first
/// SIGINT, polled between stages — while blocked inside `audio::decode`'s
/// read that poll never runs, so a real caller's single Ctrl-C on a wedged
/// pipe does nothing, and a second one exits immediately. This test drives
/// exactly that sequence against a process blocked reading an open (never
/// closed) stdin pipe, with a bounded wait so a regression to "hangs
/// forever" fails the test instead of hanging the suite.
///
/// auris links sherpa-onnx's ONNX Runtime, so the delay between process
/// start and `install_interrupt_handler` actually arming the handler is not
/// fixed — on a busy machine (e.g. a concurrent `cargo build`) it can run
/// long enough that the first SIGINT below arrives before the handler is
/// armed, in which case the OS default disposition (terminate) fires
/// instead of the handler being exercised. That is a timing race, not a
/// regression, so this retries with a fresh process rather than either
/// papering over it with an ever-longer fixed sleep (which just moves the
/// race, it never removes it) or asserting on it as if it disproved the
/// handler exists. A child that dies any other way — a different signal, or
/// a normal exit with any code — is a real regression and fails immediately,
/// no retry.
#[cfg(unix)]
#[test]
fn double_sigint_interrupts_a_blocked_read() {
    use std::os::unix::process::ExitStatusExt;
    use std::time::{Duration, Instant};

    // Eight rather than a token two or three: on a loaded machine a single
    // attempt loses this race far more often than not (observed 8 runs in 10
    // needing a retry, one needing three), so a small ceiling would fail a
    // correct binary on a slow CI box. An attempt costs ~1.5 s and only the
    // losing ones are paid.
    const MAX_ATTEMPTS: u32 = 8;

    for attempt in 1..=MAX_ATTEMPTS {
        let home = tempfile::tempdir().expect("tempdir");
        let mut child = Command::new(env!("CARGO_BIN_EXE_auris"))
            .args(["--no-daemon", "--quiet"])
            .env("AURIS_HOME", home.path())
            .env_remove("AURIS_MODEL")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn auris");

        // Kept open (never written to, never dropped) until the process has
        // exited: it must be blocked on this pipe's read, not sitting at EOF.
        let stdin = child.stdin.take().expect("child stdin");

        std::thread::sleep(Duration::from_millis(1000));

        let pid = child.id() as i32;

        // First SIGINT: the flag gets set, but nothing polls it while
        // blocked on stdin, so the process must still be alive — unless the
        // handler wasn't armed yet (see the retry logic below).
        unsafe { libc::kill(pid, libc::SIGINT) };
        std::thread::sleep(Duration::from_millis(500));

        match child.try_wait().expect("try_wait") {
            None => {
                // Handler was armed in time. Second SIGINT: it exits
                // immediately.
                unsafe { libc::kill(pid, libc::SIGINT) };

                let deadline = Instant::now() + Duration::from_secs(5);
                let status = loop {
                    if let Some(status) = child.try_wait().expect("try_wait") {
                        break status;
                    }
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("auris did not exit after a second SIGINT");
                    }
                    std::thread::sleep(Duration::from_millis(50));
                };

                drop(stdin); // held open on purpose; dropped now that the process has exited
                assert_eq!(status.code(), Some(130));
                return;
            }
            Some(status) if status.signal() == Some(libc::SIGINT) => {
                // Startup race: the handler wasn't armed yet, so the OS
                // default (terminate) fired on the first SIGINT instead.
                // Not a regression — retry with a fresh process.
                let _ = child.kill();
                let _ = child.wait();
                drop(stdin);
                eprintln!(
                    "attempt {attempt}/{MAX_ATTEMPTS}: auris was killed by raw SIGINT before \
                     arming its handler (machine load); retrying"
                );
                continue;
            }
            Some(status) => {
                let _ = child.kill();
                let _ = child.wait();
                let out = child.wait_with_output().expect("wait_with_output");
                drop(stdin);
                panic!(
                    "a single SIGINT must not exit a process blocked reading stdin: \
                     status={status:?} stdout={:?} stderr={:?}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
    }

    panic!(
        "auris never armed its ctrl-c handler within {MAX_ATTEMPTS} attempts; this looks like \
         a real startup regression, not machine load"
    );
}

/// clap owns bad-flag reporting and writes its own multi-line usage text —
/// not the single-line `auris: ...` shape every other case here checks — so
/// only the exit code and the "no transcript leaked to stdout" half of the
/// contract are pinned for this case.
#[test]
fn usage_bad_flag() {
    let out = run_auris(&["--not-a-real-flag"], b"");
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
}

/// A PATH argument that doesn't exist is a usage error, not "no transcript
/// from bad audio" — auris never even reaches `audio::decode`.
#[test]
fn usage_missing_file_path() {
    let out = run_auris(
        &["--no-daemon", "--quiet", "/nonexistent/auris-test-path.wav"],
        b"",
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let stderr = assert_single_line_stderr(&out.stderr);
    assert!(stderr.contains("failed to open"));
}

/// The acceptance check this file exists to satisfy: nothing on the
/// non-interactive failure surface exits 0 while stdout is empty — that
/// combination is exactly what would make mesa's speech driver treat a
/// failed run as a successful, silent transcription.
#[test]
fn no_failure_case_exits_0_with_empty_stdout() {
    let wav = small_valid_wav();
    let cases: Vec<(Vec<&str>, Vec<u8>)> = vec![
        (vec!["--no-daemon", "--quiet"], wav.clone()),
        (vec!["--no-daemon", "--quiet", "--no-download"], wav),
        (
            vec!["--no-daemon", "--quiet"],
            b"this is not audio at all\n".to_vec(),
        ),
        (vec!["--no-daemon", "--quiet"], Vec::new()),
        (vec!["--not-a-real-flag"], Vec::new()),
        (
            vec!["--no-daemon", "--quiet", "/nonexistent/auris-test-path.wav"],
            Vec::new(),
        ),
    ];

    for (args, stdin) in &cases {
        let out = run_auris(args, stdin);
        assert!(
            out.status.code() != Some(0) || !out.stdout.is_empty(),
            "case {args:?} exited 0 with an empty stdout"
        );
    }
}
