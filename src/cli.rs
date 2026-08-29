//! The transcribe path: audio in (stdin or `PATH`), transcript out on
//! stdout (README "Synopsis", "The contract"). This module is the client
//! half of auris; it knows how to read audio, resolve a model, and shape the
//! result the way `--format` says. By default it talks to the persistent
//! daemon ([`crate::daemon`]) instead of loading
//! [`crate::engine::Recognizer`] itself, auto-starting one if none is
//! listening (README "The daemon"); `--no-daemon` keeps the original
//! in-process path for one-off use and for debugging the daemon path
//! itself. This module also owns the `serve` / `status` / `stop`
//! subcommands, which are thin wrappers around [`crate::daemon`].

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use clap::{Parser, ValueEnum};

use crate::audio;
use crate::daemon;
use crate::engine::{EngineConfig, Recognizer};
use crate::model;
use crate::vad;
use crate::vocabulary::{Vocabulary, looks_manufactured};

/// Exit codes, copied from kokoro-rs (README "Exit codes"), not reinvented.
const OK: i32 = 0;
const NOTHING_TRANSCRIBED: i32 = 1;
const USAGE: i32 = 2;
const INTERRUPTED: i32 = 130;

/// The one message for "no transcript" — printed whether the energy gate
/// trips before the recognizer runs or the recognizer itself returns
/// nothing, so there is a single wording for mesa to match on, not two.
/// The two paths being indistinguishable from outside the process once cost
/// an investigation (mesa task 965) that was chasing a gate defect when the
/// recognizer had in fact run and decoded nothing. The wording and exit code
/// stay merged deliberately — mesa still needs one thing to match — but
/// under `verbose`, each site now adds a detail line naming which one fired.
/// That line is invisible to anything that pipes stderr, which is why it
/// costs the contract nothing.
const NOTHING_TRANSCRIBED_MSG: &str = "auris: nothing transcribed; no speech in the audio";

/// The only model auris knows how to fetch today (README "Which model, and
/// `-m`").
const DEFAULT_MODEL_NAME: &str = "parakeet-tdt-0.6b-v2-int8";

/// The five files that make a model directory "installed" (README "The
/// cache") — `bpe.vocab` included, since a half-generated model is just as
/// unusable as a half-downloaded one.
const REQUIRED_MODEL_FILES: [&str; 5] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
    "bpe.vocab",
];

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Text,
    Json,
}

#[derive(Parser)]
#[command(
    name = "auris",
    about = "Speech-to-text with Parakeet TDT 0.6B v2 (int8). \
             Reads audio from stdin or a file argument, writes the transcript to stdout.",
    version
)]
pub struct Args {
    /// `serve` / `status` / `stop` — omitted for the default transcribe path
    #[command(subcommand)]
    command: Option<Command>,

    /// Audio file to transcribe (default: read stdin)
    path: Option<PathBuf>,

    /// Model to use: a name from --list-models, or a path to a model directory
    #[arg(short = 'm', long = "model", value_name = "NAME")]
    model: Option<String>,

    /// Output shape
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Terms for hotword biasing, one `term :boost` per line
    /// (docs/vocabulary.md). With no vocabulary file, auris decodes plainly.
    #[arg(long, value_name = "FILE")]
    vocabulary_file: Option<PathBuf>,

    /// Fail rather than fetch a missing model on a real run
    #[arg(long)]
    no_download: bool,

    /// Print installed model names, one per line, and exit
    #[arg(long)]
    list_models: bool,

    /// Silero VAD speech-probability threshold
    #[arg(long, value_name = "F", default_value_t = vad::VadConfig::default().threshold)]
    vad_threshold: f32,

    /// Speech spans shorter than this are dropped
    #[arg(long, value_name = "SECS", default_value_t = vad::VadConfig::default().min_speech)]
    vad_min_speech: f32,

    /// Skip the VAD gate — for audio that is already segmented, or to
    /// measure accuracy against today's behaviour with no VAD at all
    #[arg(long)]
    no_vad: bool,

    /// Load the recognizer in-process instead of talking to a daemon
    #[arg(long)]
    no_daemon: bool,

    /// Daemon socket path (default $AURIS_HOME/auris.sock)
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Suppress the progress line
    #[arg(short, long)]
    quiet: bool,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Start (or become) the daemon
    Serve(ServeArgs),
    /// Is a daemon running, with what model
    Status(SocketArgs),
    /// Ask the daemon to exit
    Stop(SocketArgs),
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Model to use: a name from --list-models, or a path to a model directory
    #[arg(short = 'm', long = "model", value_name = "NAME")]
    model: Option<String>,

    /// Daemon socket path (default $AURIS_HOME/auris.sock)
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Suppress the progress line
    #[arg(short, long)]
    quiet: bool,

    /// Fail rather than fetch a missing model on a real run
    #[arg(long)]
    no_download: bool,
}

#[derive(clap::Args)]
struct SocketArgs {
    /// Daemon socket path (default $AURIS_HOME/auris.sock)
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,
}

pub fn main() -> i32 {
    run(Args::parse())
}

fn run(args: Args) -> i32 {
    match args.command {
        Some(Command::Serve(serve_args)) => run_serve(serve_args),
        Some(Command::Status(socket_args)) => run_status(socket_args),
        Some(Command::Stop(socket_args)) => run_stop(socket_args),
        None => run_transcribe(args),
    }
}

/// Turns Ctrl-C into a polled flag rather than an unwind — the recognizer
/// holds an ONNX Runtime session that needs no help from a panic mid-decode.
/// A second Ctrl-C kills the process immediately, for a wedged run.
fn install_interrupt_handler() -> Arc<AtomicBool> {
    let interrupt = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupt);
    let _ = ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::SeqCst) {
            std::process::exit(INTERRUPTED);
        }
    });
    interrupt
}

/// `$AURIS_HOME`, defaulting to `$HOME/.cache/auris` — computed exactly as
/// kokoro-rs computes `KOKORO_HOME` (README "The cache", `models.rs:65-71`):
/// read the env var, else `$HOME/.cache/<name>`, falling back to `.` when
/// `$HOME` is unset. Takes `home` rather than reading `$HOME` itself so the
/// one env read lives at the top of `run`, not scattered through pure
/// helpers.
pub(crate) fn auris_home(home: Option<&str>) -> PathBuf {
    if let Ok(dir) = std::env::var("AURIS_HOME") {
        return PathBuf::from(dir);
    }
    PathBuf::from(home.unwrap_or("."))
        .join(".cache")
        .join("auris")
}

/// Expands a leading `~/`, mirroring kokoro-rs's `expand_tilde`
/// (`models.rs:104-111`) exactly — nothing fancier than that is needed here
/// either. Takes `home` as a parameter, rather than reading `$HOME` itself,
/// so [`resolve_model_source`] stays a pure function safe to call from
/// tests run in parallel threads (env vars are process-global).
fn expand_tilde(path: &str, home: Option<&str>) -> PathBuf {
    match (path.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

/// Where a model directory came from, and so how a missing one should be
/// treated: a name or path the caller gave explicitly (`-m`, `AURIS_MODEL`)
/// is a hard usage error when absent (README "Which model, and `-m`",
/// "Environment"); the default name falling through unresolved just means
/// "not fetched yet" (README "`--no-download`").
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelSource {
    Explicit(PathBuf),
    Default(PathBuf),
}

/// Resolves `-m` / `AURIS_MODEL` / the default name into a model directory,
/// without touching the filesystem — precedence is `-m` beats `AURIS_MODEL`
/// beats the default name (README "Which model, and `-m`"). The
/// path-vs-name discriminator is a path separator, or a leading `~`: an
/// argument containing `/` (or starting with `~`) is a path, anything else
/// is a name looked up under `<auris_home>/models/`.
fn resolve_model_source(
    model_flag: Option<&str>,
    auris_model_env: Option<&str>,
    auris_home: &Path,
    home: Option<&str>,
) -> ModelSource {
    if let Some(m) = model_flag {
        let path = if m.contains('/') || m.starts_with('~') {
            expand_tilde(m, home)
        } else {
            auris_home.join("models").join(m)
        };
        return ModelSource::Explicit(path);
    }
    if let Some(e) = auris_model_env {
        return ModelSource::Explicit(expand_tilde(e, home));
    }
    ModelSource::Default(auris_home.join("models").join(DEFAULT_MODEL_NAME))
}

/// A model directory is "installed" only when all five required files are
/// present (README "The cache") — a half-finished download or a
/// not-yet-generated `bpe.vocab` must be invisible, not offered and then
/// broken.
fn model_dir_is_complete(dir: &Path) -> bool {
    dir.is_dir() && REQUIRED_MODEL_FILES.iter().all(|f| dir.join(f).is_file())
}

/// Scans `<auris_home>/models/` for complete model directories, sorted by
/// name. Never touches the network, never fails: a missing or unreadable
/// models directory is just an empty list, matching README
/// "`--no-download`"'s promise that `--list-models` "exits 0 even with
/// nothing installed" and must never hang.
fn list_models(auris_home: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(auris_home.join("models"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| model_dir_is_complete(&entry.path()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    names.sort();
    names
}

/// Builds the `segment` line of `--format json` output (README "`--format
/// json`"). `end_seconds` is the decoded utterance's duration; a single
/// utterance always starts at 0.0, since per-utterance segmentation within a
/// stream is the later streaming task, not this one.
fn segment_line(text: &str, end_seconds: f64) -> String {
    serde_json::json!({
        "type": "segment",
        "index": 0,
        "text": text,
        "start": 0.0,
        "end": end_seconds,
    })
    .to_string()
}

/// Builds the `transcript` line of `--format json` output — always the last
/// line on a run that produced one (README "`--format json`").
fn transcript_line(text: &str) -> String {
    serde_json::json!({
        "type": "transcript",
        "text": text,
    })
    .to_string()
}

fn run_transcribe(args: Args) -> i32 {
    let interrupt = install_interrupt_handler();
    let verbose = !args.quiet && std::io::stderr().is_terminal();
    let home_env = std::env::var("HOME").ok();
    let home = auris_home(home_env.as_deref());

    if args.list_models {
        // Never touches the network, never loads a recognizer, never hangs
        // (README "`--no-download`") — --no-download changes nothing here,
        // because there is nothing for it to refuse.
        for name in list_models(&home) {
            println!("{name}");
        }
        return OK;
    }

    if !(0.0..=1.0).contains(&args.vad_threshold) {
        eprintln!(
            "auris: --vad-threshold must be between 0.0 and 1.0, got {}",
            args.vad_threshold
        );
        return USAGE;
    }
    if !args.vad_min_speech.is_finite() || args.vad_min_speech < 0.0 {
        eprintln!(
            "auris: --vad-min-speech must be a non-negative number of seconds, got {}",
            args.vad_min_speech
        );
        return USAGE;
    }

    let mut input: Box<dyn Read> = if let Some(path) = &args.path {
        match std::fs::File::open(path) {
            Ok(f) => Box::new(f),
            Err(e) => {
                eprintln!("auris: failed to open {}: {e}", path.display());
                return USAGE;
            }
        }
    } else {
        // Audio never reaches the argument parser (README "stdin"); the
        // usage error fires iff PATH is omitted and stdin is a terminal,
        // never merely because a PATH was given.
        if std::io::stdin().is_terminal() {
            eprintln!("auris: no input given; pass a file path or pipe audio in (auris --help)");
            return USAGE;
        }
        Box::new(std::io::stdin())
    };

    let samples = match audio::decode(&mut input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("auris: {e}");
            return e.exit_code();
        }
    };
    if interrupt.load(Ordering::SeqCst) {
        return INTERRUPTED;
    }

    let env_model = std::env::var("AURIS_MODEL").ok();
    let source = resolve_model_source(
        args.model.as_deref(),
        env_model.as_deref(),
        &home,
        home_env.as_deref(),
    );
    let model_dir = match source {
        ModelSource::Default(path) => {
            if !model_dir_is_complete(&path) {
                if args.no_download {
                    eprintln!(
                        "auris: model {DEFAULT_MODEL_NAME} is not installed at {}; run without --no-download, or run `auris serve`, to fetch it",
                        path.display()
                    );
                    return NOTHING_TRANSCRIBED;
                }
                // The client fetches in its own process, before it spawns or
                // contacts the daemon: a spawned daemon only gets
                // CLIENT_CONNECT_TIMEOUT (README "The daemon",
                // src/daemon.rs) to become reachable, far too short for a
                // ~661 MB download, and this process's stderr is the
                // terminal the caller is actually watching.
                if let Err(e) = model::ensure_installed(&path, verbose) {
                    eprintln!("auris: {e}");
                    return NOTHING_TRANSCRIBED;
                }
            }
            path
        }
        // A name or path the caller gave explicitly: let Recognizer::load's
        // own validation (missing dir, missing file, missing bpe.vocab) do
        // the checking, so its USAGE-error messages stay in one place. An
        // explicit -m/AURIS_MODEL is never a trigger to download (README
        // "Which model, and `-m`").
        ModelSource::Explicit(path) => path,
    };
    if interrupt.load(Ordering::SeqCst) {
        return INTERRUPTED;
    }

    // The energy gate (`audio::is_silent`) runs here — after model
    // resolution, so a missing model still produces its existing error and
    // exit code above, but before the recognizer is ever reached. It covers
    // both the daemon and `--no-daemon` paths, since both decode from this
    // same `samples`. The real Parakeet model hallucinates "Okay." on
    // digital silence rather than returning nothing, which would otherwise
    // slip past the post-decode `text.trim().is_empty()` check below and
    // break README "Exit codes"'s exit-1-on-silence contract.
    if audio::is_silent(&samples) {
        eprintln!("{NOTHING_TRANSCRIBED_MSG}");
        if verbose {
            eprintln!("auris: silence gate tripped; the recognizer was not run");
        }
        return NOTHING_TRANSCRIBED;
    }

    // Silero VAD runs client-side, right here — the same spot `is_silent`
    // just ran, so this one gate covers both the daemon and `--no-daemon`
    // paths identically. It is a decision, not a filter (`src/vad.rs`'s
    // module doc comment): it only decides whether to run the recognizer at
    // all, and `samples` reaching the recognizer below is never touched by
    // it — an earlier design that trimmed audio down to the detected speech
    // span broke real transcripts (35 of 40 files in a sweep changed from
    // `--no-vad`), so the recognizer always sees the same, complete buffer
    // it always did. `--no-vad` is the escape hatch (mirroring
    // `--no-daemon`/`--no-download`): audio mesa already segments needs no
    // second pass.
    if !args.no_vad {
        let vad_path = model::vad_model_path(&home);
        if !vad_path.is_file() {
            if args.no_download {
                eprintln!(
                    "auris: VAD model {} is not installed at {}; run without --no-download, or run `auris serve`, to fetch it",
                    model::VAD_FILENAME,
                    vad_path.display()
                );
                return NOTHING_TRANSCRIBED;
            }
            if let Err(e) = model::ensure_vad_installed(&home, verbose) {
                eprintln!("auris: {e}");
                return NOTHING_TRANSCRIBED;
            }
        }
        if interrupt.load(Ordering::SeqCst) {
            return INTERRUPTED;
        }

        let vad = match vad::Vad::load(&vad::VadConfig {
            model: vad_path,
            threshold: args.vad_threshold,
            min_speech: args.vad_min_speech,
        }) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        };
        if !vad.has_speech(&samples) {
            eprintln!("{NOTHING_TRANSCRIBED_MSG}");
            if verbose {
                eprintln!("auris: no speech detected; the recognizer was not run");
            }
            return NOTHING_TRANSCRIBED;
        }
    }

    // Parsed and validated before the ~4 s model load, so a bad
    // --vocabulary-file fails fast (README "`--vocabulary-file`",
    // docs/vocabulary.md "Validation"). The cap warning is a diagnostic, not
    // progress, so it goes to stderr unconditionally, not just under
    // `verbose` (CLAUDE.md "stderr").
    let vocabulary = match &args.vocabulary_file {
        Some(path) => match Vocabulary::load(path) {
            Ok(v) => {
                if v.terms_dropped > 0 {
                    eprintln!(
                        "auris: vocabulary file exceeds {} terms; dropped {} in file order",
                        crate::vocabulary::MAX_TERMS,
                        v.terms_dropped
                    );
                }
                // A vocabulary with no terms (an empty file, or one that's
                // only comments and blank lines) means exactly what no
                // vocabulary means. Short-circuit here rather than handing
                // sherpa-onnx a wholly-empty hotwords string — an
                // unexercised path, not the same as the safely-ignored
                // empty *segments* an assembled non-empty string can carry.
                if v.terms.is_empty() { None } else { Some(v) }
            }
            Err(e) => {
                eprintln!("auris: {e}");
                return USAGE;
            }
        },
        None => None,
    };
    if interrupt.load(Ordering::SeqCst) {
        return INTERRUPTED;
    }

    let hotwords_string = vocabulary.as_ref().map(|v| v.hotwords_string());
    let socket_path = args
        .socket
        .clone()
        .unwrap_or_else(|| home.join("auris.sock"));

    // The default path talks to the daemon (README "The daemon"), starting
    // one if none is listening, so the ~4 s model load is paid once per
    // daemon lifetime rather than once per call. `--no-daemon` is the
    // escape hatch that keeps the original in-process behaviour verbatim.
    // The `--no-daemon` recognizer is kept in scope past this branch (rather
    // than dropped at the end of an `if`/`else` as before) so the
    // manufactured-vocabulary guard below can run its confirming decode on
    // the same already-loaded recognizer instead of paying a second ~4 s
    // model load.
    let recognizer = if args.no_daemon {
        if verbose {
            eprintln!("auris: loading {}", model_dir.display());
        }
        let load_start = Instant::now();
        let recognizer = match Recognizer::load(&EngineConfig {
            model_dir: model_dir.clone(),
            ..Default::default()
        }) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        };
        if verbose {
            eprintln!("auris: model loaded in {:?}", load_start.elapsed());
        }
        if interrupt.load(Ordering::SeqCst) {
            return INTERRUPTED;
        }
        Some(recognizer)
    } else {
        None
    };

    let text = if let Some(recognizer) = &recognizer {
        let decode_start = Instant::now();
        let result = match &hotwords_string {
            Some(h) => recognizer.decode_with_hotwords(&samples, h),
            None => recognizer.decode(&samples),
        };
        let text = match result {
            Ok(t) => t,
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        };
        if verbose {
            eprintln!("auris: decoded in {:?}", decode_start.elapsed());
        }
        text
    } else {
        let hotwords = hotwords_string.as_deref().unwrap_or("");
        match daemon::transcribe_via_daemon(&socket_path, &model_dir, &samples, hotwords) {
            Ok((text, decode_ms)) => {
                if verbose {
                    eprintln!("auris: decoded in {decode_ms:.0}ms");
                }
                text
            }
            Err(e) => {
                eprintln!("auris: {e}");
                return e.code;
            }
        }
    };
    if interrupt.load(Ordering::SeqCst) {
        return INTERRUPTED;
    }

    let text = text.trim();
    if text.is_empty() {
        // No blank line, no JSON — a run with no transcript writes nothing
        // to stdout at all (README "stdout"). The stderr line still exists
        // so mesa can tell "you didn't say anything" from a broken install,
        // which a silent exit 1 cannot distinguish.
        eprintln!("{NOTHING_TRANSCRIBED_MSG}");
        if verbose {
            eprintln!("auris: the recognizer ran and returned an empty transcript");
        }
        return NOTHING_TRANSCRIBED;
    }

    // The manufactured-vocabulary guard (mesa task 970): hotword biasing is
    // meant to nudge an existing hypothesis toward the vocabulary, not to
    // manufacture a transcript out of nothing, but on non-speech audio it
    // can do exactly that — a confident, wholly invented transcript built
    // almost entirely out of boosted terms (e.g. "The vegetable mesa mesa
    // khora khora khora ... mesa khan mesa q" from a transient click).
    // Neither lowering the boost nor trusting the heuristic below as a
    // decision worked (measured: the hallucination is non-monotonic in
    // boost, and real speech has its own legitimate vocabulary hits), so
    // this guard is a prefilter plus a confirming decode, mirroring
    // `src/vad.rs`'s "decision, not a filter" shape: `looks_manufactured`
    // only decides whether to spend a second, unbiased decode of the exact
    // same audio. If that unbiased decode also comes back with something,
    // the biased transcript stands unchanged — biasing nudged a real
    // hypothesis, it didn't invent one. Only when the unbiased decode comes
    // back empty is the biased transcript discarded. This is what makes the
    // prefilter safe to run unconditionally despite being blunt: it cannot
    // change the output for any audio that decodes to something unbiased,
    // so a false trigger costs one wasted decode and never a wrong or
    // altered transcript. (A separate, out-of-scope failure mode: at boost
    // 8.0 the same non-speech fixtures produce short runs of digit/letter
    // garbage containing no vocabulary terms at all, which this guard
    // cannot and does not catch.)
    if let Some(v) = &vocabulary
        && looks_manufactured(text, &v.terms)
    {
        let confirmation: Result<String, String> = match &recognizer {
            Some(recognizer) => recognizer.decode(&samples).map_err(|e| e.to_string()),
            None => daemon::transcribe_via_daemon(&socket_path, &model_dir, &samples, "")
                .map(|(text, _)| text)
                .map_err(|e| e.to_string()),
        };
        // A failed confirming decode is not evidence of hallucination — fall
        // through and emit the biased transcript as if the guard had never
        // triggered, same as `Ok(_)` non-empty below.
        if let Ok(unbiased) = confirmation
            && unbiased.trim().is_empty()
        {
            eprintln!("{NOTHING_TRANSCRIBED_MSG}");
            if verbose {
                eprintln!(
                    "auris: the transcript was made only of boosted vocabulary terms and \
                     the same audio decodes to nothing without the vocabulary; discarded"
                );
            }
            return NOTHING_TRANSCRIBED;
        }
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match args.format {
        Format::Text => {
            let _ = writeln!(out, "{text}");
        }
        Format::Json => {
            let duration = samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;
            let _ = writeln!(out, "{}", segment_line(text, duration));
            let _ = out.flush();
            let _ = writeln!(out, "{}", transcript_line(text));
            let _ = out.flush();
        }
    }
    // main.rs hands our return value straight to std::process::exit, which
    // skips destructors — flush explicitly rather than trust stdout's Drop.
    let _ = out.flush();

    OK
}

/// `auris serve` (README "The daemon", "Synopsis"): resolves the model the
/// same way `run_transcribe` does, then hands off to
/// [`daemon::serve`] for the accept loop.
fn run_serve(args: ServeArgs) -> i32 {
    let verbose = !args.quiet && std::io::stderr().is_terminal();
    let home_env = std::env::var("HOME").ok();
    let home = auris_home(home_env.as_deref());

    let env_model = std::env::var("AURIS_MODEL").ok();
    let source = resolve_model_source(
        args.model.as_deref(),
        env_model.as_deref(),
        &home,
        home_env.as_deref(),
    );
    let model_dir = match source {
        ModelSource::Default(path) => {
            if !model_dir_is_complete(&path) {
                if args.no_download {
                    eprintln!(
                        "auris: model {DEFAULT_MODEL_NAME} is not installed at {}; run without --no-download to fetch it",
                        path.display()
                    );
                    return NOTHING_TRANSCRIBED;
                }
                if let Err(e) = model::ensure_installed(&path, verbose) {
                    eprintln!("auris: {e}");
                    return NOTHING_TRANSCRIBED;
                }
            }
            path
        }
        // A name or path the caller gave explicitly: let Recognizer::load's
        // own validation do the checking, same as run_transcribe.
        ModelSource::Explicit(path) => path,
    };

    // "Run `auris serve` once after install" is the complete install-time
    // fetch step (README "Getting the model"), so the VAD model is fetched
    // here too, not only on first transcribe.
    let vad_path = model::vad_model_path(&home);
    if !vad_path.is_file() {
        if args.no_download {
            eprintln!(
                "auris: VAD model {} is not installed at {}; run without --no-download to fetch it",
                model::VAD_FILENAME,
                vad_path.display()
            );
            return NOTHING_TRANSCRIBED;
        }
        if let Err(e) = model::ensure_vad_installed(&home, verbose) {
            eprintln!("auris: {e}");
            return NOTHING_TRANSCRIBED;
        }
    }

    let socket_path = args
        .socket
        .clone()
        .unwrap_or_else(|| home.join("auris.sock"));

    daemon::serve(daemon::ServeConfig {
        model_dir,
        socket_path,
        verbose,
    })
}

/// `auris status` (README "The daemon", "Synopsis"): prints the daemon's
/// status line to stdout — it is the answer the caller asked for, not
/// progress (CLAUDE.md "stdout"). No daemon reachable is exit 1, mirroring
/// README "Exit codes"' "a daemon that could not be reached or started".
fn run_status(args: SocketArgs) -> i32 {
    let home_env = std::env::var("HOME").ok();
    let home = auris_home(home_env.as_deref());
    let socket_path = args.socket.unwrap_or_else(|| home.join("auris.sock"));

    match daemon::status(&socket_path) {
        Ok(info) => {
            println!(
                "model {}  pid {}  uptime {}s  requests {}",
                info.model, info.pid, info.uptime_secs, info.requests
            );
            OK
        }
        Err(_) => {
            eprintln!("auris: no daemon on {}", socket_path.display());
            NOTHING_TRANSCRIBED
        }
    }
}

/// `auris stop` (README "The daemon", "Synopsis"): asks the daemon to exit.
/// No daemon reachable is exit 1, same as [`run_status`].
fn run_stop(args: SocketArgs) -> i32 {
    let home_env = std::env::var("HOME").ok();
    let home = auris_home(home_env.as_deref());
    let socket_path = args.socket.unwrap_or_else(|| home.join("auris.sock"));

    match daemon::stop(&socket_path) {
        Ok(()) => OK,
        Err(_) => {
            eprintln!("auris: no daemon on {}", socket_path.display());
            NOTHING_TRANSCRIBED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_flag_with_slash_is_a_path() {
        let home = PathBuf::from("/home/x/.cache/auris");
        let source = resolve_model_source(Some("./local/model"), None, &home, None);
        assert_eq!(
            source,
            ModelSource::Explicit(PathBuf::from("./local/model"))
        );
    }

    #[test]
    fn model_flag_with_tilde_is_a_path_and_expands() {
        let home = PathBuf::from("/home/tildetest/.cache/auris");
        let source =
            resolve_model_source(Some("~/models/mine"), None, &home, Some("/home/tildetest"));
        assert_eq!(
            source,
            ModelSource::Explicit(PathBuf::from("/home/tildetest/models/mine"))
        );
    }

    #[test]
    fn model_flag_without_slash_is_a_name_under_auris_home() {
        let home = PathBuf::from("/home/x/.cache/auris");
        let source = resolve_model_source(Some("parakeet-tdt-0.6b-v2-int8"), None, &home, None);
        assert_eq!(
            source,
            ModelSource::Explicit(home.join("models").join("parakeet-tdt-0.6b-v2-int8"))
        );
    }

    #[test]
    fn model_flag_beats_env_var() {
        let home = PathBuf::from("/home/x/.cache/auris");
        let source = resolve_model_source(Some("from-flag"), Some("/from/env"), &home, None);
        assert_eq!(
            source,
            ModelSource::Explicit(home.join("models").join("from-flag"))
        );
    }

    #[test]
    fn env_var_beats_default_and_is_always_a_path() {
        let home = PathBuf::from("/home/x/.cache/auris");
        let source = resolve_model_source(None, Some("also-no-slash"), &home, None);
        assert_eq!(
            source,
            ModelSource::Explicit(PathBuf::from("also-no-slash"))
        );
    }

    #[test]
    fn default_name_used_when_nothing_given() {
        let home = PathBuf::from("/home/x/.cache/auris");
        let source = resolve_model_source(None, None, &home, None);
        assert_eq!(
            source,
            ModelSource::Default(home.join("models").join(DEFAULT_MODEL_NAME))
        );
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn list_models_lists_only_complete_dirs_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();

        // Complete: every required file present.
        let complete = models.join("zeta-complete");
        std::fs::create_dir_all(&complete).unwrap();
        for f in REQUIRED_MODEL_FILES {
            touch(&complete.join(f));
        }

        // Incomplete: missing bpe.vocab.
        let incomplete = models.join("alpha-incomplete");
        std::fs::create_dir_all(&incomplete).unwrap();
        for f in [
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ] {
            touch(&incomplete.join(f));
        }

        let names = list_models(tmp.path());
        assert_eq!(names, vec!["zeta-complete".to_string()]);
    }

    #[test]
    fn list_models_on_missing_directory_is_empty_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        // tmp.path()/models does not exist at all.
        let names = list_models(tmp.path());
        assert!(names.is_empty());
    }

    #[test]
    fn json_segment_line_has_the_readme_shape() {
        let line = segment_line("book a call with khora for tomorrow", 2.14);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "segment");
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "book a call with khora for tomorrow");
        assert_eq!(v["start"], 0.0);
        assert_eq!(v["end"], 2.14);
    }

    #[test]
    fn json_transcript_line_has_the_readme_shape() {
        let line = transcript_line("book a call with khora for tomorrow");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "transcript");
        assert_eq!(v["text"], "book a call with khora for tomorrow");
        // Only two fields on this line: no leftover segment fields.
        assert_eq!(v.as_object().unwrap().len(), 2);
    }

    #[test]
    fn json_lines_escape_quotes_correctly() {
        // The reason serde_json builds this instead of hand-rolled string
        // formatting: text containing a literal quote must round-trip.
        let line = transcript_line(r#"she said "khora" clearly"#);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["text"], r#"she said "khora" clearly"#);
    }
}
