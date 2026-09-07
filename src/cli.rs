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

    /// Trailing silence that ends an utterance — where one `segment` line
    /// stops and the next begins
    #[arg(long, value_name = "SECS", default_value_t = vad::VadConfig::default().min_silence)]
    vad_min_silence: f32,

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
/// json`"). `index` counts utterances from 0 and is monotonic; `start` and
/// `end` are that utterance's own speech boundaries in seconds from the
/// start of the stream — *not* the bounds of the slice handed to the
/// recognizer, which deliberately reaches further on both sides
/// (`src/vad.rs`, "decision, not a filter").
fn segment_line(index: usize, text: &str, start_seconds: f64, end_seconds: f64) -> String {
    serde_json::json!({
        "type": "segment",
        "index": index,
        "text": text,
        "start": start_seconds,
        "end": end_seconds,
    })
    .to_string()
}

/// Builds the `speech` heartbeat line (`docs/streaming.md`): the
/// speech-activity event mesa's silence timer consumes in place of the
/// partial hypotheses that document rejects, computed by the VAD pass that
/// is running anyway and costing no decode at all.
///
/// `at` is the stream-time second at which speech started or stopped. For
/// the closing line that is exactly the segment's `end` — when speech
/// stopped, not when the VAD noticed, which `docs/latency.md` makes binding
/// so mesa can backdate `heardAt` instead of starting its 2000 ms wait
/// 500 ms late. The opening line is the one place auris cannot backdate:
/// `detected()` is polled as chunks arrive and the detector does not report
/// where the span it has opened began until it closes, so `at` there is
/// when auris noticed, which is what a heartbeat needs and all it needs.
fn speech_line(active: bool, at_seconds: f64) -> String {
    serde_json::json!({
        "type": "speech",
        "active": active,
        "at": at_seconds,
    })
    .to_string()
}

/// Writes one JSON-Lines (or plain text) line and flushes it, returning
/// `false` once stdout has gone away. Every line is flushed as it is
/// written — that is what "flushed to stdout as soon as that utterance
/// ends" means for a reader on the other end of a pipe (README "stdin",
/// `docs/streaming.md`).
///
/// The return value is not decoration: a consumer that stops reading (mesa
/// exiting, a `| head`) must not wedge auris. Rust ignores SIGPIPE, so the
/// write comes back `EPIPE` instead of killing the process, and the caller
/// stops decoding audio nobody will read rather than grinding on to EOF.
fn write_line(out: &mut impl Write, line: &str) -> bool {
    writeln!(out, "{line}").and_then(|()| out.flush()).is_ok()
}

/// Decodes one utterance and applies the manufactured-vocabulary guard to
/// what comes back. `Ok(None)` means this utterance produced no transcript
/// — an empty decode, or one the guard discarded; `Err` is a real engine or
/// daemon failure, carrying the message and the exit code it maps to.
///
/// `samples` is a slice of the *original* buffer, cut at neighbouring
/// utterance boundaries rather than at this one's edges (`src/vad.rs`), so
/// a single-utterance recording passes the whole buffer through here
/// exactly as the pre-segmentation code did.
#[allow(clippy::too_many_arguments)]
fn decode_utterance(
    samples: &[f32],
    recognizer: Option<&Recognizer>,
    socket_path: &Path,
    model_dir: &Path,
    hotwords: Option<&str>,
    vocabulary: Option<&Vocabulary>,
    verbose: bool,
) -> Result<Option<String>, (String, i32)> {
    let decode_start = Instant::now();
    let text = match recognizer {
        Some(recognizer) => {
            let result = match hotwords {
                Some(h) => recognizer.decode_with_hotwords(samples, h),
                None => recognizer.decode(samples),
            };
            let text = result.map_err(|e| (e.to_string(), e.exit_code()))?;
            if verbose {
                eprintln!("auris: decoded in {:?}", decode_start.elapsed());
            }
            text
        }
        None => {
            let (text, decode_ms) = daemon::transcribe_via_daemon(
                socket_path,
                model_dir,
                samples,
                hotwords.unwrap_or(""),
            )
            .map_err(|e| (e.to_string(), e.code))?;
            if verbose {
                eprintln!("auris: decoded in {decode_ms:.0}ms");
            }
            text
        }
    };

    let text = text.trim();
    if text.is_empty() {
        if verbose {
            eprintln!("auris: the recognizer ran and returned an empty transcript");
        }
        return Ok(None);
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
    if let Some(v) = vocabulary
        && looks_manufactured(text, &v.terms)
    {
        let confirmation: Result<String, String> = match recognizer {
            Some(recognizer) => recognizer.decode(samples).map_err(|e| e.to_string()),
            None => daemon::transcribe_via_daemon(socket_path, model_dir, samples, "")
                .map(|(text, _)| text)
                .map_err(|e| e.to_string()),
        };
        // A failed confirming decode is not evidence of hallucination — fall
        // through and emit the biased transcript as if the guard had never
        // triggered, same as `Ok(_)` non-empty below.
        if let Ok(unbiased) = confirmation
            && unbiased.trim().is_empty()
        {
            if verbose {
                eprintln!(
                    "auris: the transcript was made only of boosted vocabulary terms and \
                     the same audio decodes to nothing without the vocabulary; discarded"
                );
            }
            return Ok(None);
        }
    }

    Ok(Some(text.to_string()))
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
    if !args.vad_min_silence.is_finite() || args.vad_min_silence <= 0.0 {
        // Strictly positive, unlike --vad-min-speech: zero trailing silence
        // is not "no minimum", it is a detector that never closes a span,
        // which under segmentation means a stream that never produces a
        // `segment` line at all.
        eprintln!(
            "auris: --vad-min-silence must be a positive number of seconds, got {}",
            args.vad_min_silence
        );
        return USAGE;
    }

    let input: Box<dyn Read> = if let Some(path) = &args.path {
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

    // Only the header is read here, not the audio. That keeps every error
    // about the *stream itself* — "input was empty", "not a wav file", an
    // unsupported rate — ahead of everything about models and vocabularies,
    // exactly where it was when this line decoded the whole thing
    // (tests/errors.rs), while leaving the body to be pulled in as it
    // arrives (README "stdin": stdout's first byte must not wait for
    // stdin's EOF).
    let mut stream = match audio::WavStream::open(input) {
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
    let hotwords_string = vocabulary.as_ref().map(|v| v.hotwords_string());
    let socket_path = args
        .socket
        .clone()
        .unwrap_or_else(|| home.join("auris.sock"));
    if interrupt.load(Ordering::SeqCst) {
        return INTERRUPTED;
    }

    // The default path talks to the daemon (README "The daemon"), starting
    // one if none is listening, so the ~4 s model load is paid once per
    // daemon lifetime rather than once per call. `--no-daemon` is the
    // escape hatch that keeps the original in-process behaviour verbatim.
    //
    // Under segmentation this load moves ahead of reading the audio rather
    // than after it: an utterance can close a second into a live stream,
    // and four seconds of model load sitting between that instant and
    // stdout would spend the entire latency budget `docs/latency.md`
    // allots. The cost is that silent audio now pays for a load it will
    // never use on the `--no-daemon` path; the alternative — loading
    // lazily at the first utterance — would have moved every "this model
    // directory is broken" error behind the audio, which is a contract
    // (tests/errors.rs) rather than a preference.
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

    // Resampling is the one thing that cannot be done incrementally
    // (`audio::WavStream`), so a source that is not already at 16 kHz is
    // read to EOF here and then handed to the same loop below in chunks.
    // Nothing that can be segmented live is affected: every live capture
    // that reaches auris is already at 16 kHz.
    let mut buffered = if stream.sample_rate() == audio::TARGET_SAMPLE_RATE {
        None
    } else {
        let whole = match stream.rest_resampled() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        };
        Some(
            whole
                .chunks(audio::READ_CHUNK_FRAMES)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    };
    let mut read_chunk = move || -> Result<Vec<f32>, audio::AudioError> {
        match &mut buffered {
            Some(chunks) => Ok(chunks.next().unwrap_or_default()),
            None => stream.next_chunk(audio::READ_CHUNK_FRAMES),
        }
    };

    // Every sample ever read, at 16 kHz, kept whole and never modified: the
    // slices handed to the recognizer below are cut out of this and nothing
    // else (`src/vad.rs`, "decision, not a filter").
    let mut all: Vec<f32> = Vec::new();
    let mut at_eof = false;

    // The energy gate (`audio::is_silent`) runs here — after model
    // resolution, so a missing model still produces its existing error and
    // exit code above, but before the recognizer or the VAD is ever
    // reached. It covers both the daemon and `--no-daemon` paths, since
    // both decode from this same buffer. The real Parakeet model
    // hallucinates "Okay." on digital silence rather than returning
    // nothing, which would otherwise slip past the post-decode
    // `text.trim().is_empty()` check and break README "Exit codes"'s
    // exit-1-on-silence contract.
    //
    // It runs *incrementally*: reading stops at the first chunk that proves
    // the audio is not silence, so wholly silent input still never loads
    // the VAD model, let alone the recognizer — which is what keeps
    // `--no-download` on a silent clip reporting silence rather than a
    // missing VAD model. Checking each arriving chunk against the windows
    // it overlaps (hence the overlap back into the previous chunk) is the
    // same maximum over the same windows as one whole-buffer call.
    while !at_eof {
        if interrupt.load(Ordering::SeqCst) {
            return INTERRUPTED;
        }
        let chunk = match read_chunk() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        };
        if chunk.is_empty() {
            at_eof = true;
            break;
        }
        let overlap = all.len().saturating_sub(audio::SILENCE_WINDOW_SAMPLES - 1);
        all.extend_from_slice(&chunk);
        if !audio::is_silent(&all[overlap..]) {
            break;
        }
    }
    if all.is_empty() {
        // Zero samples is a distinct cause from silence — a WAV whose data
        // chunk is empty never had anything to be silent about.
        eprintln!("auris: {}", audio::AudioError::Empty);
        return NOTHING_TRANSCRIBED;
    }
    if at_eof && audio::is_silent(&all) {
        eprintln!("{NOTHING_TRANSCRIBED_MSG}");
        if verbose {
            eprintln!("auris: silence gate tripped; the recognizer was not run");
        }
        return NOTHING_TRANSCRIBED;
    }

    // Silero VAD runs client-side, right here — the same spot `is_silent`
    // just ran, so this one component covers both the daemon and
    // `--no-daemon` paths identically. It decides where each utterance ends
    // and nothing else: the recognizer is handed slices of `all` cut at
    // *neighbouring* utterance boundaries, never the spans Silero found, so
    // one utterance is still decoded as the whole buffer, byte for byte
    // (`src/vad.rs`'s module doc comment, and the 35-of-40 corruption
    // measured when an earlier design trimmed to the spans). `--no-vad` is
    // the escape hatch (mirroring `--no-daemon`/`--no-download`): audio
    // mesa already segments needs no second pass, and gets exactly one
    // segment covering the whole buffer.
    let vad = if args.no_vad {
        None
    } else {
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
        match vad::Vad::load(&vad::VadConfig {
            model: vad_path,
            threshold: args.vad_threshold,
            min_speech: args.vad_min_speech,
            min_silence: args.vad_min_silence,
        }) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("auris: {e}");
                return e.exit_code();
            }
        }
    };
    let mut segmenter = vad.as_ref().map(|v| v.segmenter());

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let json = args.format == Format::Json;
    let rate = audio::TARGET_SAMPLE_RATE as f64;

    // A closed utterance is held back until the *next* one opens, because
    // its right edge is the next utterance's start: that is what makes the
    // slice handed to the recognizer a partition of `all` rather than a
    // crop of it, and what makes a single-utterance recording decode as the
    // whole buffer exactly as it did before segmentation. The cost is one
    // utterance of delay on each `segment` line, which is why the `speech`
    // heartbeat is written at the moment the utterance closes rather than
    // alongside its `segment` line — mesa's silence timer consumes the
    // heartbeat, not the text, and every line is self-describing and
    // ordered by `index` (`docs/streaming.md`).
    let mut pending: Option<vad::Segment> = None;
    let mut slice_start = 0usize;
    let mut index = 0usize;
    let mut texts: Vec<String> = Vec::new();
    let mut speaking = false;
    let mut stdout_open = true;
    let mut failure: Option<(String, i32)> = None;

    let mut decoded_any = false;
    let mut fed = 0usize;
    // Utterances whose right edge is now known, ready to decode and write
    // in this pass: (utterance, slice start, slice end).
    let mut ready: Vec<(vad::Segment, usize, usize)> = Vec::new();

    'stream: loop {
        if interrupt.load(Ordering::SeqCst) {
            return INTERRUPTED;
        }

        match &mut segmenter {
            Some(segmenter) => {
                if fed < all.len() {
                    segmenter.accept(&all[fed..]);
                    fed = all.len();
                }
                if at_eof {
                    // Forces the span still open at end of stream closed, so
                    // a recording that stops mid-utterance still produces it.
                    segmenter.finish();
                }

                // The heartbeat is written the moment the VAD's own
                // `detected()` flips, not when the utterance's text is
                // ready: it exists so mesa's silence timer knows speech is
                // still happening, which is worth nothing after the fact.
                //
                // One divergence, deliberate and left alone: because these
                // lines are written before any utterance has decoded, a
                // consumer that closes stdout at the very start of a
                // `--format json` run exits 1 (nothing was ever committed
                // to stdout, which is exactly what exit 1 means) where the
                // same close on the text path exits 0 (the first thing
                // text mode writes is a transcript it has already
                // produced). Both are honest about what reached the
                // consumer; making them agree would mean either inventing
                // a transcript for the JSON case or suppressing the
                // heartbeat, and neither is worth it.
                if !speaking && segmenter.speaking() {
                    speaking = true;
                    if json && !write_line(&mut out, &speech_line(true, all.len() as f64 / rate)) {
                        stdout_open = false;
                        break 'stream;
                    }
                }

                while let Some(current) = segmenter.next_segment() {
                    if speaking {
                        speaking = false;
                        let at = (current.start + current.len) as f64 / rate;
                        if json && !write_line(&mut out, &speech_line(false, at)) {
                            stdout_open = false;
                            break 'stream;
                        }
                    }
                    if let Some(previous) = pending.replace(current) {
                        let cut = current.start.max(slice_start);
                        ready.push((previous, slice_start, cut));
                        slice_start = cut;
                    }
                }
                if at_eof && let Some(last) = pending.take() {
                    ready.push((last, slice_start, all.len()));
                }
            }
            // `--no-vad`: the whole buffer is one utterance, decoded once at
            // end of stream — verbatim what auris did before segmentation.
            None => {
                if at_eof {
                    ready.push((
                        vad::Segment {
                            start: 0,
                            len: all.len(),
                        },
                        0,
                        all.len(),
                    ));
                }
            }
        }

        for (utterance, from, to) in ready.drain(..) {
            decoded_any = true;
            let text = match decode_utterance(
                &all[from..to],
                recognizer.as_ref(),
                &socket_path,
                &model_dir,
                hotwords_string.as_deref(),
                vocabulary.as_ref(),
                verbose,
            ) {
                Ok(text) => text,
                Err(e) => {
                    failure = Some(e);
                    break 'stream;
                }
            };
            let Some(text) = text else {
                continue;
            };
            let line = if json {
                segment_line(
                    index,
                    &text,
                    utterance.start as f64 / rate,
                    (utterance.start + utterance.len) as f64 / rate,
                )
            } else {
                text.clone()
            };
            index += 1;
            texts.push(text);
            if !write_line(&mut out, &line) {
                stdout_open = false;
                break 'stream;
            }
        }

        if at_eof {
            break 'stream;
        }
        match read_chunk() {
            Ok(chunk) if chunk.is_empty() => at_eof = true,
            Ok(chunk) => all.extend_from_slice(&chunk),
            Err(e) => {
                failure = Some((e.to_string(), e.exit_code()));
                break 'stream;
            }
        }
    }

    if let Some((message, code)) = failure {
        eprintln!("auris: {message}");
        // A failure that lands after text has already been committed to
        // stdout must not become a nonzero exit: "nonzero with nothing on
        // stdout" is the one signal mesa treats as failure (README "Exit
        // codes"), and unsaying lines it has already acted on is exactly
        // what `docs/streaming.md`'s no-retraction rule forbids.
        if texts.is_empty() {
            return code;
        }
    } else if texts.is_empty() {
        // No blank line, no JSON — a run with no transcript writes nothing
        // to stdout at all (README "stdout"). The stderr line still exists
        // so mesa can tell "you didn't say anything" from a broken install,
        // which a silent exit 1 cannot distinguish.
        eprintln!("{NOTHING_TRANSCRIBED_MSG}");
        if verbose && !decoded_any {
            eprintln!("auris: no speech detected; the recognizer was not run");
        }
        return NOTHING_TRANSCRIBED;
    }

    // Always the last line on a run that produced one, so the simplest
    // possible reader — read to EOF, parse the last line — is a correct
    // reader (`docs/streaming.md`). With one utterance its text is the
    // `segment` line's, unchanged from before segmentation.
    if json && stdout_open {
        write_line(&mut out, &transcript_line(&texts.join(" ")));
    }

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
        let line = segment_line(0, "book a call with khora for tomorrow", 0.0, 2.14);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["type"], "segment");
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "book a call with khora for tomorrow");
        assert_eq!(v["start"], 0.0);
        assert_eq!(v["end"], 2.14);
    }

    /// `index` counts utterances, `start` is real stream time, and neither
    /// is the hardcoded 0 they were before segmentation (mesa task 936).
    #[test]
    fn json_segment_line_carries_a_real_index_and_start() {
        let line = segment_line(2, "and one more thing", 5.5, 7.25);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["index"], 2);
        assert_eq!(v["start"], 5.5);
        assert_eq!(v["end"], 7.25);
    }

    /// `docs/latency.md` makes this binding rather than incidental: the
    /// closing heartbeat's `at` is when speech stopped, so mesa can
    /// backdate its silence timer instead of starting it a whole
    /// `--vad-min-silence` late.
    #[test]
    fn json_speech_line_closes_at_the_segment_end() {
        let segment = segment_line(0, "book a call with khora for tomorrow", 0.32, 2.46);
        let closing = speech_line(false, 2.46);
        let segment: serde_json::Value = serde_json::from_str(&segment).unwrap();
        let closing: serde_json::Value = serde_json::from_str(&closing).unwrap();
        assert_eq!(closing["type"], "speech");
        assert_eq!(closing["active"], false);
        assert_eq!(closing["at"], segment["end"]);

        let opening: serde_json::Value = serde_json::from_str(&speech_line(true, 1.20)).unwrap();
        assert_eq!(opening["active"], true);
        assert_eq!(opening["at"], 1.20);
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
