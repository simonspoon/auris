//! The persistent daemon (`auris serve`) that holds one loaded
//! [`crate::engine::Recognizer`] warm for the process lifetime, and the
//! client-side plumbing a bare `auris` invocation uses to talk to it
//! (README "The daemon"). The daemon pays the ~4 s model load once; every
//! request after that is a cheap decode against the already-warm recognizer
//! — the whole reason this module exists (`docs/engine.md`).
//!
//! ## Wire framing
//!
//! One connection carries exactly one request and one response. A request
//! is one line of JSON terminated by `\n`, followed — for `transcribe`
//! only — by exactly `samples * 4` bytes of little-endian `f32` audio:
//!
//! ```text
//! {"op":"transcribe","samples":N,"model":"<absolute model dir>","hotwords":"<string, may be empty>"}
//! {"op":"status"}
//! {"op":"stop"}
//! ```
//!
//! A response is one line of JSON terminated by `\n`:
//!
//! ```text
//! {"ok":true,"text":"...","decode_ms":<f64>}                          // transcribe
//! {"ok":true,"model":"...","pid":N,"uptime_secs":N,"requests":N}      // status
//! {"ok":true}                                                         // stop (daemon then exits)
//! {"ok":false,"code":<exit code>,"message":"..."}                     // any failure
//! ```
//!
//! The client decodes audio ([`crate::audio::decode`]) and parses the
//! vocabulary ([`crate::vocabulary::Vocabulary`]) before ever connecting, so
//! every format/usage error keeps the exact message and exit code it has
//! today (`src/cli.rs`) and the daemon only ever handles plain `f32`
//! samples — audio never reaches the daemon in any form other than that
//! (README "stdin").

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::engine::{EngineConfig, Recognizer};

/// Exit codes, mirroring `src/cli.rs`'s copy of kokoro-rs's (README "Exit
/// codes"). Duplicated rather than imported so this module stays usable
/// without depending on `cli`'s internals — the daemon is the lower layer.
const OK: i32 = 0;
const NOTHING_TRANSCRIBED: i32 = 1;
const USAGE: i32 = 2;

/// No request for this long and an idle daemon exits (README "The daemon")
/// — a warm ~1.5 GB process is not held forever on a machine that
/// transcribes once and goes quiet.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// How often the accept loop polls the nonblocking listener between idle-
/// timeout checks.
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How often a client retries connecting to a socket it just asked a spawned
/// daemon to create.
const CLIENT_CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// How long a client waits for a spawned daemon to become reachable before
/// giving up (README "a daemon that could not be reached or started" is
/// exit 1).
const CLIENT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// A sanity bound on a `transcribe` request's declared `samples` count — one
/// hour of 16 kHz mono audio — not a policy limit (mesa's own 25 MB
/// per-request cap is mesa's, documented in docs/posture.md, not auris's).
/// This exists only so a bogus or hostile `samples` value in the header
/// can't make the daemon allocate an unbounded buffer before it has read a
/// single payload byte.
const MAX_TRANSCRIBE_SAMPLES: usize = crate::audio::TARGET_SAMPLE_RATE as usize * 3600;

/// How long the daemon waits for a payload byte on an accepted connection
/// before giving up on it. Bounds a client that sends a `transcribe` header
/// and then stalls: without this, `read_exact` below blocks forever and,
/// because connections are served one at a time, wedges the daemon for
/// every other caller too (including `status` and `stop`).
const CONNECTION_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// A failure to complete an RPC — connect, spawn, write, read, or a
/// `{"ok":false,...}` response relayed from the daemon. Carries the exit
/// code the caller should return, same split as [`crate::engine::EngineError`].
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonError {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for DaemonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Resolves a relative model directory to an absolute one where possible, so
/// the same model given as `-m ./foo` and `-m /abs/foo` compares equal
/// between what a client sends and what the daemon loaded. Falls back to the
/// input unchanged if it cannot be canonicalized (e.g. it does not exist) —
/// callers only reach this after `model_dir_is_complete`/`Recognizer::load`
/// has already validated existence for the path that matters.
fn canonical_or_self(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

// ---------------------------------------------------------------------
// Request / response framing
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    Transcribe {
        samples: usize,
        model: PathBuf,
        hotwords: String,
    },
    Status,
    Stop,
}

pub fn encode_request(req: &Request) -> String {
    match req {
        Request::Transcribe {
            samples,
            model,
            hotwords,
        } => serde_json::json!({
            "op": "transcribe",
            "samples": samples,
            "model": model.to_string_lossy(),
            "hotwords": hotwords,
        })
        .to_string(),
        Request::Status => serde_json::json!({"op": "status"}).to_string(),
        Request::Stop => serde_json::json!({"op": "stop"}).to_string(),
    }
}

#[derive(Debug)]
pub enum RequestParseError {
    Json(serde_json::Error),
    MissingField(&'static str),
    UnknownOp(String),
}

impl std::fmt::Display for RequestParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestParseError::Json(e) => write!(f, "invalid request: {e}"),
            RequestParseError::MissingField(field) => {
                write!(f, "request missing field {field:?}")
            }
            RequestParseError::UnknownOp(op) => write!(f, "unknown op {op:?}"),
        }
    }
}

/// Parses one request line (README wire framing above). `line` may carry a
/// trailing `\n` — trimmed before parsing.
pub fn parse_request(line: &str) -> Result<Request, RequestParseError> {
    let v: serde_json::Value =
        serde_json::from_str(line.trim_end()).map_err(RequestParseError::Json)?;
    let op = v
        .get("op")
        .and_then(|o| o.as_str())
        .ok_or(RequestParseError::MissingField("op"))?;
    match op {
        "transcribe" => {
            let samples =
                v.get("samples")
                    .and_then(|s| s.as_u64())
                    .ok_or(RequestParseError::MissingField("samples"))? as usize;
            let model = v
                .get("model")
                .and_then(|m| m.as_str())
                .ok_or(RequestParseError::MissingField("model"))?;
            let hotwords = v
                .get("hotwords")
                .and_then(|h| h.as_str())
                .unwrap_or("")
                .to_string();
            Ok(Request::Transcribe {
                samples,
                model: PathBuf::from(model),
                hotwords,
            })
        }
        "status" => Ok(Request::Status),
        "stop" => Ok(Request::Stop),
        other => Err(RequestParseError::UnknownOp(other.to_string())),
    }
}

pub struct StatusInfo {
    pub model: String,
    pub pid: u32,
    pub uptime_secs: u64,
    pub requests: u64,
}

enum Response<'a> {
    Transcribed { text: &'a str, decode_ms: f64 },
    Status(&'a StatusInfo),
    Stopped,
    Err { code: i32, message: &'a str },
}

fn encode_response(resp: &Response) -> String {
    match resp {
        Response::Transcribed { text, decode_ms } => serde_json::json!({
            "ok": true,
            "text": text,
            "decode_ms": decode_ms,
        })
        .to_string(),
        Response::Status(info) => serde_json::json!({
            "ok": true,
            "model": info.model,
            "pid": info.pid,
            "uptime_secs": info.uptime_secs,
            "requests": info.requests,
        })
        .to_string(),
        Response::Stopped => serde_json::json!({"ok": true}).to_string(),
        Response::Err { code, message } => serde_json::json!({
            "ok": false,
            "code": code,
            "message": message,
        })
        .to_string(),
    }
}

/// Parses a response line into either the `{"ok":false,...}` error it always
/// carries or the `serde_json::Value` of a successful response, leaving the
/// op-specific field extraction to each call site (a response carries no
/// `op` field of its own — the caller already knows what it asked for).
fn parse_response_line(line: &str) -> Result<serde_json::Value, DaemonError> {
    let v: serde_json::Value = serde_json::from_str(line.trim_end()).map_err(|e| DaemonError {
        code: NOTHING_TRANSCRIBED,
        message: format!("bad response from daemon: {e}"),
    })?;
    if v.get("ok").and_then(|o| o.as_bool()) == Some(true) {
        return Ok(v);
    }
    let code = v
        .get("code")
        .and_then(|c| c.as_i64())
        .unwrap_or(NOTHING_TRANSCRIBED as i64) as i32;
    let message = v
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("daemon error")
        .to_string();
    Err(DaemonError { code, message })
}

fn parse_transcribe_response(line: &str) -> Result<(String, f64), DaemonError> {
    let v = parse_response_line(line)?;
    let text = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
    let decode_ms = v.get("decode_ms").and_then(|d| d.as_f64()).unwrap_or(0.0);
    Ok((text.to_string(), decode_ms))
}

fn parse_status_response(line: &str) -> Result<StatusInfo, DaemonError> {
    let v = parse_response_line(line)?;
    let malformed = || DaemonError {
        code: NOTHING_TRANSCRIBED,
        message: "malformed status response from daemon".to_string(),
    };
    Ok(StatusInfo {
        model: v
            .get("model")
            .and_then(|m| m.as_str())
            .ok_or_else(malformed)?
            .to_string(),
        pid: v
            .get("pid")
            .and_then(|p| p.as_u64())
            .ok_or_else(malformed)? as u32,
        uptime_secs: v
            .get("uptime_secs")
            .and_then(|u| u.as_u64())
            .ok_or_else(malformed)?,
        requests: v
            .get("requests")
            .and_then(|r| r.as_u64())
            .ok_or_else(malformed)?,
    })
}

fn parse_stop_response(line: &str) -> Result<(), DaemonError> {
    parse_response_line(line).map(|_| ())
}

// ---------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------

/// Connects to `socket_path`; if nothing is listening, spawns `auris serve`
/// against it (with `model_dir` so the spawned daemon loads the same model
/// the caller asked for) and polls until it comes up or
/// [`CLIENT_CONNECT_TIMEOUT`] elapses (README "The daemon").
fn connect_or_spawn(socket_path: &Path, model_dir: &Path) -> Result<UnixStream, DaemonError> {
    if let Ok(stream) = UnixStream::connect(socket_path) {
        return Ok(stream);
    }

    let exe = std::env::current_exe().map_err(|e| DaemonError {
        code: NOTHING_TRANSCRIBED,
        message: format!("could not start daemon: {e}"),
    })?;
    // The spawned `Child` is intentionally dropped without being waited on:
    // this process exits shortly after its own RPC completes, and the
    // daemon it just started reparents to init and keeps running
    // independently — there is nothing here that should, or needs to,
    // outlive this call.
    Command::new(exe)
        .arg("serve")
        .arg("--socket")
        .arg(socket_path)
        .arg("-m")
        .arg(model_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| DaemonError {
            code: NOTHING_TRANSCRIBED,
            message: format!("could not start daemon: {e}"),
        })?;

    let deadline = Instant::now() + CLIENT_CONNECT_TIMEOUT;
    loop {
        if let Ok(stream) = UnixStream::connect(socket_path) {
            return Ok(stream);
        }
        if Instant::now() >= deadline {
            return Err(DaemonError {
                code: NOTHING_TRANSCRIBED,
                message: format!(
                    "could not start daemon: timed out waiting for {}",
                    socket_path.display()
                ),
            });
        }
        std::thread::sleep(CLIENT_CONNECT_RETRY_INTERVAL);
    }
}

fn write_io_err(e: std::io::Error) -> DaemonError {
    DaemonError {
        code: NOTHING_TRANSCRIBED,
        message: format!("daemon connection failed: {e}"),
    }
}

/// Transcribes one utterance against the daemon at `socket_path`, starting
/// it first if nothing is listening. `hotwords` empty means "decode
/// plainly" — mirrors [`Recognizer::decode`] vs
/// [`Recognizer::decode_with_hotwords`] on the daemon side.
pub fn transcribe_via_daemon(
    socket_path: &Path,
    model_dir: &Path,
    samples: &[f32],
    hotwords: &str,
) -> Result<(String, f64), DaemonError> {
    let model_dir = canonical_or_self(model_dir);
    let stream = connect_or_spawn(socket_path, &model_dir)?;

    let header = encode_request(&Request::Transcribe {
        samples: samples.len(),
        model: model_dir,
        hotwords: hotwords.to_string(),
    });
    let mut writer = &stream;
    writer
        .write_all(header.as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(write_io_err)?;
    let payload: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    writer.write_all(&payload).map_err(write_io_err)?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(write_io_err)?;
    parse_transcribe_response(&line)
}

/// Asks the daemon at `socket_path` for its status. Unlike
/// [`transcribe_via_daemon`], never spawns one — no daemon listening is
/// reported as-is, for `auris status` to turn into "no daemon on `<path>`"
/// (README "The daemon").
pub fn status(socket_path: &Path) -> Result<StatusInfo, DaemonError> {
    let stream = UnixStream::connect(socket_path).map_err(write_io_err)?;
    let mut writer = &stream;
    writer
        .write_all(encode_request(&Request::Status).as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(write_io_err)?;
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(write_io_err)?;
    parse_status_response(&line)
}

/// Asks the daemon at `socket_path` to exit. Never spawns one, same as
/// [`status`].
pub fn stop(socket_path: &Path) -> Result<(), DaemonError> {
    let stream = UnixStream::connect(socket_path).map_err(write_io_err)?;
    let mut writer = &stream;
    writer
        .write_all(encode_request(&Request::Stop).as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(write_io_err)?;
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(write_io_err)?;
    parse_stop_response(&line)
}

// ---------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------

pub struct ServeConfig {
    pub model_dir: PathBuf,
    pub socket_path: PathBuf,
    pub verbose: bool,
}

/// Binds `path`, or — if something is already bound there — decides whether
/// that's a live daemon (a connect succeeds: print README's "daemon already
/// running" message and return `Ok(None)`, the caller's cue to exit `OK`
/// without touching the socket) or a stale one left behind by a daemon that
/// didn't unlink on exit (connect fails: unlink and rebind).
fn bind_or_detect_existing(path: &Path) -> std::io::Result<Option<UnixListener>> {
    match UnixListener::bind(path) {
        Ok(listener) => Ok(Some(listener)),
        Err(e) if e.kind() == ErrorKind::AddrInUse => {
            if UnixStream::connect(path).is_ok() {
                eprintln!("auris: daemon already running on {}", path.display());
                Ok(None)
            } else {
                std::fs::remove_file(path)?;
                UnixListener::bind(path).map(Some)
            }
        }
        Err(e) => Err(e),
    }
}

/// Checks a `transcribe` request's declared `samples` count against
/// [`MAX_TRANSCRIBE_SAMPLES`] and turns it into the payload length to read,
/// via checked arithmetic — both checks run before `handle_connection`
/// allocates anything, so a bogus or hostile header can't make the daemon
/// allocate an unbounded buffer. Returns the error message to send back
/// (as `{"ok":false,"code":USAGE,...}`) on either failure.
fn validate_transcribe_samples(samples: usize) -> Result<usize, String> {
    if samples > MAX_TRANSCRIBE_SAMPLES {
        return Err(format!(
            "samples {samples} exceeds the sanity bound of {MAX_TRANSCRIBE_SAMPLES} (one hour of 16 kHz mono audio)"
        ));
    }
    samples
        .checked_mul(4)
        .ok_or_else(|| format!("samples {samples} overflows a payload length"))
}

/// Handles one connection to completion: reads exactly one request, writes
/// exactly one response. Returns `false` when the daemon should stop serving
/// (a `stop` request was handled) and `true` otherwise.
fn handle_connection(
    stream: UnixStream,
    recognizer: &Recognizer,
    loaded_model_dir: &Path,
    pid: u32,
    started: Instant,
    request_count: &mut u64,
) -> bool {
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) | Err(_) => return true, // client disconnected without a full line
        Ok(_) => {}
    }

    let req = match parse_request(&line) {
        Ok(req) => req,
        Err(e) => {
            let mut writer = &stream;
            let msg = e.to_string();
            let _ = writeln!(
                writer,
                "{}",
                encode_response(&Response::Err {
                    code: USAGE,
                    message: &msg
                })
            );
            return true;
        }
    };

    let mut writer = &stream;
    match req {
        Request::Status => {
            let info = StatusInfo {
                model: loaded_model_dir.to_string_lossy().into_owned(),
                pid,
                uptime_secs: started.elapsed().as_secs(),
                requests: *request_count,
            };
            let _ = writeln!(writer, "{}", encode_response(&Response::Status(&info)));
            true
        }
        Request::Stop => {
            let _ = writeln!(writer, "{}", encode_response(&Response::Stopped));
            false
        }
        Request::Transcribe {
            samples,
            model,
            hotwords,
        } => {
            if model != loaded_model_dir {
                let message = format!(
                    "daemon is running model {}; requested {}",
                    loaded_model_dir.display(),
                    model.display()
                );
                let _ = writeln!(
                    writer,
                    "{}",
                    encode_response(&Response::Err {
                        code: NOTHING_TRANSCRIBED,
                        message: &message,
                    })
                );
                return true;
            }

            let payload_len = match validate_transcribe_samples(samples) {
                Ok(len) => len,
                Err(message) => {
                    let _ = writeln!(
                        writer,
                        "{}",
                        encode_response(&Response::Err {
                            code: USAGE,
                            message: &message,
                        })
                    );
                    return true;
                }
            };

            let mut payload = vec![0u8; payload_len];
            if let Err(e) = reader.read_exact(&mut payload) {
                let message = format!("failed to read audio payload: {e}");
                let _ = writeln!(
                    writer,
                    "{}",
                    encode_response(&Response::Err {
                        code: NOTHING_TRANSCRIBED,
                        message: &message,
                    })
                );
                return true;
            }
            let samples_f32: Vec<f32> = payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let decode_start = Instant::now();
            let result = if hotwords.is_empty() {
                recognizer.decode(&samples_f32)
            } else {
                recognizer.decode_with_hotwords(&samples_f32, &hotwords)
            };
            *request_count += 1;

            match result {
                Ok(text) => {
                    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                    let _ = writeln!(
                        writer,
                        "{}",
                        encode_response(&Response::Transcribed {
                            text: &text,
                            decode_ms,
                        })
                    );
                }
                Err(e) => {
                    let message = e.to_string();
                    let _ = writeln!(
                        writer,
                        "{}",
                        encode_response(&Response::Err {
                            code: e.exit_code(),
                            message: &message,
                        })
                    );
                }
            }
            true
        }
    }
}

/// Runs `auris serve`: loads the recognizer once, then serves requests until
/// told to stop, an idle timeout elapses, or Ctrl-C arrives. Returns the exit
/// code for the process (README "The daemon", "Exit codes").
pub fn serve(cfg: ServeConfig) -> i32 {
    let model_dir = canonical_or_self(&cfg.model_dir);

    let listener = match bind_or_detect_existing(&cfg.socket_path) {
        Ok(Some(listener)) => listener,
        Ok(None) => return OK, // another daemon is already live; message already printed
        Err(e) => {
            eprintln!("auris: failed to bind {}: {e}", cfg.socket_path.display());
            return NOTHING_TRANSCRIBED;
        }
    };
    // Nonblocking so the accept loop can also check the idle timeout and the
    // stop flag between connections, instead of blocking forever in accept.
    if let Err(e) = listener.set_nonblocking(true) {
        eprintln!("auris: failed to configure socket: {e}");
        let _ = std::fs::remove_file(&cfg.socket_path);
        return NOTHING_TRANSCRIBED;
    }

    if cfg.verbose {
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
            let _ = std::fs::remove_file(&cfg.socket_path);
            return e.exit_code();
        }
    };
    if cfg.verbose {
        eprintln!("auris: model loaded in {:?}", load_start.elapsed());
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    {
        let stop_flag = Arc::clone(&stop_flag);
        // Best-effort: if a handler is already installed in this process
        // (it shouldn't be — `serve` runs as its own subcommand, never
        // alongside the client's interrupt handler) this just does nothing.
        let _ = ctrlc::set_handler(move || stop_flag.store(true, Ordering::SeqCst));
    }

    let pid = std::process::id();
    let started = Instant::now();
    let mut request_count: u64 = 0;
    let mut last_activity = Instant::now();

    loop {
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        if last_activity.elapsed() >= IDLE_TIMEOUT {
            if cfg.verbose {
                eprintln!("auris: idle timeout, exiting");
            }
            break;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                last_activity = Instant::now();
                // The listener is nonblocking so the accept loop can also
                // poll the idle timeout and stop flag; on some platforms
                // (macOS included) an accepted connection inherits that
                // nonblocking mode, which would make `read_line` below
                // return `WouldBlock` and drop the connection before the
                // client finishes writing. Each connection is handled to
                // completion synchronously, so put it back to blocking.
                if let Err(e) = stream.set_nonblocking(false) {
                    eprintln!("auris: failed to configure connection: {e}");
                    continue;
                }
                // Bounds a client that sends a header and then stalls: a
                // timed-out read surfaces as an `Err` to `handle_connection`,
                // which already treats a header read failure as "drop the
                // connection" and a payload read failure as a normal
                // `{"ok":false,...}` response — so this needs no new
                // handling there, only this deadline. Server-side only: a
                // client's own wait for the response must stay unbounded, a
                // long decode is legitimate and must not be cut off.
                if let Err(e) = stream.set_read_timeout(Some(CONNECTION_READ_TIMEOUT)) {
                    eprintln!("auris: failed to configure connection: {e}");
                    continue;
                }
                let keep_serving = handle_connection(
                    stream,
                    &recognizer,
                    &model_dir,
                    pid,
                    started,
                    &mut request_count,
                );
                if !keep_serving {
                    break;
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                std::thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(e) => {
                eprintln!("auris: accept failed: {e}");
                break;
            }
        }
    }

    let _ = std::fs::remove_file(&cfg.socket_path);
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_request_round_trips() {
        let req = Request::Transcribe {
            samples: 16_000,
            model: PathBuf::from("/home/x/.cache/auris/models/parakeet-tdt-0.6b-v2-int8"),
            hotwords: "khora :6.5/qorvex :5".to_string(),
        };
        let line = encode_request(&req);
        assert_eq!(parse_request(&line).unwrap(), req);
    }

    #[test]
    fn status_and_stop_requests_round_trip() {
        assert_eq!(
            parse_request(&encode_request(&Request::Status)).unwrap(),
            Request::Status
        );
        assert_eq!(
            parse_request(&encode_request(&Request::Stop)).unwrap(),
            Request::Stop
        );
    }

    #[test]
    fn transcribe_request_with_no_hotwords_parses_as_empty_string() {
        let line = r#"{"op":"transcribe","samples":10,"model":"/m"}"#;
        let req = parse_request(line).unwrap();
        assert_eq!(
            req,
            Request::Transcribe {
                samples: 10,
                model: PathBuf::from("/m"),
                hotwords: String::new(),
            }
        );
    }

    #[test]
    fn parse_request_rejects_unknown_op() {
        let err = parse_request(r#"{"op":"frobnicate"}"#).unwrap_err();
        assert!(matches!(err, RequestParseError::UnknownOp(op) if op == "frobnicate"));
    }

    #[test]
    fn parse_request_rejects_missing_field() {
        let err = parse_request(r#"{"op":"transcribe","model":"/m"}"#).unwrap_err();
        assert!(matches!(err, RequestParseError::MissingField("samples")));
    }

    #[test]
    fn parse_request_tolerates_trailing_newline() {
        let line = format!("{}\n", encode_request(&Request::Status));
        assert_eq!(parse_request(&line).unwrap(), Request::Status);
    }

    #[test]
    fn transcribe_response_parses_ok() {
        let line = r#"{"ok":true,"text":"book a call with khora","decode_ms":42.5}"#;
        let (text, decode_ms) = parse_transcribe_response(line).unwrap();
        assert_eq!(text, "book a call with khora");
        assert_eq!(decode_ms, 42.5);
    }

    #[test]
    fn transcribe_response_parses_error() {
        let line = r#"{"ok":false,"code":1,"message":"daemon is running model /a; requested /b"}"#;
        let err = parse_transcribe_response(line).unwrap_err();
        assert_eq!(err.code, 1);
        assert_eq!(err.message, "daemon is running model /a; requested /b");
    }

    #[test]
    fn status_response_parses_ok() {
        let line = r#"{"ok":true,"model":"/m","pid":123,"uptime_secs":90,"requests":4}"#;
        let info = parse_status_response(line).unwrap();
        assert_eq!(info.model, "/m");
        assert_eq!(info.pid, 123);
        assert_eq!(info.uptime_secs, 90);
        assert_eq!(info.requests, 4);
    }

    #[test]
    fn stop_response_parses_ok() {
        let line = r#"{"ok":true}"#;
        assert!(parse_stop_response(line).is_ok());
    }

    #[test]
    fn stop_response_parses_error() {
        let line = r#"{"ok":false,"code":1,"message":"nope"}"#;
        let err = parse_stop_response(line).unwrap_err();
        assert_eq!(err.code, 1);
        assert_eq!(err.message, "nope");
    }

    #[test]
    fn stale_socket_is_unlinked_and_rebound() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("auris.sock");

        // Simulate a daemon that died without unlinking: bind, then let the
        // listener drop — the fd closes (nothing is listening any more) but
        // `UnixListener`'s `Drop` does not unlink the path, leaving a dead
        // socket file behind exactly as a killed daemon would.
        {
            let _listener = UnixListener::bind(&path).unwrap();
        }
        assert!(path.exists());

        // No one is listening, so this must detect staleness, unlink, and
        // rebind successfully rather than reporting "already running".
        let result = bind_or_detect_existing(&path);
        assert!(result.is_ok(), "expected rebind to succeed: {result:?}");
        assert!(result.unwrap().is_some(), "expected a fresh listener");
    }

    #[test]
    fn live_socket_is_detected_as_already_running() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("auris.sock");
        let _listener = UnixListener::bind(&path).unwrap();

        let result = bind_or_detect_existing(&path).unwrap();
        assert!(
            result.is_none(),
            "a live listener must be reported as already running"
        );
    }

    #[test]
    fn transcribe_samples_within_bound_returns_payload_length() {
        assert_eq!(validate_transcribe_samples(16_000).unwrap(), 64_000);
        assert_eq!(
            validate_transcribe_samples(MAX_TRANSCRIBE_SAMPLES).unwrap(),
            MAX_TRANSCRIBE_SAMPLES * 4
        );
    }

    #[test]
    fn transcribe_samples_over_the_sanity_bound_is_rejected() {
        let err = validate_transcribe_samples(MAX_TRANSCRIBE_SAMPLES + 1).unwrap_err();
        assert!(err.contains("exceeds the sanity bound"));
    }

    #[test]
    fn transcribe_samples_that_would_overflow_the_payload_length_is_rejected() {
        // Comfortably past the sanity bound, so this exercises the same
        // rejection path as the bound check above in practice — the
        // checked-arithmetic guard exists for defense in depth regardless.
        let err = validate_transcribe_samples(usize::MAX).unwrap_err();
        assert!(!err.is_empty());
    }

    // --- Integration test: needs the real model, skips (not fails) without
    // it. Duplicates engine.rs's test-module skip pattern rather than
    // reusing its private helpers, same reasoning as vocabulary.rs's copy.

    /// Locates the spike Parakeet model, honouring `AURIS_TEST_MODEL_DIR` —
    /// same lookup as `engine::tests::spike_model_dir`.
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

    /// Symlinks the spike model into a temp dir with `bpe_synth.vocab`
    /// linked in as `bpe.vocab` — same layout `engine::tests` builds.
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

    fn decode_fixture(name: &str) -> Vec<f32> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("spike/fixtures/wav")
            .join(name);
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
        crate::audio::decode(file).unwrap_or_else(|e| panic!("decode {name}: {e}"))
    }

    /// Task 951's acceptance property, measured through the real wire
    /// protocol rather than assumed: the ~4 s model load happens once, in
    /// `serve`, before the accept loop ever starts — every `transcribe`
    /// request afterwards is a decode against the already-warm recognizer,
    /// never a reload. This drives an actual `serve` (on a background
    /// thread, over a real Unix socket) and two real `transcribe_via_daemon`
    /// calls, and asserts each decode is dramatically cheaper than the load
    /// that preceded it — not just that it "looks fast" by eye.
    #[test]
    fn second_decode_is_fast_after_the_one_time_load() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let model_dir = tmp.path().to_path_buf();

        let socket_dir = tempfile::tempdir().expect("tempdir");
        let socket_path = socket_dir.path().join("auris.sock");

        let load_start = Instant::now();
        let serve_cfg = ServeConfig {
            model_dir: model_dir.clone(),
            socket_path: socket_path.clone(),
            verbose: false,
        };
        let server = std::thread::spawn(move || serve(serve_cfg));

        // Poll status until the daemon is up and has finished loading —
        // `serve` only starts accepting connections once `Recognizer::load`
        // returns, so a successful status response means the load is done.
        let ready_deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if status(&socket_path).is_ok() {
                break;
            }
            assert!(Instant::now() < ready_deadline, "daemon never became ready");
            std::thread::sleep(Duration::from_millis(50));
        }
        let load_elapsed = load_start.elapsed();

        // Both a client-side wall clock *and* the daemon's self-reported
        // `decode_ms` are kept, and the assertions below lean on the
        // client-side one. That is not belt-and-braces: `decode_ms` is timed
        // from inside `handle_connection`, after the payload read, so a
        // regression that moved `Recognizer::load` to happen per request —
        // the exact regression this test exists to catch — would sit
        // *outside* the daemon's own timer and still report a fast
        // `decode_ms`. Only a clock the client starts before the RPC and
        // stops after it can see a reload hidden ahead of the server's
        // timer.
        let samples_u01 = decode_fixture("u01.wav");
        let rpc_start = Instant::now();
        let (_, decode_ms_1) =
            transcribe_via_daemon(&socket_path, &model_dir, &samples_u01, "").expect("decode u01");
        let rpc_elapsed_1 = rpc_start.elapsed();

        let samples_u04 = decode_fixture("u04.wav");
        let rpc_start = Instant::now();
        let (_, decode_ms_2) =
            transcribe_via_daemon(&socket_path, &model_dir, &samples_u04, "").expect("decode u04");
        let rpc_elapsed_2 = rpc_start.elapsed();

        stop(&socket_path).expect("stop");
        server.join().expect("server thread panicked");

        eprintln!(
            "load: {load_elapsed:?}, \
             u01: {rpc_elapsed_1:?} round trip ({decode_ms_1}ms decode), \
             u04: {rpc_elapsed_2:?} round trip ({decode_ms_2}ms decode)"
        );

        // "Dramatically cheaper": a whole warm round trip is a small
        // fraction of the one-time load, not a load-sized cost repeated per
        // request. Half the load time is a deliberately loose threshold —
        // this runs under the default parallel `cargo test`, where CPU
        // contention inflates every number, so it is sized to catch a
        // ~4 s reload per request rather than to police decode speed.
        let load_ms = load_elapsed.as_secs_f64() * 1000.0;
        let round_trip_1 = rpc_elapsed_1.as_secs_f64() * 1000.0;
        let round_trip_2 = rpc_elapsed_2.as_secs_f64() * 1000.0;
        assert!(
            round_trip_1 < load_ms / 2.0,
            "first round trip ({round_trip_1}ms) should be well under half \
             the load time ({load_ms}ms); the load looks like it is being \
             paid per request"
        );
        assert!(
            round_trip_2 < load_ms / 2.0,
            "second round trip ({round_trip_2}ms) should be well under half \
             the load time ({load_ms}ms); the load looks like it is being \
             paid per request"
        );
        // The daemon's own view must agree with the wall clock: if these
        // ever diverge sharply, time is going somewhere the daemon is not
        // measuring, which is itself the signal worth failing on.
        assert!(
            decode_ms_1 < load_ms / 2.0 && decode_ms_2 < load_ms / 2.0,
            "daemon-reported decodes ({decode_ms_1}ms, {decode_ms_2}ms) \
             should be well under half the load time ({load_ms}ms)"
        );
    }
}
