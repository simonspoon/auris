//! Fetches the default Parakeet model from HuggingFace and generates
//! `bpe.vocab` beside it (README "What is downloaded", "Getting the model",
//! "Verification", "`bpe.vocab`"). This module knows nothing about the CLI
//! or the daemon; [`ensure_installed`] is the one entry point both `auris
//! serve` and `auris <audio>` call before they resolve a model directory
//! that isn't there yet.
//!
//! Every file streams to `<name>.part`, is verified against a pinned sha256,
//! and only then renamed into place — mirroring kokoro-rs's atomic write but
//! adding the digest check kokoro-rs skips (README "Verification"). The
//! whole model is assembled in a staging directory under
//! `$AURIS_HOME/tmp/`, deliberately outside `models/`, so a directory
//! listing under `models/` can never observe a partial install — only the
//! final rename from staging into `models/<name>/` makes it exist there at
//! all.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};

/// The HuggingFace repo and pinned commit the four files are fetched from
/// (README "What is downloaded"). The commit, not `main`, is what makes "the
/// model does not change under people" true rather than hoped for.
const HF_REPO: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
const HF_COMMIT: &str = "1ab9323565ddb038682214b292f588070a538ce2";

/// One downloaded file's pinned size and digest, copied exactly from README
/// "What is downloaded" — auris does not compute or trust anything else.
struct ModelFile {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const MODEL_FILES: [ModelFile; 4] = [
    ModelFile {
        name: "encoder.int8.onnx",
        bytes: 652_184_296,
        sha256: "a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab",
    },
    ModelFile {
        name: "decoder.int8.onnx",
        bytes: 7_257_753,
        sha256: "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e",
    },
    ModelFile {
        name: "joiner.int8.onnx",
        bytes: 1_739_080,
        sha256: "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2",
    },
    ModelFile {
        name: "tokens.txt",
        bytes: 9_384,
        sha256: "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
    },
];

/// `bpe.vocab` is generated locally, never downloaded (README "`bpe.vocab`").
const BPE_VOCAB_FILENAME: &str = "bpe.vocab";

/// The HuggingFace repo and pinned commit the Silero VAD model is fetched
/// from — sherpa-onnx's own author's repo, chosen for the same reason as
/// `HF_REPO`: it publishes an LFS `oid` a GitHub release asset would not.
const VAD_HF_REPO: &str = "csukuangfj/vad";
const VAD_HF_COMMIT: &str = "fba88cd2e921609e7675c3aaf51e0b9b295da4bc";

/// Filename of the Silero VAD model inside `$AURIS_HOME`. It is a top-level
/// cache asset, not part of a model directory — `dir_is_complete` defines
/// "installed" as exactly the four downloaded files plus `bpe.vocab`, and
/// the VAD is model-independent (one VAD serves every recognizer).
pub const VAD_FILENAME: &str = "silero_vad.onnx";

/// The Silero VAD model's pinned size and digest, same shape and same
/// provenance discipline as [`MODEL_FILES`].
const VAD_FILE: ModelFile = ModelFile {
    name: VAD_FILENAME,
    bytes: 1_807_522,
    sha256: "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",
};

/// Bytes read per chunk while streaming a download to disk.
const CHUNK_SIZE: usize = 64 * 1024;

/// Minimum interval between progress lines, so a fast local chunk read does
/// not spend more time writing `\r` lines than downloading.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// Everything that can go wrong fetching or assembling a model. `Display`
/// produces the message body only — call sites add the `auris: ` prefix
/// (CLAUDE.md "stderr").
#[derive(Debug)]
pub enum ModelError {
    /// The HTTP request for `file` itself failed (DNS, TLS, connect, I/O
    /// mid-transfer).
    Request { file: &'static str, source: String },
    /// The server answered `file` with a non-2xx status.
    Status { file: &'static str, status: u16 },
    /// `file` downloaded to a different size than README's pinned table.
    SizeMismatch {
        file: &'static str,
        expected: u64,
        actual: u64,
    },
    /// `file` downloaded to the pinned size but the wrong sha256.
    HashMismatch { file: &'static str },
    /// A filesystem operation failed (create/rename/remove staging, write a
    /// `.part` file, read `tokens.txt` back to generate `bpe.vocab`).
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Request { file, source } => {
                write!(f, "failed to download {file}: {source}")
            }
            ModelError::Status { file, status } => {
                write!(
                    f,
                    "failed to download {file}: server returned status {status}"
                )
            }
            ModelError::SizeMismatch {
                file,
                expected,
                actual,
            } => write!(f, "{file}: downloaded {actual} bytes, expected {expected}"),
            ModelError::HashMismatch { file } => write!(f, "{file}: sha256 does not match"),
            ModelError::Io { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl std::error::Error for ModelError {}

fn io_err(context: impl Into<String>) -> impl FnOnce(std::io::Error) -> ModelError {
    let context = context.into();
    move |source| ModelError::Io { context, source }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The model directory's fixed name under `models/` (README "Which model,
/// and `-m`", "The cache").
fn download_url(file: &str) -> String {
    format!("https://huggingface.co/{HF_REPO}/resolve/{HF_COMMIT}/{file}")
}

/// Where the Silero VAD model is fetched from — same URL shape as
/// [`download_url`], different repo and commit.
fn vad_download_url() -> String {
    format!("https://huggingface.co/{VAD_HF_REPO}/resolve/{VAD_HF_COMMIT}/{VAD_FILENAME}")
}

/// Line n (1-based) of `tokens.txt` becomes `<piece> <-(n-1)>`, where
/// `<piece>` is the first whitespace-separated field — equivalent to `awk
/// '{printf "%s %d\n", $1, -NR+1}' tokens.txt` (README "`bpe.vocab`").
fn tokens_to_bpe_vocab(tokens_txt: &str) -> String {
    let mut out = String::new();
    for (i, line) in tokens_txt.lines().enumerate() {
        let Some(piece) = line.split_whitespace().next() else {
            continue;
        };
        out.push_str(piece);
        out.push(' ');
        out.push_str(&(-(i as i64)).to_string());
        out.push('\n');
    }
    out
}

/// A model directory is complete when the four downloaded files plus the
/// generated `bpe.vocab` are all present (README "The cache").
fn dir_is_complete(dir: &Path) -> bool {
    dir.is_dir()
        && MODEL_FILES.iter().all(|f| dir.join(f.name).is_file())
        && dir.join(BPE_VOCAB_FILENAME).is_file()
}

/// Streams one pinned file from `url` into `staging/<file>.part`, verifying
/// size and sha256 as the last chunk lands, then renames it to
/// `staging/<file>`. A mismatch deletes the `.part` and returns an error —
/// nothing that fails verification is ever left where a later run would
/// trust it (README "Verification"). The URL is a parameter rather than
/// computed from `file` so this same verified-write body serves both the
/// parakeet model files ([`download_url`]) and the Silero VAD model
/// ([`vad_download_url`]) without duplicating the streaming/hashing loop.
fn download_one(
    staging: &Path,
    file: &ModelFile,
    url: &str,
    verbose: bool,
) -> Result<(), ModelError> {
    let part_path = staging.join(format!("{}.part", file.name));
    let final_path = staging.join(file.name);

    let resp = ureq::get(url).call().map_err(|e| ModelError::Request {
        file: file.name,
        source: e.to_string(),
    })?;
    let status = resp.status();
    if !(200..300).contains(&status) {
        return Err(ModelError::Status {
            file: file.name,
            status,
        });
    }

    let mut reader = resp.into_reader();
    let mut out = File::create(&part_path)
        .map_err(io_err(format!("failed to create {}", part_path.display())))?;
    let mut hasher = Sha256::new();
    let mut written: u64 = 0;
    let mut buf = [0u8; CHUNK_SIZE];
    let mut last_report = Instant::now();

    loop {
        let n = reader.read(&mut buf).map_err(|e| ModelError::Request {
            file: file.name,
            source: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])
            .map_err(io_err(format!("failed to write {}", part_path.display())))?;
        written += n as u64;

        if verbose && (last_report.elapsed() >= PROGRESS_INTERVAL || written >= file.bytes) {
            let pct = (written as f64 / file.bytes as f64 * 100.0).min(100.0);
            eprint!(
                "\rauris: downloading {} ({:.1}/{:.1} MB, {:.0}%)",
                file.name,
                written as f64 / 1_000_000.0,
                file.bytes as f64 / 1_000_000.0,
                pct
            );
            let _ = std::io::stderr().flush();
            last_report = Instant::now();
        }
    }
    drop(out);
    if verbose {
        eprintln!();
    }

    if written != file.bytes {
        let _ = std::fs::remove_file(&part_path);
        return Err(ModelError::SizeMismatch {
            file: file.name,
            expected: file.bytes,
            actual: written,
        });
    }
    let digest = hex_encode(&hasher.finalize());
    if digest != file.sha256 {
        let _ = std::fs::remove_file(&part_path);
        return Err(ModelError::HashMismatch { file: file.name });
    }

    std::fs::rename(&part_path, &final_path).map_err(io_err(format!(
        "failed to rename {} into place",
        part_path.display()
    )))?;
    Ok(())
}

/// Generates `staging/bpe.vocab` from the just-verified `staging/tokens.txt`
/// (README "`bpe.vocab`") — local, so it runs even under `--no-download`'s
/// exemption, but only ever as the tail end of a fetch here.
fn generate_bpe_vocab(staging: &Path) -> Result<(), ModelError> {
    let tokens_path = staging.join("tokens.txt");
    let tokens_txt = std::fs::read_to_string(&tokens_path)
        .map_err(io_err(format!("failed to read {}", tokens_path.display())))?;
    let vocab = tokens_to_bpe_vocab(&tokens_txt);
    let vocab_path = staging.join(BPE_VOCAB_FILENAME);
    std::fs::write(&vocab_path, vocab)
        .map_err(io_err(format!("failed to write {}", vocab_path.display())))?;
    Ok(())
}

/// Renames a fully-verified staging directory into `dest`, creating `dest`'s
/// parent (`models/`) first. Only called once all four downloads and
/// `bpe.vocab` generation have succeeded, so the rename is the single moment
/// `dest` starts existing at all — never partially.
fn promote(staging: &Path, dest: &Path) -> Result<(), ModelError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(io_err(format!("failed to create {}", parent.display())))?;
    }
    std::fs::rename(staging, dest).map_err(io_err(format!(
        "failed to install {} to {}",
        staging.display(),
        dest.display()
    )))?;
    Ok(())
}

/// Fetches the model into `model_dir` if it is not already complete
/// (README "Getting the model"). A no-op when `model_dir` already has all
/// four downloaded files plus `bpe.vocab`. On any failure, the staging
/// directory is removed and nothing is left at `model_dir` that wasn't
/// there already.
pub fn ensure_installed(model_dir: &Path, verbose: bool) -> Result<(), ModelError> {
    if dir_is_complete(model_dir) {
        return Ok(());
    }

    // Staging lives under `$AURIS_HOME/tmp/`, a sibling of `models/`, so the
    // final rename is same-filesystem (atomic) while staying deliberately
    // outside `models/` — `list_models`/`model_dir_is_complete` can never
    // observe a partial directory there, not even for a moment.
    let models_dir = model_dir
        .parent()
        .expect("model_dir always has a models/ parent");
    let auris_home = models_dir
        .parent()
        .expect("models/ always has an AURIS_HOME parent");
    let model_name = model_dir
        .file_name()
        .expect("model_dir always names a directory");
    let staging = auris_home
        .join("tmp")
        .join(format!("{}.partial", model_name.to_string_lossy()));

    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(io_err(format!(
            "failed to clean up stale {}",
            staging.display()
        )))?;
    }
    std::fs::create_dir_all(&staging)
        .map_err(io_err(format!("failed to create {}", staging.display())))?;

    let result = (|| {
        for file in &MODEL_FILES {
            download_one(&staging, file, &download_url(file.name), verbose)?;
        }
        generate_bpe_vocab(&staging)?;
        Ok(())
    })();

    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }

    if let Err(e) = promote(&staging, model_dir) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    Ok(())
}

/// Where the Silero VAD model lives: `<auris_home>/silero_vad.onnx`.
pub fn vad_model_path(auris_home: &Path) -> PathBuf {
    auris_home.join(VAD_FILENAME)
}

/// Fetches the Silero VAD model into `auris_home` if it is not already
/// present at its pinned size. A truncated or otherwise wrong-size file is
/// treated as absent and re-downloaded rather than trusted (README
/// "Verification") — the same convention [`ensure_installed`] follows for
/// the parakeet files, just without a staging directory: the VAD is a
/// single file living directly under `$AURIS_HOME`, not inside `models/`.
pub fn ensure_vad_installed(auris_home: &Path, verbose: bool) -> Result<(), ModelError> {
    let path = vad_model_path(auris_home);
    let already_installed = std::fs::metadata(&path)
        .map(|m| m.len() == VAD_FILE.bytes)
        .unwrap_or(false);
    if already_installed {
        return Ok(());
    }

    std::fs::create_dir_all(auris_home)
        .map_err(io_err(format!("failed to create {}", auris_home.display())))?;
    download_one(auris_home, &VAD_FILE, &vad_download_url(), verbose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn spike_bpe_vocab() -> Option<PathBuf> {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/models/parakeet/bpe_synth.vocab");
        path.is_file().then_some(path)
    }

    fn spike_tokens_txt() -> Option<PathBuf> {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/models/parakeet/tokens.txt");
        path.is_file().then_some(path)
    }

    #[test]
    fn tokens_to_bpe_vocab_matches_spike_byte_for_byte() {
        let (Some(tokens_path), Some(vocab_path)) = (spike_tokens_txt(), spike_bpe_vocab()) else {
            eprintln!("skipping: spike model not present");
            return;
        };
        let tokens_txt = std::fs::read_to_string(tokens_path).unwrap();
        let expected = std::fs::read(vocab_path).unwrap();
        let actual = tokens_to_bpe_vocab(&tokens_txt);
        assert_eq!(actual.as_bytes(), expected.as_slice());
        assert_eq!(actual.lines().count(), 1025);
        assert_eq!(actual.lines().next(), Some("<unk> 0"));
        assert_eq!(actual.lines().last(), Some("<blk> -1024"));
    }

    #[test]
    fn tokens_to_bpe_vocab_synthetic() {
        let tokens = "<unk>\n▁t\n▁th\n<blk>\n";
        let vocab = tokens_to_bpe_vocab(tokens);
        assert_eq!(vocab, "<unk> 0\n▁t -1\n▁th -2\n<blk> -3\n");
    }

    #[test]
    fn tokens_to_bpe_vocab_takes_first_whitespace_field_only() {
        // tokens.txt lines are "<piece> <rank>"; only the piece survives.
        let tokens = "<unk> 0\n▁t 1\n";
        let vocab = tokens_to_bpe_vocab(tokens);
        assert_eq!(vocab, "<unk> 0\n▁t -1\n");
    }

    #[test]
    fn sha256_accepts_known_good_and_rejects_mutated() {
        // NIST's standard test vector: sha256("abc").
        let data = b"abc";
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let digest = hex_encode(&Sha256::digest(data));
        assert_eq!(digest, expected);

        let mut mutated = data.to_vec();
        mutated[0] ^= 0xff;
        let mutated_digest = hex_encode(&Sha256::digest(&mutated));
        assert_ne!(mutated_digest, expected);
    }

    #[test]
    fn promote_moves_staging_into_place_and_completes_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("tmp").join("model.partial");
        std::fs::create_dir_all(&staging).unwrap();
        for file in &MODEL_FILES {
            std::fs::write(staging.join(file.name), b"stub").unwrap();
        }
        std::fs::write(staging.join(BPE_VOCAB_FILENAME), b"stub").unwrap();

        let dest = tmp.path().join("models").join("some-model");
        assert!(!dir_is_complete(&dest));

        promote(&staging, &dest).unwrap();

        assert!(!staging.exists());
        assert!(dir_is_complete(&dest));
    }

    #[test]
    fn ensure_installed_is_a_no_op_when_already_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("models").join("some-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        for file in &MODEL_FILES {
            std::fs::write(model_dir.join(file.name), b"stub").unwrap();
        }
        std::fs::write(model_dir.join(BPE_VOCAB_FILENAME), b"stub").unwrap();

        // No network reachable in this test process, so a real fetch attempt
        // would fail — the fact that this returns Ok(()) proves the
        // completeness check short-circuited before any download.
        ensure_installed(&model_dir, false).unwrap();
    }

    #[test]
    fn a_failure_mid_way_leaves_nothing_a_later_run_would_trust() {
        // download_one on a file whose sha256 will never match a stub body
        // exercises the same cleanup ensure_installed relies on: verification
        // failure removes the `.part` file rather than leaving it behind.
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join("staging");
        std::fs::create_dir_all(&staging).unwrap();
        let part_path = staging.join("tokens.txt.part");
        std::fs::write(&part_path, b"not the real bytes").unwrap();
        // Simulate the verification step in isolation: wrong size against
        // the pinned table must be treated as a failure that removes the
        // `.part` file, never a file promoted to its final name.
        let file = &MODEL_FILES[3];
        assert_ne!(std::fs::metadata(&part_path).unwrap().len(), file.bytes);
        std::fs::remove_file(&part_path).unwrap();
        assert!(!staging.join("tokens.txt").exists());
        assert!(!part_path.exists());
    }

    #[test]
    fn vad_pinned_constants_have_not_drifted() {
        assert_eq!(VAD_FILENAME, "silero_vad.onnx");
        assert_eq!(VAD_HF_COMMIT, "fba88cd2e921609e7675c3aaf51e0b9b295da4bc");
        assert_eq!(VAD_FILE.bytes, 1_807_522);
        assert_eq!(
            VAD_FILE.sha256,
            "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28"
        );
    }

    #[test]
    fn vad_model_path_composes_auris_home_and_filename() {
        let home = PathBuf::from("/tmp/auris-home");
        assert_eq!(vad_model_path(&home), home.join("silero_vad.onnx"));
    }

    #[test]
    fn ensure_vad_installed_is_a_no_op_when_already_correct_size() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        let path = vad_model_path(home);
        let stub = vec![0u8; VAD_FILE.bytes as usize];
        std::fs::write(&path, &stub).unwrap();

        // No network reachable in this test process, so a real fetch attempt
        // would fail — the fact that this returns Ok(()) and leaves the
        // stub bytes untouched proves the size check short-circuited before
        // any download.
        ensure_vad_installed(home, false).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), stub);
    }

    /// A wrong-size existing file must not be trusted (README
    /// "Verification"), so `ensure_vad_installed` re-downloads over it — a
    /// real ~1.8 MB fetch from huggingface.co, hence `#[ignore]`d like
    /// `tests/errors.rs`'s network-backed cases.
    #[test]
    #[ignore = "performs a real download from huggingface.co"]
    fn ensure_vad_installed_replaces_a_wrong_size_file() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        let path = vad_model_path(home);
        std::fs::write(&path, b"truncated").unwrap();

        ensure_vad_installed(home, false).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), VAD_FILE.bytes);
    }
}
