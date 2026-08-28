//! Parses, validates and assembles the `--vocabulary-file` term list
//! (docs/vocabulary.md "The file format", "Bounds", "The cap",
//! "Validation") into the per-stream hotwords string
//! [`crate::engine::Recognizer::decode_with_hotwords`] takes. This module
//! owns parsing only. The post-ASR correction pass (docs/correction.md)
//! today lives outside the crate, as the spike `vocab_correct.py`, and reads
//! the same hotwords file through its own, more permissive parser — the two
//! can disagree on what counts as a valid line. Once that pass is ported in,
//! it must reuse [`Term`], with boosts discarded, so "the correction
//! vocabulary and the biasing vocabulary cannot drift apart" becomes true by
//! construction rather than by convention.
//!
//! docs/vocabulary.md settled the shape of the file and the numbers below;
//! this module is a direct translation of its "Validation" and "The cap"
//! tables, not a re-derivation of them.

use std::fmt;
use std::path::{Path, PathBuf};

/// A vocabulary is capped at 128 terms (docs/vocabulary.md "The cap"): past
/// this many, terms are dropped in file order — the file's order is the
/// caller's priority signal, and auris does not reorder it.
pub const MAX_TERMS: usize = 128;

/// The upper end of the accepted `:boost` range, `(0.0, 8.0]`
/// (docs/vocabulary.md "Bounds"). Not a recommendation — it's a stop past
/// which the measured sweep shows the transcript coming apart.
const MAX_BOOST: f32 = 8.0;

/// A term longer than this many bytes is rejected as "not a name"
/// (docs/vocabulary.md "Validation").
const MAX_TERM_BYTES: usize = 64;

/// One parsed line of a vocabulary file: a term, and the optional per-word
/// boost that overrides the global `hotwords_score` fixed in
/// [`crate::engine::EngineConfig`] (docs/vocabulary.md "The one thing that
/// really is construction state"). Public so the correction pass
/// (docs/correction.md) can reuse the term list with boosts discarded.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub text: String,
    pub boost: Option<f32>,
}

/// Everything that can go wrong reading or validating a vocabulary file.
/// Every validation variant carries the 1-based line number and the
/// offending line's content (after stripping its comment), so the
/// diagnostic can name both, per docs/vocabulary.md "Validation": "auris
/// rejects a vocabulary file, with an `auris: ` diagnostic naming the
/// offending line."
#[derive(Debug)]
pub enum VocabularyError {
    /// The file itself could not be read.
    Unreadable(PathBuf, std::io::Error),
    /// A term contains `/`, the per-stream phrase separator with no escape.
    SlashInTerm(usize, String),
    /// A `:boost` token that does not parse as a number (`std::stof` has no
    /// guard against this on the sherpa-onnx side; a malformed value there
    /// is a hard crash, not a skip).
    NonNumericBoost(usize, String),
    /// `term:boost` with no space — silently parsed by sherpa-onnx as one
    /// out-of-vocabulary token and dropped. Rejected rather than passed
    /// through.
    NoSpaceBeforeBoost(usize, String),
    /// A `:`-led token where a term word was expected: a colon separated
    /// from its value by whitespace (`mesa : 3.0`), or a second boost token
    /// (`mesa :3.0 :4.0`). Neither is the glued `term:boost` spelling above.
    StrayBoostToken(usize, String),
    /// A boost outside `(0.0, 8.0]`.
    BoostOutOfRange(usize, String, f32),
    /// The term is empty.
    EmptyTerm(usize),
    /// The term is longer than 64 bytes.
    TermTooLong(usize, String),
    /// The term contains a control character.
    ControlCharacter(usize, String),
}

impl fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VocabularyError::Unreadable(path, e) => {
                write!(f, "failed to read vocabulary file {}: {e}", path.display())
            }
            VocabularyError::SlashInTerm(line, content) => write!(
                f,
                "line {line}: term contains '/', the per-stream phrase separator: {content:?}"
            ),
            VocabularyError::NonNumericBoost(line, content) => {
                write!(f, "line {line}: boost is not a number: {content:?}")
            }
            VocabularyError::NoSpaceBeforeBoost(line, content) => write!(
                f,
                "line {line}: \"term:boost\" needs a space before the colon \
                 (\"term :boost\"), or sherpa-onnx silently drops it: {content:?}"
            ),
            VocabularyError::StrayBoostToken(line, content) => write!(
                f,
                "line {line}: a boost token appears where a term word was expected \
                 (a colon separated from its value by whitespace, or a second boost): \
                 {content:?}"
            ),
            VocabularyError::BoostOutOfRange(line, content, boost) => write!(
                f,
                "line {line}: boost {boost} is outside (0.0, 8.0]: {content:?}"
            ),
            VocabularyError::EmptyTerm(line) => write!(f, "line {line}: term is empty"),
            VocabularyError::TermTooLong(line, term) => write!(
                f,
                "line {line}: term is {} bytes, over the 64-byte limit: {term:?}",
                term.len()
            ),
            VocabularyError::ControlCharacter(line, content) => write!(
                f,
                "line {line}: term contains a control character: {content:?}"
            ),
        }
    }
}

impl std::error::Error for VocabularyError {}

/// A parsed, validated, capped vocabulary — ready to become a per-stream
/// hotwords string ([`Vocabulary::hotwords_string`]), or to feed the
/// correction pass (docs/correction.md) via [`Vocabulary::terms`] with
/// boosts discarded.
#[derive(Debug)]
pub struct Vocabulary {
    pub terms: Vec<Term>,
    /// How many terms past [`MAX_TERMS`] were dropped, in file order. Zero
    /// means the file was under the cap. Exceeding the cap is a warning, not
    /// an error (docs/vocabulary.md "The cap").
    pub terms_dropped: usize,
}

impl Vocabulary {
    /// Reads and parses `path`. The only I/O in this module — kept separate
    /// from [`Vocabulary::parse`] so parsing itself stays testable on plain
    /// strings.
    pub fn load(path: &Path) -> Result<Vocabulary, VocabularyError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| VocabularyError::Unreadable(path.to_path_buf(), e))?;
        Self::parse(&contents)
    }

    /// Parses a vocabulary file's contents (docs/vocabulary.md "The file
    /// format"): UTF-8, one entry per line, `#` to end of line a comment,
    /// blank lines ignored, and applies the cap (docs/vocabulary.md "The
    /// cap") to the result.
    pub fn parse(contents: &str) -> Result<Vocabulary, VocabularyError> {
        let mut terms = Vec::new();
        for (i, raw_line) in contents.lines().enumerate() {
            let line_no = i + 1;
            if let Some(term) = parse_line(line_no, raw_line)? {
                terms.push(term);
            }
        }

        let terms_dropped = terms.len().saturating_sub(MAX_TERMS);
        terms.truncate(MAX_TERMS);
        Ok(Vocabulary {
            terms,
            terms_dropped,
        })
    }

    /// The per-stream hotwords string (docs/vocabulary.md "What becomes of
    /// the names"): terms joined with `/`, each rendered `term :boost` when
    /// a boost was given and bare otherwise. Byte-compatible with the
    /// `hotwords_file` syntax `src/engine.rs`'s
    /// `per_stream_hotwords_match_hotwords_file_byte_for_byte` test relies
    /// on — an empty term list needs no guard here, since a leading,
    /// trailing or doubled `/` is safe (sherpa-onnx drops empty segments
    /// silently).
    pub fn hotwords_string(&self) -> String {
        self.terms
            .iter()
            .map(|term| match term.boost {
                // `{:?}` (not `{}`) is what keeps e.g. 3.0 rendering as
                // "3.0" rather than Display's "3" — matching the file
                // format's own spelling.
                Some(boost) => format!("{} :{boost:?}", term.text),
                None => term.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }
}

/// Parses one line: strips its comment, and returns `None` for a
/// blank/comment-only line or `Some(term)` for a valid entry. `line_no` is
/// 1-based, for error messages only.
fn parse_line(line_no: usize, raw_line: &str) -> Result<Option<Term>, VocabularyError> {
    let content = match raw_line.find('#') {
        Some(i) => &raw_line[..i],
        None => raw_line,
    };
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let last = *tokens.last().expect("non-empty trimmed line has a token");

    // The boost is its own whitespace-separated token, spelled ":BOOST"
    // (docs/vocabulary.md "The file format"). Anything else with a `:` in
    // it — most importantly a term's last word glued straight to the boost,
    // "khora:6.5" — is the no-space spelling that sherpa-onnx silently
    // mis-parses, caught below.
    let (term_tokens, boost) = match last.strip_prefix(':') {
        Some(rest) => {
            let value: f32 = rest
                .parse()
                .map_err(|_| VocabularyError::NonNumericBoost(line_no, trimmed.to_string()))?;
            (&tokens[..tokens.len() - 1], Some(value))
        }
        None => (&tokens[..], None),
    };

    // A term token that *starts* with ':' is a stray or duplicated boost
    // token ("mesa : 3.0", "mesa :3.0 :4.0"), not the glued spelling below —
    // the colon there sits in the middle of a word, never at its start.
    if term_tokens.iter().any(|t| t.starts_with(':')) {
        return Err(VocabularyError::StrayBoostToken(
            line_no,
            trimmed.to_string(),
        ));
    }
    if term_tokens.iter().any(|t| t.contains(':')) {
        return Err(VocabularyError::NoSpaceBeforeBoost(
            line_no,
            trimmed.to_string(),
        ));
    }

    let term = term_tokens.join(" ");
    if term.is_empty() {
        return Err(VocabularyError::EmptyTerm(line_no));
    }
    if term.contains('/') {
        return Err(VocabularyError::SlashInTerm(line_no, trimmed.to_string()));
    }
    if term.len() > MAX_TERM_BYTES {
        return Err(VocabularyError::TermTooLong(line_no, term));
    }
    if term.chars().any(|c| c.is_control()) {
        return Err(VocabularyError::ControlCharacter(
            line_no,
            trimmed.to_string(),
        ));
    }
    if let Some(boost) = boost
        && !(boost > 0.0 && boost <= MAX_BOOST)
    {
        return Err(VocabularyError::BoostOutOfRange(
            line_no,
            trimmed.to_string(),
            boost,
        ));
    }

    Ok(Some(Term { text: term, boost }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(v: &Vocabulary) -> Vec<(&str, Option<f32>)> {
        v.terms.iter().map(|t| (t.text.as_str(), t.boost)).collect()
    }

    #[test]
    fn empty_file_is_an_empty_vocabulary() {
        let v = Vocabulary::parse("").expect("parse");
        assert!(v.terms.is_empty());
        assert_eq!(v.terms_dropped, 0);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let v = Vocabulary::parse("# a comment\n\nmesa :3.0\n   \n# trailing\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(3.0))]);
    }

    #[test]
    fn bare_term_has_no_boost() {
        let v = Vocabulary::parse("mesa\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", None)]);
    }

    #[test]
    fn term_with_boost() {
        let v = Vocabulary::parse("khora :6.5\n").expect("parse");
        assert_eq!(terms(&v), vec![("khora", Some(6.5))]);
    }

    #[test]
    fn multi_word_phrase() {
        let v = Vocabulary::parse("wire up :3.0\n").expect("parse");
        assert_eq!(terms(&v), vec![("wire up", Some(3.0))]);
    }

    #[test]
    fn inline_comment_after_a_term_is_stripped() {
        let v = Vocabulary::parse("mesa :3.0 # our own product\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(3.0))]);
    }

    #[test]
    fn cap_truncates_to_first_128_in_file_order_and_reports_it() {
        let mut file = String::new();
        for i in 0..150 {
            file.push_str(&format!("term{i}\n"));
        }
        let v = Vocabulary::parse(&file).expect("parse");
        assert_eq!(v.terms.len(), MAX_TERMS);
        assert_eq!(v.terms_dropped, 22);
        assert_eq!(v.terms[0].text, "term0");
        assert_eq!(v.terms[127].text, "term127");
    }

    #[test]
    fn rejects_slash_in_term() {
        let err = Vocabulary::parse("mesa/auris :3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::SlashInTerm(1, _)));
    }

    #[test]
    fn rejects_non_numeric_boost() {
        let err = Vocabulary::parse("mesa :abc\n").unwrap_err();
        assert!(matches!(err, VocabularyError::NonNumericBoost(1, _)));
    }

    #[test]
    fn rejects_term_colon_boost_with_no_space() {
        let err = Vocabulary::parse("khora:6.5\n").unwrap_err();
        assert!(matches!(err, VocabularyError::NoSpaceBeforeBoost(1, _)));
    }

    #[test]
    fn rejects_space_between_colon_and_boost_value() {
        let err = Vocabulary::parse("mesa : 3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::StrayBoostToken(1, _)));
    }

    #[test]
    fn rejects_a_second_boost_token() {
        let err = Vocabulary::parse("mesa :3.0 :4.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::StrayBoostToken(1, _)));
    }

    #[test]
    fn rejects_boost_of_zero() {
        let err = Vocabulary::parse("mesa :0.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::BoostOutOfRange(1, _, _)));
    }

    #[test]
    fn rejects_boost_above_the_cap() {
        let err = Vocabulary::parse("mesa :8.1\n").unwrap_err();
        assert!(matches!(err, VocabularyError::BoostOutOfRange(1, _, _)));
    }

    #[test]
    fn accepts_boost_at_the_upper_bound() {
        let v = Vocabulary::parse("mesa :8.0\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(8.0))]);
    }

    #[test]
    fn rejects_empty_term() {
        let err = Vocabulary::parse(":3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::EmptyTerm(1)));
    }

    #[test]
    fn rejects_term_over_64_bytes() {
        let long_term = "a".repeat(65);
        let err = Vocabulary::parse(&long_term).unwrap_err();
        assert!(matches!(err, VocabularyError::TermTooLong(1, _)));
    }

    #[test]
    fn accepts_term_at_64_bytes() {
        let term = "a".repeat(64);
        let v = Vocabulary::parse(&term).expect("parse");
        assert_eq!(v.terms[0].text.len(), 64);
    }

    #[test]
    fn rejects_control_character_in_term() {
        let err = Vocabulary::parse("me\u{0007}sa :3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::ControlCharacter(1, _)));
    }

    #[test]
    fn load_reports_unreadable_file() {
        let err = Vocabulary::load(Path::new("/nonexistent/auris-vocab-test-file")).unwrap_err();
        assert!(matches!(err, VocabularyError::Unreadable(_, _)));
    }

    #[test]
    fn hotwords_string_matches_the_production_fixture_shape() {
        // spike/fixtures/hotwords.txt, inlined here so this test doesn't
        // depend on the model spike being present.
        let file = "mesa :3.0\nauris :3.0\nkhora :6.5\nqorvex :5.0\nhelios :3.0\nkokoro :3.0\n";
        let v = Vocabulary::parse(file).expect("parse");
        assert_eq!(
            v.hotwords_string(),
            "mesa :3.0/auris :3.0/khora :6.5/qorvex :5.0/helios :3.0/kokoro :3.0"
        );
    }

    #[test]
    fn hotwords_string_bare_term_has_no_colon() {
        let v = Vocabulary::parse("mesa\nauris :3.0\n").expect("parse");
        assert_eq!(v.hotwords_string(), "mesa/auris :3.0");
    }

    // --- Integration tests: need the real model, skip (not fail) without it.
    //
    // These duplicate engine.rs's test-module skip pattern rather than
    // reusing its private helpers, so this module's tests don't reach into
    // engine.rs's test internals for a handful of small setup functions.

    use std::path::PathBuf;

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
    /// linked in as `bpe.vocab` — same layout `engine::tests` builds, and
    /// for the same reason: the engine needs that exact filename, and
    /// spike/ must not be copied or renamed in place.
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

    /// The production term list (docs/vocabulary.md's "Before and after"
    /// example, and `spike/fixtures/hotwords.txt`).
    fn production_vocabulary() -> Vocabulary {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spike/fixtures/hotwords.txt");
        Vocabulary::load(&path).expect("load production vocabulary")
    }

    /// docs/vocabulary.md "Before and after": u03 decoded with and without
    /// the production vocabulary, same recognizer, same audio. Both mesa
    /// names land only with the vocabulary; the doc's own transcripts are
    /// quoted here after re-running them to confirm they still hold.
    #[test]
    fn u03_decodes_mesa_names_correctly_only_with_vocabulary() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = crate::engine::EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = crate::engine::Recognizer::load(&cfg).expect("load");
        let samples = decode_fixture("u03.wav");

        let plain = recognizer.decode(&samples).expect("decode plain");
        let vocabulary = production_vocabulary();
        let biased = recognizer
            .decode_with_hotwords(&samples, &vocabulary.hotwords_string())
            .expect("decode biased");

        eprintln!("plain:  {plain:?}");
        eprintln!("biased: {biased:?}");

        let plain_lower = plain.to_lowercase();
        let biased_lower = biased.to_lowercase();

        // docs/vocabulary.md: unbiased renders qorvex as "Corvex".
        assert!(
            !plain_lower.contains("qorvex"),
            "expected the unbiased decode to miss qorvex, got {plain:?}"
        );
        // Biased lands both mesa names.
        assert!(
            biased_lower.contains("mesa"),
            "expected the biased decode to contain mesa, got {biased:?}"
        );
        assert!(
            biased_lower.contains("qorvex"),
            "expected the biased decode to contain qorvex, got {biased:?}"
        );
    }

    /// An empty vocabulary (a file with only comments/blank lines) means
    /// exactly what no vocabulary means. cli.rs short-circuits to plain
    /// `decode()` for this case rather than calling `decode_with_hotwords`
    /// with an empty string — an unexercised sherpa-onnx path, not the same
    /// as the safely-ignored empty *segments* a non-empty joined string can
    /// carry (docs/vocabulary.md "What becomes of the names"). This test
    /// mirrors that dispatch directly against the engine and checks the
    /// output is byte-identical either way.
    #[test]
    fn empty_vocabulary_decodes_identically_to_no_vocabulary() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = crate::engine::EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = crate::engine::Recognizer::load(&cfg).expect("load");
        let samples = decode_fixture("u01.wav");

        let vocabulary = Vocabulary::parse("# no terms\n\n").expect("parse");
        assert!(vocabulary.terms.is_empty(), "fixture is meant to be empty");

        // What cli.rs's dispatch actually calls for an empty vocabulary
        // (plain `decode()`), compared against the empty hotwords string
        // the pre-fix code would have sent to `decode_with_hotwords`
        // instead — proving the short-circuit changes nothing observable,
        // while still being the only call this module makes on the path
        // sherpa-onnx has never been checked against.
        let via_short_circuit = recognizer.decode(&samples).expect("decode plain");
        let via_empty_hotwords_string = recognizer
            .decode_with_hotwords(&samples, &vocabulary.hotwords_string())
            .expect("decode with empty hotwords string");
        assert_eq!(
            via_short_circuit, via_empty_hotwords_string,
            "an empty vocabulary must decode identically to no vocabulary at all"
        );
    }

    /// The restated no-hallucination property (docs/vocabulary.md "Ordinary
    /// English words are not safe to include"): a curated vocabulary of
    /// mesa's own names does not hallucinate those names into recordings
    /// that do not contain them. For each fixture, every mesa name absent
    /// from its reference text must also be absent from the biased decode
    /// of that fixture's audio.
    #[test]
    fn curated_vocabulary_does_not_hallucinate_absent_names() {
        let Some(model_dir) = spike_model_dir() else {
            return;
        };
        let tmp = symlinked_model_dir(&model_dir);
        let cfg = crate::engine::EngineConfig {
            model_dir: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let recognizer = crate::engine::Recognizer::load(&cfg).expect("load");
        let vocabulary = production_vocabulary();

        // (fixture, reference text) — spike/fixtures/utterances.tsv.
        let fixtures = [
            (
                "u01.wav",
                "close task 924 and open a follow up under auris for the parakeet spike",
            ),
            (
                "u02.wav",
                "hey mesa add a note to khora that headless mode is the default now",
            ),
            (
                "u03.wav",
                "create a task in mesa called wire up qorvex to the ios simulator and assign it to me",
            ),
            (
                "u04.wav",
                "helios needs an index refresh before task 118 can start",
            ),
            (
                "u05.wav",
                "remind me to check kokoro's latency numbers against whisper before we pick a winner",
            ),
            (
                "u06.wav",
                "so basically what happened is auris transcribed helios as hell EOS again which is the third time this week and I think we need a custom vocabulary list for mesa terms before we ship, can you open a bug for that",
            ),
            (
                "u07.wav",
                "mark task 730 done and task 731 blocked on khora",
            ),
            (
                "u08.wav",
                "qorvex tap failed twice on the android emulator, log it under auris and link it to task 902",
            ),
        ];
        let names = ["mesa", "auris", "khora", "qorvex", "helios", "kokoro"];

        for (wav, reference) in fixtures {
            let samples = decode_fixture(wav);
            let biased = recognizer
                .decode_with_hotwords(&samples, &vocabulary.hotwords_string())
                .unwrap_or_else(|e| panic!("decode {wav}: {e:?}"));
            let biased_lower = biased.to_lowercase();
            eprintln!("{wav}: biased: {biased_lower:?}");

            for name in names {
                if !reference.contains(name) {
                    assert!(
                        !biased_lower.contains(name),
                        "{wav}: biased decode introduced {name:?}, which is not in \
                         the reference, into {biased:?}"
                    );
                }
            }
        }
    }
}
