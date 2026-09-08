#!/usr/bin/env python3
"""Post-ASR vocabulary correction for auris (mesa task 922's algorithm, ported).

Every engine benchmarked in `docs/spike-results.md` (section 3) reliably mangles
this project's own names: `khora` comes back as "qora"/"Quora"/"Korra"/"Cora",
`qorvex` as "Corvex"/"korvex", `auris` as "Aurus"/"Aorus"/"auras", `kokoro` as
"Kakoro"/"Kakaro". None of that is fixable by better prompting or hotword
biasing alone (RESULTS.md section 6, "What this does not fix") -- it needs a
correction pass over the transcript text after decoding.

This module is a faithful port of the sibling project mesa's existing pass
(`mesa/frontend/src/liveRecognition.ts`, functions `soundKey`,
`COMMON_ENGLISH`, `buildVocabulary`, `correctVocabulary` -- mesa task 922).
Same letter foldings, same fold ordering, same gating thresholds (min token
length 4, min sound-key length 2), same ambiguity rule, same COMMON_ENGLISH
guard list. Consistency with mesa's already-fought edge cases matters more
than reinventing the algorithm here.

One deliberate divergence from mesa: mesa substitutes the vocabulary's stored
spelling verbatim, on the theory that its correction is *introducing* a
proper noun's real spelling into browser text. auris instead preserves the
matched word's original capitalisation pattern -- Titlecase in, Titlecase out
("Corvex tap failed" -> "Qorvex tap failed", not "qorvex tap failed"); ALLCAPS
in, ALLCAPS out; anything else gets the canonical (lowercase, as stored in the
hotwords file) spelling. auris is correcting mid-sentence dictated transcript
text, not injecting a name into UI chrome, so keeping the sentence's own
casing reads naturally. Scoring in `score.py` lowercases both sides before
comparing, so this choice cannot affect any metric -- it only affects the
text a person would actually read.

A second, larger divergence: mesa's sound-key match is used ALONE as the
correction gate, and at dictionary scale that is not safe for auris. auris's
terms are short and vowel-light, so their sound keys come out at only 2
characters (`mesa`->`ms`, `auris`->`ar`, `khora`->`kr`, `helios`->`hl`), and a
2-character key is a common prefix of hundreds of ordinary words -- no
curated COMMON_ENGLISH list, mesa's 237 words included, can enumerate enough
of the language to block all of them. A scan of `/usr/share/dict/web2`
(every word of at least 4 letters not already in COMMON_ENGLISH) found 321
words the sound-key-only rule would rewrite, including real sentences like
"I saw a mouse in the house" -> "...a mesa in the house", "The town hall
meeting starts soon" -> "...the town helios meeting", and "That was a
curious choice" -> "...a khora choice". This is a scale problem, not a
mesa-porting bug: mesa's browser context and small COMMON_ENGLISH list were
tuned for mesa's own vocabulary, and the fold behaves exactly as mesa
designed it to -- it is simply too permissive for a wider, unscreened corpus.

auris therefore requires a SECOND, independent gate before a correction
fires: the sound keys must match (mesa's rule, unchanged) AND the matched
word must be within edit distance 2 of the term's spelling
(`_levenshtein(word.lower(), term) <= 2`, plain letters, no keys involved).
This is a pure tightening -- it can only reject a correction the sound-key
rule would have made, never add one that rule would not have -- so every
real mishearing found in `docs/spike-results.md` still passes (e.g. "corvex" is
edit distance 1 from "qorvex"; "Aorus" is distance 2 from "auris"). What it
newly rejects is the collateral damage above: "hall"/"hole"/"holy"/"hill" are
each 4+ edits from "helios", "cross"/"cherry"/"chair"/"query" are 4+ edits
from "khora", and so on -- close enough in *sound* to collide on a 2-3
character key, but nowhere near close enough in *spelling* to plausibly be
the same word misheard. One accepted loss from this gate: mesa's own
"chorus" -> "khora" fold (distance 3) no longer fires under auris. mesa wants
that correction for its own reasons; auris does not need it -- "chorus" is
not a mishearing any engine in this benchmark ever produces (it does not
appear in `docs/spike-results.md` section 3), and it is itself an ordinary English
word that should not be rewritten by default. `docs/correction.md` has the
full scan and the auris-specific COMMON_ENGLISH addendum this gate still
needed on top.

Term-to-term correction is deliberately not attempted: none of the six
hotword terms will ever be corrected to a *different* hotword term. Reasons:
(a) the pass has no acoustic or contextual evidence available to it -- by the
time it runs, all it has is decoded text, so it cannot distinguish a
correctly-decoded "qorvex" from a misdecoded "khora"; rewriting exact terms
would trade a rare error (an occasional missed correction) for a systematic
one (breaking every genuine mention of whichever term loses the coin flip).
(b) the observed khora->qorvex confusion in RESULTS.md came only from
whisper's initial *prompt* pulling decoded text toward any prompt term, a
failure mode specific to that mechanism -- the chosen engine (parakeet with
hotword biasing, per docs/engine.md) does not exhibit it; no parakeet run at
a usable boost inserted a name that was not spoken. (c) structurally, the
ported algorithm already makes this impossible on its own: the six terms fold
to six distinct sound keys (asserted below), and `build_vocabulary`'s
ambiguity rule means that if two *did* ever collide on a key, the key would
be dropped entirely rather than guessed at -- see `_selftest`.

Term source: terms come from the hotwords file that also feeds the recognizer
(`bench/fixtures/hotwords.txt` by default), so the correction vocabulary and
the biasing vocabulary can never drift apart. `read_hotword_terms` parses one
`term :boost` line per term and discards the boost -- only the term list is
shared, not the tuning.

Usage:
    python3 vocab_correct.py --hotwords F IN.tsv       # corrected id<TAB>text TSV to stdout
    python3 vocab_correct.py --hotwords F --text "..."  # corrected sentence to stdout
    python3 vocab_correct.py --selftest
"""
import re
import sys

# --- term source -----------------------------------------------------------

DEFAULT_HOTWORDS_PATH = "bench/fixtures/hotwords.txt"


def read_hotword_terms(path):
    """Parse a hotwords file (`term :boost` per line, blank lines and `#`
    comments ignored) and return the list of terms, boost discarded."""
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            term = line.split(":", 1)[0].strip()
            if term:
                terms.append(term)
    return terms


# --- sound_key: faithful port of mesa's soundKey ----------------------------

def sound_key(word):
    """A rough phonetic fold (ported from mesa's `soundKey`, mesa task 922),
    used to catch a recognizer mishearing a name for an ordinary word (or
    another spelling) that sounds like it -- "khora" as "chorus", "helios" as
    "helius".

    Plain soundex will not do: it keeps the first *letter* rather than the
    first *sound*, so "chorus" (C-) and "khora" (K-) are already different
    codes before either word is folded any further. Metaphone comes closer
    but maps `ch` to `x`, landing "chorus" and "khora" on different codes
    again. This fold instead normalises the *spelling* toward the sound
    first -- `ch`, `kh`, `ck` and `qu` all become `k`, the same target `c`
    itself falls to except before `e`/`i`/`y` -- and only then keeps one
    representative letter, so both words collapse onto the same key.

    The steps, in the order that makes that collapse happen:
    1. Lowercase and strip everything that is not a-z -- punctuation and
       digits carry no sound.
    2. Fold digraphs and single letters that spell one sound multiple ways,
       left to right over the whole string. `gh` is dropped outright (a
       silent letter pair); `x` becomes `ks` and `z` becomes `s` because both
       are just voiced/voiceless spellings of sounds already in the
       alphabet; `y` is left as a vowel, treated like `a`/`e`/`i`/`o`/`u` by
       the vowel-stripping step below. This has to run before step 4, or
       `ch`/`kh`/`ck` would already have lost the letter that makes them a
       digraph.
    3. Drop one trailing `s` -- the recognizer's stray plural/sibilant
       ("helius" for "Helios") must not be what keeps two spellings apart.
    4. Keep the first character verbatim (soundex's one idea worth keeping),
       then drop every vowel *after* it -- vowels are the least reliably
       heard part of a word.
    5. Collapse runs of the same character to one, the way soundex does for a
       doubled consonant.

    Step 5 (collapse doubled letters) runs *before* step 4 (drop vowels), and
    the order is the whole difference between a fold that works and one that
    eats the flagship case (mesa task 922). Collapsing afterwards would merge
    two consonants a vowel had kept apart: `kokoro` strips to `kkr` and then
    to `kr`, which is exactly `khora`'s key -- so the ambiguity rule in
    `build_vocabulary` would cancel both, and the one mishearing this whole
    function exists to correct would stop being correctable. A vowel between
    two of the same consonant is a syllable, and soundex has always counted
    it. Preserve this ordering exactly.
    """
    s = re.sub(r"[^a-z]", "", word.lower())
    s = s.replace("gh", "")
    s = s.replace("ph", "f")
    s = s.replace("ch", "k")
    s = s.replace("kh", "k")
    s = s.replace("ck", "k")
    s = s.replace("qu", "k")
    s = s.replace("q", "k")
    s = s.replace("wh", "w")
    s = s.replace("x", "ks")
    s = s.replace("z", "s")
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    # Doubled letters collapse before the vowels come out, not after (see
    # docstring above) -- this ordering is what keeps kokoro's key distinct
    # from khora's.
    s = re.sub(r"(.)\1+", r"\1", s)
    s = re.sub(r"s$", "", s)
    if len(s) > 1:
        s = s[0] + re.sub(r"[aeiouy]", "", s[1:])
    return s


# --- COMMON_ENGLISH: faithful port -------------------------------------------

COMMON_ENGLISH = frozenset([
    # top function words
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
    "his", "from", "they", "she", "her", "been", "than", "its", "who", "did",
    "yes", "get", "has", "him", "how", "man", "new", "now", "old", "see",
    "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she",
    "too", "use", "want", "need", "will", "well", "were", "when", "what",
    "where", "which", "while", "would", "could", "should", "about", "after",
    "again", "against", "because", "before", "being", "below", "between",
    "both", "down", "during", "each", "few", "further", "here", "into",
    "itself", "just", "more", "most", "once", "only", "other", "over",
    "own", "same", "some", "such", "then", "there", "these", "those",
    "through", "under", "until", "very", "your", "yours", "ours", "theirs",
    "them", "their", "have", "having", "does", "doing", "done", "shall",
    "must", "can", "cannot", "like", "make", "made", "take", "took",
    "come", "came", "go", "goes", "went", "gone", "look", "looked",
    "give", "gave", "find", "found", "know", "knew", "think", "thought",
    "good", "great", "little", "long", "right", "still", "never", "always",
    "today", "tomorrow", "yesterday", "time", "year", "work", "life",
    "world", "hand", "part", "place", "case", "week", "point", "fact",
    "group", "number", "room", "area", "money", "story", "water", "family",
    "word", "body", "music", "level", "child", "eye", "day", "thing",
    "people", "name", "home", "country", "company", "system", "program",
    "question", "government", "power", "issue", "side", "kind", "head",
    "house", "service", "friend", "father", "mother", "sister", "brother",
    # sound-alikes to the words we correct *toward* -- the actual guard
    "course", "cores", "chores", "corps", "care", "call", "called", "calls",
    "class", "close", "closed", "coarse", "chore", "core",
    "cause", "cost", "cold", "code", "cloud", "crowd", "clock", "clerk",
    "quote", "quiet", "quick", "quite", "question",
    "help", "helper", "health", "held", "hell", "hello",
    "lock", "locked", "lucky", "local", "logic",
    "saw", "sonnet", "song", "sound", "south", "shape", "sharp",
    "open", "opens", "opened", "opening", "opus", "office", "often",
    "clip", "client", "clean",
])


# --- build_vocabulary: faithful port -----------------------------------------

def build_vocabulary(terms):
    """Build a sound-key -> canonical-term correction table from `terms`
    (ported from mesa's `buildVocabulary`, mesa task 922). Each name is split
    on non-letters into tokens; each token is its own candidate. A token
    under 4 characters, one in COMMON_ENGLISH, or one whose sound key comes
    out under 2 characters, is dropped. If two different surviving tokens
    land on the same sound key with different canonical spellings, the key
    is dropped entirely rather than guessed at."""
    claims = {}
    ambiguous = set()
    for name in terms:
        for token in re.split(r"[^A-Za-z]+", name):
            if len(token) < 4:
                continue
            if token.lower() in COMMON_ENGLISH:
                continue
            key = sound_key(token.lower())
            if len(key) < 2:
                continue
            existing = claims.get(key)
            if existing is None:
                claims[key] = token
            elif existing.lower() != token.lower():
                ambiguous.add(key)
    for key in ambiguous:
        claims.pop(key, None)
    return claims


# --- auris addendum to COMMON_ENGLISH ---------------------------------------

# auris's own addition on top of mesa's COMMON_ENGLISH, kept as a SEPARATE
# constant so the ported set above stays diff-able against mesa's. Found by
# scanning /usr/share/dict/web2 for words the sound-key + edit-distance gate
# (see module docstring) still lets through, and that are common enough in
# ordinary dictation to be worth guarding explicitly rather than accepting as
# collateral. See docs/correction.md for the full scan.
#
# Deliberately NOT included, despite passing both gates, because each is a
# real mishearing this pass must still fix -- adding it here would silence a
# required correction, not just block collateral damage: "cora" and "kora"
# (both literal `khora` misdecodes seen in the raw whisper transcripts, e.g.
# spike/results/raw/whisper-medium.en-noprompt.tsv u07 "blocked on Kora"),
# and "auras" (a literal `auris` misdecode, docs/spike-results.md section 3,
# whisper medium.en no-prompt: "under auras"). Each is also an ordinary
# English word (a person's name, a musical instrument, the plural of
# "aura"), so this pass will occasionally rewrite a genuine use of one of
# these three words -- that trade is accepted because the alternative is
# breaking a correction the whole feature exists to make.
AURIS_COMMON_ENGLISH = frozenset([
    "muse", "muses", "messy", "aria", "arias", "aura", "masa", "mesas",
])


# --- levenshtein: the second correction gate --------------------------------

def _levenshtein(a, b):
    """Plain stdlib edit distance between two strings, no third-party deps.
    Used as the second, independent gate `correct_text` applies on top of
    mesa's sound-key match (see module docstring): a 2-character sound key
    collides with hundreds of ordinary words, and edit distance on the raw
    letters is what tells a real mishearing ("corvex", distance 1 from
    "qorvex") apart from an unrelated word that merely sounds similar
    ("chair", distance 4 from "khora")."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


MAX_EDIT_DISTANCE = 2


# --- correct_text: faithful port with the casing divergence -----------------

_WORD_RE = re.compile(r"[A-Za-z']+")
_LEADING_APOSTROPHES_RE = re.compile(r"^'+")
_POSSESSIVE_RE = re.compile(r"'s$", re.IGNORECASE)
_TRAILING_APOSTROPHES_RE = re.compile(r"'+$")


def _match_case(canonical, original):
    """Apply `original`'s capitalisation pattern to `canonical` (the
    divergence from mesa documented in the module docstring): ALLCAPS stays
    ALLCAPS, Titlecase stays Titlecase, anything else gets the canonical
    spelling as stored."""
    if original.isupper():
        return canonical.upper()
    if original[:1].isupper() and original[1:].islower():
        return canonical[:1].upper() + canonical[1:]
    return canonical


def correct_text(text, vocab):
    """Rewrite whole words in `text` toward `vocab` (ported from mesa's
    `correctVocabulary`, mesa task 922). Operates on `[A-Za-z']+` spans, so
    all punctuation and whitespace pass through unchanged. A word under 4
    characters, or one whose lowercase form is in COMMON_ENGLISH (mesa's set
    plus auris's own addendum), is left alone without even computing a sound
    key. Otherwise its sound key is looked up, and it is replaced only when
    BOTH there is a sound-key hit whose spelling differs from the word
    case-insensitively AND the word is within `MAX_EDIT_DISTANCE` letters of
    the hit term (see module docstring for why the second gate exists); the
    replacement carries the matched word's own capitalisation pattern (see
    `_match_case`).

    Leading and trailing apostrophes are split off before any of that and
    reattached after. A trailing possessive `'s` is handled the same way for
    the same reason: the vocabulary stores bare terms ("kokoro", not
    "kokoro's"), so matching and replacing on the whole span would either
    miss the term (the `'s` changes its sound key) or silently drop the
    possessive from the output. "Kakoro's" must become "Kokoro's", not
    "Kokoro". A bare trailing apostrophe ("Corvex'") is handled the same way
    so it survives the round trip instead of being silently dropped. A
    leading apostrophe ("'Korra") is split off purely so it does not defeat
    the Titlecase detection in `_match_case` (which otherwise sees a
    lowercase-first "original" and falls back to the bare canonical
    spelling)."""
    if not vocab:
        return text

    def repl(m):
        word = m.group(0)
        lead_m = _LEADING_APOSTROPHES_RE.match(word)
        lead = lead_m.group(0) if lead_m else ""
        rest = word[len(lead):]

        suffix = ""
        base = rest
        possessive = _POSSESSIVE_RE.search(rest)
        if possessive:
            base = rest[: possessive.start()]
            suffix = rest[possessive.start():]
        else:
            trailing = _TRAILING_APOSTROPHES_RE.search(rest)
            if trailing:
                base = rest[: trailing.start()]
                suffix = rest[trailing.start():]

        if len(base) < 4:
            return word
        lower = base.lower()
        if lower in COMMON_ENGLISH or lower in AURIS_COMMON_ENGLISH:
            return word
        hit = vocab.get(sound_key(lower))
        if hit is None or hit.lower() == lower:
            return word
        if _levenshtein(lower, hit.lower()) > MAX_EDIT_DISTANCE:
            return word
        return lead + _match_case(hit, base) + suffix

    return _WORD_RE.sub(repl, text)


# --- CLI ---------------------------------------------------------------------

def _selftest():
    # sound_key: the worked examples the fold is designed to satisfy.
    assert sound_key("chorus") == sound_key("khora")
    assert sound_key("helius") == sound_key("helios")
    assert sound_key("helium") != sound_key("helios")
    assert sound_key("kokoro") != sound_key("khora")

    terms = read_hotword_terms(DEFAULT_HOTWORDS_PATH)
    assert terms == ["mesa", "auris", "khora", "qorvex", "helios", "kokoro"], terms

    vocab = build_vocabulary(terms)

    # (c) structural guarantee behind the term-to-term decision: all six
    # terms fold to distinct sound keys, so build_vocabulary could never
    # confuse one hotword term for another even if it tried.
    keys = [sound_key(t) for t in terms]
    assert len(set(keys)) == len(keys), keys

    # 1. Ordinary English is left alone -- the headline requirement.
    for s in [
        "core of the problem",
        "the core of it",
        "of course",
        "cores",
        "care",
        "corps",
        "coarse",
        "chores",
        "quote",
        "quiet",
        "hello",
        "help",
        "open a bug for that",
    ]:
        assert correct_text(s, vocab) == s, s

    # 1b. The edit-distance gate: real sentences that collide with a hotword
    # term's sound key but are nowhere near its spelling must survive
    # untouched. Each of these was an actual collateral rewrite before the
    # gate existed (see docs/correction.md for the full dictionary scan).
    for s in [
        "I saw a mouse in the house.",
        "The town hall meeting starts soon.",
        "That was a curious choice.",
        "We drove past the quarry.",
        "Ask her a query about it.",
        "The Quaker meeting is Sunday.",
    ]:
        assert correct_text(s, vocab) == s, s

    # 1c. Regression tripwire for the defect the edit-distance gate exists to
    # fix: a fixed inline list of collateral words pulled from the original
    # /usr/share/dict/web2 scan (321 ordinary words the sound-key-only rule
    # rewrote -- see docs/correction.md), checked without touching the
    # dictionary file itself so this stays portable across hosts. If someone
    # later loosens MAX_EDIT_DISTANCE or turns the AND into an OR, this is
    # what catches it -- every one of these shares a sound key with a
    # hotword term but is nowhere near it in spelling, and must stay
    # untouched.
    collateral_words = [
        "mouse", "mice", "muse", "maze",
        "hall", "hole", "holy", "holly", "hill", "hull", "heel", "hail",
        "cross", "crass", "cure", "curry", "cherry", "cheer", "chair", "choir",
        "quarry", "query", "queer", "quaker", "curious", "chorus",
    ]
    for word in collateral_words:
        sentence = f"we talked about the {word} yesterday"
        assert correct_text(sentence, vocab) == sentence, word

    # 2. Real mishearings, from docs/spike-results.md section 3, are fixed.
    for bad, good in [
        ("qora", "khora"), ("Quora", "khora"), ("Korra", "khora"), ("Cora", "khora"),
        ("Corvex", "qorvex"), ("korvex", "qorvex"),
        ("Aurus", "auris"), ("Aorus", "auris"), ("auras", "auris"), ("AORUS", "auris"),
        ("Kakoro", "kokoro"), ("Kakaro", "kokoro"),
        ("Helius", "helios"),
    ]:
        corrected = correct_text(bad, vocab)
        assert corrected.lower() == good, (bad, corrected, good)

    # 3. Surrounding text and casing pattern are preserved exactly.
    assert correct_text("blocked on Korra.", vocab) == "blocked on Khora."
    assert correct_text("Kakoro's latency numbers", vocab) == "Kokoro's latency numbers"

    # 3b. Apostrophe handling: a bare trailing apostrophe survives (it used
    # to be silently dropped), and a leading apostrophe no longer defeats
    # Titlecase detection (it used to fall back to the bare lowercase
    # canonical spelling, dropping the apostrophe too).
    assert correct_text("Corvex' settings", vocab) == "Qorvex' settings"
    assert correct_text("blocked on 'Korra again", vocab) == "blocked on 'Khora again"

    # 4. Exact terms are fixed points; qorvex is never rewritten to khora.
    sentence = "mesa auris khora qorvex helios kokoro"
    assert correct_text(sentence, vocab) == sentence
    assert correct_text("qorvex", vocab) == "qorvex"

    # 5. Short words untouched.
    assert correct_text("cor", vocab) == "cor"

    # 6. Ambiguity rule: two invented terms sharing a sound key are both
    # dropped, so neither is corrected toward.
    ambiguous_vocab = build_vocabulary(["zamble", "zambel"])
    assert correct_text("I heard zambl today", ambiguous_vocab) == "I heard zambl today"
    assert sound_key("zamble") == sound_key("zambel")
    assert sound_key("zamble") not in ambiguous_vocab

    # 7. Accepted loss from the edit-distance gate: mesa's own
    # "chorus" -> "khora" fold (same sound key, but edit distance 3) no
    # longer fires for auris. Not a mishearing any engine in this benchmark
    # produces, and "chorus" is itself ordinary English -- see
    # docs/correction.md.
    assert sound_key("chorus") == sound_key("khora")
    assert _levenshtein("chorus", "khora") > MAX_EDIT_DISTANCE
    assert correct_text("the chorus swelled", vocab) == "the chorus swelled"

    # 8. _read_tsv skips a malformed (no-tab) line rather than crashing, and
    # still returns every well-formed row around it.
    import tempfile
    import os
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as f:
            f.write("u01\tfirst line\nmalformed no tab here\nu02\tsecond line\n")
        rows = _read_tsv(path)
        assert rows == [("u01", "first line"), ("u02", "second line")], rows
    finally:
        os.remove(path)

    print("selftest OK", file=sys.stderr)


def _read_tsv(path):
    """Read `id<TAB>text` lines. A line with no tab is malformed input, not
    something this pass should crash the whole run over -- it is skipped
    with a warning to stderr, and every well-formed line is still
    processed."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                print(f"{path}:{lineno}: no tab, skipping: {line!r}", file=sys.stderr)
                continue
            uid, text = line.split("\t", 1)
            rows.append((uid, text))
    return rows


def main(argv):
    if "--selftest" in argv:
        _selftest()
        return 0

    hotwords_path = DEFAULT_HOTWORDS_PATH
    text = None
    tsv_path = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--hotwords":
            i += 1
            hotwords_path = argv[i]
        elif arg == "--text":
            i += 1
            text = argv[i]
        else:
            tsv_path = arg
        i += 1

    vocab = build_vocabulary(read_hotword_terms(hotwords_path))

    if text is not None:
        print(correct_text(text, vocab))
        return 0

    if tsv_path is None:
        print(f"usage: {sys.argv[0]} --hotwords F IN.tsv | --hotwords F --text '...' | --selftest",
              file=sys.stderr)
        return 1

    for uid, line_text in _read_tsv(tsv_path):
        print(f"{uid}\t{correct_text(line_text, vocab)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
