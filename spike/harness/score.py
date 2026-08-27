#!/usr/bin/env python3
"""WER + name-accuracy scorer for auris ASR fixtures.

Usage:
    python3 spike/harness/score.py REF.tsv HYP.tsv
    python3 spike/harness/score.py --selftest

Input format: both REF.tsv and HYP.tsv are tab-separated `id<TAB>text` lines,
one utterance per line, matched by id.

Normalisation (applied to both reference and hypothesis text before scoring):
  1. Lowercase.
  2. Strip all punctuation (anything that isn't a letter, digit, or
     whitespace).
  3. Collapse repeated whitespace to single spaces and strip ends.
  4. Number normalisation so digit and spelled-out forms compare equal:
       - Any spelled-out number word run (ones, teens, tens, "hundred") is
         converted into the digit sequence it stands for. "nine two four"
         is three separate tokens ("nine","two","four") which convert to
         "9 2 4" -- digit-by-digit, matching how people dictate numbers.
         "nine hundred twenty four" is a single number word run and
         converts to the number's value split into digits: "9 2 4".
       - Any digit run already in the text ("924") is expanded into its
         individual digits, space separated ("9 2 4"), so both sides land
         in the same "sequence of single digits" space regardless of
         whether the source wrote "924" or spoke "nine two four" or
         "nine hundred twenty four".
  This is applied identically to ref and hyp so word counts / alignment
  stay comparable.

Metrics:
  - Overall WER via Levenshtein edit distance over the normalised word
    lists (concatenated across all matched utterances), reporting
    substitutions/deletions/insertions and total reference word count.
  - "Name accuracy": recall of the mesa vocabulary terms
    {mesa, auris, khora, qorvex, helios, kokoro} in the hypothesis.
    Per utterance and per term: correct = min(ref_count, hyp_count) of
    that term (as whole normalised words), missed = ref_count - correct.
    Aggregated across all utterances into per_term and overall
    name_accuracy = name_correct / name_total.

Output: a single JSON object printed to stdout.
"""
import json
import re
import sys

VOCAB_TERMS = ["mesa", "auris", "khora", "qorvex", "helios", "kokoro"]

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000}
NUMBER_WORDS = set(ONES) | set(TENS) | set(SCALES)


def _words_to_number(words):
    """Convert a contiguous run of number words (e.g. ['nine','hundred',
    'twenty','four']) into an integer, using standard English number-word
    composition rules (ones/tens add, scale words multiply the running
    total-so-far and get added to an accumulator)."""
    total = 0
    current = 0
    for w in words:
        if w in ONES:
            current += ONES[w]
        elif w in TENS:
            current += TENS[w]
        elif w in SCALES:
            current = current * SCALES[w] if current else SCALES[w]
            if SCALES[w] >= 1000:
                total += current
                current = 0
    return total + current


def _spell_numbers_to_digits(tokens):
    """Replace contiguous runs of number words in a token list with their
    digit-string equivalent, split into individual digit tokens.

    A run containing a tens or scale word ("twenty", "hundred", ...) is
    treated as one composed number ("nine hundred twenty four" -> 924 ->
    "9 2 4"). A run made only of ones/teens words (0-19) is instead
    dictation-style digit-by-digit -- each word converts to its own value
    independently ("nine two four" -> "9", "2", "4"), not summed.
    """
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] in NUMBER_WORDS:
            j = i
            while j < n and tokens[j] in NUMBER_WORDS:
                j += 1
            run = tokens[i:j]
            if any(w in TENS or w in SCALES for w in run):
                value = _words_to_number(run)
                out.extend(list(str(value)))
            else:
                for w in run:
                    out.extend(list(str(ONES[w])))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def _expand_digit_runs(tokens):
    """Split any token that is a run of digits into individual digit
    tokens, e.g. '924' -> ['9','2','4']. Leaves non-digit tokens alone."""
    out = []
    for tok in tokens:
        if tok.isdigit():
            out.extend(list(tok))
        else:
            out.append(tok)
    return out


def normalize(text):
    """Lowercase, strip punctuation, collapse whitespace, and map numbers
    (digit or spelled-out) to a shared digit-sequence representation.
    Returns a list of normalised word tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = _spell_numbers_to_digits(tokens)
    tokens = _expand_digit_runs(tokens)
    return tokens


def levenshtein_ops(ref, hyp):
    """Word-level Levenshtein alignment. Returns (distance, sub, del, ins)."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = min edits to turn ref[:i] into hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrack to classify edits.
    i, j = n, m
    sub = del_ = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            del_ += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return dp[n][m], sub, del_, ins


def read_tsv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, text = line.split("\t", 1)
            rows[uid] = text
    return rows


def score(ref_path, hyp_path):
    ref_rows = read_tsv(ref_path)
    hyp_rows = read_tsv(hyp_path)

    total_sub = total_del = total_ins = total_ref_words = 0
    per_term = {t: {"ref": 0, "correct": 0} for t in VOCAB_TERMS}
    per_utterance = []

    for uid in sorted(ref_rows):
        ref_text = ref_rows[uid]
        hyp_text = hyp_rows.get(uid, "")

        ref_tokens = normalize(ref_text)
        hyp_tokens = normalize(hyp_text)

        _, sub, del_, ins = levenshtein_ops(ref_tokens, hyp_tokens)
        total_sub += sub
        total_del += del_
        total_ins += ins
        total_ref_words += len(ref_tokens)

        utt_name_total = 0
        utt_name_correct = 0
        for term in VOCAB_TERMS:
            ref_count = ref_tokens.count(term)
            if ref_count == 0:
                continue
            hyp_count = hyp_tokens.count(term)
            correct = min(ref_count, hyp_count)
            per_term[term]["ref"] += ref_count
            per_term[term]["correct"] += correct
            utt_name_total += ref_count
            utt_name_correct += correct

        utt_wer = (sub + del_ + ins) / len(ref_tokens) if ref_tokens else 0.0
        per_utterance.append({
            "id": uid,
            "wer": utt_wer,
            "sub": sub,
            "del": del_,
            "ins": ins,
            "ref_words": len(ref_tokens),
            "name_total": utt_name_total,
            "name_correct": utt_name_correct,
        })

    wer = (total_sub + total_del + total_ins) / total_ref_words if total_ref_words else 0.0

    name_total = sum(v["ref"] for v in per_term.values())
    name_correct = sum(v["correct"] for v in per_term.values())
    name_accuracy = name_correct / name_total if name_total else 0.0

    per_term_out = {
        t: {
            "ref": v["ref"],
            "correct": v["correct"],
            "accuracy": (v["correct"] / v["ref"]) if v["ref"] else 0.0,
        }
        for t, v in per_term.items()
    }

    return {
        "wer": wer,
        "sub": total_sub,
        "del": total_del,
        "ins": total_ins,
        "ref_words": total_ref_words,
        "name_accuracy": name_accuracy,
        "name_total": name_total,
        "name_correct": name_correct,
        "per_term": per_term_out,
        "per_utterance": per_utterance,
    }


def _selftest():
    # normalize: digits and spelled-out numbers land in the same space.
    assert normalize("task 924") == ["task", "9", "2", "4"]
    assert normalize("task nine two four") == ["task", "9", "2", "4"]
    assert normalize("task nine hundred twenty four") == ["task", "9", "2", "4"]
    assert normalize("Auris, khora!") == ["auris", "khora"]

    # levenshtein_ops: simple known case.
    dist, sub, del_, ins = levenshtein_ops(["a", "b", "c"], ["a", "x", "c"])
    assert dist == 1 and sub == 1 and del_ == 0 and ins == 0, (dist, sub, del_, ins)

    dist, sub, del_, ins = levenshtein_ops(["a", "b", "c"], ["a", "b"])
    assert dist == 1 and sub == 0 and del_ == 1 and ins == 0, (dist, sub, del_, ins)

    dist, sub, del_, ins = levenshtein_ops(["a", "b"], ["a", "b", "c"])
    assert dist == 1 and sub == 0 and del_ == 0 and ins == 1, (dist, sub, del_, ins)

    print("selftest OK", file=sys.stderr)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)

    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} REF.tsv HYP.tsv", file=sys.stderr)
        sys.exit(1)

    result = score(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
