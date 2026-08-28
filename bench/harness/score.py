#!/usr/bin/env python3
"""WER + name-accuracy + punctuation-boundary scorer for auris ASR fixtures.

This is spike/harness/score.py plus a sentence-boundary punctuation F1
metric and per-utterance output. Do not edit spike/harness/score.py from
here; it stays the frozen reference. Everything WER- and name-accuracy
related below is unchanged from that file.

Usage:
    python3 bench/harness/score.py REF.tsv HYP.tsv
    python3 bench/harness/score.py REF.tsv HYP.tsv --csv per_utterance.csv
    python3 bench/harness/score.py --selftest

Input format: both REF.tsv and HYP.tsv are tab-separated `id<TAB>text` lines,
one utterance per line, matched by id.

Normalisation (applied to both reference and hypothesis text before scoring
WER and name accuracy):
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
  - "Name accuracy": alignment-based F1 of the mesa vocabulary terms
    {mesa, auris, khora, qorvex, helios, kokoro} between ref and hyp.
    Each utterance is word-aligned (Levenshtein backtrack); a vocab term
    only counts as a hit when it lands on a "match" op in that alignment.
    A ref vocab term that is substituted or deleted is a miss (hurts
    recall); a hyp vocab term that is inserted, or is the hyp side of a
    substitution, is a false positive (hurts precision). This means a
    hypothesis that inserts vocabulary terms it never heard (babble) or
    swaps one vocab term for another (khora -> qorvex) is penalised, not
    just credited for whatever it happens to overlap on. name_accuracy is
    the F1 of precision and recall aggregated across all terms and
    utterances; per_term breaks the same precision/recall/f1 down by term.
  - "Punctuation sentence-boundary F1": see `_sentence_boundaries` and
    `_score_punct` docstrings below for the precise definition. Summary:
    operates on RAW (un-normalised) text; a boundary is a position after a
    word that ends in '.', '!', or '?'; the trailing boundary at the very
    end of an utterance is excluded on both sides; boundaries are anchored
    to a simple-normalised (lowercase, punctuation-stripped, no number
    expansion) word index and compared via a Levenshtein word alignment
    between ref and hyp, so a hyp boundary counts as a true positive only
    when it maps onto a ref boundary at the same aligned word position.
    Also reports the cruder `any_terminal_punct_rate` (fraction of
    utterances whose hyp has any terminal punctuation mark at all) and raw
    comma counts, neither of which is alignment-based -- both are stated
    as crude, blunt signals, not F1-quality metrics.

Output: a single JSON object printed to stdout. `per_utterance` is now a
dict keyed by utterance id (each with ref/hyp text, WER components, vocab
term hits, and punctuation tp/fp/fn). `--csv PATH` additionally writes a
per-utterance CSV.
"""
import csv
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


def align_ops(ref, hyp):
    """Word-level Levenshtein alignment. Returns a list of (op, ref_idx,
    hyp_idx) tuples in left-to-right order. op is one of "match", "sub",
    "del", "ins"; ref_idx/hyp_idx are the 0-based indices into ref/hyp,
    None where not applicable ("del" has no hyp_idx, "ins" has no
    ref_idx). Backtracking prefers a match whenever tokens are equal and
    the DP cost allows it."""
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

    # Backtrack to classify edits, then reverse into left-to-right order.
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1

    ops.reverse()
    return ops


def levenshtein_ops(ref, hyp):
    """Word-level Levenshtein alignment. Returns (distance, sub, del, ins)."""
    ops = align_ops(ref, hyp)
    sub = sum(1 for op, _, _ in ops if op == "sub")
    del_ = sum(1 for op, _, _ in ops if op == "del")
    ins = sum(1 for op, _, _ in ops if op == "ins")
    return sub + del_ + ins, sub, del_, ins


def _score_names(ref_tokens, hyp_tokens, per_term):
    """Score mesa vocabulary term hits/misses/false-positives for one
    utterance from its word alignment, accumulating ref/hyp/correct
    counts into per_term (keyed by VOCAB_TERMS). Returns (hits, misses,
    false_positives) for this utterance."""
    hits = misses = false_pos = 0
    for op, ri, hi in align_ops(ref_tokens, hyp_tokens):
        if op == "match":
            term = ref_tokens[ri]
            if term in VOCAB_TERMS:
                per_term[term]["ref"] += 1
                per_term[term]["hyp"] += 1
                per_term[term]["correct"] += 1
                hits += 1
            continue
        if ri is not None:
            term = ref_tokens[ri]
            if term in VOCAB_TERMS:
                per_term[term]["ref"] += 1
                misses += 1
        if hi is not None:
            term = hyp_tokens[hi]
            if term in VOCAB_TERMS:
                per_term[term]["hyp"] += 1
                false_pos += 1
    return hits, misses, false_pos


def _vocab_hit_list(ref_tokens, hyp_tokens):
    """Per-utterance detail for the vocab terms present in ref_tokens: a
    list of {"term": ..., "hit": bool} in ref order, hit True only when
    the term landed on a "match" alignment op (same rule _score_names
    uses for recall)."""
    out = []
    for op, ri, _hi in align_ops(ref_tokens, hyp_tokens):
        if ri is not None:
            term = ref_tokens[ri]
            if term in VOCAB_TERMS:
                out.append({"term": term, "hit": op == "match"})
    return out


def _simple_norm_word(word):
    """Lowercase and strip everything but letters/digits. Unlike
    normalize(), this does NOT expand numbers -- it exists only to anchor
    punctuation-boundary positions to a word index that's stable across
    ref/hyp regardless of number formatting or casing."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _sentence_boundaries(text):
    """Tokenize RAW text on whitespace and find sentence boundaries.

    A "sentence boundary" is a position after word index i in the word
    sequence, where the raw token at index i ends with '.', '!', or '?'.
    Positions are reported as indices into a simple-normalised word list
    (see _simple_norm_word) -- lowercase, punctuation stripped, no number
    expansion -- so that a boundary can later be compared across ref and
    hyp texts of different lengths via a word alignment. A raw token that
    is pure punctuation (nothing alphanumeric survives stripping) is
    dropped rather than counted as its own word.

    The boundary following the very last word is always excluded: every
    utterance ends with one trivially (has a full stop or not, the end of
    the utterance is still "the end"), and counting it would inflate
    precision/recall for free.

    Returns (norm_tokens, boundary_indices) where boundary_indices is a
    set of indices into norm_tokens; index k means "a boundary follows
    norm_tokens[k]".
    """
    norm_tokens = []
    boundaries = set()
    for tok in text.split():
        nw = _simple_norm_word(tok)
        if not nw:
            continue
        norm_tokens.append(nw)
        if tok[-1] in ".!?":
            boundaries.add(len(norm_tokens) - 1)
    if norm_tokens:
        boundaries.discard(len(norm_tokens) - 1)
    return norm_tokens, boundaries


def _score_punct(ref_text, hyp_text):
    """Score sentence-boundary punctuation for one utterance.

    Boundaries are computed on raw ref/hyp text per _sentence_boundaries,
    then ref and hyp simple-normalised word lists are Levenshtein-aligned
    (align_ops) so each hyp word index maps to the ref word index it was
    matched or substituted against (insertions have no ref counterpart).
    A hyp boundary is a true positive if the ref word index it maps to is
    itself a ref boundary; otherwise it's a false positive. A ref
    boundary with no hyp boundary mapped onto it is a false negative.

    Returns a dict with tp/fp/fn/ref_boundaries/hyp_boundaries for this
    utterance.
    """
    ref_norm, ref_b = _sentence_boundaries(ref_text)
    hyp_norm, hyp_b = _sentence_boundaries(hyp_text)

    hi_to_ri = {hi: ri for op, ri, hi in align_ops(ref_norm, hyp_norm) if hi is not None}

    tp = 0
    fp = 0
    hit_ref = set()
    for hb in hyp_b:
        ri = hi_to_ri.get(hb)
        if ri is not None and ri in ref_b:
            tp += 1
            hit_ref.add(ri)
        else:
            fp += 1
    fn = len(ref_b - hit_ref)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ref_boundaries": len(ref_b),
        "hyp_boundaries": len(hyp_b),
    }


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


def score_rows(ref_rows, hyp_rows):
    total_sub = total_del = total_ins = total_ref_words = 0
    per_term = {t: {"ref": 0, "hyp": 0, "correct": 0} for t in VOCAB_TERMS}
    per_utterance = {}

    total_punct_tp = total_punct_fp = total_punct_fn = 0
    total_ref_boundaries = total_hyp_boundaries = 0
    any_punct_count = 0
    total_comma_ref = total_comma_hyp = 0
    n_utt = 0

    for uid in sorted(ref_rows):
        ref_text = ref_rows[uid]
        hyp_text = hyp_rows.get(uid, "")
        n_utt += 1

        ref_tokens = normalize(ref_text)
        hyp_tokens = normalize(hyp_text)

        _, sub, del_, ins = levenshtein_ops(ref_tokens, hyp_tokens)
        total_sub += sub
        total_del += del_
        total_ins += ins
        total_ref_words += len(ref_tokens)

        hits, misses, false_pos = _score_names(ref_tokens, hyp_tokens, per_term)
        vocab_hits = _vocab_hit_list(ref_tokens, hyp_tokens)

        punct = _score_punct(ref_text, hyp_text)
        total_punct_tp += punct["tp"]
        total_punct_fp += punct["fp"]
        total_punct_fn += punct["fn"]
        total_ref_boundaries += punct["ref_boundaries"]
        total_hyp_boundaries += punct["hyp_boundaries"]
        has_terminal_punct = bool(re.search(r"[.!?]", hyp_text))
        any_punct_count += 1 if has_terminal_punct else 0
        comma_ref = ref_text.count(",")
        comma_hyp = hyp_text.count(",")
        total_comma_ref += comma_ref
        total_comma_hyp += comma_hyp

        utt_wer = (sub + del_ + ins) / len(ref_tokens) if ref_tokens else 0.0
        per_utterance[uid] = {
            "id": uid,
            "ref": ref_text,
            "hyp": hyp_text,
            "wer": utt_wer,
            "sub": sub,
            "del": del_,
            "ins": ins,
            "ref_words": len(ref_tokens),
            "name_total": hits + misses,
            "name_correct": hits,
            "name_hyp_total": hits + false_pos,
            "name_false_pos": false_pos,
            "vocab_terms": vocab_hits,
            "punct_tp": punct["tp"],
            "punct_fp": punct["fp"],
            "punct_fn": punct["fn"],
            "ref_boundaries": punct["ref_boundaries"],
            "hyp_boundaries": punct["hyp_boundaries"],
            "hyp_has_terminal_punct": has_terminal_punct,
            "comma_ref": comma_ref,
            "comma_hyp": comma_hyp,
        }

    wer = (total_sub + total_del + total_ins) / total_ref_words if total_ref_words else 0.0

    name_total = sum(v["ref"] for v in per_term.values())
    name_hyp_total = sum(v["hyp"] for v in per_term.values())
    name_hits = sum(v["correct"] for v in per_term.values())
    name_missed = name_total - name_hits
    name_false_pos = name_hyp_total - name_hits
    name_recall = name_hits / name_total if name_total else 0.0
    name_precision = name_hits / name_hyp_total if name_hyp_total else 0.0
    name_f1 = (
        2 * name_precision * name_recall / (name_precision + name_recall)
        if (name_precision + name_recall) else 0.0
    )

    per_term_out = {}
    for t, v in per_term.items():
        precision = (v["correct"] / v["hyp"]) if v["hyp"] else 0.0
        recall = (v["correct"] / v["ref"]) if v["ref"] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_term_out[t] = {
            "ref": v["ref"],
            "hyp": v["hyp"],
            "correct": v["correct"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": f1,
        }

    # Punctuation boundary precision/recall/f1 follow the same "0 when the
    # denominator is 0" convention as name accuracy above, for consistency:
    # a ref or hyp with zero boundaries (e.g. a single-sentence utterance)
    # yields 0.0 for the corresponding side rather than an undefined value.
    punct_precision = (
        total_punct_tp / total_hyp_boundaries if total_hyp_boundaries else 0.0
    )
    punct_recall = (
        total_punct_tp / total_ref_boundaries if total_ref_boundaries else 0.0
    )
    punct_f1 = (
        2 * punct_precision * punct_recall / (punct_precision + punct_recall)
        if (punct_precision + punct_recall) else 0.0
    )
    any_terminal_punct_rate = any_punct_count / n_utt if n_utt else 0.0

    return {
        "wer": wer,
        "sub": total_sub,
        "del": total_del,
        "ins": total_ins,
        "ref_words": total_ref_words,
        "name_accuracy": name_f1,
        "name_f1": name_f1,
        "name_precision": name_precision,
        "name_recall": name_recall,
        "name_total": name_total,
        "name_correct": name_hits,
        "name_hyp_total": name_hyp_total,
        "name_hits": name_hits,
        "name_missed": name_missed,
        "name_false_pos": name_false_pos,
        "per_term": per_term_out,
        "punct_boundary_precision": punct_precision,
        "punct_boundary_recall": punct_recall,
        "punct_boundary_f1": punct_f1,
        "tp": total_punct_tp,
        "fp": total_punct_fp,
        "fn": total_punct_fn,
        "ref_boundaries": total_ref_boundaries,
        "hyp_boundaries": total_hyp_boundaries,
        "any_terminal_punct_rate": any_terminal_punct_rate,
        "comma_count_ref": total_comma_ref,
        "comma_count_hyp": total_comma_hyp,
        "per_utterance": per_utterance,
    }


def score(ref_path, hyp_path):
    return score_rows(read_tsv(ref_path), read_tsv(hyp_path))


def write_csv(result, path):
    """Write the per-utterance CSV: id, ref_words, wer, name_hits,
    name_misses, name_fp, punct_tp, punct_fp, punct_fn."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "id", "ref_words", "wer", "name_hits", "name_misses", "name_fp",
            "punct_tp", "punct_fp", "punct_fn",
        ])
        for uid, u in sorted(result["per_utterance"].items()):
            writer.writerow([
                uid,
                u["ref_words"],
                u["wer"],
                u["name_correct"],
                u["name_total"] - u["name_correct"],
                u["name_false_pos"],
                u["punct_tp"],
                u["punct_fp"],
                u["punct_fn"],
            ])


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

    # align_ops: reconstructs the same sub/del/ins counts as levenshtein_ops,
    # covers the full length of both sequences, and prefers match over sub
    # whenever tokens are equal.
    ops = align_ops(["a", "b", "c"], ["a", "x", "c"])
    assert [o[0] for o in ops] == ["match", "sub", "match"], ops
    assert ops[0] == ("match", 0, 0) and ops[2] == ("match", 2, 2), ops
    ops = align_ops(["a", "b", "c"], ["a", "b"])
    kinds = [o[0] for o in ops]
    assert kinds.count("match") == 2 and kinds.count("del") == 1, ops
    ops = align_ops(["a", "b"], ["a", "b", "c"])
    kinds = [o[0] for o in ops]
    assert kinds.count("match") == 2 and kinds.count("ins") == 1, ops

    # --- name scoring: alignment-based precision/recall/F1 ---

    # Perfect case: ref == hyp on an utterance with names -> P=R=F1=1.
    ref_rows = {"u": "hey mesa add a note to khora that headless mode is the default now"}
    hyp_rows = {"u": "hey mesa add a note to khora that headless mode is the default now"}
    r = score_rows(ref_rows, hyp_rows)
    assert r["name_precision"] == 1.0 and r["name_recall"] == 1.0 and r["name_f1"] == 1.0, r

    # Swap case: khora replaced by qorvex in the hyp. khora recall drops to
    # 0, qorvex precision drops to 0 (it's a hallucinated term, ref count 0).
    hyp_rows_swap = {"u": "hey mesa add a note to qorvex that headless mode is the default now"}
    r = score_rows(ref_rows, hyp_rows_swap)
    assert r["per_term"]["khora"]["recall"] == 0.0, r["per_term"]["khora"]
    assert r["per_term"]["qorvex"]["precision"] == 0.0, r["per_term"]["qorvex"]
    assert r["per_term"]["qorvex"]["ref"] == 0, r["per_term"]["qorvex"]

    # Insertion case: an extra vocab term (auris) appears where the ref
    # said something else -> hurts precision, counts as a false positive.
    ref_rows_ins = {"u": "create a task in mesa called wire up qorvex to the ios simulator"}
    hyp_rows_ins = {"u": "create a task in mesa called wire up korvex to the auris simulator"}
    r = score_rows(ref_rows_ins, hyp_rows_ins)
    assert r["name_precision"] < 1.0, r
    assert r["name_false_pos"] >= 1, r

    # Babble case (u08): the recall-only measure the old code used would
    # have scored this transcript as near-perfect on names even though it
    # is babble with no real recognition happening. The new alignment-based
    # F1 must catch that.
    ref_u08 = "qorvex tap failed twice on the android emulator, log it under auris and link it to task 902"
    hyp_u08 = "qorvex qo' qo' auris auris auris koko"
    ref_tokens = normalize(ref_u08)
    hyp_tokens = normalize(hyp_u08)
    old_style_correct = sum(
        min(ref_tokens.count(t), hyp_tokens.count(t)) for t in VOCAB_TERMS
    )
    old_style_total = sum(ref_tokens.count(t) for t in VOCAB_TERMS)
    old_style_recall = old_style_correct / old_style_total if old_style_total else 0.0
    assert old_style_recall >= 0.99, old_style_recall  # documents what the old metric said

    # This single utterance's hyp does contain one genuine "qorvex" in the
    # right place (recall == 1.0 -- both ref vocab tokens found a match),
    # so its F1 isn't driven all the way down by recall; it's precision
    # (hurt by the repeated "auris") that takes the hit. F1 still drops
    # from a "perfect" 1.0 under the old recall-only measure to ~0.67.
    r = score_rows({"u08": ref_u08}, {"u08": hyp_u08})
    assert r["name_accuracy"] < 0.70, r["name_accuracy"]

    # Extend to a full babble run: u01-u07 are the real parakeet decode from
    # spike/results/hotwords/hot_bpe_7.0.tsv (contextual biasing pushed to
    # score 7.0, where the beam starts emitting mesa terms as filler), and
    # u08 is the score-10 transcript quoted in spike/RESULTS.md section 6.
    # References are the real ones from spike/fixtures/utterances.tsv. This
    # is the primary "scores badly" demonstration -- observed new F1 is
    # 0.4286 (P=0.3103, R=0.6923), vs 0.6923 under the old recall-only
    # measure, which ranked this garbage above half the real benchmark.
    babble_ref = {
        "u01": "close task 924 and open a follow up under auris for the parakeet spike",
        "u02": "hey mesa add a note to khora that headless mode is the default now",
        "u03": "create a task in mesa called wire up qorvex to the ios simulator and assign it to me",
        "u04": "helios needs an index refresh before task 118 can start",
        "u05": "remind me to check kokoro's latency numbers against whisper before we pick a winner",
        "u06": (
            "so basically what happened is auris transcribed helios as hell EOS "
            "again which is the third time this week and I think we need a "
            "custom vocabulary list for mesa terms before we ship, can you open "
            "a bug for that"
        ),
        "u07": "mark task 730 done and task 731 blocked on khora",
        "u08": ref_u08,
    }
    babble_hyp = {
        "u01": "Close task 9 hundred 4 qorvex mesa khora qoris- qorake kid qor",
        "u02": "Hey mesa add a note to qor that edless mote is the default now.",
        "u03": "Create a task in mesa khora qorvex to the auris auris it to mesa",
        "u04": "Helios mesa mesa qorvex khora heli can khora q",
        "u05": "indind mesa qi kokoro's aum mesa mesa isper qor",
        "u06": (
            "so basically what helios auris transcribed helios' auris mesa 3rd "
            "helios qor mesa qm qorvex q can you open a bug for that "
        ),
        "u07": "mark q 7 hon q 7 2 qor",
        "u08": hyp_u08,
    }

    old_correct = old_total = 0
    for uid in babble_ref:
        rt = normalize(babble_ref[uid])
        ht = normalize(babble_hyp.get(uid, ""))
        for t in VOCAB_TERMS:
            rc = rt.count(t)
            old_total += rc
            old_correct += min(rc, ht.count(t))
    old_recall = old_correct / old_total if old_total else 0.0

    r = score_rows(babble_ref, babble_hyp)
    # Observed: new F1 ~= 0.4286 (P ~= 0.3103, R ~= 0.6923). Assert an honest
    # bound above that, not a value tightened to make a point.
    assert r["name_accuracy"] < 0.50, r["name_accuracy"]
    # The regression this whole task is about: the old recall-only measure
    # (observed ~= 0.6923) is substantially higher than the new F1 on the
    # exact same babble transcript.
    assert old_recall > r["name_accuracy"] + 0.20, (old_recall, r["name_accuracy"])

    # --- punctuation sentence-boundary F1 ---

    # (a) hyp identical to ref, two sentences -> one internal boundary on
    # each side, aligned to the same word -> P=R=F1=1.0.
    ref_two = {"u": "Close task nine two four. Open a follow up under auris."}
    hyp_two = {"u": "Close task nine two four. Open a follow up under auris."}
    r = score_rows(ref_two, hyp_two)
    assert r["ref_boundaries"] == 1 and r["hyp_boundaries"] == 1, r
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0, r
    assert r["punct_boundary_precision"] == 1.0
    assert r["punct_boundary_recall"] == 1.0
    assert r["punct_boundary_f1"] == 1.0
    assert r["any_terminal_punct_rate"] == 1.0

    # (b) hyp with all punctuation stripped -> no hyp boundaries at all, so
    # recall is 0 (nothing found the one ref boundary) and precision is
    # defined as 0 (0 hyp boundaries -> 0/0 -> 0.0, same convention as
    # name accuracy above). any_terminal_punct_rate is 0 too.
    hyp_stripped = {"u": "close task nine two four open a follow up under auris"}
    r = score_rows(ref_two, hyp_stripped)
    assert r["hyp_boundaries"] == 0, r
    assert r["punct_boundary_recall"] == 0.0, r
    assert r["punct_boundary_precision"] == 0.0, r
    assert r["any_terminal_punct_rate"] == 0.0, r

    # (c) hyp moves the full stop to the wrong word -> exactly one false
    # positive (boundary claimed where ref has none) and one false negative
    # (the real ref boundary was missed). Same words on both sides, so
    # every word aligns 1:1 by identity -- only the punctuation moved.
    ref_moved = {"u": "one two three. four five six."}
    hyp_moved = {"u": "one two. three four five six."}
    r = score_rows(ref_moved, hyp_moved)
    assert r["ref_boundaries"] == 1 and r["hyp_boundaries"] == 1, r
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1, r

    # (d) single-sentence ref/hyp -> the only boundary either side could
    # have is the trailing one, which is excluded -> ref_boundaries == 0.
    # Both precision and recall are then 0/0, defined as 0.0 by the same
    # "0 when the denominator is 0" convention used everywhere else in this
    # file (see per_term / name_precision / name_recall above) -- not 1.0,
    # even though ref and hyp agree perfectly that there's no internal
    # boundary. This is a documented, deliberate choice, not an oversight.
    ref_one = {"u": "one two three."}
    hyp_one = {"u": "one two three."}
    r = score_rows(ref_one, hyp_one)
    assert r["ref_boundaries"] == 0, r
    assert r["hyp_boundaries"] == 0, r
    assert r["punct_boundary_precision"] == 0.0, r
    assert r["punct_boundary_recall"] == 0.0, r
    assert r["punct_boundary_f1"] == 0.0, r

    # comma counts are a raw sum, no alignment.
    ref_commas = {"u": "hello, world, and then some."}
    hyp_commas = {"u": "hello world and then some."}
    r = score_rows(ref_commas, hyp_commas)
    assert r["comma_count_ref"] == 2, r
    assert r["comma_count_hyp"] == 0, r

    print("selftest OK", file=sys.stderr)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)

    csv_path = None
    if "--csv" in sys.argv:
        csv_idx = sys.argv.index("--csv")
        try:
            csv_path = sys.argv[csv_idx + 1]
        except IndexError:
            print("usage: --csv PATH", file=sys.stderr)
            sys.exit(2)

    positional = [
        a for i, a in enumerate(sys.argv[1:], start=1)
        if a != "--csv" and sys.argv[i - 1] != "--csv"
    ]
    if len(positional) != 2:
        print(f"usage: {sys.argv[0]} REF.tsv HYP.tsv [--csv PATH]", file=sys.stderr)
        sys.exit(1)

    result = score(positional[0], positional[1])
    if csv_path:
        write_csv(result, csv_path)
    print(json.dumps(result, indent=2))
