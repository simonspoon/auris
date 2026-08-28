#!/usr/bin/env python3
"""Mechanical check for cross-utterance bleed in a captured transcript TSV.

Chrome's `webkitSpeechRecognition` finalization is nondeterministic
(`bench/harness/webspeech/README.md`, `capture.py`'s own handling of
`window.__wsFinal`): a result for utterance N can finalize LATE, after
`capture.py` has already moved on and reset the recognizer for utterance
N+1, so N's leftover words land at the front of N+1's row instead of N's
own. This is not a theoretical concern -- it was OBSERVED in a real run of
this harness: utterance u30's trailing words opened u31's transcript,
because u30's recognizer took until after the N+1 reset to finalize.

Why this matters more than a single bad row: a bleed doesn't just corrupt
utterance N+1's score -- N's own row is now silently missing the words that
leaked forward (N looks like it under-transcribed), and N+1's row is scored
against the wrong reference-adjacent content (its real transcript is
padded with someone else's words at the front, which name-accuracy and WER
will both charge to the wrong utterance). One bleed event corrupts the
attribution of two rows, not one, and any claim built on the resulting
numbers is not measuring what it says it's measuring.

Detection method: for each utterance i (i > 0, in the reference file's own
order -- see `read_tsv_ordered` below for why this must not be sorted),
compare how well hyp[i] opens against its OWN reference's start (the
longest common word prefix, `own_overlap`) versus how well it opens
against ANY tail-fragment of the PREVIOUS utterance's reference (the best
longest-common-prefix over every suffix start of ref[i-1], `prev_overlap`).
A hyp that opens more like the end of the previous utterance than the
start of its own (`prev_overlap >= MIN_OVERLAP and prev_overlap >
own_overlap`) is flagged as a bleed suspect. This is a heuristic, not a
proof -- a short or garbled hyp could coincidentally share a few words with
the previous reference -- so `MIN_OVERLAP` is kept at 3 words to keep false
positives rare, and every flagged id prints both overlap numbers so a
human can judge plausibility rather than trust a bare pass/fail.

Word matching reuses `score.py`'s own `normalize()` (lowercase, punctuation
stripped, numbers folded to digits) rather than reimplementing it, so a
match here means the same thing "match" means to the scorer itself.

A detector that has never fired is indistinguishable from one that cannot
fire. `--selftest` plants a known synthetic bleed (the tail of one
utterance's reference glued onto the front of the next one's hypothesis)
and asserts this script actually catches it, before trusting a clean
result on real data to mean anything.

Usage:
    python3 bench/harness/detect_bleed.py REF.tsv HYP.tsv
    python3 bench/harness/detect_bleed.py --selftest

Exit codes: 0 clean (no suspects found), 1 one or more ids flagged, 2 usage
error.
"""
import sys

sys.path.insert(0, "bench/harness")
from score import normalize  # noqa: E402

MIN_OVERLAP = 3


def read_tsv_ordered(path):
    """Read id<TAB>text lines, preserving FILE order, not sorted order like
    score.py's read_tsv -- bleed detection is inherently about sequence
    (utterance N's leftovers landing in N+1's row), so re-sorting by id
    would silently break the very adjacency this check depends on."""
    ids = []
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, text = line.split("\t", 1)
            ids.append(uid)
            rows[uid] = text
    return ids, rows


def longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def best_suffix_overlap(hyp_tokens, prev_ref_tokens):
    """Best longest-common-prefix between hyp_tokens and ANY suffix of
    prev_ref_tokens -- does hyp open like some tail fragment of the
    previous utterance's reference, not necessarily its very last word?"""
    best = 0
    for start in range(len(prev_ref_tokens)):
        overlap = longest_common_prefix_len(hyp_tokens, prev_ref_tokens[start:])
        if overlap > best:
            best = overlap
    return best


def find_bleed(ref_ids, ref_rows, hyp_rows):
    """Return a list of (uid, prev_uid, own_overlap, prev_overlap) for every
    id flagged as a bleed suspect, in ref_ids order."""
    flagged = []
    for i in range(1, len(ref_ids)):
        uid = ref_ids[i]
        prev_uid = ref_ids[i - 1]
        hyp_text = hyp_rows.get(uid, "")
        if not hyp_text.strip():
            continue  # nothing to bleed into an empty transcript

        hyp_tokens = normalize(hyp_text)
        own_tokens = normalize(ref_rows[uid])
        prev_tokens = normalize(ref_rows[prev_uid])

        own_overlap = longest_common_prefix_len(hyp_tokens, own_tokens)
        prev_overlap = best_suffix_overlap(hyp_tokens, prev_tokens)

        if prev_overlap >= MIN_OVERLAP and prev_overlap > own_overlap:
            flagged.append((uid, prev_uid, own_overlap, prev_overlap))
    return flagged


def _selftest():
    # Plant a known bleed: the last few words of ref["a"] land on the front
    # of hyp["b"], ahead of b's own content. This must be caught.
    ref_rows = {
        "a": "close task nine two four and open a follow up under auris",
        "b": "hey mesa add a note to khora",
        "c": "mark task seven thirty done",
    }
    ref_ids = ["a", "b", "c"]
    bled_hyp = "follow up under auris hey mesa add a note to khora"
    hyp_rows = {"a": ref_rows["a"], "b": bled_hyp, "c": ref_rows["c"]}

    flagged = find_bleed(ref_ids, ref_rows, hyp_rows)
    assert len(flagged) == 1, flagged
    uid, prev_uid, own_overlap, prev_overlap = flagged[0]
    assert uid == "b" and prev_uid == "a", flagged
    assert own_overlap == 0, flagged  # "follow" != "hey", no match at all
    assert prev_overlap >= MIN_OVERLAP, flagged

    # A clean run (each hyp opens like its own reference) must not fire.
    clean_hyp_rows = dict(ref_rows)
    assert find_bleed(ref_ids, ref_rows, clean_hyp_rows) == []

    # A short, unrelated hyp that happens to share a couple of words with
    # the previous reference must not fire below MIN_OVERLAP.
    near_miss_hyp_rows = dict(ref_rows)
    near_miss_hyp_rows["b"] = "auris hey mesa add a note to khora"
    flagged = find_bleed(ref_ids, ref_rows, near_miss_hyp_rows)
    assert flagged == [], flagged  # "auris" alone is below MIN_OVERLAP

    print("selftest OK", file=sys.stderr)


def main(argv):
    if "--selftest" in argv:
        _selftest()
        return 0

    if len(argv) != 2:
        print(f"usage: {sys.argv[0]} REF.tsv HYP.tsv | --selftest", file=sys.stderr)
        return 2
    ref_path, hyp_path = argv

    ref_ids, ref_rows = read_tsv_ordered(ref_path)
    _hyp_ids, hyp_rows = read_tsv_ordered(hyp_path)

    flagged = find_bleed(ref_ids, ref_rows, hyp_rows)

    if not flagged:
        print(f"no bleed suspects found ({len(ref_ids)} ids checked)")
        return 0

    print(f"SUSPECT BLEED in {len(flagged)}/{len(ref_ids)} id(s):")
    for uid, prev_uid, own_overlap, prev_overlap in flagged:
        print(
            f"  {uid} (after {prev_uid}): own_overlap={own_overlap} "
            f"prev_overlap={prev_overlap}"
        )
        print(f"    hyp:      {hyp_rows[uid]!r}")
        print(f"    own ref:  {ref_rows[uid]!r}")
        print(f"    prev ref: {ref_rows[prev_uid]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
