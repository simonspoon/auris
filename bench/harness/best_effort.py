#!/usr/bin/env python3
"""Derive a best-effort Web Speech transcript from capture.py's own two
committed outputs, so the benchmark's headline number is reproducible from
committed code, not a hand-carried artifact.

Chrome promotes a `webkitSpeechRecognition` result to `isFinal` on its own
nondeterministic schedule -- a genuinely good recognition can simply never
get promoted. Observed on this corpus: a control utterance produced a
word-perfect INTERIM transcript while its final stayed empty through the
full 10-second poll `capture.py` gives it after `stop()`. `capture.py`'s
`--out` correctly stays final-only -- that is its documented contract, and
changing it would make the raw capture lie about what Chrome actually
finalized. This script does not touch that contract; it builds a DERIVED
view on top of it, after the fact, from `--out` (final results,
id<TAB>transcript) and `--diagnostics-out` (id<TAB>micRMSpeak<TAB>wsError
<TAB>wsInterim, which `capture.py` already records for exactly this reason
-- see that script's handling of `window.__wsInterim`).

The rule: best-effort transcript = the final transcript if it's non-empty,
else the last interim result read for that utterance, else empty.

Why best-effort is the number BENCHMARK.md leads with, and strict is not
discarded: when testing this project's own claim to beat the browser
baseline, the baseline deserves its strongest case, not a case weakened by
an implementation quirk of Chrome's finalization timing that has nothing
to do with how well the recognizer actually heard the utterance -- that is
what best-effort gives it. Strict (finals only) is reported alongside as
the conservative bound, because interim results are provisional by spec
and a real product built on this API would only ever render finals; a
strict number is "what a user would actually see," while best-effort is
"the best case for what the recognizer actually understood."

Usage:
    python3 bench/harness/best_effort.py RAW.tsv DIAGNOSTICS.tsv OUT.tsv
    python3 bench/harness/best_effort.py --selftest

Prints finalized / fell_back_to_interim / empty counts to stderr -- these
are quoted directly in BENCHMARK.md and must stay regenerable, not
folklore.

Exit codes: 0 on success, 2 on usage error.
"""
import sys


def read_tsv_dict(path, ncols):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # pad short rows (e.g. no wsError/wsInterim text) to ncols
            parts += [""] * (ncols - len(parts))
            rows[parts[0]] = parts[1:ncols]
    return rows


def derive_best_effort(raw, diag, ids):
    """raw: id -> [final]. diag: id -> [micRMSpeak, wsError, wsInterim].
    ids: the id order to emit in. Returns (rows, counts) where rows is a
    list of (id, best_text) and counts is the finalized/fell_back/empty
    tally."""
    rows = []
    counts = {"finalized": 0, "fell_back_to_interim": 0, "empty": 0}
    for utt_id in ids:
        final = raw.get(utt_id, [""])[0].strip()
        interim = diag.get(utt_id, ["", "", ""])[2].strip()
        if final:
            best = final
            counts["finalized"] += 1
        elif interim:
            best = interim
            counts["fell_back_to_interim"] += 1
        else:
            best = ""
            counts["empty"] += 1
        rows.append((utt_id, best))
    return rows, counts


def _selftest():
    raw = {"a": ["hello world"], "b": [""], "c": [""]}
    diag = {
        "a": ["0.1", "", "hello wor"],  # final present -- must win over interim
        "b": ["0.1", "", "khora needs a second pass"],  # empty final -- falls back
        "c": ["0.1", "no-speech", ""],  # both empty -- stays empty
    }
    rows, counts = derive_best_effort(raw, diag, ["a", "b", "c"])
    assert rows == [
        ("a", "hello world"),
        ("b", "khora needs a second pass"),
        ("c", ""),
    ], rows
    assert counts == {"finalized": 1, "fell_back_to_interim": 1, "empty": 1}, counts

    # a missing diagnostics row must not crash -- treated as no interim.
    rows, counts = derive_best_effort({"d": [""]}, {}, ["d"])
    assert rows == [("d", "")], rows
    assert counts["empty"] == 1, counts

    print("selftest OK", file=sys.stderr)


def main(argv):
    if "--selftest" in argv:
        _selftest()
        return 0

    if len(argv) != 3:
        print(f"usage: {sys.argv[0]} RAW.tsv DIAGNOSTICS.tsv OUT.tsv | --selftest",
              file=sys.stderr)
        return 2
    raw_path, diag_path, out_path = argv

    raw = read_tsv_dict(raw_path, 2)   # id -> [final]
    diag = read_tsv_dict(diag_path, 4)  # id -> [micRMSpeak, wsError, wsInterim]
    ids = list(raw.keys())

    rows, counts = derive_best_effort(raw, diag, ids)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for utt_id, best in rows:
            out_f.write(f"{utt_id}\t{best}\n")

    print(f"wrote {out_path}: {len(rows)} rows", file=sys.stderr)
    print(f"finalized={counts['finalized']} "
          f"fell_back_to_interim={counts['fell_back_to_interim']} "
          f"empty={counts['empty']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
