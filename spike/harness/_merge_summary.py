#!/usr/bin/env python3
"""Merge per-config fragment + score JSON files into spike/results/summary.json.

Usage:
    python3 _merge_summary.py RESULTS_DIR config1 config2 ...

For each config, reads RESULTS_DIR/raw/<config>.fragment.json (timing/memory,
written by run_bench.sh) and RESULTS_DIR/raw/<config>.score.json (WER/name
accuracy, written by score.py), merges them into one object, and writes the
whole list as a JSON array to RESULTS_DIR/summary.json.
"""
import json
import os
import sys


def main():
    results_dir = sys.argv[1]
    configs = sys.argv[2:]
    raw_dir = os.path.join(results_dir, "raw")

    summary = []
    for config in configs:
        frag_path = os.path.join(raw_dir, f"{config}.fragment.json")
        score_path = os.path.join(raw_dir, f"{config}.score.json")

        if not os.path.exists(frag_path):
            summary.append({"config": config, "status": "failed", "error": "no fragment written"})
            continue

        with open(frag_path, encoding="utf-8") as f:
            entry = json.load(f)

        if os.path.exists(score_path):
            with open(score_path, encoding="utf-8") as f:
                score = json.load(f)
            entry["wer"] = score["wer"]
            entry["sub"] = score["sub"]
            entry["del"] = score["del"]
            entry["ins"] = score["ins"]
            entry["name_accuracy"] = score["name_accuracy"]
            entry["per_term"] = score["per_term"]

        summary.append(entry)

    out_path = os.path.join(results_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
