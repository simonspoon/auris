#!/usr/bin/env python3
"""Write a JSON object from key=value argv pairs (used by run_bench.sh).

Values are coerced: "true"/"false" -> bool, "null" -> None, numeric-looking
strings -> int/float, everything else stays a string. The required
`_path=...` argument is popped and used as the output file path.

Usage:
    python3 _write_fragment.py _path=out.json config=foo warm_decode_seconds=1.23
"""
import json
import sys


def coerce(v):
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "null":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def main():
    out = {}
    for kv in sys.argv[1:]:
        k, _, v = kv.partition("=")
        out[k] = coerce(v)
    path = out.pop("_path")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
