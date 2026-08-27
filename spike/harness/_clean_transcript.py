#!/usr/bin/env python3
"""Read a whisper-cli transcript on stdin, strip bracketed timestamps and
blank lines, collapse to a single line, and print it (no trailing newline
issues -- caller adds the tab/id when writing the tsv row)."""
import re
import sys

text = sys.stdin.read()
lines = []
for line in text.splitlines():
    line = re.sub(r"\[[^\]]*\]", "", line)
    line = line.strip()
    if line:
        lines.append(line)
print(" ".join(lines).strip())
