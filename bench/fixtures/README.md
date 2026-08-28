# bench/fixtures

Recordings kept as evidence for a specific finding, outside the scored
corpora. Nothing here is iterated by `bench/harness/rescore_wavs.sh` or
`run_auris.sh` — those read `bench/results/acoustic-wav/` and
`bench/corpus/wav/`, keyed by id against `bench/corpus/utterances.tsv`. A
file here is referenced by hand, from the task or doc that explains it.

## `u02-16h38-marginal.wav`

md5 `acdcb192ffcb268a50e5f91822c75f63`, 16 kHz mono, 3.91 s. The 16:38
acoustic capture of u02 ("hey mesa add a note to khora that headless mode
is the default now"), from the run that preceded the one committed as
`bench/results/acoustic-wav/u02.wav` (md5
`f3cb8a920017956942d43eb8a0dac229`) — a different recording of the same
sentence, kept because the committed one does not show what this one does.

On this file the recognizer returns an **empty transcript unbiased** and a
correct one biased:

```
auris --no-daemon -q -m spike/models/parakeet bench/fixtures/u02-16h38-marginal.wav
  -> exit 1, "auris: nothing transcribed; no speech in the audio"
auris --no-daemon -q -m spike/models/parakeet \
      --vocabulary-file spike/fixtures/hotwords.txt \
      bench/fixtures/u02-16h38-marginal.wav
  -> exit 0, "mesa at a note to khora that headless mode is the default now."
```

Deterministic, 5/5 each way. The hotword beam is the difference between a
usable decode and none at all on a marginal room recording.

**This is not the silence gate**, and mistaking it for one cost an
investigation (mesa task 965). Measured with `audio::is_silent`'s own
algorithm — max RMS over a 30 ms sliding window — this file reads
**0.040201**, 40x `SILENCE_RMS_THRESHOLD` (1e-3, `src/audio.rs`). The gate
returns false and `src/cli.rs:373` is never reached; the exit 1 comes from
`src/cli.rs:485`, where the recognizer itself returned nothing. Both sites
print the same `NOTHING_TRANSCRIBED_MSG` on purpose (`src/cli.rs:31-33`:
one wording for mesa to match on), which is exactly what made an empty
decode read as a gate defect from outside the process.

Both sites still print that one message and exit 1 — unchanged, deliberate.
What changed is that auris now adds a verbose-only stderr detail line at
each site, so a terminal user (never mesa, which pipes stderr) can tell
them apart. On this file that line reads "auris: the recognizer ran and
returned an empty transcript".
