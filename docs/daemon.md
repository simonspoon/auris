# The warm process

Date: 2026-08-28. Task 951.

**Decision: `auris serve` holds one `Recognizer` warm behind a Unix-socket
daemon; a bare `auris` invocation is a thin client that auto-starts one if
none is listening. One connection carries exactly one request and one
response — no multiplexing, no persistent client connection. Implemented in
`src/daemon.rs`.**

This records what task 951 asked to be settled in writing, now that the code
exists: who holds the process, the wire framing, the lifecycle, and how a
vocabulary or model change interacts with the warm recognizer.

## Who holds the process

Already argued at length in `README.md`'s opening: auris holds it, not mesa,
because a long-lived auris reading framed utterances off mesa's stdin would
move the cost to the wrong side. That doc does not spell out the cost on
mesa's end, so it goes here: the rejected alternative would have forced mesa
to invent a framing protocol and rewrite the three-thread stdin/stdout/stderr
drain that `mesa/src/core/speech.rs` documents as load-bearing (`docs/engine.md`
lines 24-31) — just to save auris a socket. Putting the persistence inside
auris means mesa's existing spawn-per-call driver needs no changes at all.

## Framing

`src/daemon.rs`'s module header is the source of truth; this is the summary.
One connection, one request, one response, both terminated by `\n`:

```text
{"op":"transcribe","samples":N,"model":"<absolute model dir>","hotwords":"<string, may be empty>"}
{"op":"status"}
{"op":"stop"}
```

```text
{"ok":true,"text":"...","decode_ms":<f64>}                          // transcribe
{"ok":true,"model":"...","pid":N,"uptime_secs":N,"requests":N}      // status
{"ok":true}                                                         // stop (daemon then exits)
{"ok":false,"code":<exit code>,"message":"..."}                     // any failure
```

For `transcribe`, the JSON header line is followed by exactly `samples * 4`
bytes of little-endian `f32` audio — the payload is a flat sample array, no
container, no compression.

The client ([`crate::audio::decode`], [`crate::vocabulary::Vocabulary`])
decodes audio and parses the vocabulary file *before* it ever connects, so
every format or usage error keeps the exact message and exit code it has on
the `--no-daemon` path (`src/cli.rs`) — the daemon never sees a malformed
WAV, a bad `--vocabulary-file`, or anything else that isn't already valid
`f32` samples and a plain hotwords string. This is also why the daemon's
error surface is narrower than the client's: it can refuse a sample count or
a model mismatch, but it cannot produce a "bad audio format" error, because
audio in that sense never reaches it. The same is true of silence: the
energy gate (`audio::is_silent`) also runs client-side, on the decoded
samples, before a connection is even opened — a silent request never
reaches the daemon socket at all, and the daemon never spends a decode on
it.

`MAX_TRANSCRIBE_SAMPLES` (one hour of 16 kHz mono, `src/daemon.rs:71-77`) is
a sanity bound on the framing — it exists so a bogus or hostile `samples`
value in the header can't make the daemon allocate an unbounded buffer
before it has read a single payload byte. It is not a policy limit; mesa's
own 25 MB per-request cap on the audio route is mesa's own, and lives in
`docs/posture.md`, not here.

## Lifecycle

**Starting.** Eagerly via `auris serve`, or auto-started by the first client
that finds nothing listening (`connect_or_spawn`, `src/daemon.rs:309`):
the client spawns `auris serve --socket <path> -m <model_dir>` with all
three stdio streams to `/dev/null`, does not wait on the child, and polls
the socket every 100 ms for up to 30 s (`CLIENT_CONNECT_TIMEOUT`) before
giving up. The spawned daemon reparents to init and keeps running after the
client that started it exits — nothing here is meant to, or needs to,
outlive the call that triggered it.

**Idling out.** No request for `IDLE_TIMEOUT` (300 s) and the daemon exits
on its own. A warm recognizer is ~1.5 GB resident (`docs/engine.md`); holding
that forever on a machine that transcribes once and then goes quiet is not a
trade worth making by default, so the daemon reclaims itself rather than
waiting to be told.

**Dying mid-utterance.** If the daemon dies while a client is mid-request,
the client's read fails and it exits nonzero with no transcript on stdout —
exactly the one signal mesa's driver treats as failure (README "Exit
codes"). No partial transcript is ever emitted; there is no partial state to
emit, since the daemon writes its one response line only after the decode
that produced it has finished.

**Exiting cleanly.** The socket is unlinked on every reachable exit path —
idle timeout, `stop` request, and Ctrl-C (`serve`'s loop in
`serve`, `src/daemon.rs:621`, always falls through to `std::fs::remove_file`
before returning). A SIGKILL skips it, and so would an
unhandled panic unwinding past it — there is no panic path in `serve` or
`handle_connection` today (sample counts are bounds-checked, decode failures
are all `Result`), but "the socket is always unlinked" is a property of the
happy paths, not a guarantee. Either way what is left behind is a stale
socket file bound to nothing. `bind_or_detect_existing`
(`src/daemon.rs:442`) is what the next `auris serve` or client
auto-start recovers from that: a bind failing with `AddrInUse` is resolved
by trying to connect — success means a live daemon really is there (print
the message, exit `OK`, touch nothing), failure means the socket is stale
(unlink it and rebind).

**A `transcribe` for a different model is refused, not silently served
wrong.** `handle_connection` (`src/daemon.rs:478`, the model check at
`531-546`) compares the request's `model` against the daemon's loaded model directory and answers
`{"ok":false,"code":NOTHING_TRANSCRIBED,...}` naming both paths if they
differ, rather than decoding against a recognizer the caller didn't ask for.

**The honest limitation: the accept loop is sequential.** `serve`'s loop
calls `listener.accept()`, then `handle_connection` to completion, then
loops back to `accept()` — one connection is served start to finish before
the next is even accepted (`serve`'s loop, `src/daemon.rs:621`). A `status`
or `stop` call arriving while a `transcribe` is mid-decode waits behind it; a wedged
`transcribe` connection is bounded only by `CONNECTION_READ_TIMEOUT` (30 s,
server-side, on reading the payload — the *response* wait is deliberately
unbounded, since a long decode is legitimate and must not be cut off). This
was reviewed and is not fixed here; see "What this does not decide" below.

## Rebuilds

A vocabulary term list is a per-request property and never rebuilds the
recognizer — `docs/vocabulary.md` and README "The daemon" already settle
this; `handle_connection` bears it out mechanically, since `hotwords` on a
`Transcribe` request is passed straight to `decode_with_hotwords` against
the one already-loaded `Recognizer`, with nothing in between that touches
construction. `-m`, by contrast, is a daemon-startup property: switching
models always means starting a different daemon against a different socket
or stopping and restarting the existing one. There is no in-place model
swap.

## Measurements

Host: same as `docs/engine.md`/`docs/latency.md` (Intel Core i9-9880H,
macOS, CPU only), quiet at the time of the run. Produced 2026-08-28 by
`spike/harness/warm_daemon_bench.sh` against the 8 fixtures in
`spike/fixtures/wav/` (45.429 s total audio) with
`spike/fixtures/hotwords.txt`.

| | RTF | per-call |
|---|---|---|
| decode-only against the warm recognizer | **0.061 – 0.071** | 2.79 – 3.23 s decode / 45.429 s audio, four runs |
| through the warm daemon, end-to-end client wall clock | 0.101 – 0.103 | first utterance 542 – 557 ms, remaining 7 avg 575 – 590 ms |
| `--no-daemon`, in-process load every call | 0.49 – 0.57 | 2.5 – 3.4 s each |

Spike baselines for comparison (`spike/RESULTS.md` §6/§4): 0.082 warm
decode-only, 0.634 spawned-per-utterance.

Read these carefully, because the three rows answer different questions and
only one of them is comparable to the spike's headline number.

**Row 1 is the like-for-like one**, and it is the acceptance number for task
951: four quiet runs read 0.0711, 0.0704, 0.0688 and 0.0613, measured by
`engine::tests::decode_only_warm_rtf_matches_spike_methodology` the same way
the spike measured its 0.082 — one `Recognizer::load`, then all 8 fixtures
decoded in-process with the production per-word hotwords string, summing
decode time only. A range is quoted rather than a single figure because the
spread across four runs on an otherwise-idle host is itself ~14%, which is
the honest precision of this measurement. It lands slightly *better* than the
spike's 0.082 on the same host, which is the expected result for the same
model, the same decoding method and the same terms: nothing about the daemon
changed how a decode runs, which is precisely the claim being tested.

That test is `#[ignore]`d, deliberately. Under the default parallel
`cargo test`, alongside the other tests that each load their own 1.5 GB
recognizer, the identical decode reads **0.4523** — six times slower, purely
from CPU contention, and an absolute timing assertion there would be
measuring the machine rather than auris. Run it alone:

```sh
cargo test --release decode_only_warm_rtf -- --ignored --nocapture
```

The *invariant* underneath it — the load is paid once, later decodes are
cheap — is not left to an absolute threshold. It is asserted relatively,
against the load time measured in the same process, by
`daemon::tests::second_decode_is_fast_after_the_one_time_load`, which runs on
every `cargo test` and skips when the model is absent.

That test measures the **client-side round trip**, not just the daemon's
self-reported `decode_ms`, and the distinction is the whole point of the
guard. `decode_ms` is timed from inside `handle_connection`, after the
payload read — so a regression that moved `Recognizer::load` to happen per
request would fall *outside* the daemon's own timer and still report a fast
`decode_ms`, passing a test that trusted it. A clock the client starts
before the RPC and stops after it cannot be fooled that way. Measured on a
quiet host: a 2.25 s load against 395 ms and 350 ms round trips.

**Row 2 is what a caller actually pays**, and it is deliberately larger:
~0.10 is **end-to-end client wall clock** — process spawn, WAV parse,
resample, socket round-trip and teardown, all included. It is *not*
comparable to 0.082 and should never be quoted as if it were. The ~0.03 RTF
between rows 1 and 2 is roughly 450–550 ms per utterance of per-call process
and IPC overhead — spawn, clap parse, connect, WAV decode and resample — not
a colder recognizer. It is also the overhead row 3 pays identically, which is
why row 2 against row 3 is the fair comparison and row 2 against the spike is
not.

**Row 3 is what row 2 is judged against**: the same binary, the same per-call
overhead, with and without the warm recognizer behind it — ~5x slower, and
the only honest measure of what the daemon buys. Its per-call cost is that
same ~500 ms of overhead plus a fresh ~2.2–3 s model load, every single
call.

The load-paid-once property this whole design rests on is evidenced by the
first utterance (556.8 ms) being indistinguishable from the later average
(583.4 ms) — there is no per-call load spike once the daemon is up. If the
~4 s load were being paid per call, the first utterance would not look like
the rest; it does, so it isn't.

One more caveat, in the spirit of `docs/engine.md`'s fixture caveat: this run
was taken on a quiet machine, and contended runs moved the absolute numbers
substantially — `--no-daemon` passes read 1.04 and 1.58 RTF with another
`cargo test` competing for the CPU, against 0.49–0.57 quiet, with single
utterances stretching to 18.8 s that have nothing to do with auris. The
daemon-vs-`--no-daemon` *ratio* held across every run, contended or not; the
absolute numbers did not. Treat the table above as one host's
numbers on one quiet afternoon, not a portable constant.

## What this does not decide

- **The sequential accept loop under a concurrent caller.** mesa's driver is
  spawn-per-call today, so it never holds two auris connections open at
  once, and this was never exercised against a caller that does. Whether
  that needs fixing — a thread per connection, a request queue — is open.
- **The stale-socket TOCTOU when two clients auto-start at once.** Two
  processes racing `connect_or_spawn` can both fail to connect, both spawn
  an `auris serve`, and have the second `bind_or_detect_existing` unlink the
  first daemon's live socket out from under it. Reviewed, judged out of
  scope for task 951 (mesa's driver has never issued concurrent auris calls
  in practice), and not fixed here.
- **Whether 300 s is the right idle timeout.** Picked as a round number
  that keeps a machine that transcribes once and goes quiet from holding
  1.5 GB forever; not measured against real usage patterns, and there is no
  data yet on how often a 300 s window forces a reload that a longer one
  would have avoided.
