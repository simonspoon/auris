# Where the audio goes, and where it doesn't

Date: 2026-08-27. Task 930.

**Decision: retention is off. Audio is transcribed and dropped — never
written to `live_turns`, never to disk, never logged — matching what the
speak routes already do with text. A debug-retention setting, off unless
set, may keep raw audio under the mesa data directory for a rolling
window, default seven days, pruned on write; the page must say plainly
that it is on while it is on, and it is never on under `--lan`. A
request is capped at 25 MB, refused past it with a `413` naming the
limit. And `--lan` gets no audio route at all: not gated, absent —
`require_agent_access` relaxes rather than refuses out there, so the
refusal has to be that the route was never registered.**

## The trade this task actually is

Every other auris doc argues for a capability. This one has to argue for
a boundary, and the honest version of that argument is a trade, not a
win.

Today, per `docs/live.md:1489-1502`, mesa's own audio path is absolute:
"no audio ever reaches the server: recognition is the **browser's**,
running in the page, and mesa receives only the text it produced." That
sounds like the private answer. It is not. The *audio* never leaves the
page, but the *recognition* — the thing that turns audio into words —
may leave the machine entirely: for Chrome and Safari, `SpeechRecognition`
may ship the utterance to Google's or Apple's own service, and the same
line says so and stops there: "a thing worth knowing and not something
mesa can answer for." mesa's boundary was drawn at the browser's front
door, and the front door opens onto the internet.

auris moves the boundary. After this route lands, the audio does leave
the page — it reaches `mesa serve`, an HTTP body that did not exist
before. What it does not do is leave the *machine*: recognition runs in
a persistent local process (`docs/latency.md`, `docs/engine.md`), and
nothing about the utterance's content reaches a third party. **The
boundary moves from the page to the machine.** Everything below argues
why the machine is the stronger place to draw it.

The argument is not "auris is newer, so more private." The page boundary
never protected the *content* of the speech from anyone in particular —
it protected it from mesa, while handing it, on two of three engines, to
a company that isn't mesa either. A boundary that keeps a datum from its
own application while letting it reach an unrelated one is not a privacy
boundary; it is an accident of where the code happened to run. Moving it
to the edge of the machine is the first time this system has drawn it on
purpose — and the price is named, not hidden: a route now exists where
none did, an HTTP body now carries audio, and the fully-local promise
`docs/live.md` made — mic capture and VAD in the page, "a local whisper"
— is retired by an engine that is not whisper. `docs/engine.md` replaced
whisper with parakeet tdt-0.6b-v2 int8 via sherpa-onnx before this doc
was written; anywhere older text still says "a local whisper," that
phrase is superseded.

## The premise this task was written on is wrong

The task named three module headers as carrying the "audio never leaves
the page" claim that auris invalidates. Checked against the live files,
that is wrong about two of the three, in ways worth separating so the
write-up edits the file that actually says it rather than the one that
was merely named.

**`src/core/speech.rs` carries no such claim, and needs no edit.** Its
header (`speech.rs:1-30`) is entirely about the mesa-to-person path:
`kokoro-rs` synthesis, streamed audio, the subprocess contract. There is
no sentence in it about speech *input* or one-directionality. It sits on
the output side of a pipe this task does not touch.

**`liveRecognition.ts`'s header does carry the claim, at its own last
paragraph.** The module's doc comment runs from line 1 to line 52 — one
block, not several — and closes with exactly the sentence at issue,
`liveRecognition.ts:48-51`: "The audio never leaves the page: recognition
is the browser's, mesa still ships no speech-to-text and no route
accepts an audio body." That is a header sentence in a 708-line file,
not one buried deep in the body. `liveDevices.ts:31-33` carries the
matching claim in its own header: "The audio still never leaves the
page: a chosen track is handed to the browser's own recognizer, mesa
ships no speech-to-text and no route accepts an audio body." **Both
frontend module headers carry it. Only the Rust one does not** — the two
edits at the end of this doc are to the files that actually say it.

**"One-directional, server to browser" appears twice in `docs/live.md`,
and auris invalidates exactly one.** The instance at `docs/live.md:34-38`
sits in the "mesa → person is speech" bullet and describes the TTS
*output* path — "the audio path stays **one-directional, server to
browser**, exactly as it was before this feature." auris adds an *input*
path; it does not make the output path bidirectional, so that sentence
stays exactly as written. The instance inside the absent-bullet,
`docs/live.md:1489-1502`, is the one auris invalidates, because it
asserts the same phrase about the system as a whole rather than about
TTS specifically. Deleting the wrong instance would delete a sentence
that is still true.

## Retention: off, because "off" costs nothing to keep

`live_turns` has no audio column: `store.rs:548-559` lists `id,
session_id, role, text, action, target, created_at, delivered_at,
played_at`. There is no ninth column waiting for bytes, and adding one is
not on the table for this decision — a turn is text, spoken or typed, and
stays that way. `Store::add_live_turn` (`store.rs:4087`, the `INSERT` at
`store.rs:4157`) is, by its own doc comment, "the single write path for
turns." One write path is what makes "off" a property of the code rather
than a promise about it: retention-off is enforced by simply never
reaching for that path with anything but the transcript string the
decoder already produced. There is nowhere else in the schema an audio
byte could land by accident.

The precedent for storing nothing at all already exists, on the
*output* side, in the two speak routes. `GET /api/inbox/{id}/speak`
(`api.rs:2219-2266`, registered `api.rs:794`) and `GET
/api/live/turns/{id}/speak` (registered `api.rs:819`) both synthesise
audio on the fly and stream it straight to the response body. The doc
comment says the whole of the policy: "Nothing is stored and nothing is
cached — the button is a read, repeated as often as it is pressed"
(`api.rs:2200-2202`). Transcribe-and-drop is that same sentence read
backwards. The speak routes take text in, hand audio out, and keep
neither. auris takes audio in, hands text out, and keeps neither. The
shape is identical; only the direction of the arrow changed.

## Debug retention: opt-in, bounded, and it has to say so on the page

A debugging aid needs an escape hatch, because "off, no exceptions" is
also "no recording to diff a bad transcript against while the engine is
still being tuned." The escape hatch is a setting, off unless set, that
keeps raw audio under the mesa data directory for a rolling window,
default seven days when turned on, pruned on write.

`default_db_path()` (`store.rs:67-75`) resolves to
`directories::ProjectDirs::from("", "", "mesa").data_dir()` —
`~/Library/Application Support/mesa/` on macOS — and today that
directory holds exactly one thing: `mesa.db` and its WAL/SHM siblings.
Kept audio would be the first non-database files ever written there,
which argues for a short window and an off default: this is not a
directory mesa has ever asked to grow, and the setting is built to be
turned off again once the engine is settled rather than to become a
feature.

The mechanism is `~/.mesa/config.json` (`config_file()`,
`config.rs:193-205`), a sectioned JSON document of `Option<T>` fields.
The nearest precedent for shape is `watchers.todo-concurrency`: a
private `Option<u32>` (`config.rs:1041`), a public reader
(`config.rs:1069-1071`), surfaced to the page as `ConfigWatchers`
(`types.rs:252-260`) carrying **both** the configured value and a
`_default` — "the built-in limit mesa ships... so the editor can show
what blank means without hardcoding it" (`types.rs:257-258`). There is
no boolean anywhere in mesa's config sections today; a debug-retention
flag would be the first `Option<bool>`, and it follows the `Option<T>` +
`_default` shape on principle rather than copying a boolean precedent
that does not exist.

That shape is not incidental to "the page must say plainly it is on
while it is on." A config value read fresh by a route and returned in a
GET response is what makes a setting visible to a page without the page
guessing at it from behavior — the same reason `todo_concurrency` is a
value the editor reads rather than infers from watching the watcher. A
flag that only showed itself by files accumulating on disk would fail
its own requirement.

Never on under `--lan` needs no new mechanism: the `--lan` section below
shows the audio route itself absent out there, and the same absence
keeps a debug flag — which would otherwise be a liability wearing a
feature flag — off with it.

## Size: 25 MB, and the trap the axum default sets for it

25 MB is one recording, not a corpus. 16 kHz 16-bit mono PCM runs about
32 KB/s, so a ten-minute utterance is roughly 19 MB; 25 MB leaves margin
without inviting a session's whole transcript through in one request.
Enforcing it is two layers, not one, because axum sets a default that
would silently break the requirement before the cap is even checked.

mesa already hit this trap once, on the attachments route, and left the
fix on record: axum's stock `DefaultBodyLimit` is 2 MiB and applies to
the whole router by default. mesa overrides it exactly once today, at
`api.rs:745`, `.layer(DefaultBodyLimit::max(ATTACHMENT_BODY_LIMIT))` on
`POST /api/tasks/{id}/attachments`, where `ATTACHMENT_BODY_LIMIT`
(`api.rs:1660-1668`) is sized to `MAX_ATTACHMENT_BYTES * 4/3 + 1 MiB` —
base64 expansion plus JSON headroom — specifically because, unoverridden,
"axum's own 2 MiB `DefaultBodyLimit` would reject an at-cap upload with
a bare non-JSON 413... before `Store`'s own size check ever runs"
(`api.rs:1662-1664`). The same trap waits for the audio route: without
its own `DefaultBodyLimit::max()` layer above 25 MB, any recording past
about 65 seconds at 16 kHz mono — 2 MiB — dies as a bodyless axum 413
that names no limit, breaking this decision's own requirement that the
413 name the cap. The cap is therefore two things: an axum layer sized
above 25 MB, and mesa's own check at 25 MB producing the JSON `413` body
the requirement asks for — one without the other either lies about the
limit or never fires.

`MAX_ATTACHMENT_BYTES` is itself 25 MiB (`attachments.rs:9`), so the
attachment route's actual ceiling, after base64 and JSON overhead, is
closer to 34 MiB — larger than the audio cap, not smaller. The
comparison worth making is not "biggest body mesa has ever accepted" but
"biggest body accepted under the axum *default*": every route mesa has
not explicitly overridden tops out at 2 MiB, and 25 MB puts the audio
route in the same small company as attachments — needing its own layer
because what it accepts is naturally larger than text.

The status code is deliberately not the one mesa's other size limit
uses. `LIVE_TEXT_MAX` (`store.rs:885`, `8192` characters), enforced in
`add_live_turn` (`store.rs:4110-4114`), maps to **422**
`UNPROCESSABLE_ENTITY` (`api.rs:1109-1126`). Audio uses **413**: 422
says "I read your input and it is invalid," fitting a string mesa parses
and measures exactly. 413 names a *body* too large to accept — an
over-cap recording is refused before it is read, at the axum layer, not
after decoding starts and fails. 422 for audio would claim mesa
inspected a recording it never let in the door.

`liveRecognition.ts:273-277` mirrors `LIVE_TEXT_MAX` today as
`HELD_MAX`, its comment noting it is "the server's own
`LIVE_TEXT_MAX`... mirrored here because this is where the text is
assembled" — a hand-maintained duplicate literal, two `8192`s and a
comment promising they match, no config route or shared constants file.
The audio cap is mirrored the same way: the person has to meet the
limit before they finish speaking, not learn it from a `413` after the
fact, and a round-trip to fetch it defeats the point. A known cost,
accepted for a known reason — `HELD_MAX` already pays it.

## Segmentation makes the cap cheaper to defend, not just to hit

`docs/streaming.md` ships one-shot transcription first and VAD-segmented
transcription next, which matters here: **the 25 MB limit is per
request, and a long session is many requests, not one.** `docs/latency.md`
holds a *final* segment to under 18.3 s at the measured warm RTF, backed
by `max_speech_duration = 8.0` on the VAD — every earlier segment is
flushed on its own pause, well under a minute and nowhere near 25 MB.
The cap only has to survive the worst single utterance spoken without
pausing, not a whole conversation. A design that uploaded the whole
recording after silence, which `docs/latency.md` already rules out on
latency grounds, would also have made this cap far harder to set —
25 MB would then have to cover an entire session, and either the cap
grows or long sessions start failing partway through. Segmentation is
why 25 MB per request is enough.

## `--lan`: the gate that relaxes, and the route that must not be there

The recorded decision — no audio under `--lan` — is right, and the
mechanism first proposed for it is not. `require_agent_access`
(`api.rs:4742-4753`) reads:

```
if state.lan {
    return require_lan_page_access(addr, headers, state.port);
}
require_loopback(addr)?;
require_local_host(headers, state.port)?;
require_local_origin(headers)?;
Ok(())
```

That is not an authentication check with a `--lan` carve-out. There is
no token, no environment variable, no config value anywhere in it —
every branch is a peer-address, `Host`, or `Origin` check, identity of
the *page*, not of the *person*. Under `--lan` it swaps a check, it does
not add one: `require_local_host`'s exact `localhost:<port>` /
`127.0.0.1:<port>` match (`api.rs:4710-4724`) becomes
`require_lan_agent_host`'s far looser rule, any IP-literal host on the
serve port with DNS names refused as anti-rebinding
(`api.rs:4778-4816`). **`require_agent_access` does not refuse under
`--lan` — it relaxes.** Any device already on the network passes it, by
design; that is what "serve to the LAN" means. Routing the audio
endpoint through that gate and expecting it to answer no out there would
be asking a door that is deliberately unlocked to stay shut.

The consequence is not "add a stronger gate," because `docs/live.md`
already recorded, for a different route, that no gate here reaches
strong enough: `mesa live look` is CLI-only, and `docs/live.md:570-580`
is explicit that "no gate in this codebase is strong enough to make that
route acceptable, so there is no route: the capability is reachable only
by something already running as the person." The audio route belongs in
the same category, for the same reason a screen capture does — decoding
whatever a stranger on the network handed the process is not something a
peer-address check should be asked to authorize. The refusal has to be
structural: the route absent from `router()` when `state.lan` is true,
not present and returning 403 — the `live look` shape, not a stronger
check but no check at all, because the capability does not exist to be
checked.

The honest counter-argument sits in the same paragraph that supplies the
precedent, and deserves an answer. `docs/live.md:570-580` itself calls
the no-auth `--lan` posture "defensible for reading tasks and dictating
an utterance" — and dictating an utterance is what this route sounds
like it does. The distinction is what crosses the wire. The existing
case is a browser posting *text it already recognised* — a string
already reviewed by whoever is looking at the screen it came from, no
different in kind from any other task-CRUD body `--lan` already accepts.
The audio route accepts *undifferentiated recorded sound* from any
device on the port, handed to a decoder running as the machine's owner
with no step where a person confirmed what it contains. "Post the
sentence you already spoke into your own browser" and "transcribe
whatever this stranger recorded" are not the same capability in
different clothes — the first is task input, the second is decoding an
unknown recording as a privileged process. The `--lan` posture
defensible for the first was never asked to defend the second.

Every route `require_agent_access` already gates — both speak routes,
agent-spawn, live-turn actions, the attach/terminal websocket, script
run and save, some two dozen call sites — keeps that gate as it is; none
accept a body a machine, rather than a person at a keyboard, could have
produced. The gate's 403 is unaffected — the
`"rejected Host {host:?}: ..."` message raised at `api.rs:4722` and
`api.rs:4815`, wrapped in the standard `{"error":{"code","message"}}`
body every `ApiError` gets (`api.rs:1141-1145`) — since the audio route
under `--lan` never reaches it.

## The rewrite, drafted

This lands with the route, not before it. It keeps the true half of the
old sentence rather than deleting it, because the browser path remains
the fallback wherever `SpeechRecognition` is what a browser offers.

```markdown
- **Speech-to-text of mesa's own — the browser's, and now also auris's.**
  Where auris is running, mesa streams the microphone to it instead of
  to the browser's built-in recognizer: `mesa serve` accepts a bounded
  audio request (25 MB per request, `413` past it naming the limit —
  `docs/posture.md`) and hands it to a persistent local decoder, so
  recognition never leaves the machine and the non-Chromium browsers,
  which never had `SpeechRecognition` at all, get speech input for the
  first time. mesa's own vocabulary reaches that decoder as per-word
  hotword bias *before* the audio is decoded (`docs/vocabulary.md`)
  rather than being repaired in the text afterward, which is what the
  browser path still has to do. Retention is off: the audio is
  transcribed and dropped, matching what the speak routes already do
  with text, and never written to `live_turns`, which has no column for
  it (`docs/posture.md`). A debug-retention setting, off by default, can
  keep a rolling window of raw audio for tuning the engine; it is never
  on under `--lan` and the page says plainly when it is on.

  Where auris is not running, mesa falls back to exactly what it always
  did: recognition is the **browser's**, running in the page, and mesa
  receives only the text it produced (task 873). The recognition
  quality, the language and the privacy question are therefore the
  browser's on that path — which, for Chrome and Safari, means the
  speech may be sent to *their* service, a thing worth knowing and not
  something mesa can answer for, and it is exactly this gap that auris
  exists to close for whoever turns it on. Where there is no recognizer
  at all, the person's own system dictation types into the text field,
  exactly as it always did.

  The trade is real and stated on purpose: an HTTP body now carries
  audio, where none did before, and it reaches `mesa serve` rather than
  staying in the page. What is bought back for it is that the audio
  never reaches anyone but the machine it was spoken on — no third
  party, including the browser vendors, ever sees it. Under `--lan` the
  audio route does not exist: it is absent from the router entirely
  rather than gated, the same shape as `mesa live look`'s CLI-only
  posture, because "transcribe whatever this stranger recorded" is not
  a request an unauthenticated LAN peer should be able to make of a
  decoder running as the owner. The one thing mesa does to the resulting
  text before anything else sees it, on either path, is correct
  mishearings of its own vocabulary against a small local table (mesa
  task 922, above, and `docs/correction.md` on the auris side) — plain
  text matching, not a second model call.
```

## The other sites that ship with it

Four more edits land alongside the route, none of them today:

- **`docs/live.md:30-31`** — the intro bullet's closing clause needs the
  same browser-path-or-auris split as the rewrite above, since it is the
  first place a reader meets the claim.
- **`docs/live.md:1503-1506`** — the sub-bullet naming "a fully local
  pipeline... a local whisper" as not-here-yet. auris retires it, and it
  also names the wrong engine: `docs/engine.md` replaced whisper with
  parakeet tdt-0.6b-v2 int8 well before this decision, so it is
  superseded twice over.
- **`liveDevices.ts:31-33`** and **`liveRecognition.ts:48-51`** — both
  module headers get the same browser-or-auris split, since both are
  read by whoever next touches these paths and should not assert an
  absolute that is no longer true when auris is running.

**`src/core/speech.rs` is not edited**, because its header never made
the claim this task invalidates — it is the TTS path, untouched by audio
input entirely. **`docs/live.md:34-38` is not edited**, because its
"one-directional, server to browser" sentence is about that same TTS
path and stays true: auris adds an input route, it does not turn the
output route into two-way traffic.

## Cross-references

`docs/engine.md` is why local recognition is worth this trade at all —
before task 924's numbers, "local" would have meant an accuracy cut
nobody would take. `docs/latency.md` is why the route holds a persistent
process rather than spawning one per request, which is also why a
one-shot spawn was never on the table for size or retention either: a
route that loads a model per call has no chance of the 300 ms budget
regardless of what it does with the bytes after. `docs/streaming.md` is
why the 25 MB cap only ever has to cover one segment. `docs/vocabulary.md`
and `docs/correction.md` are the gain side of the trade: the vocabulary
reaching the decoder before decoding, and the correction pass after it,
are what the browser path cannot do and auris can — most of the reason
the audio body is worth accepting at all.

## What this does not decide

- The exact wire shape of the audio request — chunked POST, websocket, or
  something else `docs/streaming.md` settles. This doc fixes what
  happens to the bytes once they arrive, not how they arrive.
- Whether the debug-retention window is browsable from the page or left
  as files an operator finds on disk. The requirement is only that the
  page says the setting is on; a viewer for kept recordings, if wanted,
  is a separate decision.
- Whether seven days is the right default once the engine has been in
  real use. Picked here as a starting bound for a feature meant to be
  temporary, not a measured optimum.
- What happens to a debug recording already on disk when the setting is
  turned off. "Pruned on write" bounds growth going forward; it does not
  say whether turning the setting off deletes what is already there.
- The `--lan` gate for routes that already exist. `require_agent_access`
  relaxing rather than refusing under `--lan` is true today for those
  routes and is not a call to re-gate them — only a reason the new audio
  route cannot rely on it the way they do.
