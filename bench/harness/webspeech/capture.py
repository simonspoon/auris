#!/usr/bin/env python3
"""Web Speech (webkitSpeechRecognition) baseline capture, driven over CDP.

Why this exists: bench/harness/run_auris.sh scores auris against the
40-utterance bench/corpus set. A Web Speech number on the same corpus is the
obvious point of comparison, but Chrome's speech recognizer will not take
audio from a file the way auris does -- see README.md in this directory for
the two measured findings that make this harness as complicated as it is:

  1. Chrome's fake audio device CAN feed a WAV into getUserMedia, but only
     with --disable-features=AudioServiceSandbox in addition to the fake
     device flags (README.md, finding 1).
  2. webkitSpeechRecognition does NOT read from that fake device at all --
     it opens the system default audio input directly, bypassing
     getUserMedia's media-stream pipeline entirely (README.md, finding 2).

So the only way to get a WAV file into the actual speech recognizer is an
OS-level loopback set as the macOS default input, with the corpus played
into the default *output*. This script assumes that loopback is already
installed and selected in System Settings -- see README.md for the one-time
operator setup. It does NOT set up the loopback itself.

Because finding 2 makes "the recognizer got it wrong" and "the recognizer
never heard anything" look identical from the outside (both are silence /
no-speech), capture.html also runs a WebAudio RMS meter on getUserMedia in
the same page and session. That meter reads the real system default input
(this script deliberately does NOT pass Chrome's fake-device flags -- see
the comment on CHROME_FLAGS below for why), so it moves in lockstep with
whatever is actually reaching the recognizer. This script polls it and
aborts loudly if the very first utterance shows zero peak RMS, rather than
silently writing "no-speech" for every row and letting someone score that
as "Web Speech is bad at this corpus."

There are two ways to get corpus audio into the recognizer -- --mode
loopback (above) and --mode acoustic, which plays each WAV out of the
default speakers and lets the built-in microphone hear it: no driver
install needed, at the cost of reproducibility (room noise, speaker volume,
mic placement all matter) and of degrading BOTH engines, not just Web
Speech -- see README.md "Two modes" for the full trade-off and for why
acoustic mode also re-records what the mic actually heard, so auris can be
rescored on the same degraded audio rather than compared unfairly against
its own clean corpus.

Usage:
  python3 capture.py --corpus bench/corpus/utterances.tsv \\
      --wav-dir bench/corpus/wav --out bench/results/webspeech-raw.tsv \\
      --mode loopback --device "Background Music"

  python3 capture.py --corpus bench/corpus/utterances.tsv \\
      --wav-dir bench/corpus/wav --out bench/results/webspeech-raw.tsv \\
      --mode acoustic --mic-device 0 --ids u09,u10
"""
import argparse
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

try:
    import websocket  # websocket-client, verified installed separately
except ImportError:
    print("auris: capture.py requires the 'websocket-client' package "
          "(pip3 install websocket-client)", file=sys.stderr)
    raise

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HERE = Path(__file__).resolve().parent


def read_corpus(path, ids=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, text = line.split("\t", 1)
            if ids is None or utt_id in ids:
                rows.append((utt_id, text))
    if ids is not None:
        found = {utt_id for utt_id, _ in rows}
        missing = [i for i in ids if i not in found]
        if missing:
            raise ValueError(f"--ids not found in corpus: {', '.join(missing)}")
    return rows


def wav_duration_seconds(path):
    # stdlib `wave` reads PCM WAV headers directly -- no ffprobe dependency
    # needed just to print a duration estimate. Every fixture wav in this
    # repo is PCM (see bench/corpus/synthesize.sh / record.sh, pcm_s16le).
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def serve_capture_page(directory, port):
    """Serve `directory` (capture.html's own dir) over plain http.server on
    `port`, in a background thread. Chrome loads capture.html from here
    rather than file://, since file:// pages get a different (stricter)
    getUserMedia permission model in some Chrome versions."""
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **kw
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def wait_for_devtools(port, timeout=15.0):
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1)
            return
        except Exception as e:  # noqa: BLE001 -- just polling for readiness
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"Chrome devtools port {port} never came up: {last_err}")


def open_tab(port, url):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/json/new?{url}", timeout=5
    ) as resp:
        target = json.loads(resp.read())
    return target["webSocketDebuggerUrl"]


class CDP:
    """Just enough Chrome DevTools Protocol to drive one tab: send a
    command, wait for its matching response by id. No event queue, no
    reconnect logic -- this script only needs Runtime.evaluate."""

    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._next_id = 1

    def send(self, method, params=None):
        msg_id = self._next_id
        self._next_id += 1
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == msg_id:
                if "error" in reply:
                    raise RuntimeError(f"CDP {method} failed: {reply['error']}")
                return reply.get("result", {})
            # else: an event notification, not our reply -- ignore it

    def evaluate(self, expr, await_promise=False):
        result = self.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": await_promise},
        )
        exc = result.get("exceptionDetails")
        if exc:
            raise RuntimeError(f"page threw: {exc}")
        return result.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass


def launch_chrome(chrome_bin, devtools_port, user_data_dir):
    # Deliberately NOT the fake-device flags from README.md finding 1
    # (--use-fake-device-for-media-stream / --use-file-for-fake-audio-capture).
    # Those make getUserMedia read a fixed file baked in at launch, which
    # would sever the mic-RMS diagnostic from the real loopback device --
    # the meter would show a constant nonzero peak regardless of whether the
    # OS loopback is actually wired, and the "abort if first-utterance RMS
    # is 0" check below would never fire even when it should. We only need
    # --use-fake-ui-for-media-stream, to auto-accept the permission prompt
    # so getUserMedia opens the real system default input headlessly.
    args = [
        chrome_bin,
        f"--remote-debugging-port={devtools_port}",
        "--remote-allow-origins=*",
        "--use-fake-ui-for-media-stream",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--headless=new",
    ]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_wav(ffmpeg_bin, wav_path, device):
    # Plays wav_path into the macOS default OUTPUT device, which must
    # already be the loopback's playback side (see README.md operator
    # setup) for any of this to reach the speech recognizer's input side.
    # `-audio_device_index` also accepts a device name in recent ffmpeg
    # builds; we pass whatever --device was given through unchanged. The
    # trailing "-" is a required-but-unused output filename argument for
    # the audiotoolbox muxer -- it writes to the device, not to a file.
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-f", "audiotoolbox", "-audio_device_index", str(device),
        "-",
    ]
    subprocess.run(cmd, check=True)


def play_wav_default_speakers(wav_path):
    # Acoustic mode plays out of whatever the current default output is --
    # normally the built-in speakers -- rather than a specific device, so
    # this uses macOS's own `afplay` (blocks until playback ends) instead of
    # routing through ffmpeg/audiotoolbox like the loopback path does.
    subprocess.run(["afplay", str(wav_path)], check=True)


def start_mic_recording(ffmpeg_bin, mic_device, out_path):
    # Records the built-in mic (or whatever --mic-device names) to a WAV in
    # the same format as the corpus fixtures (bench/corpus/synthesize.sh:
    # 16 kHz mono pcm_s16le), started slightly before playback so the start
    # of the utterance isn't clipped. Returns the Popen handle; caller stops
    # it with stop_mic_recording after playback ends.
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "avfoundation", "-i", f":{mic_device}",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def stop_mic_recording(proc):
    # ffmpeg's avfoundation input only writes a valid WAV header/trailer on
    # a clean shutdown -- kill -9 leaves a truncated file. SIGINT is
    # ffmpeg's documented "finish up and exit" signal, same technique
    # spike/fixtures/record.sh already uses for the same reason.
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Capture a Web Speech (webkitSpeechRecognition) baseline "
        "over the bench/corpus fixtures, via an OS-level audio loopback. "
        "See README.md in this directory before running."
    )
    parser.add_argument("--corpus", required=True, help="path to utterances.tsv (id<TAB>text)")
    parser.add_argument("--wav-dir", required=True, help="directory of <id>.wav files")
    parser.add_argument("--out", required=True, help="output TSV: id<TAB>transcript")
    parser.add_argument(
        "--diagnostics-out", default=None,
        help="output TSV: id<TAB>micRMSpeak (default: <out dir>/webspeech-diagnostics.tsv)",
    )
    parser.add_argument(
        "--mode", choices=["loopback", "acoustic"], default="loopback",
        help="loopback: play into an OS-level virtual audio device (needs a driver "
        "install, see README.md). acoustic: play out of the default speakers and "
        "record the built-in mic hearing it (no install, not reproducible). Default: loopback.",
    )
    parser.add_argument(
        "--device", default=None,
        help="[loopback mode] ffmpeg -audio_device_index value (or device name) "
        "for the default output / loopback. Required for --mode loopback.",
    )
    parser.add_argument(
        "--mic-device", default=None,
        help="[acoustic mode] ffmpeg -f avfoundation input index for the built-in "
        "mic. Required for --mode acoustic. List with: "
        "ffmpeg -f avfoundation -list_devices true -i \"\"",
    )
    parser.add_argument(
        "--acoustic-out-dir", default=None,
        help="[acoustic mode] where the re-recorded room-audio wavs go, one per "
        "utterance -- auris must be rescored against these, not the clean corpus "
        "wavs, for a fair comparison (see README.md 'Two modes'). "
        "Default: bench/results/acoustic-wav next to --out's directory.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="[acoustic mode] skip the interactive confirmation before playing "
        "audio aloud and taking over the microphone",
    )
    parser.add_argument(
        "--ids", default=None,
        help="comma-separated utterance ids to run, e.g. u09,u10 -- for spot-checking "
        "before committing to the full corpus. Default: every id in --corpus.",
    )
    parser.add_argument("--port", type=int, default=9450, help="Chrome remote-debugging port")
    parser.add_argument("--chrome-bin", default=DEFAULT_CHROME, help="path to Chrome binary")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="path to ffmpeg binary")
    parser.add_argument(
        "--settle-margin", type=float, default=1.5,
        help="seconds to wait after playback ends before reading __wsFinal, "
        "for the recognizer to finish its final result",
    )
    parser.add_argument(
        "--ready-timeout", type=float, default=15.0,
        help="seconds to wait for window.__wsReady before giving up",
    )
    args = parser.parse_args()

    if args.mode == "loopback" and not args.device:
        parser.error("--device is required for --mode loopback")
    if args.mode == "acoustic" and not args.mic_device:
        parser.error("--mic-device is required for --mode acoustic")

    ids = None
    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]

    try:
        corpus = read_corpus(args.corpus, ids=ids)
    except ValueError as e:
        print(f"auris: {e}", file=sys.stderr)
        return 2
    if not corpus:
        print(f"auris: {args.corpus} has no utterances", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = (
        Path(args.diagnostics_out) if args.diagnostics_out
        else out_path.parent / "webspeech-diagnostics.tsv"
    )
    acoustic_out_dir = (
        Path(args.acoustic_out_dir) if args.acoustic_out_dir
        else out_path.parent / "acoustic-wav"
    )

    if not shutil.which(args.ffmpeg_bin) and not os.path.isfile(args.ffmpeg_bin):
        print(f"auris: ffmpeg binary not found: {args.ffmpeg_bin}", file=sys.stderr)
        return 2

    if args.mode == "acoustic":
        acoustic_out_dir.mkdir(parents=True, exist_ok=True)
        if not shutil.which("afplay"):
            print("auris: acoustic mode needs afplay (macOS only)", file=sys.stderr)
            return 2

        total_seconds = sum(
            wav_duration_seconds(Path(args.wav_dir) / f"{utt_id}.wav") for utt_id, _ in corpus
        )
        # +1s settle per utterance is a rough pad, not the real per-utterance
        # cost computed in the loop below -- this is only for the warning.
        est_minutes = (total_seconds + len(corpus) * (args.settle_margin + 1.0)) / 60.0
        print(
            f"auris: ACOUSTIC MODE -- this will play {len(corpus)} utterance(s) "
            f"(~{est_minutes:.1f} min total) out loud through your speakers and "
            "record your microphone while it does. Don't talk or make noise near "
            "the mic until it finishes.",
            file=sys.stderr,
        )
        if not args.yes:
            try:
                input("Press Enter to continue, or Ctrl-C to abort... ")
            except (EOFError, KeyboardInterrupt):
                print("\nauris: aborted", file=sys.stderr)
                return 130

    page_port = free_port()
    httpd = serve_capture_page(HERE, page_port)
    user_data_dir = tempfile.mkdtemp(prefix="auris-webspeech-")
    chrome_proc = None
    cdp = None
    try:
        chrome_proc = launch_chrome(args.chrome_bin, args.port, user_data_dir)
        wait_for_devtools(args.port)
        ws_url = open_tab(args.port, f"http://127.0.0.1:{page_port}/capture.html")
        cdp = CDP(ws_url)
        cdp.send("Runtime.enable")
        cdp.send("Page.enable")

        deadline = time.time() + args.ready_timeout
        ready = False
        while time.time() < deadline:
            if cdp.evaluate("window.__wsReady === true"):
                ready = True
                break
            time.sleep(0.2)
        if not ready:
            print("auris: capture.html never became ready (window.__wsReady)", file=sys.stderr)
            return 1

        with open(out_path, "w", encoding="utf-8") as out_f, \
             open(diagnostics_path, "w", encoding="utf-8") as diag_f:
            for i, (utt_id, _text) in enumerate(corpus):
                wav_path = Path(args.wav_dir) / f"{utt_id}.wav"
                if not wav_path.is_file():
                    print(f"auris: missing wav for {utt_id}: {wav_path}", file=sys.stderr)
                    return 2

                cdp.evaluate("window.__wsReset()")
                time.sleep(0.3)  # let onend/restart settle before playing

                if args.mode == "loopback":
                    play_wav(args.ffmpeg_bin, wav_path, args.device)
                else:
                    # Acoustic mode: record the mic for the same stretch of
                    # time the corpus wav plays through the speakers, so
                    # bench/results/acoustic-wav/<id>.wav holds what the
                    # room actually produced -- the signal Web Speech heard
                    # -- for auris to be rescored against (README.md "Two
                    # modes"). ~0.5s of lead-in/lead-out margin around
                    # playback so the recording isn't clipped at either end.
                    rec_path = acoustic_out_dir / f"{utt_id}.wav"
                    recorder = start_mic_recording(args.ffmpeg_bin, args.mic_device, rec_path)
                    time.sleep(0.5)
                    try:
                        play_wav_default_speakers(wav_path)
                    finally:
                        time.sleep(0.5)
                        stop_mic_recording(recorder)
                time.sleep(args.settle_margin)

                mic_rms = cdp.evaluate("window.__micRMS") or 0
                final = cdp.evaluate("window.__wsFinal") or []
                transcript = " ".join(final).strip()

                if i == 0 and mic_rms == 0:
                    if args.mode == "loopback":
                        hint = (
                            "the loopback is not wired up (not set as both macOS "
                            "default output and default input). See README.md "
                            "'Operator setup'."
                        )
                    else:
                        hint = (
                            "the default output/input devices aren't actually "
                            "connected (muted speakers, wrong --mic-device, OS "
                            "input permission not granted to Chrome)."
                        )
                    print(
                        f"auris: mic RMS peak is 0 on the first utterance -- {hint} "
                        "Every number after this one would be garbage.",
                        file=sys.stderr,
                    )
                    return 1

                out_f.write(f"{utt_id}\t{transcript}\n")
                out_f.flush()
                diag_f.write(f"{utt_id}\t{mic_rms}\n")
                diag_f.flush()
                print(f">>> {utt_id}: rms={mic_rms:.4f} transcript={transcript!r}", file=sys.stderr)

        return 0
    finally:
        if cdp is not None:
            cdp.close()
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_proc.kill()
        httpd.shutdown()
        shutil.rmtree(user_data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
