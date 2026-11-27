#!/usr/bin/env python3
"""Play GTA and record the session for imitation learning.

The emulator runs at real time in a background thread. Two front ends
share it and record identically:

  web      the game in a browser: frames over a WebSocket, input back
  desktop  a native pygame window, for anyone with a screen on the
           machine running the emulator (a local desktop, or RDP/VNC)

See PLAY.md for the design.

  $ .venv/bin/python tools/play.py                 # then open http://localhost:8765
  $ .venv/bin/python tools/play.py --ui desktop    # needs a display and pygame
  $ .venv/bin/python tools/play.py --verify recordings/<session>

A recording is the start savestate plus the button mask held on every
emulated frame; --verify replays it and checks the final RAM is identical.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import sys
import threading
import time
from array import array
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import DETERMINISTIC_CORE_OPTIONS, GTA1Config, find_disc  # noqa: E402
from gta1env.libretro import JOYPAD, PSX_BUTTON, LibretroCore  # noqa: E402
from gta1env.ram import RamMap  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(HERE, "recordings")
# Separate from scratch/saves so memory-card writes never touch the files
# that training processes use.
SAVE_DIR = os.path.join(HERE, "scratch", "play_saves")

BIT = {name: JOYPAD[key] for name, key in PSX_BUTTON.items()}
INPUT_TIMEOUT = 1.0   # release everything if the page goes quiet


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_core(bios_mode):
    cfg = GTA1Config()
    rom = cfg.rom_path or find_disc()
    os.makedirs(SAVE_DIR, exist_ok=True)
    options = dict(DETERMINISTIC_CORE_OPTIONS)
    options["pcsx_rearmed_bios"] = bios_mode
    core = LibretroCore(cfg.core_path, cfg.bios_dir, SAVE_DIR, options=options)
    av = core.load_game(rom)
    return core, av, cfg, rom, options


def mask_of(names):
    m = 0
    for n in names:
        if n in BIT:
            m |= 1 << BIT[n]
    return m


# ------------------------------------------------------------------ session


class Session:
    """Owns the core. Only the emulator thread touches it."""

    def __init__(self, args):
        self.args = args
        self.core, self.av, cfg, self.rom, self.options = open_core(args.bios_mode)
        self.ram = self.core.memory()
        self.ram_map = RamMap.load(cfg.ram_map_path)
        self.ram_names = sorted(self.ram_map.fields)
        with open(args.state, "rb") as fh:
            self.core.unserialize(fh.read())
        self.cfg = cfg
        self.fps = self.av["fps"] or 60.0

        self.inputs = array("H")
        self.ram_rows = []
        self.mask = 0
        self.last_input = time.monotonic()
        # Wait for the page: nothing is stepped or recorded until the
        # browser sends its first input.
        self.paused = True
        self.connected = False
        self.running = True
        self.payload = None            # JPEG bytes (web) or RGB array (desktop)
        self.frame_id = -1
        self.cond = threading.Condition()
        # Flow control: at most `inflight` frames are sent but not yet
        # acked as drawn, so nothing queues in the network (lag stays near
        # one round trip) while the frame rate isn't capped at 1/RTT.
        self.credits = 0              # frames the browser may still receive
        self.pending = False          # an encoded frame awaits the sender
        self.sent_at = {}
        self.seen_id = -1             # last frame the browser drew
        self.seen = array("i")        # seen_id at every emulated frame
        self.rtt_ms = 0.0
        self.measured_fps = 0.0
        self.started = datetime.now(timezone.utc)
        self.saved_to = None

    def set_mask(self, mask):
        if not self.connected:
            self.connected = True
            self.paused = False
        self.mask = mask
        self.last_input = time.monotonic()

    def ack(self, frame_id):
        sent = self.sent_at.pop(frame_id, None)
        if sent is not None:
            rtt = (time.perf_counter() - sent) * 1000
            self.rtt_ms = rtt if not self.rtt_ms else 0.9 * self.rtt_ms + 0.1 * rtt
            self.seen_id = max(self.seen_id, frame_id)
            with self.cond:
                self.credits = min(self.credits + 1, self.args.inflight)

    def take_frame(self, timeout):
        """The newest frame for the front end, or (None, None) after
        `timeout`. Taking it spends one credit; ack() gives it back."""
        with self.cond:
            self.cond.wait_for(lambda: self.pending or not self.running,
                               timeout)
            if not self.pending:
                return None, None
            self.pending = False
            self.credits -= 1
            self.sent_at[self.frame_id] = time.perf_counter()
            return self.payload, self.frame_id

    def loop(self):
        period = 1.0 / self.fps
        deadline = time.perf_counter()
        tick, t0 = 0, time.perf_counter()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                deadline = time.perf_counter()
                continue
            if time.monotonic() - self.last_input > INPUT_TIMEOUT:
                self.mask = 0
            mask = self.mask
            frame_id = len(self.inputs)
            show = self.credits > 0 and not self.pending
            self.core.set_buttons(mask)
            self.core.run(convert=show)
            self.inputs.append(mask)
            self.seen.append(self.seen_id)
            if self.ram_names:
                vals = self.ram_map.read_all(self.ram)
                self.ram_rows.append([vals[n] for n in self.ram_names])
            if show and self.core.frame is not None:
                payload = self.core.frame          # a fresh array per frame
                if self.args.ui == "web":
                    bgr = cv2.cvtColor(payload, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [
                        cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                    payload = buf.tobytes() if ok else None
                if payload is not None:
                    with self.cond:
                        self.payload = payload
                        self.frame_id = frame_id
                        self.pending = True
                        self.cond.notify_all()

            tick += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                self.measured_fps = tick / (now - t0)
                tick, t0 = 0, now
            deadline += period
            delay = deadline - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:          # fell far behind: resync, don't burst
                deadline = now

    def status(self):
        vals = self.ram_map.read_all(self.ram) if self.ram_names else {}
        return {"frames": len(self.inputs), "fps": round(self.measured_fps, 1),
                "paused": self.paused, "mask": self.mask,
                "seconds": round(len(self.inputs) / self.fps, 1),
                "rtt_ms": round(self.rtt_ms, 1),
                "saved_to": self.saved_to,
                **{k: round(v, 2) if isinstance(v, float) else v
                   for k, v in vals.items()}}

    def save(self, end_reason):
        if self.saved_to or not self.inputs:
            return self.saved_to
        sid = self.started.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(REC_DIR, sid)
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(self.args.state, os.path.join(out, "start.state"))
        np.save(os.path.join(out, "inputs.npy"),
                np.frombuffer(self.inputs, dtype=np.uint16))
        np.save(os.path.join(out, "seen.npy"),
                np.frombuffer(self.seen, dtype=np.int32))
        if self.ram_rows:
            np.save(os.path.join(out, "ram.npy"),
                    np.asarray(self.ram_rows, dtype=np.float64))
        lag = np.arange(len(self.seen)) - np.frombuffer(self.seen, np.int32)
        lag = lag[np.frombuffer(self.seen, np.int32) >= 0]
        meta = {
            "session_id": sid,
            "player": self.args.player,
            "started": self.started.isoformat(),
            "ended": datetime.now(timezone.utc).isoformat(),
            "end_reason": end_reason,
            "n_frames": len(self.inputs),
            "fps": self.fps,
            "start_state_source": os.path.relpath(self.args.state, HERE),
            "start_state_sha256": sha256_file(self.args.state),
            "core_sha256": sha256_file(self.cfg.core_path),
            "disc": os.path.basename(self.rom),
            "bios_mode": self.args.bios_mode,
            "core_options": self.options,
            "ram_fields": self.ram_names,
            "final_ram_sha256": hashlib.sha256(self.ram.tobytes()).hexdigest(),
            "final_ram": self.ram_map.read_all(self.ram),
            # frames between the newest frame and the one on screen
            "display_lag_frames": {
                "median": float(np.median(lag)) if len(lag) else None,
                "p95": float(np.percentile(lag, 95)) if len(lag) else None},
        }
        with open(os.path.join(out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.saved_to = os.path.relpath(out, HERE)
        print(f"saved {len(self.inputs)} frames -> {self.saved_to}", flush=True)
        return self.saved_to


# ---------------------------------------------------------------- web page

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>GTA1Sim play</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center}
 canvas{width:960px;max-width:100vw;image-rendering:pixelated;background:#000;
     margin-top:12px;outline:2px solid #333}
 #bar{margin:8px;font-family:ui-monospace,monospace}
 .rec{color:#f55} table{border-collapse:collapse;margin:8px}
 td{padding:2px 10px;border-bottom:1px solid #222} kbd{color:#fff}
 button{margin:4px;padding:6px 14px}
</style></head><body>
<canvas id="screen" width="320" height="240"></canvas>
<div id="bar">connecting…</div>
<div><button id="pause">Pause (P)</button><button id="stop">Stop &amp; save</button></div>
<table>
<tr><td><kbd>↑ / W</kbd> forward (cross)</td><td><kbd>↓ / S</kbd> reverse (triangle)</td><td><kbd>← → / A D</kbd> steer</td></tr>
<tr><td><kbd>Space</kbd> brake (R2)</td><td><kbd>J</kbd> fire (circle)</td><td><kbd>K</kbd> enter/exit car (square)</td></tr>
<tr><td><kbd>L</kbd> special (R1)</td><td><kbd>Q / E</kbd> prev/next weapon</td><td><kbd>Enter</kbd> start · <kbd>Tab</kbd> select</td></tr>
</table>
<script>
const KEYS = {ArrowUp:"cross",KeyW:"cross",ArrowDown:"triangle",KeyS:"triangle",
  ArrowLeft:"left",KeyA:"left",ArrowRight:"right",KeyD:"right",
  Space:"r2",KeyJ:"circle",KeyK:"square",KeyL:"r1",KeyQ:"l1",KeyE:"l2",
  Enter:"start",Tab:"select"};
// Standard gamepad mapping -> PS1 buttons
const PAD = {0:"cross",1:"circle",2:"square",3:"triangle",4:"l1",5:"r1",
  6:"l2",7:"r2",8:"select",9:"start",12:"up",13:"down",14:"left",15:"right"};
const held = new Set();
let last = "", lastSent = 0;
const canvas = document.getElementById("screen"), ctx = canvas.getContext("2d");
// One socket both ways: frames down, input and draw-acks up.
const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://")
                         + location.host + "/ws");
ws.binaryType = "arraybuffer";
ws.onmessage = async ev => {
  const id = new DataView(ev.data).getUint32(0, true);
  const bmp = await createImageBitmap(new Blob([ev.data.slice(4)], {type: "image/jpeg"}));
  if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
    canvas.width = bmp.width; canvas.height = bmp.height; }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  ws.send("a " + id);              // drawn: the server may send the next one
};
ws.onclose = () => { document.getElementById("bar").textContent = "disconnected"; };
function send(force){
  const pad = new Set();
  for (const gp of navigator.getGamepads ? navigator.getGamepads() : []) {
    if (!gp) continue;
    gp.buttons.forEach((b,i)=>{ if (b.pressed && PAD[i]) pad.add(PAD[i]); });
    if (gp.axes[0] < -0.5) pad.add("left"); if (gp.axes[0] > 0.5) pad.add("right");
  }
  const body = [...new Set([...held, ...pad])].sort().join(",");
  const now = performance.now();
  if (body !== last || force || now - lastSent > 250) {
    last = body; lastSent = now;
    if (ws.readyState === 1) ws.send("i " + body);
  }
}
addEventListener("keydown", e => {
  if (e.code === "KeyP" && !e.repeat) { togglePause(); e.preventDefault(); return; }
  if (KEYS[e.code]) { held.add(KEYS[e.code]); e.preventDefault(); send(); }
});
addEventListener("keyup", e => {
  if (KEYS[e.code]) { held.delete(KEYS[e.code]); e.preventDefault(); send(); }
});
addEventListener("blur", () => { held.clear(); send(true); });
(function poll(){ send(); requestAnimationFrame(poll); })();
function togglePause(){ fetch("/pause", {method:"POST"}); }
document.getElementById("pause").onclick = togglePause;
document.getElementById("stop").onclick = () =>
  fetch("/stop", {method:"POST"}).then(r=>r.json()).then(s=>{
    document.getElementById("bar").textContent = "saved to " + s.saved_to; });
setInterval(() => fetch("/status").then(r=>r.json()).then(s => {
  if (s.saved_to) { document.getElementById("bar").textContent = "saved to " + s.saved_to; return; }
  const extra = Object.entries(s).filter(([k])=>!["frames","fps","paused","mask","seconds","saved_to"].includes(k))
    .map(([k,v])=>k+"="+v).join("  ");
  document.getElementById("bar").innerHTML =
    (s.paused ? "PAUSED" : '<span class="rec">● REC</span>') +
    `  ${s.seconds}s  ${s.frames} frames  ${s.fps} fps  ${extra}`;
}).catch(()=>{}), 500);
</script></body></html>"""


def make_handler(sess):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                self._json(sess.status())
            elif self.path == "/ws":
                self._websocket()
            else:
                self.send_error(404)

        def _websocket(self):
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(400, "websocket upgrade expected")
                return
            accept = base64.b64encode(hashlib.sha1(
                (key + WS_GUID).encode()).digest()).decode()
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.close_connection = True
            sock = self.connection
            # Inputs are tiny packets: don't let Nagle hold them back.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            closed = threading.Event()

            def reader():
                try:
                    while not closed.is_set():
                        msg = ws_recv(self.rfile)
                        if msg is None:
                            break
                        cmd, _, arg = msg.partition(" ")
                        if cmd == "i":
                            sess.set_mask(mask_of(b for b in arg.split(",") if b))
                        elif cmd == "a":
                            sess.ack(int(arg))
                except (OSError, ValueError):
                    pass
                closed.set()
                with sess.cond:
                    sess.cond.notify_all()

            threading.Thread(target=reader, daemon=True).start()
            with sess.cond:                   # a new page: fresh credits
                sess.credits = sess.args.inflight
                sess.sent_at = {}
            try:
                while sess.running and not closed.is_set():
                    jpg, fid = sess.take_frame(timeout=0.25)
                    if jpg is not None:
                        ws_send(sock, struct.pack("<I", fid) + jpg)
            except OSError:
                pass
            closed.set()
            with sess.cond:
                sess.credits = 0
                sess.pending = False

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            if self.path == "/pause":
                sess.paused = not sess.paused
                self._json(sess.status())
            elif self.path == "/stop":
                sess.paused = True
                time.sleep(0.1)         # let the emulator thread settle
                sess.save("stop")
                self._json(sess.status())
            else:
                self.send_error(404)
    return Handler


# --------------------------------------------- minimal WebSocket (RFC 6455)

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_send(sock, payload):
    """One unfragmented binary frame (server frames are never masked)."""
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x82, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x82, 126, n)
    else:
        head = struct.pack("!BBQ", 0x82, 127, n)
    sock.sendall(head + payload)


def ws_recv(rfile):
    """Next text message from the browser, or None when the socket closes.
    Browsers send small single-frame messages, so fragments aren't handled."""
    while True:
        head = rfile.read(2)
        if len(head) < 2:
            return None
        opcode, n = head[0] & 0x0F, head[1] & 0x7F
        if n == 126:
            n = struct.unpack("!H", rfile.read(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", rfile.read(8))[0]
        key = rfile.read(4) if head[1] & 0x80 else b"\0\0\0\0"
        data = rfile.read(n)
        if len(data) < n:
            return None
        data = bytes(b ^ key[i & 3] for i, b in enumerate(data))
        if opcode == 0x8:                      # close
            return None
        if opcode == 0x1:                      # text; ignore ping/pong/binary
            return data.decode()


# ----------------------------------------------------------------- desktop


def run_desktop(sess):
    """Native window with pygame. Input is sampled from the keyboard and any
    game controller on every pass of the loop, a few hundred times a
    second; each frame is drawn as soon as the emulator produces it."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from pygame._sdl2 import controller

    keymap = {
        pygame.K_UP: "cross", pygame.K_w: "cross",
        pygame.K_DOWN: "triangle", pygame.K_s: "triangle",
        pygame.K_LEFT: "left", pygame.K_a: "left",
        pygame.K_RIGHT: "right", pygame.K_d: "right",
        pygame.K_SPACE: "r2", pygame.K_j: "circle", pygame.K_k: "square",
        pygame.K_l: "r1", pygame.K_q: "l1", pygame.K_e: "l2",
        pygame.K_RETURN: "start", pygame.K_TAB: "select",
    }
    # SDL's controller API gives every pad the same Xbox-style layout.
    padmap = {
        pygame.CONTROLLER_BUTTON_A: "cross", pygame.CONTROLLER_BUTTON_B: "circle",
        pygame.CONTROLLER_BUTTON_X: "square", pygame.CONTROLLER_BUTTON_Y: "triangle",
        pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "l1",
        pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "r1",
        pygame.CONTROLLER_BUTTON_BACK: "select", pygame.CONTROLLER_BUTTON_START: "start",
        pygame.CONTROLLER_BUTTON_DPAD_UP: "up", pygame.CONTROLLER_BUTTON_DPAD_DOWN: "down",
        pygame.CONTROLLER_BUTTON_DPAD_LEFT: "left",
        pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "right",
    }

    pygame.init()
    controller.init()
    pads = {}
    w, h = 320, 240
    scale = sess.args.scale
    screen = pygame.display.set_mode((w * scale, h * scale), pygame.RESIZABLE)
    pygame.display.set_caption("GTA1Sim")
    with sess.cond:
        sess.credits = sess.args.inflight
    sess.set_mask(0)                       # window is up: start the clock
    print("playing in a window: P pauses, Esc saves and quits", flush=True)
    title_at = 0.0
    try:
        while sess.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return
                    if ev.key == pygame.K_p:
                        sess.paused = not sess.paused
                elif ev.type == pygame.CONTROLLERDEVICEADDED:
                    pads[ev.device_index] = controller.Controller(ev.device_index)

            held = pygame.key.get_pressed()
            names = {n for k, n in keymap.items() if held[k]}
            for pad in pads.values():
                names.update(n for b, n in padmap.items() if pad.get_button(b))
                if pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT) > 16000:
                    names.add("l2")
                if pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT) > 16000:
                    names.add("r2")
                x = pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX)
                if x < -16000:
                    names.add("left")
                elif x > 16000:
                    names.add("right")
            sess.set_mask(mask_of(names))

            frame, fid = sess.take_frame(timeout=0.004)
            if frame is not None:
                fh, fw = frame.shape[:2]
                surf = pygame.image.frombuffer(frame.tobytes(), (fw, fh), "RGB")
                sw, sh = screen.get_size()
                k = min(sw / fw, sh / fh)          # keep the aspect ratio
                dw, dh = int(fw * k), int(fh * k)
                screen.fill((0, 0, 0))
                screen.blit(pygame.transform.scale(surf, (dw, dh)),
                            ((sw - dw) // 2, (sh - dh) // 2))
                pygame.display.flip()
                sess.ack(fid)

            now = time.monotonic()
            if now - title_at > 0.5:
                title_at = now
                st = sess.status()
                extra = "  ".join(f"{k}={v}" for k, v in st.items()
                                  if k in sess.ram_names)
                pygame.display.set_caption(
                    f"GTA1Sim  {'PAUSED' if st['paused'] else 'REC'}  "
                    f"{st['seconds']}s  {st['fps']} fps  {extra}   "
                    "[P pause, Esc save+quit]")
    finally:
        pygame.quit()


# ------------------------------------------------------------------ verify


def verify(rec, bios_mode):
    with open(os.path.join(rec, "meta.json")) as fh:
        meta = json.load(fh)
    core, _, cfg, _, _ = open_core(meta.get("bios_mode", bios_mode))
    if sha256_file(cfg.core_path) != meta["core_sha256"]:
        print("WARNING: core build differs from the one that recorded this")
    with open(os.path.join(rec, "start.state"), "rb") as fh:
        core.unserialize(fh.read())
    inputs = np.load(os.path.join(rec, "inputs.npy"))
    t = time.perf_counter()
    for m in inputs:
        core.set_buttons(int(m))
        core.run(convert=False)
    dt = time.perf_counter() - t
    got = hashlib.sha256(core.memory().tobytes()).hexdigest()
    ok = got == meta["final_ram_sha256"]
    print(f"replayed {len(inputs)} frames in {dt:.1f}s: "
          f"{'IDENTICAL' if ok else 'MISMATCH'} final RAM")
    return 0 if ok else 1


# -------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ui", choices=["web", "desktop"], default="web")
    p.add_argument("--scale", type=int, default=3,
                   help="desktop window size as a multiple of 320x240")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--state", default=os.path.join(HERE, "config", "boot.state"),
                   help="savestate to start from (read only)")
    p.add_argument("--player", default="anon")
    p.add_argument("--bios-mode", default="auto")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--inflight", type=int, default=3,
                   help="frames sent but not yet drawn (1 = lowest lag, "
                        "but frame rate capped at 1/round-trip)")
    p.add_argument("--verify", metavar="RECORDING",
                   help="replay a recording and check the final RAM matches")
    args = p.parse_args()

    if args.verify:
        sys.exit(verify(args.verify, args.bios_mode))

    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    sess = Session(args)
    emu = threading.Thread(target=sess.loop, daemon=True)
    emu.start()
    if args.ui == "desktop":
        try:
            run_desktop(sess)
        except KeyboardInterrupt:
            pass
        finally:
            sess.paused = True
            time.sleep(0.1)
            sess.save("quit")
            sess.running = False
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sess))
    server.daemon_threads = True
    print(f"open http://localhost:{args.port}  (Ctrl-C saves and quits)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sess.paused = True
        time.sleep(0.1)
        sess.save("quit")
        sess.running = False
        with sess.cond:
            sess.cond.notify_all()
        server.server_close()


if __name__ == "__main__":
    main()
