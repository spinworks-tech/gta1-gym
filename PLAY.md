# Playing in the browser and recording sessions

Design note for letting people play GTA1Sim from a web browser and
record their sessions as demonstrations for imitation learning.

**Status.** Steps 1–3 of the [build order](#build-order) work locally:

```bash
.venv/bin/python tools/play.py                       # open http://localhost:8765
.venv/bin/python tools/play.py --ui desktop             # native window (local desktop, RDP)
.venv/bin/python tools/play.py --verify recordings/<session>
.venv/bin/python tools/replay.py recordings/*        # -> <session>/dataset_fs4_84g.npz
```

`play.py` uses the standard library plus OpenCV, with a minimal
WebSocket implementation of its own. A single socket carries frames
down and input plus draw acknowledgements up, with at most 3 frames in
flight (`--inflight`), so nothing queues in the network. With a
single frame in flight the frame rate is capped at 1/round-trip: a real
42 ms LAN round trip showed only 16 fps on screen. With 3 in flight a
simulated 40 ms round trip shows 56 fps. Display lag is about one
round trip and is recorded in `seen.npy` and in `meta.json`
(`display_lag_frames`). An earlier MJPEG + POST version was noticeably
laggy through the VS Code tunnel because frames queued up in buffers.
Multi-player and hosting are not built.

**Desktop mode.** `--ui desktop` opens a pygame window instead of
serving a page, for anyone with a screen on the machine running the
emulator: a local desktop, or a remote desktop session (RDP, VNC). It
shares the same emulator thread and writes identical recordings,
including `seen.npy`. Keyboard and any SDL game controller are read on
every pass of the loop, and each frame is drawn as soon as it is
produced (offscreen test: 58 fps on screen, 1 frame behind). P pauses,
Esc or closing the window saves and quits. Over RDP the remote-desktop
protocol becomes the transport, so lag depends on it rather than on
this code; smaller windows (`--scale 2`) are cheaper to send.

## Goals

- Anyone with the link plays in a browser: no install, keyboard or
  gamepad.
- Every session is recorded in a form that replays **exactly** through
  the same pipeline the RL agent uses.
- Recordings become `(obs, action, reward, done)` datasets for
  behaviour cloning or for pretraining the PPO policy in `train_ppo.py`.
- The emulator can run on a remote host and serve several players at
  once.

Non-goals for now: multiplayer, audio, mobile touch controls.

## Architecture

```
 browser                                   server (one host)
 ───────                                   ─────────────────
 <canvas>  ◄── frames (JPEG, WebSocket) ── session worker  ── pcsx_rearmed core
 keyboard/ ─── held-button mask ─────────►  (one process      stepped at 60 fps
 gamepad       + last frame id seen          per player)      wall-clock paced
                                                │
                                                └─► recordings/<session>/
                    ▲
                    │ HTTP: lobby, start/stop, consent
              tools/play.py  (aiohttp: static page + WebSocket + worker pool)
```

- **`tools/play.py`** is a web server. It serves one static page, accepts
  a WebSocket per player and starts one **session worker** process per
  player. libretro cores keep global state, so one core per process is
  required, the same rule as `SubprocVecEnv`.
- **The session worker** owns a `LibretroCore`, loads the start state and
  runs a loop paced by wall-clock time:
  1. read the latest button mask from the socket (non-blocking);
  2. `core.set_buttons(mask)`, `core.run()`;
  3. append the mask to the recording;
  4. encode the frame and send it, dropping the frame if the socket is
     still busy with the previous one.

  The emulator never waits for the network. A slow connection costs
  frames on screen, not game speed, so the recording stays a clean
  60 Hz input stream.
- **The browser page** draws frames to a `<canvas>`, tracks which keys
  or gamepad buttons are held and sends the full mask whenever it
  changes (plus a heartbeat). Sending the whole mask, not key events,
  means a lost or reordered message can't leave a button stuck.

### Controls

The page maps keys to PS1 buttons, which reach the game through
`CONTROLS` in [gta1env/ram.py](gta1env/ram.py) (Layout One):

| key | PS1 button | in game |
|---|---|---|
| ↑ / W | cross | forward / accelerate |
| ↓ / S | triangle | backward / reverse |
| ← → / A D | left, right | steer |
| Space | r2 | brake |
| J | circle | fire |
| K | square | enter / exit vehicle |
| L | r1 | special |
| Q / E | l1 / l2 | previous / next weapon |
| Enter | start | pause |
| Tab | select | repeat message |

Gamepads go through the browser Gamepad API with the standard
DualShock layout. The key map lives in the page, so changing it never
affects recordings: those store PS1 buttons, not keys.

## Recording format

The emulator is deterministic: the same savestate plus the same
per-frame inputs gives a byte-identical trajectory (see TODO.md). So a
recording is the inputs, not the video:

```
recordings/<session_id>/
  start.state     savestate the session began from
  inputs.npy      uint16[n_frames]   button mask held on each emulated frame
  seen.npy        uint32[n_frames]   last frame id the client had displayed
  ram.npy         int32[n_frames, k] RAM-map fields per frame (score, x, y…)
  meta.json       see below
```

`meta.json` pins everything replay depends on:

```json
{
  "session_id": "…", "player": "anon-7f3a", "started": "2026-11-26T16:00:00Z",
  "core_sha256": "…", "disc_sha256": "…", "bios_mode": "auto",
  "core_options": { "...": "DETERMINISTIC_CORE_OPTIONS as used" },
  "start_state": "config/boot.state", "n_frames": 54000,
  "final_ram_sha256": "…", "end_reason": "quit | disconnect | timeout"
}
```

About 7 KB per minute of play for `inputs.npy`, so storage is not a
concern.

- **`final_ram_sha256`** lets replay prove that it reproduced the session
  exactly.
- **`seen.npy`** records what the player could actually see when they
  pressed each button. The gap between it and the current frame is the
  player's latency, which helps filter or correct laggy demonstrations.
- **`ram.npy`** is redundant with replay but cheap. It allows quick
  filtering (by final score or distance driven) without replaying
  anything.

**Pausing.** Pausing the session (not the in-game pause) stops the
emulator, so no frames pass and `inputs.npy` stays continuous: no
segments needed. The server also starts paused and only begins when the
page sends its first input, so recordings don't open with idle frames.

## Replay: turning recordings into datasets

`tools/replay.py` rebuilds training data from recordings:

1. Load `start.state`, check the core and disc hashes against
   `meta.json`, refuse on mismatch.
2. Feed `inputs.npy` frame by frame and check `final_ram_sha256` at the
   end.
3. Group frames into windows of `frameskip` (default 4) and map each
   window to one of the discrete actions in `DEFAULT_ACTIONS`. Take
   every button pressed at any point in the window, then the closest
   action by number of differing buttons. A majority vote would drop
   taps: fire and enter/exit are usually held for only a frame or two.
   The emulator is still driven by the raw per-frame masks, so the
   trajectory is the one actually played; only the label is
   approximated.
4. Run the same `_preprocess` and frame stack as `GTA1Env`, and compute
   reward with the same `RewardShaper`.
5. Write `.npz`: `obs uint8 (N, C, H, W)`, `actions int64 (N,)`,
   `rewards float32`, `dones bool`, plus the raw masks and RAM fields.

Because observations are regenerated, a new observation setup (colour,
128×128, frameskip 2, a different action set) only needs a re-run of
replay, not new players.

**Action-set mismatch.** People hold combinations that aren't among the
14 actions (forward + steer + fire) and tap faster than 4 frames.
Replay reports what fraction of windows needed an approximate match and
which combinations were missing. If that fraction is high, the fix is
to extend `DEFAULT_ACTIONS`, not to hide the mismatch.

## Hosting

**Capacity.** Real time is 60 fps and one core sustains about 1,800 fps
on this Xeon, so emulation costs about 1/30 of a core per player. JPEG
encoding at 60 fps costs more than emulation; encoding at 30 fps
(the game doesn't need more on screen) and at the native 320×240 keeps a
player well under a tenth of a core. A 16-thread box should hold
dozens of concurrent players. Measure before promising a number.

**Bandwidth.** About 20–40 KB per JPEG frame at 30 fps is 5–10 Mbit/s
per player. Fine on a LAN. For public hosting, switch the video path to
WebRTC with H.264 (aiortc), which cuts this tenfold. The input and
recording design doesn't change.

**Latency.** Streaming adds a round trip to every input. Host close to
the players. Up to about 100 ms is playable for GTA's top-down driving,
and `seen.npy` measures it per session either way.

**Session limits.** One worker per player, a cap on concurrent sessions
(extra players wait in a lobby), an idle timeout (no input for 2 minutes
ends the session), and a maximum session length.

**Deployment.** A container image with the core, the Python env and
`play.py` behind a reverse proxy that terminates TLS (WebSockets need
`wss://` on a public host). The disc image and BIOS are mounted at run
time, never baked into the image.

## Access, consent and legal

- **Copyright.** The disc and BIOS are copyrighted. Streaming the game to
  the public is distributing it, whatever the technical setup. Keep the
  server behind a login, limited to people who own the game or are part
  of the project, and never expose the disc, BIOS or savestates for
  download. Savestates contain game RAM, so treat recordings the same
  way.
- **Consent.** Show a notice before play starts: the session is recorded
  and used to train models. Record nothing until it's accepted.
- **Privacy.** Store an anonymous player id, not a name or email. Inputs
  and RAM are all that's kept: no IP addresses in `meta.json`.
- **Abuse.** Logins, a per-account session cap, and rate limits on the
  WebSocket.

## Build order

1. **Recorder core, no web.** A worker that steps a core at 60 fps from
   a mask source and writes the recording format above. Test it against
   `tests/mock_core.c`: record random masks, replay, get identical RAM.
2. **`tools/replay.py`.** Recording → `.npz`, with the hash checks and
   the action-mismatch report.
3. **Local play.** `tools/play.py` with the canvas page and JPEG over
   WebSocket, one player, reached through VS Code port forwarding.
   Record a few real sessions and replay them.
4. **Multi-player.** Worker pool, lobby, limits, consent screen, login.
5. **Hosting.** Container, TLS proxy, then WebRTC video if bandwidth
   demands it.
6. **Use the data.** Behaviour-cloning baseline on the `.npz` files, then
   PPO initialised from it.

## Open questions

- **Start states.** One `boot.state` for everyone, or a small set
  (different spawn points, mid-mission) to widen coverage?
- **Who plays.** Only project members, or a wider group that owns the
  game? This decides how much of the hosting section is needed.
- **Demonstration quality.** Keep everything and weight by score, or
  filter to sessions that complete missions?
- **Input rate.** Should recordings keep the 60 Hz per-frame masks
  (proposed), or also store a pre-quantised 15 Hz action stream for
  convenience?
