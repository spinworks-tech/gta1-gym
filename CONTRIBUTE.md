# Contributing play sessions

GTA1Sim trains agents to play the original 1997 *Grand Theft Auto*
(PlayStation). Recordings of people playing let the agent learn from
human examples (imitation learning) before it starts learning by trial
and error, and that is where you can help.

A contribution is a short file of your controller inputs, typically a
few KiB per session. It contains no game data and no video. This guide
covers the one-time setup, recording, and how to submit.

## How it works, and why your setup must match exactly

The emulator is deterministic: the same starting point plus the same
button presses on every frame always produces the same game, byte for
byte. So we don't need your video, only your inputs. We replay them on
our side and rebuild everything the agent needs.

That only works if your emulator behaves identically to ours. In
practice that means:

| | must match | why |
|---|---|---|
| **Disc** | Grand Theft Auto (USA), SLUS-00106 | other regions and revisions run different code |
| **BIOS** | SCPH-1001 (NTSC-U, v2.2) | the game calls BIOS code constantly |
| **Emulator core** | `pcsx_rearmed`, the pinned commit | emulation differences change the game |
| **Start point** | the reference start state | every session starts at the same moment |

You don't need to compare hashes by hand. You build the start state
yourself, by booting the game from power-on with a fixed sequence of
button presses, and compare its hash with ours. If it matches, your
disc, BIOS and core all behave identically, and your sessions will
replay on our side. If it doesn't, `tools/contrib.py check` shows which
part differs.

## What you need

- Linux on x86-64, Python 3.12+, GCC and Make.
- **Your own** copy of the game and BIOS. We never distribute them and
  never ask you to send them.

### Disc: `roms/`

A Redump-verified rip of *Grand Theft Auto (USA)*, as `.cue` plus
`.bin` files with these exact names:

| file | sha256 |
|---|---|
| `Grand Theft Auto (USA).cue` | `d3c221b0612c4d9d0b94241945775115cafe760fa0cf1667cf3b493be1e61c2d` |
| `Grand Theft Auto (USA) (Track 01).bin` | `fd7bc3271d638208de6fedf20f193e21a0f89680077f2cac49156511310c9572` |
| `… (Track 02)` to `(Track 11).bin` | listed in [config/contrib.json](config/contrib.json) |

Track 01 is the game data; tracks 02–11 are CD audio, which the
emulator has switched off. If your files carry other names, rename them
to the Redump names and fix the names inside the `.cue` to match.

### BIOS: `bios/`

`SCPH1001.BIN`, 524,288 bytes:

```
md5     924e392ed05558ffdb115408c263dccf
sha256  71af94d1e47a68c11e8fdb9f8368040601514a42a5a399cda48c7d3bff1e99d3
```

Keep it as the only BIOS file in `bios/`. The emulator's built-in BIOS
replacement (`--bios-mode HLE`) behaves differently, so sessions
recorded with it can't be accepted.

### Emulator core

Build `pcsx_rearmed` at the pinned commit:

```bash
git clone https://github.com/libretro/pcsx_rearmed.git third_party/pcsx_rearmed
git -C third_party/pcsx_rearmed checkout 0f544ce1d18ac7dd114cb8f5507f01a29d3cd58c
make -C third_party/pcsx_rearmed -f Makefile.libretro platform=unix -j$(nproc)
```

Your binary's hash will usually differ from ours, because compiler
versions differ. That's expected and fine: the start-state check below
is what matters.

## 1. Set up

Install as in the [README](README.md#install), plus `pygame` if you
want to play in a desktop window instead of a browser. Then:

```bash
.venv/bin/python tools/contrib.py check
```

This boots the game from power-on with the fixed button sequence (a few
seconds) and compares the result with the reference:

```
start state  MATCHES the reference
  wrote config/contrib_start.state; play with --state config/contrib_start.state
```

If it says `DIFFERS`, it lists each disc file, the BIOS and the core
with `ok` or `BAD`. Fix whatever is marked `BAD` and run it again.

`config/contrib_start.state` contains game memory. Keep it to yourself,
like the disc.

## 2. Play and record

Always pass the reference start state and a handle (any name you like,
not your real name):

```bash
# in a browser: open http://localhost:8765
.venv/bin/python tools/play.py --state config/contrib_start.state --player <handle>

# or in a window, if you're at the machine's own screen (or RDP/VNC)
.venv/bin/python tools/play.py --ui desktop --state config/contrib_start.state --player <handle>
```

Each run is one session, starting at the first mission briefing.
Recording starts when the page or window opens. **Stop & save** (in the
browser) or **Esc** (in the window) ends it, and it lands in
`recordings/<date-time>/`.

| key | action |
|---|---|
| ↑ / W | forward |
| ↓ / S | reverse |
| ← → / A D | steer |
| Space | brake |
| J | fire |
| K | enter / exit vehicle |
| L | special |
| Q / E | previous / next weapon |
| Enter | start |
| P | pause recording |

Any game controller works too.

**What makes a useful session:**

- **Just play.** Drive around, take jobs from the phones, do missions,
  get chased. Natural play is the point; there is no right way to do it.
- **Longer is better.** Aim for 5 minutes or more. Several short
  sessions are fine too.
- **Pause with P** if you step away. A paused session records nothing,
  so you don't need to stop.
- **Laggy connection?** If the browser shows a high `rtt_ms`, play in
  the desktop window or closer to the machine. Lag is measured and
  recorded, so laggy sessions can be filtered out later rather than
  silently mixed in.

## 3. Pack and submit

```bash
.venv/bin/python tools/contrib.py pack recordings/<session> [more sessions…]
```

Before packing, each session is replayed and checked, so you can't
submit a broken one. It's refused if it wasn't recorded from the
reference start state or doesn't replay identically. Accepted sessions
become `contrib_out/<handle>_<session>.zip`.

**Send the zip files via:** _[to be decided by the maintainers]_

## What you're sending

Each zip holds four files for one session:

| file | contents |
|---|---|
| `inputs.npy` | which buttons were held on each frame (60 per second) |
| `seen.npy` | which frame was on your screen at each moment (measures lag) |
| `ram.npy` | score and player position on each frame |
| `meta.json` | your handle, start and end time, setup hashes, a summary of lag |

No video, no savestate, no game data, no IP address, nothing from your
machine beyond the hashes of your game and emulator files.

**Licence of contributed data:** _[to be decided by the maintainers]_.
By submitting you agree that your sessions may be used to train and
evaluate models in this project.

## Rules

- Use your own copy of the game and BIOS. Don't share them, or
  `contrib_start.state`, with anyone, including us.
- Play by hand. No scripts, macros or turbo buttons: the point is human
  play.
- One handle per person, please, so sessions can be grouped by player.

## For maintainers

```bash
.venv/bin/python tools/contrib.py check                     # once, to build the start state
.venv/bin/python tools/contrib.py unpack submissions/*.zip  # -> recordings/contrib/<handle>_<session>/
.venv/bin/python tools/replay.py recordings/contrib/*       # -> datasets
```

`unpack` replays every session against your own start state and
rejects anything that doesn't reproduce exactly: a different setup,
corrupted files, or edited inputs. Accepted sessions get the start state
copied in locally, so `replay.py` and `play.py --verify` work on them
unchanged.

**Not yet verified across machines.** The start-state hash is
reproducible across processes on the reference machine (gcc 13.3,
x86-64). Whether another CPU or compiler gives the same hash is what
the first outside contributors will tell us. If it doesn't, the
fallback is to compare a deterministic replay of inputs instead of the
whole state.
