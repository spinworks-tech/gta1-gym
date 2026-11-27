#!/usr/bin/env python3
"""Drive the emulator by hand over SSH -- no display server involved.

You need this once per setup: boot the disc, get through the BIOS and the
menus into actual gameplay, then `boot` to save the savestate every
training reset will start from. It also prints an ASCII preview of the
frame, so you can see where you are without copying PNGs around.

  $ .venv/bin/python tools/drive.py
  > f 600              advance 600 frames, no buttons
  > start 4            tap START (4 frames held), then release
  > up,circle 30       hold UP + CIRCLE for 30 frames
  > view               ASCII preview of the current frame
  > png shot.png       write the frame to a PNG
  > boot               save config/boot.state (training resets use this)
  > snap before        keep a named RAM snapshot for tools/ram_scan.py
  > quit
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import DETERMINISTIC_CORE_OPTIONS, GTA1Config  # noqa: E402
from gta1env.libretro import JOYPAD, PSX_BUTTON, LibretroCore  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.path.join(HERE, "scratch", "snapshots")
STATE_DIR = os.path.join(HERE, "scratch", "states")
RAMP = " .:-=+*#%@"

HELP = """commands:
  f N                 advance N frames with nothing held
  <buttons> N         hold buttons for N frames then release (comma separated)
                      names: up down left right cross circle square triangle
                             l1 r1 l2 r2 start select  (or raw: A B X Y L R ...)
  view [W]            ASCII preview of the current frame (default width 78)
  png [path]          save the current frame as a PNG
  info                resolution, frame count, pixel format
  save NAME / load NAME     savestate under scratch/states/
  boot                save the current state as config/boot.state
  snap NAME           save a RAM snapshot under scratch/snapshots/
  peek ADDR [SIZE]    read an integer from RAM (hex ok: 0x800a1234)
  find VALUE [SIZE]   list RAM offsets currently holding VALUE
  reset               hard reset the console
  help / quit"""


def ascii_frame(frame, width=78):
    if frame is None:
        return "(no frame yet)"
    h, w = frame.shape[:2]
    # PS1 pixels are tall; terminal cells are taller still.
    height = max(1, int(width * h / w * 0.45))
    ys = (np.linspace(0, h - 1, height)).astype(int)
    xs = (np.linspace(0, w - 1, width)).astype(int)
    gray = frame[:, :, :3].mean(axis=2)[np.ix_(ys, xs)]
    idx = np.clip((gray / 255.0 * (len(RAMP) - 1)).astype(int),
                  0, len(RAMP) - 1)
    return "\n".join("".join(RAMP[i] for i in row) for row in idx)


def parse_buttons(token):
    out = []
    for raw in token.split(","):
        name = raw.strip()
        if not name:
            continue
        if name.upper() in JOYPAD:
            out.append(name.upper())
        elif name.lower() in PSX_BUTTON:
            out.append(PSX_BUTTON[name.lower()])
        else:
            raise KeyError(name)
    return out


def main():
    cfg = GTA1Config()
    for d in (SNAP_DIR, STATE_DIR, cfg.save_dir):
        os.makedirs(d, exist_ok=True)

    rom = cfg.rom_path
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        rom = sys.argv[1]
    options = dict(DETERMINISTIC_CORE_OPTIONS)
    if "--hle" in sys.argv:
        options["pcsx_rearmed_bios"] = "HLE"
        print("using HLE BIOS (no BIOS image needed)")
    core = LibretroCore(cfg.core_path, cfg.bios_dir, cfg.save_dir,
                        options=options, enable_audio=False, verbose=False)
    av = core.load_game(rom)
    ram = core.memory()
    print(f"loaded {os.path.basename(rom)}: {av}")
    print(f"system RAM: {ram.nbytes/1024/1024:.0f} MiB")
    print(HELP)

    frames = 0
    script = []
    if "--script" in sys.argv:
        with open(sys.argv[sys.argv.index("--script") + 1]) as fh:
            script = [ln.strip() for ln in fh
                      if ln.strip() and not ln.startswith("#")]

    while True:
        if script:
            line = script.pop(0)
            print(f"> {line}")
        else:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("quit", "q", "exit"):
                break
            if cmd == "help":
                print(HELP)
            elif cmd == "f":
                n = int(parts[1]) if len(parts) > 1 else 1
                core.set_buttons(())
                for _ in range(n):
                    core.run()
                frames += n
                print(f"+{n} frames (total {frames})")
            elif cmd == "view":
                width = int(parts[1]) if len(parts) > 1 else 78
                print(ascii_frame(core.frame, width))
            elif cmd == "png":
                path = parts[1] if len(parts) > 1 else os.path.join(
                    HERE, "scratch", f"frame_{frames:07d}.png")
                from PIL import Image
                if core.frame is None:
                    print("no frame yet")
                else:
                    Image.fromarray(core.frame).save(path)
                    print(f"wrote {path} ({core.width}x{core.height})")
            elif cmd == "info":
                print(f"frames={frames} res={core.width}x{core.height} "
                      f"pixfmt={core.pixel_format} fps={av['fps']:.3f}")
            elif cmd == "save":
                path = os.path.join(STATE_DIR, f"{parts[1]}.state")
                with open(path, "wb") as fh:
                    fh.write(core.serialize())
                print(f"saved {path}")
            elif cmd == "load":
                path = os.path.join(STATE_DIR, f"{parts[1]}.state")
                with open(path, "rb") as fh:
                    core.unserialize(fh.read())
                print(f"loaded {path}")
            elif cmd == "boot":
                path = cfg.boot_state
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(core.serialize())
                print(f"saved boot state -> {path}")
                print("every env reset will now start from exactly here")
            elif cmd == "snap":
                path = os.path.join(SNAP_DIR, f"{parts[1]}.npy")
                np.save(path, ram.copy())
                print(f"saved RAM snapshot {path}")
            elif cmd == "peek":
                addr = int(parts[1], 0) & 0x1FFFFF
                size = int(parts[2]) if len(parts) > 2 else 4
                raw = ram[addr:addr + size].tobytes()
                print(f"0x{addr:06x} = {int.from_bytes(raw, 'little')} "
                      f"(hex {raw[::-1].hex()})")
            elif cmd == "find":
                value = int(parts[1], 0)
                size = int(parts[2]) if len(parts) > 2 else 4
                dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[size]
                view = ram[:ram.size // size * size].view(dtype)
                hits = np.flatnonzero(view == value)
                print(f"{hits.size} matches" + (
                    ": " + ", ".join(f"0x{h*size:06x}" for h in hits[:20])
                    if hits.size else ""))
            elif cmd == "reset":
                core.reset()
                frames = 0
                print("console reset")
            else:
                buttons = parse_buttons(cmd)
                n = int(parts[1]) if len(parts) > 1 else 1
                core.set_buttons(buttons)
                for _ in range(n):
                    core.run()
                core.set_buttons(())
                frames += n
                print(f"held {'+'.join(buttons)} for {n} frames "
                      f"(total {frames})")
        except KeyError as exc:
            print(f"unknown button {exc}; see 'help'")
        except (IndexError, ValueError) as exc:
            print(f"bad arguments: {exc}")

    core.close()
    print("closed")


if __name__ == "__main__":
    main()
