#!/usr/bin/env python3
"""Work out what each button does, by pressing it and looking.

Restores the boot savestate, holds one button, and compares the result
against doing nothing: how much of the screen changed, and how much of
RAM. Buttons that move the player or fire a weapon stand out clearly.
A PNG per button is written for eyeballing.

    .venv/bin/python tools/probe_controls.py --frames 60
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import DETERMINISTIC_CORE_OPTIONS, GTA1Config, find_disc  # noqa: E402
from gta1env.libretro import LibretroCore  # noqa: E402

BUTTONS = ["up", "down", "left", "right", "cross", "circle", "square",
           "triangle", "l1", "r1", "l2", "r2", "select"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60,
                    help="how long to hold each button")
    ap.add_argument("--state", default=None, help="savestate to start from")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = GTA1Config()
    state_path = args.state or cfg.boot_state
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scratch", "controls")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(state_path):
        raise SystemExit(f"no savestate at {state_path}; make one with "
                         f"tools/drive.py first")

    core = LibretroCore(cfg.core_path, cfg.bios_dir, cfg.save_dir,
                        options=DETERMINISTIC_CORE_OPTIONS)
    core.load_game(cfg.rom_path or find_disc())
    with open(state_path, "rb") as fh:
        blob = fh.read()
    ram = core.memory()

    def run_with(buttons):
        core.unserialize(blob)
        core.set_buttons(buttons)
        for _ in range(args.frames):
            core.run()
        core.set_buttons(())
        return core.frame.copy(), ram.copy()

    base_frame, base_ram = run_with(())
    Image.fromarray(base_frame).save(os.path.join(outdir, "_idle.png"))

    print(f"holding each button for {args.frames} frames from "
          f"{os.path.basename(state_path)}")
    print(f"{'button':<10} {'pixels changed':>14} {'RAM bytes changed':>18}")
    print("-" * 46)
    rows = []
    for name in BUTTONS:
        frame, snap = run_with([name])
        if frame.shape != base_frame.shape:
            pixels = float("nan")
        else:
            pixels = float((frame != base_frame).any(axis=2).mean() * 100)
        rambytes = int((snap != base_ram).sum())
        Image.fromarray(frame).save(os.path.join(outdir, f"{name}.png"))
        rows.append((name, pixels, rambytes))
        print(f"{name:<10} {pixels:13.2f}% {rambytes:18d}")

    print()
    movers = [r for r in rows if r[1] > 1.0]
    if movers:
        print("clearly does something: "
              + ", ".join(f"{n} ({p:.0f}%)" for n, p, _ in
                          sorted(movers, key=lambda r: -r[1])))
    quiet = [r[0] for r in rows if r[1] <= 1.0]
    if quiet:
        print("little or no visible change: " + ", ".join(quiet))
    print(f"\nPNGs in {outdir}/ — compare each against _idle.png")


if __name__ == "__main__":
    main()
