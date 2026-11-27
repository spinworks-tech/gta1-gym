#!/usr/bin/env python3
"""Check a contributor's setup, and pack / unpack contributed play sessions.

Everyone starts from the same start state, which each person builds from
their own disc and BIOS by booting from power-on with a fixed button
script (config/contrib.json). If two setups emulate identically, they
produce a byte-identical state, so its hash is an end-to-end check of
disc + BIOS + core build. A submission then needs only the inputs, never
a savestate: no game data leaves anyone's machine.

  contributor:
    $ .venv/bin/python tools/contrib.py check            # builds config/contrib_start.state
    $ .venv/bin/python tools/play.py --state config/contrib_start.state --player <handle>
    $ .venv/bin/python tools/contrib.py pack recordings/<session> [...]

  maintainer:
    $ .venv/bin/python tools/contrib.py unpack submissions/*.zip
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from play import open_core, sha256_file  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(HERE, "config", "contrib.json")
START = os.path.join(HERE, "config", "contrib_start.state")
OUT_DIR = os.path.join(HERE, "contrib_out")
IN_DIR = os.path.join(HERE, "recordings", "contrib")
PACKED = ("inputs.npy", "seen.npy", "ram.npy", "meta.json")


def spec():
    with open(SPEC_PATH) as fh:
        return json.load(fh)


def sha(b):
    return hashlib.sha256(b).hexdigest()


_CORE = None


def core_once(bios_mode):
    """One core per process: libretro cores keep global state, so they are
    never closed and reopened here, only reloaded from a savestate."""
    global _CORE
    if _CORE is None:
        _CORE = open_core(bios_mode)
    return _CORE


def build_start_state(sp):
    core, _, cfg, rom, _ = core_once(sp["bios_mode"])
    for button, n in sp["boot_script"]:
        core.set_buttons(() if button == "none" else (button,))
        for _ in range(n):
            core.run(convert=False)
    core.set_buttons(())
    return core.serialize(), cfg, rom


def replay_hash(start_blob, inputs, bios_mode):
    core = core_once(bios_mode)[0]
    core.unserialize(start_blob)
    for m in inputs:
        core.set_buttons(int(m))
        core.run(convert=False)
    return sha(core.memory().tobytes())


# ------------------------------------------------------------------- check


def check(_args):
    sp = spec()
    blob, cfg, rom = build_start_state(sp)
    ok = sha(blob) == sp["start_state_sha256"]
    print(f"start state  {'MATCHES the reference' if ok else 'DIFFERS from the reference'}")
    if ok:
        with open(START, "wb") as fh:
            fh.write(blob)
        print(f"  wrote {os.path.relpath(START, HERE)}; play with "
              f"--state {os.path.relpath(START, HERE)}")
        return 0

    # Diagnose which ingredient differs.
    print("\nlooking for the difference:")
    disc_dir = os.path.dirname(rom)
    for name, want in sp["disc"]["files"].items():
        path = os.path.join(disc_dir, name)
        got = sha256_file(path) if os.path.exists(path) else None
        print(f"  {'ok ' if got == want else 'BAD'} disc  {name}"
              + ("" if got else "  (missing)"))
    bios = [f for f in os.listdir(cfg.bios_dir)
            if f.lower().endswith((".bin", ".rom"))]
    for f in bios:
        got = sha256_file(os.path.join(cfg.bios_dir, f))
        print(f"  {'ok ' if got == sp['bios']['sha256'] else 'BAD'} bios  {f}")
    if len(bios) > 1:
        print("  note: more than one BIOS file; keep only "
              f"{sp['bios']['name']} in {cfg.bios_dir}")
    core = sha256_file(cfg.core_path)
    same = core == sp["core"]["reference_sha256"]
    print(f"  {'ok ' if same else '-- '} core  "
          + ("identical build" if same else
             f"different build (expected for another compiler); "
             f"use commit {sp['core']['commit'][:12]}"))
    print("\nThe start state must match before sessions can be contributed.")
    return 1


# -------------------------------------------------------------------- pack


def pack(args):
    sp = spec()
    if not os.path.exists(START):
        sys.exit("run `tools/contrib.py check` first")
    with open(START, "rb") as fh:
        start = fh.read()
    if sha(start) != sp["start_state_sha256"]:
        sys.exit(f"{START} is not the reference start state; rerun check")
    os.makedirs(OUT_DIR, exist_ok=True)
    for rec in args.recordings:
        with open(os.path.join(rec, "meta.json")) as fh:
            meta = json.load(fh)
        if meta["start_state_sha256"] != sp["start_state_sha256"]:
            print(f"SKIP {rec}: not recorded from the reference start state "
                  f"(play with --state {os.path.relpath(START, HERE)})")
            continue
        inputs = np.load(os.path.join(rec, "inputs.npy"))
        if replay_hash(start, inputs, sp["bios_mode"]) != meta["final_ram_sha256"]:
            print(f"SKIP {rec}: does not replay identically")
            continue
        name = f"{meta['player']}_{meta['session_id']}.zip"
        out = os.path.join(OUT_DIR, name)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in PACKED:
                if os.path.exists(os.path.join(rec, f)):
                    z.write(os.path.join(rec, f), f"{meta['session_id']}/{f}")
        print(f"packed {rec} ({len(inputs)} frames, "
              f"{len(inputs) / meta['fps'] / 60:.1f} min) -> "
              f"{os.path.relpath(out, HERE)} ({os.path.getsize(out) // 1024} KiB)")
    return 0


# ------------------------------------------------------------------ unpack


def unpack(args):
    sp = spec()
    if not os.path.exists(START):
        sys.exit("run `tools/contrib.py check` first")
    with open(START, "rb") as fh:
        start = fh.read()
    bad = 0
    for path in args.zips:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            sessions = {n.split("/")[0] for n in names}
            for sid in sessions:
                if not all(f"{sid}/{f}" in names for f in ("inputs.npy", "meta.json")):
                    print(f"REJECT {path}:{sid}: missing files")
                    bad += 1
                    continue
                meta = json.loads(z.read(f"{sid}/meta.json"))
                if meta.get("start_state_sha256") != sp["start_state_sha256"]:
                    print(f"REJECT {path}:{sid}: different start state")
                    bad += 1
                    continue
                inputs = np.load(z.open(f"{sid}/inputs.npy"))
                if replay_hash(start, inputs, sp["bios_mode"]) != meta["final_ram_sha256"]:
                    print(f"REJECT {path}:{sid}: does not replay identically here")
                    bad += 1
                    continue
                dest = os.path.join(IN_DIR, f"{meta['player']}_{sid}")
                os.makedirs(dest, exist_ok=True)
                for f in PACKED:
                    if f"{sid}/{f}" in names:
                        with open(os.path.join(dest, f), "wb") as fh:
                            fh.write(z.read(f"{sid}/{f}"))
                with open(os.path.join(dest, "start.state"), "wb") as fh:
                    fh.write(start)
                print(f"accepted {path}:{sid} ({len(inputs)} frames) -> "
                      f"{os.path.relpath(dest, HERE)}")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verify the setup and build the start state")
    pk = sub.add_parser("pack", help="package recordings for submission")
    pk.add_argument("recordings", nargs="+")
    up = sub.add_parser("unpack", help="verify and import submissions")
    up.add_argument("zips", nargs="+")
    args = p.parse_args()
    sys.exit({"check": check, "pack": pack, "unpack": unpack}[args.cmd](args))


if __name__ == "__main__":
    main()
