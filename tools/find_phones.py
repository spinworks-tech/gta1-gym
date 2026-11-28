#!/usr/bin/env python3
r"""Locate phone booths by teleporting the player across the map.

Phones are the entry point to every mission in GTA, and no trained policy
has ever reached one -- so the reward function cannot shape toward them
until we know where they are. Scanning RAM for their coordinates does not
work: 88,000 constant int32 pairs survive a plausible-range filter.

Visiting the map works instead. The player's position at ``PLAYER_X``/
``PLAYER_Y`` is writable, so this teleports to each grid point, waits for
the phone's message to appear, and records where it does.

Two things make this correct rather than merely plausible:

* **Reset between probes.** A phone message stays on screen once shown,
  so without a reset every later point in the sweep also "hits" and the
  result is one enormous false region. A savestate reload is cheap.
* **Enough settle frames.** The message takes roughly 30 frames to
  appear. At 4 frames a known phone reads as empty -- the first sweep
  that found nothing was wrong for exactly this reason.

    python tools/find_phones.py --x0 5500 --x1 7000 --y0 9000 --y1 10000
    python tools/find_phones.py --verify 6100 9575
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from gta1env.env import GTA1Config, GTA1Env   # noqa: E402

SCALE = 1.52587890625e-05          # 16.16 fixed point
PLAYER_X, PLAYER_Y = 0x2C96C, 0x2C970
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "config", "phones.json")


class Prober:
    def __init__(self, settle=30, seed=0):
        self.env = GTA1Env(GTA1Config(episode_steps=10**9),
                           render_mode="rgb_array")
        self.i32 = self.env.ram.view(np.int32)
        self.settle = settle
        self.seed = seed

    def probe(self, x, y):
        """Teleport to (x, y) from a clean reset; return the message."""
        self.env.reset(seed=self.seed)
        self.i32[PLAYER_X // 4] = int(x / SCALE)
        self.i32[PLAYER_Y // 4] = int(y / SCALE)
        info = None
        for _ in range(self.settle):
            _, _, _, _, info = self.env.step(0)
        return ((info.get("centre_message", "") or "").strip(),
                (info.get("message", "") or "").strip())

    def close(self):
        self.env.close()


def sweep(p, x0, x1, y0, y1, step, quiet=False):
    hits = []
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    for x in xs:
        row = ""
        for y in ys:
            centre, _ = p.probe(float(x), float(y))
            row += "#" if centre else "."
            if centre:
                hits.append((float(x), float(y), centre))
        if not quiet:
            print(f"  x={x:6.0f}  {row}", flush=True)
    return hits, xs, ys


def cluster(hits, gap):
    """Group hit points into regions; one region is one phone."""
    out = []
    for x, y, msg in hits:
        for g in out:
            if (g["msg"] == msg and abs(x - g["x"][-1]) <= gap * 2
                    and min(abs(y - yy) for yy in g["y"]) <= gap * 2):
                g["x"].append(x); g["y"].append(y)
                break
        else:
            out.append({"msg": msg, "x": [x], "y": [y]})
    return [{"message": g["msg"],
             "x": round(float(np.mean(g["x"])), 1),
             "y": round(float(np.mean(g["y"])), 1),
             "x_range": [min(g["x"]), max(g["x"])],
             "y_range": [min(g["y"]), max(g["y"])],
             "points": len(g["x"])} for g in out]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", type=float, default=5500)
    ap.add_argument("--x1", type=float, default=7000)
    ap.add_argument("--y0", type=float, default=9000)
    ap.add_argument("--y1", type=float, default=10000)
    ap.add_argument("--step", type=float, default=75,
                    help="grid spacing; a trigger zone is ~200 units across")
    ap.add_argument("--settle", type=int, default=30,
                    help="frames to wait for the message (below ~30 misses)")
    ap.add_argument("--verify", nargs=2, type=float, metavar=("X", "Y"),
                    help="probe one point and print what it says")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    p = Prober(settle=args.settle)
    try:
        if args.verify:
            centre, brief = p.probe(*args.verify)
            print(f"({args.verify[0]:.0f}, {args.verify[1]:.0f})")
            print(f"  centre: {centre!r}")
            print(f"  brief : {brief!r}")
            return
        nx = len(np.arange(args.x0, args.x1, args.step))
        ny = len(np.arange(args.y0, args.y1, args.step))
        print(f"sweeping {nx}x{ny} = {nx*ny} points, "
              f"step {args.step:.0f}, settle {args.settle}", flush=True)
        hits, xs, ys = sweep(p, args.x0, args.x1, args.y0, args.y1,
                             args.step, args.quiet)
    finally:
        p.close()

    phones = cluster(hits, args.step)
    print(f"\n{len(hits)} hit points -> {len(phones)} pager zone(s)")
    for ph in phones:
        print(f"  ({ph['x']:.0f}, {ph['y']:.0f})  {ph['points']:3d} pts  "
              f"x{ph['x_range']} y{ph['y_range']}")
        print(f"      {ph['message'][:70]!r}")
    if phones:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # A sweep hit is a centre (pager) message, not a job phone -- real
        # phones answer in the brief channel. Keep them apart, and keep
        # whatever else the file records.
        doc = {"note": "world coords; found by teleport sweep"}
        if os.path.exists(args.out):
            with open(args.out) as fh:
                doc = json.load(fh)
        prev = doc.get("pagers", [])
        merged = {(round(q["x"]), round(q["y"])): q for q in prev}
        for ph in phones:
            merged[(round(ph["x"]), round(ph["y"]))] = dict(ph, kind="pager")
        doc["pagers"] = sorted(merged.values(), key=lambda q: (q["x"], q["y"]))
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"\nwrote {args.out} ({len(merged)} total)")


if __name__ == "__main__":
    main()
