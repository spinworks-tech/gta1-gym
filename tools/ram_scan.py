#!/usr/bin/env python3
"""Find where GTA keeps cash, health and the rest in the PS1's 2 MiB RAM.

There is no trustworthy published RAM map for GTA (PS1), so you locate
the values the same way a cheat-engine user would: snapshot RAM in
states where you know the value (or know only which way it moved), then
intersect the candidates. Take snapshots with `snap NAME` in
tools/drive.py.

Typical session -- you can see $100 on the HUD, pick up cash, now $250:

  > snap cash100            (in drive.py)
  > ... collect money ...
  > snap cash250
  $ tools/ram_scan.py known --at cash100=100 --at cash250=250
  $ tools/ram_scan.py known --at cash100=100 --at cash250=250 --size 2
  $ tools/ram_scan.py set cash 0x800a1234 --size 4 --note "HUD cash"

With only a direction, not a number (health dropping, say):

  $ tools/ram_scan.py delta --snaps full,hurt,hurt2 --pattern dec,dec --size 1
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.ram import CONFIG_PATH, RamMap  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_DIR = os.path.join(HERE, "scratch", "snapshots")

DTYPES = {
    (1, False): np.uint8, (1, True): np.int8,
    (2, False): np.uint16, (2, True): np.int16,
    (4, False): np.uint32, (4, True): np.int32,
}


def load_snap(name):
    path = name if os.path.exists(name) else os.path.join(
        SNAP_DIR, f"{name}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"no such snapshot: {name} (looked in {SNAP_DIR})")
    arr = np.load(path)
    if arr.dtype != np.uint8 or arr.size < 1 << 20:
        raise SystemExit(f"{path} does not look like a RAM snapshot")
    return arr


def views(ram, size, signed, unaligned):
    """Yield (byte_offsets, values) for every alignment we care about."""
    dtype = DTYPES[(size, signed)]
    shifts = range(size) if unaligned else (0,)
    for shift in shifts:
        body = ram[shift:]
        count = body.size // size
        vals = body[:count * size].view(dtype)
        offs = np.arange(count, dtype=np.int64) * size + shift
        yield offs, vals


def cmd_known(args):
    pairs = []
    for item in args.at:
        if "=" not in item:
            raise SystemExit(f"--at wants SNAP=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        pairs.append((load_snap(name), int(value, 0), name))

    dtype = DTYPES[(args.size, args.signed)]
    found = []
    for shift in (range(args.size) if args.unaligned else (0,)):
        seqs = []
        for ram, _, _ in pairs:
            body = ram[shift:]
            count = body.size // args.size
            seqs.append(body[:count * args.size].view(dtype))
        n = min(s.size for s in seqs)
        mask = np.ones(n, dtype=bool)
        for seq, (_, value, _) in zip(seqs, pairs):
            mask &= (seq[:n] == value)
            if not mask.any():
                break
        found.extend((int(i) * args.size + shift)
                     for i in np.flatnonzero(mask))

    found.sort()
    print(f"{len(found)} candidate address(es) for size={args.size} "
          f"signed={args.signed}")
    for off in found[:args.limit]:
        line = f"  0x{0x80000000 + off:08X}  (offset 0x{off:06X})"
        for ram, _, name in pairs:
            val = int(ram[off:off + args.size].view(dtype)[0])
            line += f"  {name}={val}"
        print(line)
    if len(found) > args.limit:
        print(f"  ... {len(found) - args.limit} more; narrow it with another "
              f"snapshot at a different value")
    if not found:
        print("  Nothing matched. The value may be scaled (cash shown in $ "
              "but stored in units), packed, or a different width -- retry "
              "with --size 1/2/4, --signed, or --unaligned.")


def _write_candidates(path, offsets):
    with open(path, "w") as fh:
        for off in offsets:
            fh.write("%d\n" % off)


def _read_candidates(path):
    with open(path) as fh:
        return {int(line) for line in fh if line.strip()}


def cmd_delta(args):
    names = args.snaps.split(",")
    if len(names) < 2:
        raise SystemExit("--snaps needs at least two snapshot names")
    rams = [load_snap(n) for n in names]
    pattern = args.pattern.split(",")
    if len(pattern) != len(rams) - 1:
        raise SystemExit(f"--pattern needs {len(rams)-1} entries for "
                         f"{len(rams)} snapshots")

    dtype = DTYPES[(args.size, args.signed)]
    found = []
    for shift in (range(args.size) if args.unaligned else (0,)):
        seqs = []
        for ram in rams:
            body = ram[shift:]
            count = body.size // args.size
            seqs.append(body[:count * args.size].view(dtype).astype(np.int64))
        mask = np.ones(seqs[0].size, dtype=bool)
        for i, rule in enumerate(pattern):
            a, b = seqs[i], seqs[i + 1]
            if rule == "inc":
                mask &= b > a
            elif rule == "dec":
                mask &= b < a
            elif rule == "same":
                mask &= b == a
            elif rule == "ge":          # non-decreasing: tolerates pauses
                mask &= b >= a
            elif rule == "le":          # non-increasing
                mask &= b <= a
            elif rule == "any":
                pass
            else:
                raise SystemExit(f"unknown rule {rule!r}; "
                                 "use inc/dec/ge/le/same/any")
            if not mask.any():
                break
        if args.changed:
            # Non-strict rules admit values that never move at all.
            mask &= seqs[-1] != seqs[0]
        idx = np.flatnonzero(mask)
        found.extend((int(i) * args.size + shift) for i in idx)

    found.sort()
    if args.only:
        keep = _read_candidates(args.only)
        before = len(found)
        found = [o for o in found if o in keep]
        print(f"intersected with {args.only}: {before} -> {len(found)}")
    if args.save:
        _write_candidates(args.save, found)
        print(f"saved {len(found)} candidate(s) to {args.save}")
    print(f"{len(found)} address(es) matching {args.pattern} "
          f"(size={args.size} signed={args.signed})")
    for off in found[:args.limit]:
        vals = [int(r[off:off + args.size].view(dtype)[0]) for r in rams]
        print(f"  0x{0x80000000 + off:08X}  (offset 0x{off:06X})  "
              + " -> ".join(str(v) for v in vals))
    if len(found) > args.limit:
        print(f"  ... {len(found) - args.limit} more; add snapshots and a "
              f"longer pattern (including 'same' steps) to narrow it")


def cmd_set(args):
    ram_map = RamMap.load(args.config)
    ram_map.set(args.name, int(args.address, 0), size=args.size,
                signed=args.signed, scale=args.scale, note=args.note)
    path = ram_map.save(args.config)
    print(f"{args.name} -> 0x{int(args.address, 0):08X} "
          f"(size {args.size}) written to {path}")
    if args.name in ("cash", "score", "health"):
        print("The reward function picks this up automatically on the next "
              "env construction.")


def cmd_list(args):
    ram_map = RamMap.load(args.config)
    if ram_map.is_empty:
        print(f"{args.config}: no fields yet -- reward will be 0 until you "
              f"add at least one of cash/score/health")
        print(f"recognised names: {', '.join(RamMap.KNOWN_KEYS)}")
        return
    print(json.dumps({k: f.to_json() for k, f in ram_map.fields.items()},
                     indent=2, sort_keys=True))


def cmd_dump(args):
    off = int(args.address, 0) & 0x1FFFFF
    for name in args.snaps.split(","):
        ram = load_snap(name)
        raw = ram[off:off + 8].tobytes()
        u8, u16, u32 = raw[0], int.from_bytes(raw[:2], "little"), \
            int.from_bytes(raw[:4], "little")
        print(f"  {name:>16}  u8={u8:<5} u16={u16:<7} u32={u32:<12} "
              f"bytes={raw.hex(' ')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(size=dict(type=int, default=4, choices=(1, 2, 4)),
                  limit=dict(type=int, default=40))

    p = sub.add_parser("known", help="addresses holding known values")
    p.add_argument("--at", action="append", required=True,
                   metavar="SNAP=VALUE")
    p.add_argument("--size", **common["size"])
    p.add_argument("--signed", action="store_true")
    p.add_argument("--unaligned", action="store_true")
    p.add_argument("--limit", **common["limit"])
    p.set_defaults(func=cmd_known)

    p = sub.add_parser("delta", help="addresses moving a given direction")
    p.add_argument("--snaps", required=True, help="comma separated, in order")
    p.add_argument("--pattern", required=True,
                   help="comma separated inc/dec/ge/le/same/any, one per gap")
    p.add_argument("--size", **common["size"])
    p.add_argument("--signed", action="store_true")
    p.add_argument("--unaligned", action="store_true")
    p.add_argument("--limit", **common["limit"])
    p.add_argument("--changed", action="store_true",
                   help="require the value to actually differ end to end")
    p.add_argument("--save", help="write surviving offsets to a file")
    p.add_argument("--only", help="keep only offsets listed in this file")
    p.set_defaults(func=cmd_delta)

    p = sub.add_parser("set", help="record an address in config/ram_map.json")
    p.add_argument("name", help=f"one of: {', '.join(RamMap.KNOWN_KEYS)}")
    p.add_argument("address")
    p.add_argument("--size", **common["size"])
    p.add_argument("--signed", action="store_true")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("list", help="show the current RAM map")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("dump", help="show one address across snapshots")
    p.add_argument("address")
    p.add_argument("--snaps", required=True)
    p.set_defaults(func=cmd_dump)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
