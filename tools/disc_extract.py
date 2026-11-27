#!/usr/bin/env python3
"""List / extract the ISO9660 file system of the GTA (PS1) data track.

The data track is MODE2/2352: every raw 2352-byte sector carries 2048
bytes of file data at offset 24 (Form 1). No external tools needed.

    python tools/disc_extract.py --list
    python tools/disc_extract.py --out scratch/disc        # everything

Output stays under scratch/ (gitignored): these are the game's own,
copyrighted files, extracted from your disc for local analysis only.
"""

import argparse
import glob
import os
import struct

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, DATA, OFF = 2352, 2048, 24


class Disc:
    def __init__(self, path):
        self.fh = open(path, "rb")

    def sector(self, lba):
        self.fh.seek(lba * RAW + OFF)
        return self.fh.read(DATA)

    def read(self, lba, size):
        out = bytearray()
        n = (size + DATA - 1) // DATA
        for i in range(n):
            out += self.sector(lba + i)
        return bytes(out[:size])

    def walk(self, lba=None, size=None, prefix=""):
        if lba is None:
            pvd = self.sector(16)
            assert pvd[1:6] == b"CD001", "no ISO9660 volume descriptor"
            root = pvd[156:190]
            lba, size = struct.unpack_from("<I", root, 2)[0], \
                struct.unpack_from("<I", root, 10)[0]
        data = self.read(lba, size)
        pos = 0
        while pos < len(data):
            ln = data[pos]
            if ln == 0:                       # pad to next sector
                pos = (pos // DATA + 1) * DATA
                continue
            rec = data[pos:pos + ln]
            ext, sz = struct.unpack_from("<I", rec, 2)[0], \
                struct.unpack_from("<I", rec, 10)[0]
            flags, nl = rec[25], rec[32]
            name = rec[33:33 + nl]
            pos += ln
            if name in (b"\x00", b"\x01"):
                continue
            name = name.decode("ascii", "replace").split(";")[0]
            path = f"{prefix}/{name}"
            if flags & 2:
                yield from self.walk(ext, sz, path)
            else:
                yield path, ext, sz


def find_track1():
    hits = sorted(glob.glob(os.path.join(HERE, "roms", "*Track 01*.bin")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(HERE, "roms", "*.bin")))
    if not hits:
        raise SystemExit("no .bin data track in roms/")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=None, help="data track (default: roms/)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "scratch", "disc"))
    args = ap.parse_args()
    disc = Disc(args.bin or find_track1())
    files = list(disc.walk())
    if args.list:
        for path, lba, size in files:
            print(f"{size:>10d}  lba {lba:>6d}  {path}")
        print(f"{len(files)} files")
        return
    for path, lba, size in files:
        dst = os.path.join(args.out, path.lstrip("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(disc.read(lba, size))
    print(f"extracted {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
