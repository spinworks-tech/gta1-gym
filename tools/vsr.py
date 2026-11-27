#!/usr/bin/env python3
"""Read GTA (PS1) .VSR archives: a tree of named chunks.

Layout, worked out from MISSION.VSR / NYC.VSR:

  0x00  "3RSV"  u32 data_size  u32 dir_size  ...  u32 n_entries @0x18
  0x24  "SFFO" u32 0  "CRID" u32 dir_bytes u32 ...
  0x38  n_entries x 40-byte entries:
          u32 offset   u32 size   u16 name_id  u16 index  u16 type
          u16 0xcccc   u32 -1     ...          u32 n_children @+0x18
        type 0x16 = folder (its children follow, depth-first), 0x0e = file
  then  "NRSV" u32 ... "TRTS" u32 bytes u32 n_names, n_names u32 offsets
        (relative to the count field), NUL-terminated names

    python tools/vsr.py scratch/disc/MISSION.VSR            # list
    python tools/vsr.py scratch/disc/MISSION.VSR --out DIR  # extract
"""

import argparse
import os
import struct

ENTRY = 0x28


def read_vsr(path):
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:4] != b"3RSV":
        raise ValueError(f"{path}: not a VSR archive")
    n = struct.unpack_from("<I", b, 0x18)[0]
    first = 0x38
    end = first + n * ENTRY
    # name table
    t = b.index(b"TRTS", end)
    count = struct.unpack_from("<I", b, t + 8)[0]
    offs = struct.unpack_from(f"<{count}I", b, t + 12)
    base = t + 8                       # offsets count from the count field
    names = [b[base + o:b.index(b"\0", base + o)].decode("latin-1")
             for o in offs]
    # data offsets are relative to the end of the header block
    data_base = len(b) - struct.unpack_from("<I", b, 4)[0]
    raw = []
    for i in range(n):
        p = first + i * ENTRY
        off, size, name_id, idx, kind = struct.unpack_from("<IIHHH", b, p)
        kids = struct.unpack_from("<I", b, p + 0x18)[0]
        raw.append({"off": off, "size": size, "kind": kind, "kids": kids,
                    "name": names[name_id] if name_id < len(names)
                    else f"#{name_id}", "index": idx})
    # rebuild the tree (depth-first, folders announce their child count)
    out, pos = [], 0

    def walk(prefix):
        nonlocal pos
        e = raw[pos]
        pos += 1
        path = "" if pos == 1 else f"{prefix}/{e['name']}"
        if e["kind"] == 0x16:
            for _ in range(e["kids"]):
                walk(path)
        else:
            data = b[data_base + e["off"]:data_base + e["off"] + e["size"]]
            out.append((path, e, data))
    walk("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vsr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for path, e, data in read_vsr(args.vsr):
        print(f"{e['size']:>9d}  {path}   head={data[:16].hex()}")
        if args.out:
            dst = os.path.join(args.out, path.lstrip("/"))
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(data)


if __name__ == "__main__":
    main()
