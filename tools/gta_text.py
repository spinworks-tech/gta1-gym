#!/usr/bin/env python3
"""Decode every text string in GTA (PS1): all levels, all languages.

The text lives in MISSION.VSR (see tools/vsr.py):

  HUFF_TABLES/<LANG>       Huffman tree for that language
  NON_INT/<LANG>           in-game messages shared by all levels
  FRONTEND/<LANG>          menus
  MISSION<n>/<LANG>        one level's briefings, pager messages, names
  MISSION<n>/MISSION       the level's compiled mission script (binary)

Text table:  u32 index_bytes, then (u32 id, u32 offset) pairs, u32
data_bytes, then the bitstreams. Message ids match PC GTA1's numbering.

Huffman scheme, read off the game's own decoder (SLUS_001.06,
0x8010a5b0, called per character by get_text at 0x800c9d60):

  table = u16 words; word 0 is the root index, the tree follows.
  node word w:  children at node + (w >> 2)  (bit 0)  and  +1  (bit 1)
                w & 2 -> the 0-child is a leaf, w & 1 -> the 1-child is
                a leaf; a leaf's low byte is the character.
  bits are read MSB-first; character 0 ends the string.

    python tools/gta_text.py                       # all -> scratch/text/
    python tools/gta_text.py --lang ENGLISH --print

Output stays in scratch/ (gitignored): it is the game's own text.
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vsr import read_vsr  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ("ENGLISH", "FRENCH", "GERMAN", "ITALIAN", "SPECIAL", "JAPANESE")


class Huffman:
    def __init__(self, blob):
        words = struct.unpack_from(f"<{(len(blob) - 4) // 2}H", blob, 4)
        self.root, self.tbl = words[0], words[1:]

    def decode(self, data, pos):
        out, byte, mask = bytearray(), 0, 0
        tbl = self.tbl
        while True:
            node = self.root
            while True:
                if mask == 0:
                    byte, pos, mask = data[pos], pos + 1, 0x80
                w = tbl[node]
                node += w >> 2
                flag = 2
                if byte & mask:
                    node += 1
                    flag = 1
                mask >>= 1
                if w & flag:
                    c = tbl[node] & 0xFF
                    break
            if c == 0:
                return bytes(out)
            out.append(c)


def decode_table(blob, huff):
    n = struct.unpack_from("<I", blob, 0)[0]
    pairs = [struct.unpack_from("<II", blob, 4 + 8 * i)
             for i in range(n // 8)]
    base = 4 + n + 4
    return {int(i): huff.decode(blob, base + off).decode("latin-1")
            for i, off in pairs}


def decode_all(vsr_path):
    chunks = {path.strip("/"): data for path, _, data in read_vsr(vsr_path)}
    out = {}
    for lang in LANGS:
        hkey = f"HUFF_TABLES/{lang}"
        if hkey not in chunks:
            continue
        huff = Huffman(chunks[hkey])
        out[lang] = {}
        for key, data in chunks.items():
            group, _, name = key.rpartition("/")
            if name == lang and group != "HUFF_TABLES":
                out[lang][group] = decode_table(data, huff)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsr", default=os.path.join(HERE, "scratch", "disc",
                                                  "MISSION.VSR"))
    ap.add_argument("--out", default=os.path.join(HERE, "scratch", "text"))
    ap.add_argument("--lang", default=None)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    text = decode_all(args.vsr)
    os.makedirs(args.out, exist_ok=True)
    for lang, groups in text.items():
        if args.lang and lang != args.lang:
            continue
        with open(os.path.join(args.out, f"{lang.lower()}.json"), "w") as fh:
            json.dump(groups, fh, indent=1, ensure_ascii=False)
        total = sum(len(g) for g in groups.values())
        print(f"{lang}: {total} strings in "
              + ", ".join(f"{g} ({len(v)})" for g, v in groups.items()))
        if args.print:
            for g, msgs in groups.items():
                for i, s in sorted(msgs.items()):
                    print(f"[{g} {i}] {s}")


if __name__ == "__main__":
    main()
