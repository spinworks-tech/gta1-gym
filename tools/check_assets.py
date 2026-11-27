#!/usr/bin/env python3
"""Verify the two files you must supply yourself: a PS1 BIOS and the disc.

Neither can be downloaded here -- they are copyrighted. Dump them from
hardware/media you own, drop them in bios/ and roms/, then run this.
"""

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOS_DIR = os.path.join(HERE, "bios")
ROM_DIR = os.path.join(HERE, "roms")

# A PS1 BIOS identifies itself near the end of the ROM, e.g.
#   "System ROM Version 2.2 12/04/95 A"
# with a trailing region letter. That beats a checksum table: it needs no
# maintenance and tells you the region, which must match your disc.
ROM_ID_OFFSET = 0x7FF32
REGIONS = {"A": "America (NTSC-U)", "E": "Europe (PAL)",
           "J": "Japan (NTSC-J)", "I": "Japan (NTSC-J, early)"}
BIOS_SIZE = 524288

DISC_EXT = (".cue", ".chd", ".bin", ".img", ".pbp", ".m3u", ".iso")


def md5(path, limit=None):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        read = 0
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


def identify_bios(path):
    """Read the BIOS's own version string. Returns (version, date, region)."""
    with open(path, "rb") as fh:
        fh.seek(ROM_ID_OFFSET)
        blob = fh.read(64).split(b"\x00")[0]
    text = blob.decode("ascii", errors="replace").strip()
    m = re.match(r"System ROM Version\s+(\S+)\s+(\S+)\s*(\S*)", text)
    if not m:
        return None
    version, date, region = m.group(1), m.group(2), m.group(3)
    return version, date, region


def check_bios():
    print("== PS1 BIOS ==")
    print("  (optional: the core can use HLE instead -- see the README)")
    if not os.path.isdir(BIOS_DIR):
        print(f"  none: {BIOS_DIR} does not exist")
        return True
    files = [f for f in sorted(os.listdir(BIOS_DIR))
             if f.lower().endswith((".bin", ".rom"))]
    if not files:
        print(f"  none in {BIOS_DIR} -- run with HLE, or add a dump")
        return True
    for name in files:
        path = os.path.join(BIOS_DIR, name)
        size = os.path.getsize(path)
        ident = identify_bios(path)
        if ident is None:
            print(f"  ??  {name}  {size/1024:.0f} KiB  "
                  f"no version string -- not a PS1 BIOS?")
            continue
        version, date, region = ident
        region_name = REGIONS.get(region, f"unknown region {region!r}")
        ok = size == BIOS_SIZE
        print(f"  {'OK  ' if ok else '??  '}{name}  {size/1024:.0f} KiB  "
              f"v{version} {date}  {region_name}")
        print(f"        md5={md5(path)}")
        if not ok:
            print(f"        warning: a PS1 BIOS is exactly {BIOS_SIZE} bytes")
        if region == "A":
            print("        pair with an NTSC-U disc (SLUS-*)")
        elif region == "E":
            print("        pair with a PAL disc (SLES-*); 50 Hz is ~20% "
                  "slower to emulate than NTSC")
    return True


def check_disc():
    print("== GTA (PlayStation) disc image ==")
    if not os.path.isdir(ROM_DIR):
        print(f"  MISSING: {ROM_DIR} does not exist")
        return False
    files = [f for f in sorted(os.listdir(ROM_DIR))
             if f.lower().endswith(DISC_EXT)]
    if not files:
        print(f"  MISSING: no disc image in {ROM_DIR}")
        print("  Accepted: .cue (+.bin), .chd, .img, .pbp, .m3u, .iso")
        print("  Note: a .cue must sit next to the .bin it references.")
        return False
    for name in files:
        path = os.path.join(ROM_DIR, name)
        size = os.path.getsize(path)
        print(f"  found {name}  {size/1e6:.1f} MB")
        if name.lower().endswith(".cue"):
            with open(path, errors="replace") as fh:
                body = fh.read()
            for line in body.splitlines():
                line = line.strip()
                if line.upper().startswith("FILE"):
                    ref = line.split('"')[1] if '"' in line else line.split()[1]
                    target = os.path.join(ROM_DIR, ref)
                    mark = "OK" if os.path.exists(target) else "BROKEN REF"
                    print(f"       -> {ref}  [{mark}]")
    return True


def check_core():
    print("== libretro core ==")
    core = os.path.join(HERE, "third_party", "pcsx_rearmed",
                        "pcsx_rearmed_libretro.so")
    if os.path.exists(core):
        print(f"  OK  {core}  ({os.path.getsize(core)/1e6:.1f} MB)")
        return True
    print(f"  MISSING: {core} -- run make -f Makefile.libretro platform=unix")
    return False


if __name__ == "__main__":
    results = [check_core(), check_bios(), check_disc()]
    print()
    if all(results):
        print("All assets present. Next: tools/drive.py to reach a playable "
              "state and save a boot savestate.")
        sys.exit(0)
    print("Missing assets above must be supplied before the env can run.")
    sys.exit(1)
