"""On-screen messages read from RAM and mapped to the game's message ids.

The game keeps decoded *text*, not ids, in its display buffers (see
RESOURCES.md). Every string is known ahead of time, decoded from the disc
by tools/gta_text.py, so the text on screen maps back to its id exactly.
`sprintf` templates ("%d Lives Remaining!") are matched as patterns.

Two display channels, both at fixed addresses (static data of the
executable, so stable across savestates):

  brief   scrolling message box, filled by 0x800ba7b8
          text 0x800444DC, on-screen flag 0x80019FF8 (u8),
          kind 0x8001A00C (u32), scroll countdown 0x8001A004 (u32)
  centre  second channel, filled by 0x800bb288 / 0x800bb540
          text 0x800448C8                          -- not yet seen live

The banner at 0x800443CC shows the district name when entering one, and
the vehicle name when getting into a car ("Southwest Park", "Penetrator").
"""

import json
import os
import re

RAM_BASE = 0x80000000
BRIEF_TEXT = 0x800444DC
BRIEF_ON = 0x80019FF8
BRIEF_KIND = 0x8001A00C
BRIEF_TIMER = 0x8001A004
CENTRE_TEXT = 0x800448C8
BANNER_TEXT = 0x800443CC

MISSION_COMPLETE = 2500
MISSION_FAILED = 2501

# Prefer per-level text, then shared in-game text, then menus, when the
# same string has several ids.
GROUP_ORDER = ("MISSION", "NON_INT", "FRONTEND")


def _cstr(ram, addr, n=256):
    off = addr - RAM_BASE
    raw = bytes(ram[off:off + n])
    return raw.split(b"\0", 1)[0].decode("latin-1")


def _u32(ram, addr):
    off = addr - RAM_BASE
    return int.from_bytes(bytes(ram[off:off + 4]), "little", signed=True)


class MessageTable:
    """text -> (group, id), with sprintf templates as regexes."""

    def __init__(self, path):
        with open(path) as fh:
            groups = json.load(fh)

        def rank(g):
            return next((i for i, p in enumerate(GROUP_ORDER)
                         if g.startswith(p)), len(GROUP_ORDER))
        self.exact, self.patterns = {}, []
        for group in sorted(groups, key=rank):
            for mid, text in groups[group].items():
                key = (group, int(mid))
                if "%" in text:
                    rx = re.escape(text)
                    rx = re.sub(r"%(?:\\-)?\d*(?:\\\.\d+)?[ds]", "(.+?)", rx)
                    self.patterns.append((re.compile(rx + r"\Z"), key))
                else:
                    self.exact.setdefault(text, key)

    def lookup(self, text):
        if not text:
            return None
        hit = self.exact.get(text)
        if hit:
            return hit
        for rx, key in self.patterns:
            if rx.match(text):
                return key
        return None


class MessageReader:
    """Per-env tracker: current messages, ids, and new-message events."""

    def __init__(self, table_path=None):
        self.table = None
        if table_path and os.path.exists(table_path):
            self.table = MessageTable(table_path)
        self.reset()

    def reset(self):
        self._prev = {"brief": ("", 0), "centre": ""}

    def read(self, ram):
        brief = _cstr(ram, BRIEF_TEXT)
        on = int(ram[BRIEF_ON - RAM_BASE])
        centre = _cstr(ram, CENTRE_TEXT)
        out = {"message": brief if on else "",
               "message_on": bool(on),
               "message_kind": _u32(ram, BRIEF_KIND) if on else -1,
               "centre_message": centre,
               "banner": _cstr(ram, BANNER_TEXT, 64)}
        new = []
        pb, pon = self._prev["brief"]
        if on and (not pon or brief != pb):
            new.append(brief)
        if centre and centre != self._prev["centre"]:
            new.append(centre)
        self._prev = {"brief": (brief, on), "centre": centre}

        def ident(text):
            key = self.table.lookup(text) if self.table else None
            return key if key else (None, -1)
        out["message_group"], out["message_id"] = ident(out["message"])
        out["centre_group"], out["centre_id"] = ident(centre)
        out["new_message_ids"] = [ident(t)[1] for t in new]
        return out
