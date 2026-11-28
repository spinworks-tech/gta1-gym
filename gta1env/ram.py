"""Reading game state straight out of the PS1's 2 MiB main RAM.

The addresses are NOT known up front. GTA (PS1) has no published RAM map
worth trusting, so they are discovered empirically with tools/ram_scan.py
and stored in config/ram_map.json. Until that file has real entries the
env runs fine but gives you a reward of 0 -- by design, so a broken
reward signal is loud rather than silent.

PS1 main RAM is mirrored at 0x80000000 / 0xA0000000 / 0x00000000. The
libretro SYSTEM_RAM pointer is the raw 2 MiB, so an address such as
0x800A1234 becomes offset 0x000A1234.
"""

import json
import math
import os
import struct

import numpy as np

RAM_SIZE = 2 * 1024 * 1024
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config", "ram_map.json")

# How the agent's intents map onto a DualShock. Taken from the game's own
# "Control Selection" screen (Layout One), not guessed: the d-pad only
# steers -- forwards and backwards are face buttons.
CONTROLS = {
    "forward": "cross",        # accelerate, or walk on foot
    "backward": "triangle",    # reverse
    "brake": "r2",
    "turn_left": "left",
    "turn_right": "right",
    "fire": "circle",          # Attack/Fire
    "enter_exit": "square",    # Enter/Exit Vehicle
    "special": "r1",
    "prev_weapon": "l1",
    "next_weapon": "l2",
    "repeat_message": "select",
    "pause": "start",
}

# The discrete action set the policy chooses from. Each entry is a set of
# simultaneously held intents, held for the whole frameskip window.
DEFAULT_ACTIONS = [
    (),                                     # 0  coast
    ("forward",),                           # 1
    ("backward",),                          # 2
    ("turn_left",),                         # 3
    ("turn_right",),                        # 4
    ("forward", "turn_left"),               # 5
    ("forward", "turn_right"),              # 6
    ("backward", "turn_left"),              # 7
    ("backward", "turn_right"),             # 8
    ("brake",),                             # 9
    ("fire",),                              # 10
    ("forward", "fire"),                    # 11
    ("enter_exit",),                        # 12
    ("next_weapon",),                       # 13
]


def normalize(addr):
    """Map any PS1 virtual address onto a 0..2MiB RAM offset."""
    return addr & 0x001FFFFF


class Field:
    """One scalar in RAM: where it lives, how wide it is, how to read it."""

    FORMATS = {
        ("u", 1): "<B", ("i", 1): "<b",
        ("u", 2): "<H", ("i", 2): "<h",
        ("u", 4): "<I", ("i", 4): "<i",
    }

    def __init__(self, address, size=4, signed=False, scale=1.0, note=""):
        self.address = int(address)
        self.offset = normalize(self.address)
        self.size = int(size)
        self.signed = bool(signed)
        self.scale = float(scale)
        self.note = note
        key = ("i" if self.signed else "u", self.size)
        if key not in self.FORMATS:
            raise ValueError(f"unsupported field width {self.size}")
        self.fmt = self.FORMATS[key]

    def read(self, ram):
        raw = ram[self.offset:self.offset + self.size].tobytes()
        if len(raw) < self.size:
            return 0.0
        return struct.unpack(self.fmt, raw)[0] * self.scale

    def to_json(self):
        return {"address": self.address, "size": self.size,
                "signed": self.signed, "scale": self.scale, "note": self.note}


class RamMap:
    """A named collection of Fields, loaded from / saved to JSON."""

    # Everything the reward function and the info dict can use. Unset keys
    # are simply absent from the observation info.
    KNOWN_KEYS = ("cash", "score", "health", "armor", "lives", "wanted",
                  "player_x", "player_y", "player_z", "in_vehicle",
                  "mission_id", "level_complete", "game_over")

    def __init__(self, fields=None, path=None):
        self.fields = dict(fields or {})
        self.path = path

    @classmethod
    def load(cls, path=CONFIG_PATH):
        if not os.path.exists(path):
            return cls(path=path)
        with open(path) as fh:
            blob = json.load(fh)
        fields = {k: Field(**v) for k, v in blob.get("fields", {}).items()}
        return cls(fields, path=path)

    def save(self, path=None):
        path = path or self.path or CONFIG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"fields": {k: f.to_json()
                                  for k, f in self.fields.items()}},
                      fh, indent=2, sort_keys=True)
        return path

    def set(self, name, address, size=4, signed=False, scale=1.0, note=""):
        self.fields[name] = Field(address, size, signed, scale, note)

    def read_all(self, ram):
        return {name: f.read(ram) for name, f in self.fields.items()}

    @property
    def is_empty(self):
        return not self.fields

    def __contains__(self, name):
        return name in self.fields


class RewardShaper:
    """Turns successive RAM reads into a scalar reward.

    Deliberately simple and transparent: progress in cash/score is the
    signal, dying is a penalty, and everything is clipped so one corrupt
    read cannot blow up an update.
    """

    def __init__(self, cash_scale=0.01, score_scale=0.01, death_penalty=1.0,
                 health_scale=0.0, step_penalty=0.0, clip=5.0,
                 distance_scale=0.0, max_step_distance=50.0,
                 respawn_distance=1500.0, novelty_scale=0.0,
                 novelty_cell=256.0):
        self.cash_scale = cash_scale
        self.score_scale = score_scale
        self.death_penalty = death_penalty
        self.health_scale = health_scale
        self.step_penalty = step_penalty
        self.clip = clip
        # Score is very sparse (roughly one event per few thousand steps),
        # so movement can be used as a dense bootstrap signal.
        self.distance_scale = distance_scale
        self.max_step_distance = max_step_distance
        # The lives field (0x2C446) now detects death exactly; this jump test
        # is kept as a fallback for RAM maps without it. Death is inferred
        # from a large jump in position. Measured over 40k steps: ordinary motion peaks
        # near 32 units/step, entering or leaving a vehicle snaps the player
        # by up to ~512, and a real teleport was ~5200. The threshold sits
        # well above the vehicle snaps -- a lower one would penalise getting
        # into cars. Heuristic: set to 0 to disable.
        self.respawn_distance = respawn_distance
        # Distance is PATH LENGTH, and path length is farmable: measured
        # from a state where the car is wedged against the waterfront, the
        # trained policy earned 746 units of "travel" over 600 steps while
        # moving a net 15 -- it learned to vibrate in place. Novelty pays
        # once per world cell per episode instead, so jitter earns nothing
        # and only going somewhere new pays.
        self.novelty_scale = novelty_scale
        self.novelty_cell = novelty_cell
        self.visited = set()
        self.prev = None

    def reset(self, state):
        self.prev = dict(state)
        self.visited = set()
        self._mark_cell(state)

    def _mark_cell(self, state):
        """Return True if this position is a cell not yet seen this episode."""
        if not self.novelty_scale:
            return False
        x, y = state.get("player_x"), state.get("player_y")
        if x is None or y is None:
            return False
        cell = (int(math.floor(x / self.novelty_cell)),
                int(math.floor(y / self.novelty_cell)))
        if cell in self.visited:
            return False
        self.visited.add(cell)
        return True

    def __call__(self, state):
        if self.prev is None:
            self.reset(state)
            return 0.0, False
        r = -self.step_penalty
        for key, scale in (("cash", self.cash_scale),
                           ("score", self.score_scale),
                           ("health", self.health_scale)):
            if scale and key in state and key in self.prev:
                delta = state[key] - self.prev[key]
                # A huge jump usually means a reload/respawn, not progress.
                if abs(delta) < 1e6:
                    r += scale * delta
        dead = False
        if "player_x" in state and "player_y" in state and self.prev:
            dx = state["player_x"] - self.prev.get("player_x", 0.0)
            dy = state["player_y"] - self.prev.get("player_y", 0.0)
            moved = math.hypot(dx, dy)
            if self.respawn_distance and moved >= self.respawn_distance:
                dead = True            # teleported: respawned after dying
            elif self.distance_scale:
                r += self.distance_scale * min(moved, self.max_step_distance)
        if "health" in state and state["health"] <= 0 < self.prev.get(
                "health", 0):
            dead = True
        if "lives" in state and state["lives"] < self.prev.get("lives", 0):
            dead = True
        if "game_over" in state and state["game_over"]:
            dead = True
        if not dead and self._mark_cell(state):
            r += self.novelty_scale
        if dead:
            r -= self.death_penalty
        self.prev = dict(state)
        return float(np.clip(r, -self.clip, self.clip)), dead
