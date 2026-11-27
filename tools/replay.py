#!/usr/bin/env python3
"""Turn recorded play sessions into imitation-learning datasets.

Replays each recording from tools/play.py through the same emulator,
preprocessing, frame stack and reward shaper as GTA1Env, and writes:

  obs        uint8  (N, C, H, W)   what the agent would have seen
  actions    int64  (N,)           human input mapped to the env's action set
  rewards    float32 (N,)          RewardShaper output for that step
  terminated bool (N,)             shaper saw a death (the human played on)
  truncated  bool (N,)             last step of the session
  exact      bool (N,)             input was exactly that action, all frames
  masks      uint16 (N, frameskip) raw per-frame button masks
  ram_<name> float64 (N,)          RAM-map fields after the step

Step t means: obs[t] is observed, the human holds actions[t] for
`frameskip` frames, rewards[t] is earned. The first window of every
session is used to produce obs[0], like the no-op window in
GTA1Env.reset(), so it carries no label.

The emulator is always driven by the recorded per-frame masks, never by
the quantised action, so the trajectory is the one the human actually
played; the final RAM is checked against the recording's hash.

  $ .venv/bin/python tools/replay.py recordings/<session> [...]
  $ .venv/bin/python tools/replay.py recordings/* --frameskip 2 --color --size 128
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import GTA1Config, GTA1Env  # noqa: E402
from gta1env.libretro import JOYPAD, PSX_BUTTON  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Same memory-card folder as tools/play.py, so replays never touch the one
# that training processes use.
SAVE_DIR = os.path.join(HERE, "scratch", "play_saves")

BIT_NAME = {JOYPAD[key]: name for name, key in PSX_BUTTON.items()}


def mask_names(mask):
    return "+".join(BIT_NAME.get(b, f"bit{b}")
                    for b in range(16) if mask >> b & 1) or "none"


def mask_from_buttons(buttons):
    m = 0
    for name in buttons:
        key = PSX_BUTTON.get(name.lower(), name.upper())
        m |= 1 << JOYPAD[key]
    return m


class Quantiser:
    """Map a window of per-frame masks onto the env's discrete actions."""

    def __init__(self, button_sets):
        self.masks = np.array([mask_from_buttons(b) for b in button_sets])

    def __call__(self, window):
        # Every button pressed at any point in the window. A majority vote
        # would drop taps -- fire and enter/exit are usually held for a
        # frame or two -- and the env holds the action for the whole
        # window anyway.
        mask = 0
        for m in window:
            mask |= int(m)
        # Nearest action by number of differing buttons. Ties prefer an
        # action whose buttons are all actually held, then the lowest index.
        diff = [bin(mask ^ int(a)).count("1") for a in self.masks]
        subset = [(int(a) & ~mask) == 0 for a in self.masks]
        action = min(range(len(self.masks)),
                     key=lambda i: (diff[i], not subset[i], i))
        exact = diff[action] == 0 and len(set(window)) == 1
        return action, exact, mask


def replay(env, rec):
    with open(os.path.join(rec, "meta.json")) as fh:
        meta = json.load(fh)
    inputs = np.load(os.path.join(rec, "inputs.npy")).astype(np.uint16)
    with open(os.path.join(rec, "start.state"), "rb") as fh:
        env.core.unserialize(fh.read())

    fs = env.cfg.frameskip
    quant = Quantiser(env._button_sets)
    env._stack[:] = 0
    env.shaper.reset(env._read_state())

    def window(masks):
        last = len(masks) - 1
        for i, m in enumerate(masks):
            env.core.set_buttons(int(m))
            env.core.run(convert=(i == last))
        return env.core.frame

    n_win = len(inputs) // fs
    if n_win < 2:
        raise ValueError(f"{rec}: only {len(inputs)} frames, need {2 * fs}")

    # Window 0 produces the first observation, like the no-op in reset().
    obs0 = env._push(window(inputs[:fs]))
    env.shaper.reset(env._read_state())

    n = n_win - 1
    out = {
        "obs": np.empty((n,) + obs0.shape, dtype=np.uint8),
        "actions": np.empty(n, dtype=np.int64),
        "rewards": np.empty(n, dtype=np.float32),
        "terminated": np.zeros(n, dtype=bool),
        "truncated": np.zeros(n, dtype=bool),
        "exact": np.empty(n, dtype=bool),
        "masks": np.empty((n, fs), dtype=np.uint16),
    }
    ram = {k: np.empty(n) for k in env.ram_map.fields}
    obs = obs0
    unmatched = Counter()
    for t in range(n):
        w = inputs[(t + 1) * fs:(t + 2) * fs]
        action, exact, mask = quant(list(w))
        out["obs"][t] = obs
        out["actions"][t] = action
        out["exact"][t] = exact
        out["masks"][t] = w
        if not exact:
            unmatched[mask] += 1
        obs = env._push(window(w))
        state = env._read_state()
        out["rewards"][t], out["terminated"][t] = env.shaper(state)
        for k in ram:
            ram[k][t] = state[k]
    out["truncated"][-1] = True

    # Frames past the last whole window are played but not labelled.
    window(inputs[n_win * fs:]) if len(inputs) % fs else None
    got = hashlib.sha256(env.core.memory().tobytes()).hexdigest()
    if got != meta["final_ram_sha256"]:
        raise RuntimeError(
            f"{rec}: replay diverged from the recording (final RAM differs); "
            "different core build, options or start state?")
    out.update({f"ram_{k}": v for k, v in ram.items()})
    return out, unmatched


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+")
    p.add_argument("--frameskip", type=int, default=GTA1Config.frameskip)
    p.add_argument("--size", type=int, default=GTA1Config.obs_width,
                   help="square observation size")
    p.add_argument("--color", action="store_true", help="RGB instead of gray")
    p.add_argument("--frame-stack", type=int, default=GTA1Config.frame_stack)
    p.add_argument("--distance-scale", type=float,
                   default=GTA1Config.distance_scale)
    p.add_argument("--out", help="output name inside each recording "
                   "(default: dataset_fs<F>_<S><g|c>.npz)")
    args = p.parse_args()

    recs = [r for r in args.recordings
            if os.path.exists(os.path.join(r, "meta.json"))]
    if not recs:
        sys.exit("no recordings (directories with meta.json) given")
    bios = {json.load(open(os.path.join(r, "meta.json")))["bios_mode"]
            for r in recs}
    if len(bios) > 1:
        sys.exit(f"recordings mix bios modes {bios}; replay them separately")

    cfg = GTA1Config(
        frameskip=args.frameskip, obs_width=args.size, obs_height=args.size,
        grayscale=not args.color, frame_stack=args.frame_stack,
        distance_scale=args.distance_scale, noop_max=0,
        bios_mode=bios.pop(), save_dir=SAVE_DIR)
    env = GTA1Env(cfg)
    name = args.out or (f"dataset_fs{args.frameskip}_{args.size}"
                        f"{'c' if args.color else 'g'}.npz")

    for rec in recs:
        data, unmatched = replay(env, rec)
        path = os.path.join(rec, name)
        np.savez_compressed(path, **data)
        n = len(data["actions"])
        exact = data["exact"].mean()
        hist = np.bincount(data["actions"], minlength=len(env.actions))
        print(f"{rec}: {n} steps, final RAM identical, "
              f"return {data['rewards'].sum():.2f}, "
              f"deaths {int(data['terminated'].sum())} -> {path}")
        print(f"  exact action match {exact:.1%}")
        print("  actions: " + "  ".join(
            f"{i}:{c}" for i, c in enumerate(hist) if c))
        for mask, c in unmatched.most_common(5):
            print(f"  approximated {c:5d}x  {mask_names(mask)}")
    env.close()


if __name__ == "__main__":
    main()
