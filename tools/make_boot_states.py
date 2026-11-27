#!/usr/bin/env python3
"""Build a directory of diverse episode start states.

Every episode starting from one identical savestate is a real limitation:
the agent sees the same opening frames, spawns in the same street, and
explores one corner of Liberty City. This walks away from that state with
random actions and snapshots along the way, so resets sample a spread of
positions, times of day and vehicle situations.

    python tools/make_boot_states.py --count 16 --out config/boot_states

The result is a drop-in for the env's boot_state, which samples a
directory per reset:

    python train_ppo.py --boot-state config/boot_states ...

Random actions turn out to explore very badly: measured over ~11k steps
the player never left a 240x335 unit box, because random input mills in
place instead of travelling. Pass --policy to drive with a trained
checkpoint instead, which is what actually covers ground:

    python tools/make_boot_states.py --count 16 --policy runs/x/checkpoint.pt \
        --min-steps 1000 --max-steps 4000

Neither mode gives *intent*: states parked next to a mission phone need
hand-driving with tools/drive.py, and can be dropped into the same
directory alongside these.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import GTA1Config, GTA1Env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=16,
                    help="how many start states to write")
    ap.add_argument("--out", default="config/boot_states",
                    help="directory to fill with NN.state files")
    ap.add_argument("--min-steps", type=int, default=200,
                    help="random steps before the first snapshot")
    ap.add_argument("--max-steps", type=int, default=1200,
                    help="random steps between later snapshots")
    ap.add_argument("--boot-state", default="",
                    help="state to walk away from (default: the config one)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bios-mode", default="auto")
    ap.add_argument("--policy", default=None,
                    help="checkpoint to drive with instead of random "
                         "actions; random input barely moves the player")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg_kwargs = {"bios_mode": args.bios_mode, "episode_steps": 10 ** 9}
    if args.boot_state:
        cfg_kwargs["boot_state"] = args.boot_state
    env = GTA1Env(GTA1Config(**cfg_kwargs))
    rng = np.random.default_rng(args.seed)

    agent = None
    if args.policy:
        import torch
        from train_ppo import ActorCritic
        chans = env.cfg.frame_stack * (1 if env.cfg.grayscale else 3)
        agent = ActorCritic(chans, env.action_space.n,
                            (env.cfg.obs_height, env.cfg.obs_width))
        ck = torch.load(args.policy, map_location="cpu", weights_only=True)
        agent.load_state_dict(ck["model"])
        agent = agent.to(args.device).eval()
        print(f"driving with {args.policy} (update {ck.get('update')})",
              flush=True)

    def pick(obs):
        if agent is None:
            return int(rng.integers(0, env.action_space.n))
        import torch
        with torch.no_grad():
            o = torch.as_tensor(obs, device=args.device).unsqueeze(0)
            return int(agent.act(o.float().div_(255.0))[0].item())
    os.makedirs(args.out, exist_ok=True)

    try:
        obs, _ = env.reset(seed=args.seed)
        written = []
        for i in range(args.count):
            steps = int(rng.integers(args.min_steps, args.max_steps + 1))
            for _ in range(steps):
                obs, _, term, trunc, _ = env.step(pick(obs))
                if term or trunc:
                    obs, _ = env.reset()
            state = env._read_state()
            blob = env.core.serialize()
            path = os.path.join(args.out, f"{i:02d}.state")
            with open(path, "wb") as fh:
                fh.write(blob)
            x = state.get("player_x", float("nan"))
            y = state.get("player_y", float("nan"))
            written.append((path, x, y, state.get("score", 0)))
            print(f"{path}  after {steps:5d} steps  "
                  f"pos ({x:9.1f}, {y:9.1f})  score {state.get('score', 0)}",
                  flush=True)
    finally:
        env.close()

    # Spread is the whole point, so report it rather than assume it.
    pts = [(x, y) for _, x, y, _ in written if not math.isnan(x)]
    if len(pts) > 1:
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        d = [math.hypot(xs[i] - xs[j], ys[i] - ys[j])
             for i in range(len(pts)) for j in range(i + 1, len(pts))]
        print(f"\n{len(written)} states in {args.out}")
        print(f"x range {xs.min():.0f}..{xs.max():.0f}  "
              f"y range {ys.min():.0f}..{ys.max():.0f}")
        print(f"pairwise distance: min {min(d):.0f}  "
              f"median {float(np.median(d)):.0f}  max {max(d):.0f}")
        if min(d) < 100:
            print("note: some states are nearly co-located; raise --max-steps")


if __name__ == "__main__":
    main()
