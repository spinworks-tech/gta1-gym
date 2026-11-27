#!/usr/bin/env python3
"""Measure real throughput: emulated frames/s and agent steps/s.

Run this before training. It tells you how many parallel emulators this
box actually sustains, which is the number that decides your wall-clock
budget -- not the GPU.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gta1env.env import GTA1Config, GTA1Env  # noqa: E402
from gta1env.vec import SubprocVecEnv  # noqa: E402


def bench_single(steps, cfg_kwargs):
    env = GTA1Env(GTA1Config(**cfg_kwargs))
    env.reset()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(rng.integers(env.action_space.n))
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    skip = env.cfg.frameskip
    env.close()
    print(f"  1 env : {steps/dt:8.1f} steps/s  "
          f"{steps*skip/dt:8.1f} emulated fps  "
          f"({steps*skip/dt/60:.1f}x realtime)")
    return steps / dt


def bench_parallel(num_envs, steps, cfg_kwargs, device=None):
    vec = SubprocVecEnv(num_envs, cfg_kwargs=cfg_kwargs)
    vec.reset()
    rng = np.random.default_rng(0)
    skip = GTA1Config(**cfg_kwargs).frameskip

    for _ in range(5):                       # warm up the JIT / page cache
        vec.step(rng.integers(vec.num_actions, size=num_envs))

    t0 = time.perf_counter()
    for _ in range(steps):
        vec.step(rng.integers(vec.num_actions, size=num_envs))
        if device:
            vec.obs_to_torch(device)
    dt = time.perf_counter() - t0
    vec.close()

    total = steps * num_envs
    print(f"  {num_envs:2d} envs: {total/dt:8.1f} steps/s  "
          f"{total*skip/dt:8.1f} emulated fps  "
          f"({total*skip/dt/60:.1f}x realtime)  "
          f"{total/dt/num_envs:6.1f} steps/s per env")
    return total / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--envs", default="1,4,8,12,16",
                    help="comma separated worker counts to sweep")
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--device", default=None,
                    help="also time the GPU upload, e.g. cuda")
    args = ap.parse_args()

    cfg_kwargs = {"frameskip": args.frameskip, "episode_steps": 10 ** 9}
    print(f"frameskip={args.frameskip}  steps/measurement={args.steps}  "
          f"cpus={os.cpu_count()}")
    print("(realtime = 60 emulated fps; PS1 GTA logic runs at 30 Hz)")

    best = 0.0
    for token in args.envs.split(","):
        n = int(token)
        rate = bench_single(args.steps, cfg_kwargs) if n == 1 else \
            bench_parallel(n, args.steps, cfg_kwargs, args.device)
        best = max(best, rate)

    print(f"\nbest aggregate: {best:.0f} agent steps/s "
          f"= {best*3600/1e6:.2f}M steps/hour")
    print("A 10M-step PPO run therefore takes roughly "
          f"{10e6/best/3600:.1f} hours of sampling.")


if __name__ == "__main__":
    main()
