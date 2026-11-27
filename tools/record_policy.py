#!/usr/bin/env python3
r"""Record a trained policy playing GTA, as an mp4 with a score overlay.

Scoring episodes are rare and the spread is enormous -- one episode in
forty scored 15600 against a median of 500 -- so recording the first
episode that runs is almost always recording a dull one. This searches
first and records second: it plays `--search` episodes keeping only the
score, then replays the best seed with video capture. Both the emulator
and the action sampler are seeded per episode, so the replay reproduces
the searched episode exactly.

    python tools/record_policy.py --ckpt runs/gta1_run2/checkpoint.pt \
        --search 24 --out scratch/run2_best.mp4
"""
import argparse, os, sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from gta1env.env import GTA1Config, GTA1Env
from train_ppo import ActorCritic


def checkpoint_cfg(path):
    """Observation geometry is part of the policy's shape, so it has to
    come from the checkpoint rather than from the env defaults."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    a = blob.get("args") or {}
    if not isinstance(a, dict):
        a = vars(a)
    cfg = {}
    for key in ("obs_width", "obs_height", "frame_stack"):
        if a.get(key) is not None:
            cfg[key] = a[key]
    if a.get("color") is not None:
        cfg["grayscale"] = not a["color"]
    return blob, cfg, a


def play(env, agent, device, seed, max_steps, writer=None, scale=2,
         fps_note=""):
    """One episode. Returns (score_gained, steps, distance)."""
    torch.manual_seed(seed)
    obs, info = env.reset(seed=seed)
    score0 = info.get("score")
    score = score0 or 0
    prev = (info.get("player_x"), info.get("player_y"))
    dist = 0.0
    for t in range(max_steps):
        with torch.no_grad():
            o = torch.from_numpy(obs).to(device).float().div_(255.0)[None]
            action = int(agent.act(o)[0].item())
        obs, reward, term, trunc, info = env.step(action)
        if info.get("score") is not None:
            score = info["score"]
        x, y = info.get("player_x"), info.get("player_y")
        if prev[0] is not None and x is not None:
            d = float(np.hypot(x - prev[0], y - prev[1]))
            if d < 1500:
                dist += d
        prev = (x, y)

        if writer is not None:
            frame = env.render()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if scale != 1:
                    bgr = cv2.resize(
                        bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale),
                        interpolation=cv2.INTER_NEAREST)
                gained = score - (score0 or 0)
                hud = [f"score {gained:+.0f}", f"step {t}/{max_steps}",
                       f"dist {dist:,.0f}"]
                if fps_note:
                    hud.append(fps_note)
                for i, text in enumerate(hud):
                    pos = (8, 20 + i * 18)
                    # outline first so the text survives a bright frame
                    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(bgr)
        if term or trunc:
            break
    return score - (score0 or 0), t + 1, dist


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/gta1_run2/checkpoint.pt")
    ap.add_argument("--out", default="scratch/policy.mp4")
    ap.add_argument("--search", type=int, default=16,
                    help="episodes to try before recording the best")
    ap.add_argument("--episode-steps", type=int, default=4500)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap recorded steps (default: whole episode)")
    ap.add_argument("--fps", type=int, default=30,
                    help="15 is real time (60fps / frameskip 4)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None,
                    help="record this seed and skip the search")
    ap.add_argument("--random", action="store_true",
                    help="record a random policy instead, for comparison")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob, geom, ck_args = checkpoint_cfg(args.ckpt)
    cfg = dict(geom)
    cfg["episode_steps"] = args.episode_steps
    print(f"geometry {cfg.get('obs_width', 84)}x{cfg.get('obs_height', 84)} "
          f"stack {cfg.get('frame_stack', 4)}; "
          f"checkpoint update {blob.get('update')}", flush=True)

    env = GTA1Env(GTA1Config(**cfg), render_mode="rgb_array")
    try:
        nact = env.action_space.n
        if args.random:
            class _Rand:
                def act(self, o):
                    return (torch.randint(0, nact, (1,)),)
            agent = _Rand()
        else:
            agent = ActorCritic(env.observation_space.shape[0], nact,
                                obs_hw=env.observation_space.shape[1:]
                                ).to(device)
            agent.load_state_dict(blob["model"])
            agent.eval()

        cap = args.max_steps or args.episode_steps
        if args.seed is None:
            results = []
            for s in range(args.search):
                sc, steps, dist = play(env, agent, device, s, cap)
                results.append((sc, s, steps, dist))
                print(f"  seed {s:3d}  score {sc:7.0f}  steps {steps:5d}  "
                      f"dist {dist:8.0f}", flush=True)
            results.sort(reverse=True)
            best_score, best_seed = results[0][0], results[0][1]
            scores = np.array([r[0] for r in results])
            print(f"\nbest seed {best_seed} with score {best_score:.0f} "
                  f"(median {np.median(scores):.0f} over {len(scores)})",
                  flush=True)
        else:
            best_seed = args.seed

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        h, w = env.core.height, env.core.width
        note = "real time" if args.fps == 15 else f"{args.fps / 15:g}x speed"
        writer = cv2.VideoWriter(args.out,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (w * args.scale, h * args.scale))
        if not writer.isOpened():
            raise RuntimeError(f"could not open {args.out} for writing")
        try:
            sc, steps, dist = play(env, agent, device, best_seed, cap,
                                   writer=writer, scale=args.scale,
                                   fps_note=note)
        finally:
            writer.release()
        size = os.path.getsize(args.out) / 1e6
        print(f"wrote {args.out} -- {steps} frames, {size:.1f} MB, "
              f"score {sc:.0f}, distance {dist:.0f}, "
              f"{steps / args.fps:.0f}s at {args.fps} fps ({note})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
