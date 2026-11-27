r"""Evaluate a checkpoint against a random policy.

Episodic return cannot say whether the agent learned the game or just
the movement bonus, because the shaped return mixes the two. This
reports the raw quantities separately -- in-game score gained and world
distance travelled -- for a checkpoint and for a random policy over the
same number of episodes, so a return can be decomposed against the
shaper's weights.

    python tools/eval_policy.py --ckpt runs/gta1_run1/checkpoint.pt \
        --envs 6 --steps 9200
"""
import argparse, math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from gta1env.vec import SubprocVecEnv
from gta1env.env import GTA1Config
from train_ppo import ActorCritic


def rollout(vec, agent, device, steps, nact, rng):
    n = vec.num_envs
    score0 = [None] * n
    score_end = [0.0] * n
    dist = np.zeros(n)
    prev = [None] * n
    eps = []  # (score_gained, distance)
    obs = vec.reset()
    for t in range(steps):
        if agent is None:
            a = rng.integers(0, nact, size=n)
        else:
            with torch.no_grad():
                o = vec.obs_to_torch(device)
                a = agent.act(o)[0].cpu().numpy()
        vec.step_async(a)
        _, _, _, _, infos = vec.step_wait()
        for i, info in enumerate(infos):
            s, x, y = info.get("score"), info.get("player_x"), info.get("player_y")
            if score0[i] is None and s is not None:
                score0[i] = s
            if prev[i] is not None and x is not None:
                d = math.hypot(x - prev[i][0], y - prev[i][1])
                if d < 1500:
                    dist[i] += d
            prev[i] = (x, y)
            if s is not None:
                score_end[i] = s
            if info.get("episode_ended"):
                eps.append((score_end[i] - (score0[i] or 0), dist[i]))
                score0[i], dist[i], prev[i] = None, 0.0, None
    return eps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/gta1_run1/checkpoint.pt")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--boot-state", default=None,
                    help="file or directory of start states")
    ap.add_argument("--episode-steps", type=int, default=4500)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = {"episode_steps": args.episode_steps,
           "distance_scale": 0.001}
    if args.boot_state:
        cfg["boot_state"] = os.path.abspath(args.boot_state)
    rng = np.random.default_rng(0)
    out = {}
    for label in ("random", "policy"):
        vec = SubprocVecEnv(args.envs, cfg, base_seed=1234)
        try:
            nact = vec.num_actions
            agent = None
            if label == "policy":
                agent = ActorCritic(vec.single_obs_shape[0], nact, obs_hw=vec.single_obs_shape[1:]).to(device)
                ck = torch.load(args.ckpt, map_location=device, weights_only=True)
                agent.load_state_dict(ck["model"])
                agent.eval()
                print(f"loaded checkpoint from update {ck.get('update')}", flush=True)
            out[label] = rollout(vec, agent, device, args.steps, nact, rng)
        finally:
            vec.close()
        e = out[label]
        if e:
            sc = np.array([x[0] for x in e]); ds = np.array([x[1] for x in e])
            print(f"{label:7s} n={len(e):3d}  score {sc.mean():8.1f} "
                  f"(max {sc.max():.0f})  distance {ds.mean():10.0f}", flush=True)
        else:
            print(f"{label:7s} no completed episodes", flush=True)


if __name__ == "__main__":
    main()
