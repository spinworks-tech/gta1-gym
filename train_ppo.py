#!/usr/bin/env python3
"""PPO on the original GTA (PS1), emulators on the CPU, policy on CUDA.

Sampling is the bottleneck on this kind of box, so the loop is written
around that: N emulator processes fill a shared-memory observation
block, one pinned-buffer copy moves it to the GPU, and the update runs
entirely in VRAM.

  .venv/bin/python train_ppo.py --envs 12 --total-steps 10_000_000
"""

import argparse
import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from gta1env.env import GTA1Config
from gta1env.vec import SubprocVecEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """The Nature-DQN trunk with separate policy and value heads."""

    def __init__(self, in_channels, num_actions, obs_hw=(84, 84)):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.trunk(torch.zeros(1, in_channels, *obs_hw)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n_flat, 512)), nn.ReLU())
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, obs):
        h = self.fc(self.trunk(obs))
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--total-steps", type=int, default=10_000_000)
    ap.add_argument("--rollout", type=int, default=128,
                    help="steps per env per update")
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-coef", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--anneal-lr", action="store_true", default=True)
    ap.add_argument("--frameskip", type=int, default=4)
    # The PS1 framebuffer is 320x240 (4:3). Squashing it into a square
    # 84x84 compresses it horizontally by 25%; 112x84 keeps the ratio.
    ap.add_argument("--obs-width", type=int, default=84)
    ap.add_argument("--obs-height", type=int, default=84)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--color", dest="grayscale", action="store_false",
                    default=True,
                    help="keep RGB instead of grayscale; the game codes a "
                         "lot of meaning in colour, at 3x the channels")
    ap.add_argument("--sticky-prob", type=float, default=0.05)
    ap.add_argument("--max-episode-steps", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="bf16/fp16 autocast for the update")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-every", type=int, default=50,
                    help="updates between checkpoints")
    ap.add_argument("--save-best", action="store_true", default=True,
                    help="also keep best.pt, the highest mean-return "
                         "checkpoint; a run that collapses late would "
                         "otherwise overwrite its best policy")
    ap.add_argument("--no-save-best", dest="save_best",
                    action="store_false")
    ap.add_argument("--resume", default=None)
    # Core/content overrides: swap in a different libretro core (see the
    # README's note on SwanStation), or the mock core for a smoke test.
    ap.add_argument("--core-path", default=None)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--ram-map", default=None)
    ap.add_argument("--boot-state", default=None)
    ap.add_argument("--pipeline", action="store_true", default=True,
                    help="step two half-batches out of phase (default on)")
    ap.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                    help="step all envs in lockstep; slower, for comparison")
    ap.add_argument("--distance-scale", type=float, default=0.0,
                    help="reward per world unit moved; densifies the very "
                         "sparse score signal (try 0.002)")
    ap.add_argument("--distance-scale-final", type=float, default=None,
                    help="anneal distance-scale to this by the end of "
                         "--reward-anneal-frac of training; default: no "
                         "annealing. Use 0.0 to wean the agent off movement "
                         "reward entirely")
    ap.add_argument("--reward-anneal-frac", type=float, default=0.5,
                    help="fraction of training over which the distance "
                         "reward is annealed")
    ap.add_argument("--reward-anneal-every", type=int, default=10,
                    help="updates between reward-scale broadcasts")
    ap.add_argument("--novelty-scale", type=float, default=0.0,
                    help="reward for entering a world cell not yet seen "
                         "this episode; unlike --distance-scale this "
                         "cannot be farmed by jittering in place")
    ap.add_argument("--novelty-cell", type=float, default=256.0,
                    help="side of a novelty cell in world units")
    ap.add_argument("--score-scale", type=float, default=None,
                    help="reward per point of in-game score (env default "
                         "0.01); raise it to outweigh movement shaping")
    ap.add_argument("--bios-mode", default="auto", choices=("auto", "HLE"),
                    help="HLE needs no BIOS file; lower compatibility")
    ap.add_argument("--skip-asset-check", action="store_true",
                    help="only for the mock core; real runs need BIOS+disc")
    return ap.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"gta1_ppo_{int(time.time())}"
    out_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)
    writer.add_text("args", "\n".join(f"{k}={v}" for k, v in
                                      sorted(vars(args).items())))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    cfg_kwargs = {
        "frameskip": args.frameskip,
        "sticky_prob": args.sticky_prob,
        "episode_steps": args.max_episode_steps,
    }
    for key, value in (("core_path", args.core_path), ("rom_path", args.rom),
                       ("ram_map_path", args.ram_map),
                       ("boot_state", args.boot_state)):
        if value:
            cfg_kwargs[key] = os.path.abspath(value)
    cfg_kwargs["bios_mode"] = args.bios_mode
    cfg_kwargs["obs_width"] = args.obs_width
    cfg_kwargs["obs_height"] = args.obs_height
    cfg_kwargs["frame_stack"] = args.frame_stack
    cfg_kwargs["grayscale"] = args.grayscale
    cfg_kwargs["distance_scale"] = args.distance_scale
    cfg_kwargs["novelty_scale"] = args.novelty_scale
    cfg_kwargs["novelty_cell"] = args.novelty_cell
    if args.score_scale is not None:
        cfg_kwargs["score_scale"] = args.score_scale
    if args.skip_asset_check:
        cfg_kwargs["require_assets"] = False
    probe = GTA1Config(**cfg_kwargs)
    vec = SubprocVecEnv(args.envs, cfg_kwargs=cfg_kwargs, base_seed=args.seed)
    print(f"{args.envs} emulators up; action space {vec.num_actions}")
    if not vec.specs[0]["ram_fields"]:
        print("WARNING: config/ram_map.json is empty, so every reward will "
              "be 0. Find the addresses first with tools/ram_scan.py.")

    chans = probe.frame_stack * (1 if probe.grayscale else 3)
    agent = ActorCritic(chans, vec.num_actions,
                        (probe.obs_height, probe.obs_width)).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_update = ckpt.get("update", 0)
        print(f"resumed from {args.resume} at update {start_update}")

    batch_size = args.envs * args.rollout
    minibatch_size = batch_size // args.minibatches
    num_updates = args.total_steps // batch_size
    obs_shape = (chans, probe.obs_height, probe.obs_width)

    # Rollout storage lives on the GPU, kept in uint8 (41 MB at the default
    # sizes; float32 would be 165 MB). Peak usage including activations and
    # Adam state is ~300 MB, so VRAM is not the constraint here -- sampling is.
    obs_buf = torch.zeros((args.rollout, args.envs) + obs_shape,
                          dtype=torch.uint8, device=device)
    act_buf = torch.zeros((args.rollout, args.envs), dtype=torch.long,
                          device=device)
    logp_buf = torch.zeros((args.rollout, args.envs), device=device)
    rew_buf = torch.zeros((args.rollout, args.envs), device=device)
    done_buf = torch.zeros((args.rollout, args.envs), device=device)
    val_buf = torch.zeros((args.rollout, args.envs), device=device)

    def norm(frames):
        return frames.float().div_(255.0)

    # Two half-batches stepped out of phase: while one half's emulators are
    # busy, the other half's observations go through the GPU. Each env's own
    # trajectory stays contiguous in time, so PPO is unaffected.
    if args.pipeline and args.envs >= 2:
        half = args.envs // 2
        groups = (slice(0, half), slice(half, args.envs))
    else:
        groups = (slice(0, args.envs),)

    vec.reset(seeds=[args.seed + i for i in range(args.envs)])
    cur = {}
    for g in groups:
        o = vec.obs_to_torch(device, normalize=False, group=g)
        with torch.no_grad():
            a, lp, _, v = agent.act(norm(o))
        cur[g] = {"obs": o, "act": a, "logp": lp, "val": v,
                  "done": torch.zeros(g.stop - g.start, device=device)}
        vec.step_async(a.cpu().numpy(), g)

    # Return alone cannot say whether the agent is learning the game or
    # just the movement bonus, so track the two raw quantities too.
    ep_scores = deque(maxlen=100)
    ep_cells = deque(maxlen=100)
    cells_last = np.zeros(args.envs)
    ep_distance = deque(maxlen=100)
    score_start = [None] * args.envs
    score_last = np.zeros(args.envs)
    run_dist = np.zeros(args.envs)
    prev_pos = [None] * args.envs
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    running_ret = np.zeros(args.envs, dtype=np.float64)
    running_len = np.zeros(args.envs, dtype=np.int64)
    global_step = start_update * batch_size
    t_start = time.perf_counter()

    def distance_scale_at(update):
        """Dense movement reward gets the policy driving, but left at full
        weight it is also what the policy ends up maximising -- measured at
        80% of the return gain in the first run. Annealing it hands the
        credit back to the sparse game signals once driving is learned."""
        if args.distance_scale_final is None:
            return args.distance_scale
        span = max(1.0, args.reward_anneal_frac * num_updates)
        t = min(1.0, update / span)
        return (args.distance_scale
                + t * (args.distance_scale_final - args.distance_scale))

    cur_distance_scale = None
    best_return = float("-inf")

    for update in range(start_update, num_updates):
        if args.anneal_lr:
            frac = 1.0 - update / max(1, num_updates)
            optimizer.param_groups[0]["lr"] = frac * args.lr

        # The reward broadcast happens per group inside the rollout, in
        # the window where that group has no step in flight.
        anneal_now = (args.distance_scale_final is not None
                      and update % args.reward_anneal_every == 0)
        if anneal_now:
            cur_distance_scale = distance_scale_at(update)

        t_roll = time.perf_counter()
        for step in range(args.rollout):
            for g in groups:
                c = cur[g]
                # The action in flight was chosen from this observation.
                obs_buf[step, g] = c["obs"]
                done_buf[step, g] = c["done"]
                act_buf[step, g] = c["act"]
                logp_buf[step, g] = c["logp"]
                val_buf[step, g] = c["val"]

                _, rewards, terms, truncs, infos = vec.step_wait(g)
                if anneal_now and step == 0:
                    # This group is idle between its wait and its next
                    # async send -- the only safe window for a call().
                    vec.call("set_reward_scales",
                             distance_scale=cur_distance_scale, group=g)
                rew_buf[step, g] = torch.as_tensor(rewards, device=device)
                done = np.logical_or(terms, truncs)
                c["done"] = torch.as_tensor(done.astype(np.float32),
                                            device=device)
                c["obs"] = vec.obs_to_torch(device, normalize=False, group=g)
                with torch.no_grad():
                    a, lp, _, v = agent.act(norm(c["obs"]))
                c["act"], c["logp"], c["val"] = a, lp, v
                # Sent before the other group is serviced, so these emulators
                # work while the GPU handles that one. The send also happens
                # on the final step of a rollout, which keeps the emulators
                # busy through the optimize phase; that in-flight action is
                # recorded as step 0 of the next rollout, with the log-prob
                # and value from the policy that actually chose it.
                vec.step_async(a.cpu().numpy(), g)

                sl = slice(g.start, g.stop)
                running_ret[sl] += rewards
                running_len[sl] += 1
                for k, info in enumerate(infos):
                    i = g.start + k
                    sc = info.get("score")
                    if sc is not None:
                        if score_start[i] is None:
                            score_start[i] = sc
                        score_last[i] = sc
                    if info.get("cells") is not None:
                        cells_last[i] = info["cells"]
                    x, y = info.get("player_x"), info.get("player_y")
                    if x is not None and y is not None:
                        if prev_pos[i] is not None:
                            d = math.hypot(x - prev_pos[i][0],
                                           y - prev_pos[i][1])
                            # Same gate the shaper uses: a bigger jump is a
                            # respawn, not travel.
                            if d < probe.respawn_distance:
                                run_dist[i] += d
                        prev_pos[i] = (x, y)
                for k in np.flatnonzero(done):
                    i = g.start + k
                    ep_returns.append(running_ret[i])
                    ep_lengths.append(running_len[i])
                    if score_start[i] is not None:
                        ep_scores.append(score_last[i] - score_start[i])
                    ep_distance.append(run_dist[i])
                    ep_cells.append(cells_last[i])
                    running_ret[i] = 0.0
                    running_len[i] = 0
                    score_start[i], run_dist[i] = None, 0.0
                    prev_pos[i] = None
                global_step += g.stop - g.start
        roll_time = time.perf_counter() - t_roll

        # GAE. cur[g]["val"] is already V(final obs), so no extra forward.
        with torch.no_grad():
            next_value = torch.cat([cur[g]["val"] for g in groups])
            next_done = torch.cat([cur[g]["done"] for g in groups])
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(args.envs, device=device)
            for t in reversed(range(args.rollout)):
                if t == args.rollout - 1:
                    next_nonterm = 1.0 - next_done
                    next_val = next_value
                else:
                    next_nonterm = 1.0 - done_buf[t + 1]
                    next_val = val_buf[t + 1]
                delta = (rew_buf[t] + args.gamma * next_val * next_nonterm
                         - val_buf[t])
                last_gae = (delta + args.gamma * args.gae_lambda
                            * next_nonterm * last_gae)
                advantages[t] = last_gae
            returns = advantages + val_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        t_opt = time.perf_counter()
        clipfracs = []
        amp_dtype = torch.bfloat16 if (args.amp and device.type == "cuda") \
            else torch.float32
        for _ in range(args.epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                with torch.autocast(device_type=device.type,
                                    dtype=amp_dtype, enabled=args.amp):
                    newlogp, entropy, newval = agent.evaluate(
                        norm(b_obs[idx]), b_act[idx])
                logratio = newlogp.float() - b_logp[idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float().mean().item())

                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef,
                                          1 + args.clip_coef)).mean()

                newval = newval.float()
                v_clipped = b_val[idx] + (newval - b_val[idx]).clamp(
                    -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((newval - b_ret[idx]) ** 2,
                                         (v_clipped - b_ret[idx]) ** 2).mean()

                loss = (pg_loss + args.vf_coef * v_loss
                        - args.ent_coef * entropy.float().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         args.max_grad_norm)
                optimizer.step()
        opt_time = time.perf_counter() - t_opt

        sps = global_step / (time.perf_counter() - t_start)
        writer.add_scalar("charts/sps", sps, global_step)
        if cur_distance_scale is not None:
            writer.add_scalar("charts/distance_scale", cur_distance_scale,
                              global_step)
        writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"],
                          global_step)
        writer.add_scalar("time/rollout_s", roll_time, global_step)
        writer.add_scalar("time/optimize_s", opt_time, global_step)
        writer.add_scalar("losses/policy", pg_loss.item(), global_step)
        writer.add_scalar("losses/value", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.float().mean().item(),
                          global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)),
                          global_step)
        if ep_scores:
            writer.add_scalar("charts/episodic_score",
                              float(np.mean(ep_scores)), global_step)
        if ep_cells:
            writer.add_scalar("charts/episodic_cells",
                              float(np.mean(ep_cells)), global_step)
        if ep_distance:
            writer.add_scalar("charts/episodic_distance",
                              float(np.mean(ep_distance)), global_step)
        if ep_returns:
            writer.add_scalar("charts/episodic_return",
                              float(np.mean(ep_returns)), global_step)
            writer.add_scalar("charts/episodic_length",
                              float(np.mean(ep_lengths)), global_step)

        if update % 5 == 0:
            ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            sc = float(np.mean(ep_scores)) if ep_scores else float("nan")
            di = float(np.mean(ep_distance)) if ep_distance else float("nan")
            print(f"update {update:5d}/{num_updates}  step {global_step:9d}  "
                  f"return {ret:8.2f}  score {sc:7.1f}  dist {di:8.0f}  "
                  f"cells {float(np.mean(ep_cells)) if ep_cells else 0:5.1f}  "
                  f"{sps:6.0f} sps  "
                  f"roll {roll_time:5.2f}s opt {opt_time:5.2f}s")

        if update % args.save_every == 0 or update == num_updates - 1:
            blob = {"model": agent.state_dict(),
                    "optim": optimizer.state_dict(),
                    "update": update, "global_step": global_step,
                    "args": vars(args)}
            torch.save(blob, os.path.join(out_dir, "checkpoint.pt"))
            # Run 1 peaked at 7.1M steps and then collapsed to below its
            # starting return, so the final checkpoint was not the best
            # one -- and with a single overwritten file the best one was
            # gone. Keep it.
            if args.save_best and ep_returns:
                mean_ret = float(np.mean(ep_returns))
                if mean_ret > best_return:
                    best_return = mean_ret
                    blob["episodic_return"] = mean_ret
                    torch.save(blob, os.path.join(out_dir, "best.pt"))

    vec.close()
    writer.close()
    print(f"done; checkpoints and tensorboard logs in {out_dir}")


if __name__ == "__main__":
    main()
