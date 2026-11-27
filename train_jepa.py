#!/usr/bin/env python3
"""JEPA world model + imagination policy for GTA1 -- the "LeMario" recipe.

The PPO baseline (train_ppo.py) learns a policy straight from rewards.
This learns a *world model* first, from the game's own frames and no
labels, then trains the policy entirely inside that model:

  encoder      E(4 stacked frames) -> z in R^D
  predictor    P(z_t, a_t) -> z_{t+1}        (residual MLP + LayerNorm)
  reward head  R(z_t, a_t) -> (d score, distance, death)

  loss = mean_k ||P^k(z_t, a) - E(x_{t+k})||^2          free-running, K steps
       + lambda * SIGReg(z)                             LeJEPA anti-collapse
       + reward-head losses

No EMA target, no stop-gradient, no pixel reconstruction: SIGReg alone
keeps the latents from collapsing, by pushing every random 1-D projection
of the batch toward N(0, 1) (Epps-Pulley test on the characteristic
function). The policy pi(a|z) and value V(z) are then trained Dreamer-style
on H-step imagined rollouts, and the whole thing turns as a flywheel:

  collect (random, then policy + eps noise) -> world model from scratch on
  the cumulative corpus -> policy in imagination -> evaluate in the real game

Sources: Bai, "LeMario" (SIGReg: 1024 slices x 17 knots, lambda 0.1);
Mapel, "LeMario flywheel" (K=4 free-running, residual+LN predictor,
retrain from scratch each iteration, eval noise).

  .venv/bin/python train_jepa.py --envs 12 --iters 8 --run-name jepa1

Every stage writes into runs/<run-name>/iterNN/ and is skipped if already
done, so an interrupted run resumes by re-running the same command.
"""

import argparse
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from gta1env.env import GTA1Config
from gta1env.vec import SubprocVecEnv

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_JUMP = 1e6        # same guard as RewardShaper: a reload, not progress


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


# ===================================================================== model

class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regulariser (LeJEPA).

    Projects the batch onto `slices` random unit directions and, on each,
    compares the empirical characteristic function with that of N(0, 1)
    at `knots` points in [0, t_max], Gaussian-weighted (Epps-Pulley). Zero
    iff every projection looks standard normal, which rules out collapse
    without any target network.
    """

    def __init__(self, slices=1024, knots=17, t_max=3.0):
        super().__init__()
        self.slices = slices
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2.0 * dt)      # x2: the integrand is even
        w[[0, -1]] = dt                         # trapezoid ends
        phi = torch.exp(-0.5 * t ** 2)          # char. function of N(0,1)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("w", w * phi)

    def forward(self, z):
        """z: (..., N, D) -- statistic over N, averaged over leading dims."""
        n, d = z.shape[-2], z.shape[-1]
        a = torch.randn(d, self.slices, device=z.device)
        a = a / a.norm(dim=0, keepdim=True)
        x = (z @ a).unsqueeze(-1) * self.t                  # (..., N, M, K)
        err = ((x.cos().mean(-3) - self.phi) ** 2
               + x.sin().mean(-3) ** 2)                     # (..., M, K)
        return (err @ self.w).mean() * n


class Encoder(nn.Module):
    """CNN encoder. What matters for GTA1 is how many *native* screen
    pixels one first-layer step covers, against the size of the sprites
    (pedestrian ~7x9, arrow ~12x12, car ~20x42 on the 320x240 screen):

      nature  8x8/4, 4x4/2, 3x3/1   at 84x84:   ~15x11 native px per step
      fine    4x4/2, 3x3/2 x3       at 160x120: ~4x4   native px per step

    `model_hw` lets one dataset (stored at the env's obs size) feed
    encoders that work at a lower resolution, so variants can be compared
    on identical data.
    """

    def __init__(self, in_ch, hw, dim, width=1, kind="nature", model_hw=None):
        super().__init__()
        self.model_hw = tuple(model_hw) if model_hw else None
        hw = self.model_hw or hw
        c1, c2, c3 = 32 * width, 64 * width, 64 * width
        if kind == "nature":
            layers = [nn.Conv2d(in_ch, c1, 8, stride=4), nn.GELU(),
                      nn.Conv2d(c1, c2, 4, stride=2), nn.GELU(),
                      nn.Conv2d(c2, c3, 3, stride=1), nn.GELU()]
        elif kind == "fine":
            layers = [nn.Conv2d(in_ch, c1, 4, stride=2), nn.GELU(),
                      nn.Conv2d(c1, c2, 3, stride=2), nn.GELU(),
                      nn.Conv2d(c2, c3, 3, stride=2), nn.GELU(),
                      nn.Conv2d(c3, c3, 3, stride=2), nn.GELU()]
        else:
            raise ValueError(f"unknown encoder {kind!r}")
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, in_ch, *hw)).shape[1]
        # BatchNorm on the output, as in LeWorldModel's projector: SIGReg's
        # gradient scales with the latents themselves, so an encoder that
        # starts with tiny outputs never escapes collapse (seen with
        # `fine`: z_std 0.002, SIGReg stuck at ~26). BN pins the scale and
        # SIGReg shapes the rest.
        self.fc = nn.Sequential(nn.Linear(n, 512), nn.LayerNorm(512),
                                nn.GELU(), nn.Linear(512, dim),
                                nn.BatchNorm1d(dim))

    def forward(self, x):
        x = x.float().div(255.0)
        if self.model_hw and tuple(x.shape[-2:]) != self.model_hw:
            x = F.interpolate(x, size=self.model_hw, mode="area")
        return self.fc(self.conv(x))


class Predictor(nn.Module):
    """Residual MLP with pre-LayerNorm blocks, predicting z_{t+1} - z_t.

    Mapel's first unnormalised MLP silently collapsed to a constant output
    (dying GELUs after a gradient spike); this is the variant that held.
    The output layer starts at zero, so training begins from the
    "nothing changes" prediction rather than from noise.
    """

    def __init__(self, dim, n_actions, hidden=512, blocks=3, act_dim=64):
        super().__init__()
        self.act = nn.Embedding(n_actions, act_dim)
        self.inp = nn.Linear(dim + act_dim, hidden)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, hidden)) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, z, a):
        h = self.inp(torch.cat([z, self.act(a)], -1))
        for blk in self.blocks:
            h = h + blk(h)
        return z + self.out(h)


class RewardHead(nn.Module):
    """(z_t, a_t) -> [symlog d score, capped distance / cap, death logit]."""

    def __init__(self, dim, n_actions, hidden=256, act_dim=32):
        super().__init__()
        self.act = nn.Embedding(n_actions, act_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + act_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, z, a):
        return self.net(torch.cat([z, self.act(a)], -1))


class InverseHead(nn.Module):
    """(z_t, z_{t+1}) -> logits over a_t. An auxiliary loss on it forces
    the encoder to keep what the player controls; in GTA1 most on-screen
    change (traffic, camera zoom) is not that, and latent MSE alone ends
    up tracking it instead. Its accuracy is also the plainest check that
    actions are recoverable from the frames at all."""

    def __init__(self, dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, n_actions))

    def forward(self, z0, z1):
        return self.net(torch.cat([z0, z1 - z0], -1))


class WorldModel(nn.Module):
    def __init__(self, in_ch, hw, n_actions, args):
        super().__init__()
        mhw = ((args.model_height, args.model_width)
               if args.model_width else None)
        self.enc = Encoder(in_ch, hw, args.latent_dim, args.enc_width,
                           args.encoder, mhw)
        self.pred = Predictor(args.latent_dim, n_actions)
        self.head = RewardHead(args.latent_dim, n_actions)
        self.inv = InverseHead(args.latent_dim, n_actions)
        self.sigreg = SIGReg(args.sigreg_slices, args.sigreg_knots)
        self.dist_cap = args.max_step_distance

    def encode_window(self, win, stack):
        """win: (B, L, c, H, W) uint8 frames -> z: (B, L-stack+1, D)."""
        b, l = win.shape[:2]
        s = l - stack + 1
        stacks = torch.stack([win[:, i:i + stack].flatten(1, 2)
                              for i in range(s)], 1)       # (B, S, C, H, W)
        return self.enc(stacks.flatten(0, 1)).view(b, s, -1)

    def reward(self, z, a, scales):
        """Expected shaped reward and continuation, for imagination."""
        out = self.head(z, a)
        p_dead = torch.sigmoid(out[..., 2])
        r = (scales["score"] * symexp(out[..., 0])
             + scales["distance"] * out[..., 1].clamp(0, 1) * self.dist_cap
             - scales["death"] * p_dead)
        return r, 1.0 - p_dead


class Actor(nn.Module):
    def __init__(self, dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_actions))
        nn.init.zeros_(self.net[-1].weight)     # start uniform
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return Categorical(logits=self.net(z))


class Critic(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ==================================================================== replay

class Replay:
    """Trajectory segments kept whole in RAM, sampled as short windows.

    A segment holds T actions and T + stack frames: the first `stack`
    frames are the reset observation's own stack (no-op frames included),
    so the stack at step t is frames[t : t + stack] exactly as the agent
    saw it -- no zero-padding guesswork at episode starts.
    """

    def __init__(self, stack):
        self.stack = stack
        self.segs = {"train": [], "val": []}
        self.n = 0

    def add(self, seg):
        split = "val" if self.n % 20 == 19 else "train"
        self.n += 1
        self.segs[split].append(seg)

    def load_dir(self, path):
        for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
            with np.load(f) as d:
                self.add({k: d[k] for k in d.files})

    def frames(self, split=None):
        splits = [split] if split else ["train", "val"]
        return sum(len(s["actions"]) for sp in splits for s in self.segs[sp])

    def sample(self, batch, k, rng, split="train"):
        """Windows of k transitions: frames (B, stack+k, c, H, W) and the
        k actions / labels that connect them."""
        segs = self.segs[split] or self.segs["train"]
        n_start = np.array([max(0, len(s["actions"]) - k + 1) for s in segs],
                           dtype=np.float64)
        pick = rng.choice(len(segs), size=batch, p=n_start / n_start.sum())
        L = self.stack + k
        f0 = segs[0]["frames"]
        win = np.empty((batch, L) + f0.shape[1:], dtype=np.uint8)
        act = np.zeros((batch, k), dtype=np.int64)
        lab = np.zeros((batch, k, 4), dtype=np.float32)
        for b, i in enumerate(pick):
            s = segs[i]
            t = int(rng.integers(0, int(n_start[i])))
            win[b] = s["frames"][t:t + L]
            if k:
                sl = slice(t, t + k)
                act[b] = s["actions"][sl]
                lab[b, :, 0] = s["dscore"][sl]
                lab[b, :, 1] = s["dist"][sl]
                lab[b, :, 2] = s["dead"][sl]
                lab[b, :, 3] = s["mask"][sl]
        return win, act, lab


# =============================================================== environment

def run_episodes(vec, steps_per_env, act_fn, rng, eps, frame_c, stack,
                 respawn_distance, seed, hold=1.0):
    """Step all envs `steps_per_env` times. Returns (segments, episodes).

    act_fn(obs_uint8_numpy) -> actions, or None for uniform random. With
    act_fn, about a fraction eps of steps are exploratory (the flywheel's
    collection noise).

    Random actions are held for a geometric number of steps with mean
    `hold`. Measured on the real game: re-drawing every step (67 ms) leaves
    the player on foot and still 68% of the time, moving ~1 native pixel
    per step -- the actions' effect drowns in traffic and no world model
    or inverse-dynamics probe could find it (val accuracy = chance).
    """
    n, na = vec.num_envs, vec.num_actions
    obs, infos = vec.reset(seeds=[seed + i for i in range(n)])

    def new_seg(o, info):
        prev = None
        if info is not None and "player_x" in info:
            prev = (info.get("score", 0), info["player_x"], info["player_y"])
        return {"frames": [o[j * frame_c:(j + 1) * frame_c].copy()
                           for j in range(stack)],
                "actions": [], "dscore": [], "dist": [], "dead": [],
                "mask": [], "prev": prev}

    cur = [new_seg(obs[i], infos[i]) for i in range(n)]
    segments, episodes = [], []
    held = np.zeros(n, dtype=np.int64)
    left = np.zeros(n, dtype=np.int64)        # steps left on a held action
    p_start = 1.0 if act_fn is None else min(1.0, eps / max(hold, 1.0))

    def close(i, complete):
        s = cur[i]
        seg = {"frames": np.stack(s["frames"]),
               "actions": np.asarray(s["actions"], dtype=np.int16),
               "dscore": np.asarray(s["dscore"], dtype=np.float32),
               "dist": np.asarray(s["dist"], dtype=np.float32),
               "dead": np.asarray(s["dead"], dtype=np.float32),
               "mask": np.asarray(s["mask"], dtype=np.float32),
               "complete": np.asarray(complete)}
        if len(seg["actions"]):
            segments.append(seg)
        if complete:
            episodes.append({"score": float(seg["dscore"].sum()),
                             "distance": float(seg["dist"].sum()),
                             "length": int(len(seg["actions"])),
                             "died": bool(seg["dead"][-1]) if
                             len(seg["dead"]) else False,
                             "seg": len(segments) - 1})

    for _ in range(steps_per_env):
        a = (np.zeros(n, dtype=np.int64) if act_fn is None
             else np.asarray(act_fn(obs)))
        start = (left == 0) & (rng.random(n) < p_start)
        if start.any():
            k = int(start.sum())
            held[start] = rng.integers(na, size=k)
            left[start] = rng.geometric(1.0 / max(hold, 1.0), size=k)
        a = np.where(left > 0, held, a)
        left = np.maximum(left - 1, 0)
        obs, _, terms, truncs, infos = vec.step(a)
        for i in range(n):
            s, info = cur[i], infos[i]
            s["actions"].append(int(a[i]))
            now = None
            if "player_x" in info:
                now = (info.get("score", 0), info["player_x"],
                       info["player_y"])
            ds = dd = 0.0
            ok = now is not None and s["prev"] is not None
            if ok:
                ds = now[0] - s["prev"][0]
                if abs(ds) >= SCORE_JUMP:
                    ds = 0.0
                dd = math.hypot(now[1] - s["prev"][1], now[2] - s["prev"][2])
                if respawn_distance and dd >= respawn_distance:
                    dd = 0.0            # respawn teleport, not travel
            s["prev"] = now
            s["dscore"].append(ds)
            s["dist"].append(dd)
            s["mask"].append(float(ok))
            ended = terms[i] or truncs[i]
            s["dead"].append(float(terms[i]))
            src = info["final_obs"] if ended else obs[i]
            s["frames"].append(src[-frame_c:].copy())
            if ended:
                close(i, True)
                cur[i] = new_seg(obs[i], None)
    for i in range(n):
        close(i, False)
    return segments, episodes


def summarize(episodes, args):
    """Score and distance kept separate: shaped return alone hid that the
    naive PPO run's gain was ~80% the movement bonus."""
    if not episodes:
        return {"episodes": 0}
    sc = np.array([e["score"] for e in episodes])
    di = np.array([e["distance"] for e in episodes])
    out = {"episodes": len(episodes),
           "score_mean": float(sc.mean()), "score_std": float(sc.std()),
           "score_max": float(sc.max()),
           "distance_mean": float(di.mean()), "distance_std": float(di.std()),
           "deaths": int(sum(e["died"] for e in episodes)),
           "length_mean": float(np.mean([e["length"] for e in episodes]))}
    return out


def policy_fn(wm, actor, device):
    @torch.no_grad()
    def act(obs):
        z = wm.enc(torch.as_tensor(np.ascontiguousarray(obs), device=device))
        return actor(z).sample().cpu().numpy()
    return act


# ============================================================ world model

def wm_batch(replay, args, rng, device, split="train"):
    win, act, lab = replay.sample(args.wm_batch, args.horizon_k, rng, split)
    return (torch.as_tensor(win, device=device),
            torch.as_tensor(act, device=device),
            torch.as_tensor(lab, device=device))


def wm_losses(wm, win, act, lab, args):
    z = wm.encode_window(win, args.frame_stack)           # (B, K+1, D)
    zh, states, pred = z[:, 0], [], 0.0
    for k in range(args.horizon_k):
        states.append(zh)
        zh = wm.pred(zh, act[:, k])                       # free-running
        pred = pred + F.mse_loss(zh, z[:, k + 1])
    pred = pred / args.horizon_k
    sig = wm.sigreg(z.transpose(0, 1))                    # per time step
    # Heads see encoded AND predicted latents, so they stay valid inside
    # imagination, where only predicted ones exist.
    out = wm.head(torch.stack(states, 1).flatten(0, 1), act.flatten())
    lab = lab.flatten(0, 1)
    m = lab[:, 3]
    l_score = ((out[:, 0] - symlog(lab[:, 0])) ** 2 * m).sum() / m.sum().clamp(min=1)
    dist_t = lab[:, 1].clamp(max=args.max_step_distance) / args.max_step_distance
    l_dist = ((out[:, 1] - dist_t) ** 2 * m).sum() / m.sum().clamp(min=1)
    l_dead = F.binary_cross_entropy_with_logits(
        out[:, 2], lab[:, 2], pos_weight=torch.tensor(args.death_pos_weight,
                                                      device=out.device))
    head = l_score + l_dist + l_dead
    zi = z if args.inv_coef > 0 else z.detach()     # 0: probe only
    logits = wm.inv(zi[:, :-1].flatten(0, 1), zi[:, 1:].flatten(0, 1))
    l_inv = F.cross_entropy(logits, act.flatten())
    inv_acc = (logits.argmax(-1) == act.flatten()).float().mean()
    loss = (pred + args.sigreg_lambda * sig + args.head_coef * head
            + (args.inv_coef or 1.0) * l_inv)
    return loss, {"pred": pred.item(), "sigreg": sig.item(),
                  "score": l_score.item(), "dist": l_dist.item(),
                  "dead": l_dead.item(), "inv": l_inv.item(),
                  "inv_acc": inv_acc.item()}, z


@torch.no_grad()
def wm_diagnostics(wm, replay, args, rng, device):
    """The checks both write-ups say low loss alone cannot replace.

    pred/copy < 1      the model beats "nothing changes" (Bai: persistence
                       is a strong baseline, especially at 1 step)
    shuffled/pred > 1  predictions actually use the actions
    act_sens           spread of P(z, a) over all actions, relative to
                       latent scale: ~0 means a silently collapsed
                       predictor (Mapel's V1 failure)
    """
    wm.eval()
    win, act, lab = wm_batch(replay, args, rng, device, "val")
    z = wm.encode_window(win, args.frame_stack)
    def roll(actions):
        zh, err = z[:, 0], []
        for k in range(args.horizon_k):
            zh = wm.pred(zh, actions[:, k])
            err.append(F.mse_loss(zh, z[:, k + 1]).item())
        return err
    pred = roll(act)
    shuf = roll(act[torch.randperm(act.shape[0], device=device)])
    copy = [F.mse_loss(z[:, 0], z[:, k + 1]).item()
            for k in range(args.horizon_k)]
    na = wm.pred.act.num_embeddings
    z0 = z[:, 0]
    allp = torch.stack([wm.pred(z0, torch.full((z0.shape[0],), a,
                                               device=device))
                        for a in range(na)], 0)          # (A, B, D)
    sens = (allp.std(0).mean() / z0.std(0).mean().clamp(min=1e-8)).item()
    out = wm.head(z0, act[:, 0])
    dead = lab[:, 0, 2]
    # Reward-head losses on fixed targets are the only numbers here that
    # compare across encoders/resolutions (latent spaces differ).
    vals = [wm_losses(wm, *wm_batch(replay, args, rng, device, "val"),
                      args)[1] for _ in range(args.val_batches)]
    wm.train()
    return {**{f"val_{k}": float(np.mean([v[k] for v in vals]))
               for k in ("score", "dist", "dead", "inv_acc")},
            "pred_1": pred[0], "pred_K": pred[-1],
            "copy_1": copy[0], "copy_K": copy[-1],
            "pred_over_copy_K": pred[-1] / max(copy[-1], 1e-12),
            "shuffled_over_pred_K": shuf[-1] / max(pred[-1], 1e-12),
            "act_sens": sens,
            "z_std": z0.std(0).mean().item(),
            "z_mean_abs": z0.mean(0).abs().mean().item(),
            "p_dead_mean": torch.sigmoid(out[:, 2]).mean().item(),
            "dead_rate": dead.mean().item()}


def train_world_model(wm, replay, args, rng, device, writer, it):
    opt = torch.optim.AdamW(wm.parameters(), lr=args.wm_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.wm_lr, total_steps=args.wm_steps, pct_start=0.05)
    base = it * args.wm_steps
    t0 = time.perf_counter()
    for step in range(args.wm_steps):
        win, act, lab = wm_batch(replay, args, rng, device)
        loss, parts, _ = wm_losses(wm, win, act, lab, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(wm.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        if step % 100 == 0:
            for k, v in parts.items():
                writer.add_scalar(f"wm/{k}", v, base + step)
            writer.add_scalar("wm/grad_norm", gn.item(), base + step)
        if step % args.diag_every == 0 or step == args.wm_steps - 1:
            d = wm_diagnostics(wm, replay, args, rng, device)
            for k, v in d.items():
                writer.add_scalar(f"wm_diag/{k}", v, base + step)
            rate = (step + 1) / (time.perf_counter() - t0)
            print(f"  wm {step:6d}/{args.wm_steps}  pred {parts['pred']:.4f} "
                  f"sig {parts['sigreg']:.3f}  pred/copy@K "
                  f"{d['pred_over_copy_K']:.3f}  shuf/pred@K "
                  f"{d['shuffled_over_pred_K']:.3f}  inv_acc "
                  f"{d['val_inv_acc']:.3f}  act_sens "
                  f"{d['act_sens']:.4f}  z_std {d['z_std']:.3f}  "
                  f"{rate:.1f} it/s", flush=True)
    return wm_diagnostics(wm, replay, args, rng, device)


# =================================================== policy in imagination

def train_actor_critic(wm, replay, args, rng, device, writer, it, na,
                       actor=None, critic=None):
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    d = args.latent_dim
    actor = actor or Actor(d, na).to(device)
    critic = critic or Critic(d).to(device)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=1e-5)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
    scales = {"score": args.score_scale, "distance": args.distance_scale,
              "death": args.death_penalty}
    ret_scale = None
    base = it * args.ac_steps
    for step in range(args.ac_steps):
        win, _, _ = replay.sample(args.ac_batch, 0, rng)
        with torch.no_grad():
            z = wm.encode_window(torch.as_tensor(win, device=device),
                                 args.frame_stack)[:, 0]
        zs, logps, ents, rews, conts = [z], [], [], [], []
        for _ in range(args.imag_horizon):
            pi = actor(z)
            a = pi.sample()
            logps.append(pi.log_prob(a))
            ents.append(pi.entropy())
            with torch.no_grad():
                r, c = wm.reward(z, a, scales)
                z = wm.pred(z, a)
            rews.append(r)
            conts.append(c)
            zs.append(z)
        zs = torch.stack(zs)                                   # (H+1, B, D)
        rews, conts = torch.stack(rews), torch.stack(conts)    # (H, B)
        logps, ents = torch.stack(logps), torch.stack(ents)
        v = critic(zs)                                         # (H+1, B)
        with torch.no_grad():
            vd = v.detach()
            ret = torch.empty_like(rews)
            nxt = vd[-1]
            for h in reversed(range(args.imag_horizon)):
                nxt = rews[h] + args.gamma * conts[h] * (
                    (1 - args.lam) * vd[h + 1] + args.lam * nxt)
                ret[h] = nxt
            # Weight later steps by the chance of still being alive.
            w = torch.cumprod(torch.cat([torch.ones_like(conts[:1]),
                                         args.gamma * conts[:-1]]), 0)
            # DreamerV3 return normalisation: scale-free advantages, but
            # never amplify small returns (the floor).
            spread = (torch.quantile(ret.flatten(), 0.95)
                      - torch.quantile(ret.flatten(), 0.05))
            ret_scale = spread if ret_scale is None else (
                0.99 * ret_scale + 0.01 * spread)
            adv = (ret - vd[:-1]) / max(args.ret_floor, ret_scale.item())
        actor_loss = -(w * (logps * adv + args.ent_coef * ents)).mean()
        critic_loss = 0.5 * (w * (v[:-1] - ret) ** 2).mean()
        a_opt.zero_grad(set_to_none=True)
        c_opt.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        a_opt.step()
        c_opt.step()
        if step % 100 == 0:
            s = base + step
            writer.add_scalar("ac/actor_loss", actor_loss.item(), s)
            writer.add_scalar("ac/critic_loss", critic_loss.item(), s)
            writer.add_scalar("ac/entropy", ents.mean().item(), s)
            writer.add_scalar("ac/imag_return", ret[0].mean().item(), s)
            writer.add_scalar("ac/imag_reward", rews.mean().item(), s)
            writer.add_scalar("ac/ret_scale", ret_scale.item(), s)
        if step % 1000 == 0 or step == args.ac_steps - 1:
            print(f"  ac {step:6d}/{args.ac_steps}  imag_ret "
                  f"{ret[0].mean().item():.4f}  r/step "
                  f"{rews.mean().item():.5f}  entropy "
                  f"{ents.mean().item():.3f}  (max {math.log(na):.3f})",
                  flush=True)
    for p in wm.parameters():
        p.requires_grad_(True)
    return actor, critic


# ===================================================================== main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    # flywheel
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--init-steps", type=int, default=300_000,
                    help="random-policy frames before the first model")
    ap.add_argument("--iter-steps", type=int, default=100_000,
                    help="frames collected by each iteration's policy")
    ap.add_argument("--collect-eps", type=float, default=0.2)
    ap.add_argument("--hold", type=float, default=8.0,
                    help="mean steps a random/exploratory action is held "
                         "(geometric); 1 = re-draw every step")
    ap.add_argument("--eval-steps", type=int, default=None,
                    help="steps per env per evaluation; default one full "
                         "episode plus no-op slack")
    ap.add_argument("--warm-start", action="store_true",
                    help="initialise each iteration from the previous one "
                         "(Mapel found this falls into bad basins; off by "
                         "default)")
    # world model
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--enc-width", type=int, default=1)
    ap.add_argument("--encoder", default="nature", choices=("nature", "fine"))
    ap.add_argument("--model-width", type=int, default=None,
                    help="resize stored frames to this before encoding "
                         "(default: use the env's obs size as is)")
    ap.add_argument("--model-height", type=int, default=None)
    ap.add_argument("--horizon-k", type=int, default=4,
                    help="free-running prediction steps in the loss")
    ap.add_argument("--wm-steps", type=int, default=20_000)
    ap.add_argument("--wm-batch", type=int, default=64)
    ap.add_argument("--wm-lr", type=float, default=3e-4)
    ap.add_argument("--sigreg-lambda", type=float, default=0.1)
    ap.add_argument("--sigreg-slices", type=int, default=1024)
    ap.add_argument("--sigreg-knots", type=int, default=17)
    ap.add_argument("--head-coef", type=float, default=1.0)
    ap.add_argument("--inv-coef", type=float, default=0.0,
                    help="inverse-dynamics auxiliary loss weight. 0 = pure "
                         "JEPA: the head then trains on detached latents, "
                         "as a probe that does not shape the encoder")
    ap.add_argument("--death-pos-weight", type=float, default=20.0,
                    help="deaths are ~1 step in 50k; upweight positives")
    ap.add_argument("--diag-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # actor-critic in imagination
    ap.add_argument("--ac-steps", type=int, default=10_000)
    ap.add_argument("--ac-batch", type=int, default=256)
    ap.add_argument("--imag-horizon", type=int, default=15)
    ap.add_argument("--actor-lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=1e-3)
    ap.add_argument("--ret-floor", type=float, default=1.0)
    # reward used in imagination -- applied to the heads' outputs, so it
    # can change between iterations without recollecting anything
    ap.add_argument("--score-scale", type=float, default=0.01)
    ap.add_argument("--distance-scale", type=float, default=0.001)
    ap.add_argument("--death-penalty", type=float, default=1.0)
    ap.add_argument("--max-step-distance", type=float, default=50.0)
    # environment (same meaning and defaults as train_ppo.py)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--obs-width", type=int, default=84)
    ap.add_argument("--obs-height", type=int, default=84)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--color", dest="grayscale", action="store_false",
                    default=True)
    ap.add_argument("--sticky-prob", type=float, default=0.05)
    ap.add_argument("--max-episode-steps", type=int, default=4500)
    ap.add_argument("--core-path", default=None)
    ap.add_argument("--rom", default=None)
    ap.add_argument("--ram-map", default=None)
    ap.add_argument("--boot-state", default=None)
    ap.add_argument("--bios-mode", default="auto", choices=("auto", "HLE"))
    ap.add_argument("--skip-asset-check", action="store_true",
                    help="only for the mock core")
    return ap.parse_args()


def env_kwargs(args):
    kw = {"frameskip": args.frameskip, "sticky_prob": args.sticky_prob,
          "episode_steps": args.max_episode_steps,
          "obs_width": args.obs_width, "obs_height": args.obs_height,
          "frame_stack": args.frame_stack, "grayscale": args.grayscale,
          "bios_mode": args.bios_mode}
    for key, value in (("core_path", args.core_path), ("rom_path", args.rom),
                       ("ram_map_path", args.ram_map),
                       ("boot_state", args.boot_state)):
        if value:
            kw[key] = os.path.abspath(value)
    if args.skip_asset_check:
        kw["require_assets"] = False
    return kw


def save_segments(segments, path):
    os.makedirs(path, exist_ok=True)
    for k, seg in enumerate(segments):
        np.savez(os.path.join(path, f"seg{k:04d}.npz"), **seg)
    open(os.path.join(path, "DONE"), "w").close()


def main():
    args = parse_args()
    run = args.run_name or f"gta1_jepa_{int(time.time())}"
    out = os.path.join(HERE, "runs", run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    writer = SummaryWriter(out)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    kw = env_kwargs(args)
    probe = GTA1Config(**kw)
    frame_c = 1 if probe.grayscale else 3
    in_ch = probe.frame_stack * frame_c
    hw = (probe.obs_height, probe.obs_width)
    eval_steps = args.eval_steps or (args.max_episode_steps
                                     + probe.noop_max + 1)
    vec = SubprocVecEnv(args.envs, cfg_kwargs=kw, base_seed=args.seed)
    na = vec.num_actions
    print(f"{args.envs} emulators up; {na} actions; run dir {out}")

    def collect(path, act_fn, steps, eps, seed):
        if os.path.exists(os.path.join(path, "DONE")):
            return None
        t0 = time.perf_counter()
        segs, eps_ = run_episodes(
            vec, math.ceil(steps / args.envs), act_fn, rng, eps, frame_c,
            probe.frame_stack, probe.respawn_distance, seed, args.hold)
        save_segments(segs, path)
        n = sum(len(s["actions"]) for s in segs)
        print(f"collected {n} frames in {time.perf_counter() - t0:.0f}s "
              f"-> {path}")
        return summarize(eps_, args)

    replay = Replay(probe.frame_stack)
    data0 = os.path.join(out, "data", "iter00")
    stats = collect(data0, None, args.init_steps, 0.0, args.seed * 1000)
    if stats is not None:
        print("random-policy baseline:", stats)
        with open(os.path.join(out, "random_baseline.json"), "w") as fh:
            json.dump(stats, fh, indent=1)
    replay.load_dir(data0)

    wm = actor = critic = None
    best = float("-inf")
    for it in range(args.iters):
        d = os.path.join(out, f"iter{it:02d}")
        os.makedirs(d, exist_ok=True)
        ck = os.path.join(d, "model.pt")
        print(f"\n=== iteration {it}: {replay.frames()} frames "
              f"({len(replay.segs['train'])} train / "
              f"{len(replay.segs['val'])} val segments) ===")
        if not args.warm_start or wm is None:
            wm = WorldModel(in_ch, hw, na, args).to(device)
            actor = critic = None
        if os.path.exists(ck):
            blob = torch.load(ck, map_location=device, weights_only=False)
            wm.load_state_dict(blob["wm"])
            actor = Actor(args.latent_dim, na).to(device)
            critic = Critic(args.latent_dim).to(device)
            actor.load_state_dict(blob["actor"])
            critic.load_state_dict(blob["critic"])
            print(f"loaded {ck}")
        else:
            t0 = time.perf_counter()
            diag = train_world_model(wm, replay, args, rng, device, writer, it)
            print(f"world model done in {time.perf_counter() - t0:.0f}s:",
                  {k: round(v, 4) for k, v in diag.items()})
            if (diag["shuffled_over_pred_K"] < 1.02
                    or diag["pred_over_copy_K"] > 0.95):
                print("WARNING: the world model barely beats 'nothing "
                      "changes' or ignores the actions (shuffled/pred "
                      f"{diag['shuffled_over_pred_K']:.3f}, pred/copy "
                      f"{diag['pred_over_copy_K']:.3f}); a policy trained "
                      "in it will learn little")
            t0 = time.perf_counter()
            actor, critic = train_actor_critic(
                wm, replay, args, rng, device, writer, it, na,
                actor if args.warm_start else None,
                critic if args.warm_start else None)
            print(f"actor-critic done in {time.perf_counter() - t0:.0f}s")
            torch.save({"wm": wm.state_dict(), "actor": actor.state_dict(),
                        "critic": critic.state_dict(), "wm_diag": diag,
                        "args": vars(args), "num_actions": na}, ck)

        ev_path = os.path.join(d, "eval.json")
        if os.path.exists(ev_path):
            with open(ev_path) as fh:
                ev = json.load(fh)
        else:
            act = policy_fn(wm, actor, device)
            _, episodes = run_episodes(
                vec, eval_steps, act, rng, 0.0, frame_c, probe.frame_stack,
                probe.respawn_distance, 10_000 + it)
            ev = summarize(episodes, args)
            with open(ev_path, "w") as fh:
                json.dump(ev, fh, indent=1)
        print(f"eval iter {it}:", ev)
        if ev.get("episodes"):
            for k in ("score_mean", "distance_mean", "deaths"):
                writer.add_scalar(f"eval/{k}", ev[k], it)
            ret = (args.score_scale * ev["score_mean"]
                   + args.distance_scale * ev["distance_mean"])
            if ret > best:
                best = ret
                shutil.copyfile(ck, os.path.join(out, "best.pt"))

        if it < args.iters - 1:
            path = os.path.join(out, "data", f"iter{it + 1:02d}")
            stats = collect(path, policy_fn(wm, actor, device),
                            args.iter_steps, args.collect_eps,
                            args.seed * 1000 + it + 1)
            if stats is not None:
                print(f"collection stats (eps={args.collect_eps}):", stats)
            replay.load_dir(path)

    vec.close()
    writer.close()
    print(f"done; models, data and tensorboard logs in {out}")


if __name__ == "__main__":
    main()
