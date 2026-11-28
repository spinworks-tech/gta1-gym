# CLAUDE.md

Working notes for this repo: what the game is, what is known about its
memory, and which mistakes have already been made so they are not made
again. Read `README.md` for setup and `docs/ram-writes.md` for the
probing technique.

## What this project is

The actual 1997 *Grand Theft Auto* (PlayStation) run as a parallel RL
environment, headless, on Linux + CUDA. A pure-`ctypes` libretro frontend
drives the `pcsx_rearmed` core; rewards are read out of PS1 main RAM; PPO
trains on the pixels.

No game data ships with the repo — the user supplies their own disc image
and BIOS. Savestates embed copyrighted game RAM and are gitignored
(`config/boot.state`, `config/boot_states/`, `recordings/`, `media/`).

## The game

Top-down, open-world, Liberty City. The things that matter for RL:

**Missions come from phones.** You start on foot with no objective except
a tutorial line: *"Answer the South Park phones to get jobs."* Walking
into a live phone offers a job; completing it pays and advances the
chapter. **Phones activate one at a time** — at game start exactly one is
live in the whole district, so finding only a single phone in a map sweep
is the correct answer, not a bug.

**Score is the HUD dollar figure.** It rises from jobs, and also from
incidental carnage — running people over, explosions, vehicle damage.
Both trained runs so far earned essentially all their score the second
way. Scoring events are rare and enormously skewed: over 128 episodes the
median was 400 and the best 4200, and in that best episode 4000 of the
4200 points arrived inside a 140-step window.

**The yellow objective arrow** points at the current target. It is RGB
(238, 231, 43), luminance 212, and survives the grayscale downscale
easily — +113 median contrast at 128×96, never below +33. It is the
brightest object in the frame. If an agent ignores the objective, that is
not a perception problem.

**Districts** announce themselves with an on-screen banner (Coach,
Jugular, Bulldog, Bus, Stinger, Southwest Park…), readable from RAM.

**`enter_exit` is one toggle button** for both entering and leaving a
vehicle. This matters more than it sounds — see the reward-hacking note
below.

## RAM

PS1 main RAM is 2 MiB, exposed as a **writable** zero-copy numpy view at
`env.ram`. PS1 addresses mirror, so mask with `& 0x001FFFFF` — or just
subtract the `0x80000000` base.

### Known addresses

| what | address | offset | type |
|---|---|---|---|
| `player_x` | `0x8002C96C` | `0x2C96C` | int32, 16.16 fixed point |
| `player_y` | `0x8002C970` | `0x2C970` | int32, 16.16 fixed point |
| `score` | `0x8002C2F0` | `0x2C2F0` | uint32, HUD dollars |
| brief text | `0x800444DC` | `0x444DC` | C string |
| brief on/off | `0x80019FF8` | `0x19FF8` | byte |
| brief kind | `0x8001A00C` | `0x1A00C` | uint32 |
| centre text | `0x800448C8` | `0x448C8` | C string |
| district banner | `0x800443CC` | `0x443CC` | C string |

Coordinates are 16.16 fixed point: `world = raw * 1.52587890625e-05`.
Observed play sits around x 5000–9000, y 7000–11000.

The map lives in `config/ram_map.json`; text addresses are in
`gta1env/text.py`.

### Still unknown

**Lives and wanted level.** Not located. Death is therefore inferred by a
heuristic: a position jump of `respawn_distance` (1500 world units) in one
step is treated as a respawn. Finding the real addresses needs a scripted
death — drive into water — snapshotted with `tools/drive.py` and diffed
with `tools/ram_scan.py`.

### RAM is writable, and that is a research tool

Writing `player_x`/`player_y` genuinely relocates the player — the scene
changes and the player walks on from the new position. This is how
`tools/find_phones.py` located the first phone at **(6100, 9575)** (zone
x 6000–6200, y 9450–9700) after RAM scanning failed outright: 88,499
constant int32 pairs survive a plausible-coordinate filter.

Use it for diagnostics and for generating start states, **not during
training** — teleporting breaks position continuity, which both the
distance reward and the death heuristic depend on. Full method and
pitfalls: `docs/ram-writes.md`.

## Environment conventions

- **Episode**: starts from a savestate; `terminated` on inferred death,
  `truncated` at `episode_steps` (default 4500). Frameskip 4, so one
  agent step is 4 emulated frames and 4500 steps ≈ 5 minutes of game time.
- **Observation**: grayscale, 4-frame stack. Default 84×84; **128×96 is
  the better choice** — 4:3 so the aspect ratio is not squashed, and 1.7×
  the pixels costs only 13% throughput.
- **Actions**: 14 discrete combinations; index 12 is `enter_exit`.
- **Determinism**: the core is deterministic, all threading disabled.
  Exploration diversity comes from `noop_max` and `sticky_prob`, not the
  emulator.
- **One core per process.** libretro cores hold global state, so N envs
  means N processes (`SubprocVecEnv`), never N objects.

## Hard-won lessons — do not rediscover these

**Every movement reward so far has been hacked.** Run 1 wedged against
the waterfront and vibrated in place, earning 746 units of "travel" for 15
of net displacement. Run 2 learned to spam `enter_exit`: 34.7% of its
actions, earning 46% of its credited distance at 3.53 units per press
versus 2.48 for actually driving — toggling in and out of a vehicle
teleports the sprite and the shaper pays for it. Before trusting any score
improvement, check the action histogram and per-action distance
attribution.

**Annealing a hackable reward to zero does not unlearn it.** The
behaviour persists as a local optimum because nothing penalises it
afterwards. Prefer not shipping a hackable term to annealing one.

**At ~0.002 per-step reward, `ent-coef` is a dominant term, not a
regulariser.** `--ent-coef 0.02` pinned the policy at 93% of uniform and
it learned nothing in 2.5M steps. 0.01 is near the ceiling.

**Score is heavy-tailed; never compare means over few episodes.** One
episode in 40 scored 30× the median. An 8-episode eval reported policy
931 vs random 587; a 40-episode eval of the same checkpoint reported the
reverse. Use medians and quartiles, and at least ~40 episodes per arm.

**Seed the action sampler.** `agent.act()` samples; unseeded, two
evaluations of one checkpoint disagree on which arm won.

**`best.pt` is unreliable on annealed runs.** Selection ranks on the
shaped return, which annealing makes non-stationary, so it freezes early
and never updates. It should rank on `episodic_score`. Still unfixed.

**Beware state leaking between probes.** When sweeping the game for
anything, reset between probes and calibrate settle time against a
known-positive point. Both traps produced confidently wrong answers here
— details in `docs/ram-writes.md`.

## Where things are

```
gta1env/      env.py (Gym env), ram.py (reward shaper), text.py,
              vec.py (SubprocVecEnv), libretro.py (ctypes frontend)
train_ppo.py  PPO (CleanRL-style); train_jepa.py is the model-based path
tools/        drive.py, ram_scan.py, bench.py, eval_policy.py,
              record_policy.py, find_phones.py, play.py, replay.py
learning/     one folder per run that was actually executed, with its
              config, result and failure modes
docs/         ram-writes.md
tests/        test_pipeline.py — run with .venv/bin/python (no pytest)
```

`TODO.md` is local-only and gitignored. The README is for public users —
keep internal deliberation out of it.

## Current state

Run 2 finished: 10M steps, score median 500 vs random's 300 (1.67×,
p=0.001), but a third of its actions are the `enter_exit` exploit and it
has never started a mission. Written up in `learning/run2/`.

**Do not tune the current reward function further.** The next run needs
the objective loop: shape toward the known phone coordinates and set
`mission_complete_reward`, which defaults to 0.0 and has never been set.
