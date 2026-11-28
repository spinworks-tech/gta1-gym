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
a tutorial line: *"Answer the South Park phones to get jobs."* The South
Park phones are **a row of four booths at y ≈ 8722, x ≈ 7855 / 7919 /
7983 / 8047, all live at game start** (red markers above each); the start
arrow points at them. Standing on one accepts its job (brief in the brief
channel), the markers vanish and the others go quiet. Jobs are not timed
and survive death. Positions are in `config/phones.json`.

The centre-channel message at (6100, 9575) — *"YOU WANT YOUR JOB AT
PIZZA SHED BACK?"* — is a **pager advert, not a phone**: no job starts.
`tools/find_phones.py` detects centre messages, so it finds pagers.

Job 1 (x 7855, "Crazy Jimmy's dumped a car…") has been completed end to
end — see *Mission state* below.

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
| `lives` | `0x8002C446` | `0x2C446` | uint16; 4 at start, verified 4→3 on death |
| `wanted` | `0x8002C358` | `0x2C358` | byte 0–4; write-verified against the HUD |
| `bonus_mult` | `0x8002C442` | `0x2C442` | uint16; 1 at start |
| `armor` | `0x8002C438` | `0x2C438` | uint16 |
| ammo pistol / mg+rocket / flame | `0x8002C451` / `C452-3` / `C454` | | bytes |
| `cheat_flags` | `0x80026D84` | `0x26D84` | bitmask, see below |
| brief text | `0x800444DC` | `0x444DC` | C string |
| brief on/off | `0x80019FF8` | `0x19FF8` | byte |
| brief kind | `0x8001A00C` | `0x1A00C` | uint32 |
| centre text | `0x800448C8` | `0x448C8` | C string |
| district banner | `0x800443CC` | `0x443CC` | C string |

Coordinates are 16.16 fixed point: `world = raw * 1.52587890625e-05`.
Observed play sits around x 5000–9000, y 7000–11000.

The map lives in `config/ram_map.json`; text addresses are in
`gta1env/text.py`.

### Mission state

| what | address | notes |
|---|---|---|
| `ctrl_x`, `ctrl_y` | `0x8002C304/08` int32 | position of whatever the player controls; keeps updating in cars |
| `mission_stage` | `0x8001A000` u8 | phase byte, **not monotonic** (see below) |
| `no_job` | `0x8001A068` u8 | 1 before the first job, 0 after |
| `jobs_done` | `0x8002C44D` u8 | 0 → 1 on completing job 1; candidate, one completion seen |
| job brief text | `0x80027FD6` | C string of the current job's brief |

Job 1 played end to end (savestates in `config/mission_states/`,
gitignored):

| event | `mission_stage` | score |
|---|---|---|
| start | 4 | 0 |
| accept at phone (7855, 8722) | 3 | 0 |
| enter Jimmy's car at (10726, 6237) | 2 | +5000 |
| drive it into Dulli's garage, east face of the block at (7396, 6697), front-first heading west | 3 | **+20000** |
| timed follow-up ("talk to Jimmy at the South West Park subway") expires after ~600 steps | 2 | — |

After that, three park phones are live: the finished job's phone is gone.
**Job pay arrives as score**, so the existing score term already rewards
finishing a job; nothing extra is needed to detect completion.

**`player_x/y` freeze while in a vehicle.** They are the on-foot
position; `ctrl_x/y` follow the car. The movement rewards in runs 1–2
read `player_x/y` — check this against the `enter_exit` exploit before
the next run.

### Moving a car (diagnostics only)

The car's authoritative position is in its object record, not the copies
next to it. For job 1's car: x `0x8001C258`, y `0x8001C25C` (16.16); writing
both relocates it. The heading copies (`0x8001C2DC` and six others, 1024
per turn, 256 = east, 512 = north) are read-only; to face a direction,
drive a circle and stop when the heading reads right. Placement inside a
building is refused (the car snaps back). The garage door message
("Wait for the garage door to open…") fires near (7560–7600, 6738).

### Still unknown

The objective's world position in RAM (the arrow's target); how
`jobs_done` behaves past one completion.

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
