# Run 2: 4:3 observations, annealed movement reward, grid novelty

The run that fixed the naive baseline's reward hacking, learned a new
exploit instead, and in the process turned up the structural problem
underneath both: **the agent has never started a mission, and nothing in
the reward function asks it to.**

Verdict up front: the score gain is real but small, the policy is
spending a third of its actions on a button exploit, and no amount of
further tuning on this reward function will produce an agent that plays
GTA. The next run needs an objective, not a hyperparameter.

## What this approach was

Four changes on top of [naive/](../naive/):

| change | why |
|---|---|
| 128×96 observations, 4:3 | 84×84 squashes the aspect ratio and throws away detail; measured 22% high-frequency retention vs 37% at 160×120 |
| `distance_scale` 0.001 → 0.0 over the first 5M steps | the naive run earned ~80% of its return from the movement term rather than the game |
| grid novelty, cell 64, scale 0.05 | pays once per world cell per episode, so jitter earns nothing and only going somewhere new pays |
| `--ent-coef 0.01` | the naive run collapsed to entropy 0.707 late |

A fifth change — sampling varied start states — was **not delivered**.
Rollout cannot produce them: random play gives a median pairwise spread
of 188 world units and the trained policy gives 16, with several
byte-identical positions. Run 2 therefore trained from the single
`config/boot.state`, which is a plausible contributor to the plateau.

## Command

```bash
.venv/bin/python -u train_ppo.py \
    --envs 12 --total-steps 10000000 \
    --obs-width 128 --obs-height 96 \
    --distance-scale 0.001 --distance-scale-final 0.0 --reward-anneal-frac 0.5 \
    --novelty-scale 0.05 --novelty-cell 64 \
    --ent-coef 0.01 \
    --run-name gta1_run2
```

10M steps, 770 steps/s, **3 h 36 min** on a Xeon E5-2690 / GTX 1050 Ti.

## An aborted first attempt: `--ent-coef 0.02`

Stopped at 2.5M steps. Entropy sat at 2.46 against the ln(14) = 2.639
maximum — 93% of uniform — and score was 366 against random's 353. Per-step
rewards here are ~0.002, so the advantages are tiny and a doubled entropy
bonus simply outweighs the policy gradient, pinning the distribution
uniform. Preserved as `runs/gta1_run2_entropy020/`.

**Lesson:** when per-step reward is this small, the entropy coefficient is
not a mild regulariser — it is the dominant term. 0.01 was already near
the ceiling.

## Result: a real but small gain

40 episodes per arm, matched conditions, sampler seeded:

| | median | q25–q75 | mean | zero-score episodes |
|---|---|---|---|---|
| random | 300 | 175–400 | 328 | 4 / 40 |
| **policy** | **500** | 200–700 | 889 | **0 / 40** |

Median 1.67× random; the policy beats a random episode 70% of the time;
rank test z = 3.04, one-sided p = 0.0012. Trimmed of each arm's single
top outlier the means are 512 vs 293.

Crucially this held **after `distance_scale` reached zero at 5M steps**.
Block-mean score rose 473 (0–1M) to ~1000–1400 and stayed there, with the
movement term switched off. That is the naive run's failure mode fixed:
the gain is game score, not farmed travel.

It also plateaued. Score stopped improving around 3–4M and then
oscillated between 722 and 1386 for the remaining 6M steps.

## The new exploit: `enter_exit` pays better than driving

Action 12 is `enter_exit`, a single toggle button. The trained policy
presses it **34.7% of the time** — the most common action of all, about
five presses a second. Two actions account for two-thirds of behaviour:

| action | share of steps |
|---|---|
| `enter_exit` | 34.7% |
| `forward + fire` | 32.2% |
| the other 12 actions | 33.1% |

Entropy at the end is 1.016 of 2.639, so this is a committed policy, not
noise. Attributing credited distance to the action that earned it:

| action | steps | mean displacement | share of total distance |
|---|---|---|---|
| **`enter_exit`** | 1560 | **3.53** | **46.0%** |
| `forward + fire` | 1451 | 2.48 | 30.0% |

Toggling in and out of a vehicle teleports the player sprite between the
pavement and the driver's seat, and the shaper credits each toggle as
travel — **more per step than actually driving**. Nearly half the movement
reward was earned standing next to a car mashing one button.

This is the naive run's path-length exploit in a new costume. Grid
novelty did close the original jitter loophole, but `distance_scale` was
live for the first 5M steps and taught this; annealing it to zero removed
the payment without removing the habit, because nothing penalises it.

**Lesson:** annealing a hackable reward to zero does not unlearn what it
taught. The behaviour persists as a local optimum. Either do not ship the
hackable term, or penalise the exploit directly.

## The structural finding: no mission is ever started

Logging every on-screen message across three full episodes — 13,500
steps — exactly one mission text ever appeared:

> *"Answer the South Park phones to get jobs. Keep your eyes open..."*

The agent has never answered a phone, so it has never started a mission,
so it has never completed or failed one. Two compounding causes:

1. `mission_complete_reward` and `mission_failed_penalty` both default to
   **0.0**, and this run did not set them. The plumbing is wired and
   working — message reading is on, the text table loads, detection runs
   at `gta1env/env.py:310` — but it pays nothing.
2. Even at non-zero they would never fire. Reaching a phone is a long
   unrewarded action sequence, and the only things paying are score
   (×0.01, earned from incidental carnage) and, for the first half,
   the distance term that taught the exploit.

The agent roams six districts — Coach, Jugular, Bulldog, Bus, Stinger,
Southwest Park — while the game's entire objective loop sits untouched
behind a phone it walks past.

## It is not a perception problem

The obvious suspicion was that the yellow objective arrow is lost in the
grayscale downscale. It is not. Measured across 19 frames with the arrow
isolated, through the exact preprocessing:

| | contrast vs local background | arrow size |
|---|---|---|
| full 320×240 gray | +124 median | 84 px |
| **128×96 (current)** | **+113 median, +33 worst** | 8–14 px |
| 84×84 (old default) | +119 median | 7 px |

The arrow is RGB (238, 231, 43) → luminance 212, against backgrounds of
73–169. It never falls below +33 contrast even on pale pavement — it is
the brightest object in the frame. The agent can see where to go
perfectly well. It has no reason to go there.

## Measurements worth keeping

- 128 episodes searched: median 400, q75 600, max 4200, 8 early deaths.
  The distribution is brutally heavy-tailed — the best episode is 10× the
  median, and in the best one 4000 of 4200 points arrived in a 140-step
  window before the agent died.
- Observation resolution is cheap: 2.7× the pixels costs 27% of
  throughput. See the README's Performance section.
- The GPU is idle two-thirds of every update; sampling is the constraint.

## Two tooling flaws this exposed

**`best.pt` never updated.** It froze at update 400 / 616k steps, because
selection ranks on the shaped return — which the annealing makes
non-stationary. An early policy's 12.95 cannot be beaten once the
distance term is switched off, so the peak policy of every annealed run
goes uncaptured. It must rank on `episodic_score`. Still unfixed.

**`eval_policy.py` had three independent defects**, each able to produce a
wrong verdict on its own: the sampler was unseeded (two runs on one
checkpoint disagreed on which arm won), the observation geometry came
from env defaults (a 128×96 checkpoint would not load), and it reported
only means on a distribution where one episode in 40 scores 30× the
median. Fixed; see the commit history.

## Files here

| | |
|---|---|
| `checkpoint.pt` | final policy, update 6509 |
| `train.log` | console log |
| `events.out.tfevents.*` | tensorboard scalars |
| `best.pt` | **do not use** — frozen at update 400, see above |

Videos of this policy are produced by `tools/record_policy.py`; they live
in `media/`, which is gitignored.

## Verdict

Keep as the reference for "shaped rewards without an objective". The
reward-hacking fix worked and the score gain is statistically real, but
1.67× random is not playing the game, and a third of the policy's actions
are a button exploit.

Do not tune this further. The next run needs the objective loop: locate
phone booths in RAM, shape toward the nearest one, and set
`mission_complete_reward` so finishing a job actually pays.
