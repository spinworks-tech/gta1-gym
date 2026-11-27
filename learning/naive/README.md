# Naive run: sparse score + dense movement shaping

The first end-to-end training run on the real 1997 game. It is kept here
because it is the baseline every later approach is measured against, and
because it demonstrates a specific, instructive failure: **the agent
learned, the return went up, and almost none of the gain was the game.**

Written for: anyone picking this repo up and wondering why the reward
function looks the way it does.

## What this approach was

PPO on 84x84 grayscale frames, 4 stacked, frameskip 4. One pcsx_rearmed
emulator per worker process, 12 workers, stepped as two half-batches out
of phase so emulation overlaps GPU inference.

Reward was read straight from PS1 main RAM:

| term | weight | character |
|---|---|---|
| in-game score delta | `score_scale = 0.01` | very sparse -- order one event per few thousand steps |
| distance travelled | `distance_scale = 0.001` | dense, every step |
| death (inferred) | `death_penalty = 1.0` | rare; inferred from a position jump, no lives address known |

The distance term exists because score alone is far too sparse for PPO to
get any gradient signal from random exploration. It was meant as a
bootstrap: move first, score later.

## What an episode is

Worth being explicit, because the game has no notion of one. GTA (PS1) is
a continuous open world: it never hands us an episode boundary, so we
impose one.

An episode is a **slice of a continuing game**, not a life:

1. **Start** — restore a savestate (`retro_unserialize`). Resets never
   reboot the console, which is what makes them instant; the cost is that
   the game resumes mid-career with whatever score and position that state
   holds. Then 0–30 random no-op steps (`noop_max`) so the workers are not
   phase-locked to the same frame.
2. **Step** — one agent step holds the chosen buttons for
   `frameskip = 4` emulated frames and observes only the last. With 5%
   probability (`sticky_prob`) the previous action is repeated instead, so
   the policy cannot exploit frame-perfect determinism.
3. **End** — one of two ways:
   - `truncated` when `_steps >= episode_steps` (4500). This is what
     essentially always happens: 4500 x 4 = 18,000 frames, about **5
     minutes of game time** at 60 fps.
   - `terminated` when a death is *inferred* — see below. These do fire:
     the rolling mean episode length is **4,228** against the 4,500 cap,
     and 96% of logged points sit below 4,400, implying roughly 5-8% of
     episodes end early. (An earlier note in this repo claimed no deaths
     were ever observed; that was based on an 80k-step sample and is
     wrong over 10M.)

Two consequences a reader should keep in mind. Dying costs only the
`death_penalty` and the remainder of the episode -- there is no survival
bonus, and with ~94% of episodes timing out the agent is barely selected
on staying alive at all. And score is
cumulative in the savestate, so only **deltas** are meaningful — the
generated start states sit at score 300–500 before the agent does
anything.

## The reward function, exactly

Per step, from two successive RAM reads ([`gta1env/ram.py`](../../gta1env/ram.py),
`RewardShaper.__call__`):

```
r_t = clip( cash_scale     * d(cash)
          + score_scale    * d(score)
          + health_scale   * d(health)
          + distance_scale * min(||d(pos)||, max_step_distance)
          - death_penalty  * [dead]
          - step_penalty
          , -clip, +clip )
```

| term | default | this run |
|---|---|---|
| `cash_scale` | 0.01 | 0.01 (no cash address verified, so inert) |
| `score_scale` | 0.01 | 0.01 -- $100 on the HUD is +1.0 |
| `health_scale` | 0.0 | 0.0 (no health address) |
| `distance_scale` | 0.0 | **0.001** per world unit |
| `max_step_distance` | 50.0 | caps one step's movement credit |
| `death_penalty` | 1.0 | 1.0 |
| `step_penalty` | 0.0 | 0.0 -- no pressure to hurry |
| `clip` | 5.0 | 5.0, on the final sum |

Three guards matter more than the weights:

- **Jump rejection.** A cash/score delta of `|d| >= 1e6` is ignored: that
  is a reload or a counter wrapping, not progress.
- **Death inference.** No lives address is known, so a position jump of
  `>= respawn_distance` (1500 units) is read as "respawned after dying".
  Measured over 40,000 steps: ordinary motion peaks near **32**
  units/step, entering or leaving a vehicle snaps the player up to
  **~512**, and the one genuine teleport was **~5,200**. An earlier
  threshold of 200 fired on *getting into cars* and would have taught the
  agent to avoid vehicles -- it terminated an otherwise clean episode at
  step 298.
- **Movement and death are exclusive.** The distance credit sits in the
  `elif` branch after the death test, so a respawn teleport cannot also
  pay out as 1,500 units of travel.

Everything is differential and clipped, so one corrupt read cannot blow up
an update.

## Start states

Every episode began from **one identical savestate**
(`config/boot.state`), capped at 4500 steps. This turned out to be half
the problem -- see below.

## Command

```bash
python train_ppo.py --envs 12 --total-steps 10000000 \
    --distance-scale 0.001 --run-name gta1_run1
```

## Result: it learns, and it learns the wrong thing

Episodic return rose from **11.5 to a peak of 15.9 at 7.1M steps**, then
**collapsed to 9.65 -- below where it started** -- before partially
recovering to 14.4 by the end.

| steps | 0.0M | 2.0M | 4.0M | 6.0M | **7.0M** | 8.0M | 8.5M | **9.0M** | 9.5M |
|---|---|---|---|---|---|---|---|---|---|
| return | 11.50 | 11.93 | 12.94 | 13.80 | **14.96** | 14.28 | 12.23 | **10.38** | 13.79 |

The rise to 7M is genuine. Everything after it is a second, separate
failure, and the cause is visible in the diagnostics:

| steps | return | policy entropy | clipfrac | lr |
|---|---|---|---|---|
| 0.0M | 11.51 | 2.478 | 0.143 | 2.5e-4 |
| 4.0M | 12.93 | 2.099 | 0.166 | 1.5e-4 |
| 7.0M | **14.96** | 1.586 | 0.105 | 7.5e-5 |
| 8.5M | 12.23 | **0.707** | 0.042 | 3.8e-5 |
| 9.0M | **10.39** | 0.842 | 0.024 | 2.5e-5 |
| 9.5M | 13.81 | 1.053 | **0.004** | ~0 |

**Entropy collapse.** The action distribution starts near-uniform
(ln 14 = 2.64, measured 2.48) and decays monotonically to 0.71 -- a
nearly deterministic policy. Return falls in lockstep. `ent_coef = 0.01`
was too weak to hold exploration open, and because the learning rate was
annealing to zero at the same time, the policy could neither explore nor
recover: by 9.5M `clipfrac` is 0.004, meaning almost no update is
happening at all. The run ends frozen into a narrow, worse-than-random
behaviour.

Shaped return mixes the terms, so it cannot say *what* improved. Measuring
the raw quantities separately (`tools/eval_policy.py`, 12 episodes each,
checkpoint at update 5100) against a random policy:

| | score gained | distance travelled |
|---|---|---|
| random | 353 (max 560) | 6,521 |
| policy @5100 | **431** (max 900) | **10,501** |
| change | +22% | **+61%** |

Multiplying through by the shaper's own weights reproduces the measured
curve almost exactly, which is what makes the split trustworthy:

| | score x 0.01 | distance x 0.001 | predicted | observed |
|---|---|---|---|---|
| random | 3.53 | 6.52 | **10.05** | ~11.5 (early) |
| policy @5100 | 4.31 | 10.50 | **14.81** | ~14.9 |

**About 80% of the return gain is the distance term.** The agent learned to
drive competently and keep moving. Score rose as a side effect of being in
a car in traffic, not because the policy seeks out scoring opportunities.
At n=12 with that spread the score gain is suggestive, not established.

## Why it went this way

Three causes. The first two explain the ceiling, the third explains the
collapse:

1. **The bootstrap became the objective.** A dense term that pays every
   step dominates a term that pays once every few thousand. Movement was
   supposed to be a ladder and turned into the roof.
2. **No start-state diversity.** All 12 workers reset from the same frame,
   so they explore the same corner of Liberty City. Scoring opportunities
   sit thousands of steps away from a fixed spawn, which is exactly the
   distance random exploration cannot cover.
3. **Exploration was not maintained.** Entropy decayed unchecked while
   the learning rate annealed to zero, so the policy hardened into one
   narrow behaviour and then could not climb back out.

## Measurements worth keeping

- throughput 864-911 steps/s with 12 envs; 10M steps in ~3.2 h
- pipelining the half-batches: 597 -> 911 steps/s (1.53x)
- PPO peak VRAM 493 MB of 4096 (GTX 1050 Ti)
- single core, in-game: ~30x realtime

## What this run changed in the code

- `distance_scale` is annealable -- `--distance-scale-final`,
  `--reward-anneal-frac` -- via `GTA1Env.set_reward_scales()` broadcast over
  the vec env's existing `call()` pipe.
- `score_scale` is settable from the CLI (`--score-scale`).
- `boot_state` accepts a **directory** of `*.state` files, sampled per
  reset; `tools/make_boot_states.py` builds one.
- the trainer logs `charts/episodic_score` and `charts/episodic_distance`
  separately, so no future run needs an offline eval to be decomposed.
- `info["boot_index"]` reports which start state an episode used.

## Files here

| file | what |
|---|---|
| `checkpoint.pt` | final weights (update 6505, 9.99M steps) -- **post-collapse, not the best policy** |
| `train.log` | full console log of the run |
| `events.out.tfevents.*` | tensorboard scalars |
| `return_curve.csv` | binned episodic return |
| `eval_vs_random.log` | the policy-vs-random evaluation output |

Reproduce the evaluation:

```bash
python tools/eval_policy.py --ckpt learning/naive/checkpoint.pt \
    --envs 6 --steps 9200
```

## A tooling flaw this exposed

`--save-every` wrote to a single `checkpoint.pt`, overwriting it. Because
the run peaked at 7.1M and degraded afterwards, **the best policy was
overwritten by a worse one and is unrecoverable.** The evaluation above
survives only because it happened to be run mid-flight, at update 5100
(7.8M steps), near the peak.

The trainer now also keeps `best.pt`, the highest mean-return checkpoint
(`--save-best`, on by default).

## Verdict

Keep as a baseline, do not build on. The headline number (return 11.5 ->
15.0) is real but misleading, and this is the cautionary case for reading a
shaped return as progress. Any successor must report score and distance
separately -- which the trainer now does by default.
