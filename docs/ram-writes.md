# Writing to RAM: probing the game instead of searching it

The PS1's main RAM is exposed as a writable numpy view, not a read-only
one. That turns out to matter far more than it sounds, because it changes
what kind of question you can ask the game.

Reading RAM answers *"what is true right now?"*. Writing it answers
*"what would happen if?"* — and the second question is the one that
unblocks reward design, because you can put the agent in a situation it
would never reach on its own and see how the game responds.

This document records the capability, the first result it produced, and
the three ways it is easy to get wrong. All of it is reproducible with
`tools/find_phones.py`.

## The capability

`env.ram` is a zero-copy view of the console's 2 MiB. Writing the
player's position relocates the player for real — the scene changes, and
the player then walks on from the new position:

```python
SCALE = 1.52587890625e-05          # 16.16 fixed point
PLAYER_X, PLAYER_Y = 0x2C96C, 0x2C970

i32 = env.ram.view(np.int32)
i32[PLAYER_X // 4] = int(x / SCALE)
i32[PLAYER_Y // 4] = int(y / SCALE)
```

Measured: after writing a +400 unit offset and stepping forward, position
continued from the written value (7152 → 7156 → 7162) and the rendered
frame differed by 66.6/255 — a different part of the city, not a nudge.

**This is an authoritative write, not a mirror.** That was worth checking:
a value you can read back is not necessarily one the game obeys. The test
is to write it, then *move*, and confirm the motion continues from the new
value rather than snapping back.

## What it was built for: finding the phones

Every mission in GTA starts at a phone, and no trained policy has ever
reached one — across 13,500 logged steps the only mission text that ever
appeared was the tutorial telling the player to go answer one. The reward
function cannot shape toward phones until their locations are known.

**Searching RAM for them does not work.** Two approaches failed:

| approach | result |
|---|---|
| scan for constant int32 coordinate pairs in map range | 88,499 candidates survive — the filter is useless because most of RAM is numerically plausible |
| solve for the objective arrow's world position from its screen position | implied target varied ±364 against the player's own ±427; the arrow is not world-anchored in a way that solves |

**Visiting the map does work.** Teleport to each point of a grid, wait for
a message, record where one appears. The first phone:

```
(6100, 9575)   trigger zone x 6000-6200, y 9450-9700
"YOU WANT YOUR JOB AT PIZZA SHED BACK? CALL ME! CHAD."
```

A 2,916-point sweep over 4000×4000 world units found **exactly one**. That
is the expected answer, not a miss: GTA activates phones one at a time as
the player progresses, so a single live phone is the correct state at game
start. Re-run the sweep after completing a job to find the next one.

Results accumulate in `config/phones.json`.

## Three ways to get this wrong

Each of these produced a confidently wrong answer before being caught.
They are general traps for any probe-the-emulator experiment, not quirks
of this one.

**1. State leaks between probes.** A phone message stays on screen once
shown. A sweep without a reset therefore reports a hit at every point
after the first — the initial run "found" a region covering 2,026 of
2,209 points, and the apparent boundary was simply where the sweep
happened to pass the phone. *Reset between probes so each is independent.*
A savestate reload is cheap; this is exactly what makes the approach
affordable.

**2. Not waiting long enough.** The message takes roughly 30 frames to
appear. The corrected sweep used 4 and found **zero** hits at a location
that reliably triggers — which looked like a clean negative result and
was not. *Calibrate the settle time against a known-positive point before
trusting any negative.*

**3. Confirming a reading instead of testing it.** The first phone
location was inferred from where the message was first *observed* during
a continuous sweep. Re-probing that exact point independently showed
nothing. The inference had attributed a history-dependent effect to a
coordinate. *A location found by observation is a hypothesis; re-probe it
in isolation before recording it.*

## Other things this unblocks

**Varied start states.** Previously blocked: rollout cannot generate
them, because random play gives a median pairwise spread of 188 world
units and a trained policy gives 16, with byte-identical positions. Both
runs so far trained from one savestate. Writing position makes a
diverse start-state set a few lines of code, which removes a plausible
contributor to run 2's plateau.

**Counterfactual reward probes.** Any question of the form "does the game
reward X?" can be answered by placing the player in X and watching, rather
than waiting for a policy to stumble into it over millions of steps.

**Ground truth for shaping.** With phone coordinates known, a
distance-to-objective shaping term becomes possible — the thing run 2's
post-mortem identified as the missing piece.

## Caution

Writing RAM makes the episode unlike anything the game can produce on its
own. It is a **diagnostic and setup tool, not a training mechanism**:

* Teleporting mid-episode breaks the physics and position continuity that
  the distance reward and death heuristic both depend on.
* Start states built this way should be saved as savestates and loaded
  normally, so training itself never writes RAM.
* A teleported player can land inside geometry. Validate generated start
  states by stepping a few frames and checking the position is stable.

## See also

* `tools/find_phones.py` — the sweep, with the reset and settle defaults
  baked in
* `tools/ram_scan.py` — the read-side workflow for locating values
* `learning/run2/README.md` — why the phones matter
