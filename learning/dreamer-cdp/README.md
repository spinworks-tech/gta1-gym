# Dreamer-CDP on GTA1 (in progress, not yet run)

Model-based RL with a reconstruction-free world model: DreamerV3 with
pixel reconstruction replaced by JEPA-style prediction of the next
continuous deterministic embedding (CDP). Chosen over this repo's own
`train_jepa.py` flywheel because it keeps Dreamer's online loop, where the
policy collects its own data and reward/value losses shape the latent from
the start, and because it is published and matched DreamerV3 on Crafter,
an open-world top-down game (Hauri & Zenke, arXiv 2603.07083).

Written for: whoever picks this approach up next.

## Provenance

Vendored from https://github.com/fmi-basel/Dreamer-CDP at commit
`a851fa3` (2026-03-24): `dreamerv3/`, `embodied/`, `LICENSE` (kept),
upstream readme as `UPSTREAM_README.md`. Our changes:

- `dreamerv3/rssm.py` `calculate_encoder_output_dim`: upstream bug fixed.
  It sizes the CDP predictor's output but ignored the size presets,
  dropped the vector part whenever an image was present, and assumed
  square images. Any image + vector run (ours) crashed with a shape
  mismatch. Now mirrors the encoder's own forward pass.
- `dreamerv3/configs.yaml`: `wandb` removed from the default logger
  outputs (it needs an account and sends run data off-machine); logs are
  local `jsonl` + `scope`.
- `requirements.txt`: `nvidia-cudnn-cu12==9.5.1.17` pinned. The 9.28
  that pip resolves fails every convolution on Pascal (cuDNN status
  5003, even a 2x3x16x16 conv); 9.5.1 is what torch 2.6 uses here.

Reconstruction-free as published: the image decoder still trains, but
`dec_grad: False` stops its gradient at the latent, so it is only a
visual probe. It can be switched off for speed.

Runs in its own venv (`learning/dreamer-cdp/.venv`, Python 3.11, JAX
0.4.33 + CUDA 12) so JAX's CUDA libraries cannot clash with the main
PyTorch venv.

## Design: give the agent the state we already know

The image is kept, but everything RAM and the disc already tell us goes in
as an extra vector observation, which DreamerV3 encodes with an MLP and
fuses with the CNN embedding (no architecture change):

| key | source |
|---|---|
| position, velocity | `player_x/y` and their step difference |
| score delta | `score` |
| lives | `0x8002C446` (from CodeBreaker codes) |
| message id, on, kind | `gta1env/text.py`: id one-hot/embedding over the fixed string table |
| objective direction | the yellow arrow (RGB 238,231,43): offset of its pixel mask from screen centre, plus a present flag |
| first-phone vector | relative position to the known first phone (6100, 9575) |

Reward, from RAM, chosen to be unhackable by the moves that broke runs 1
and 2:

- potential-based shaping on distance to the current objective
  (telescopes, so vibrating or toggling `enter_exit` nets nothing),
- `mission_complete_reward` / `mission_failed_penalty` (message 2500/2501),
- score delta.

## Plan

1. **Feasibility.** JAX on the GTX 1050 Ti (Pascal, 4 GB). If it does not
   run or is too slow, train on Modal; the local GPU stays for smoke tests.
2. **Env adapter** `embodied/envs/gta1.py` wrapping `GTA1Env`, registered
   in `dreamerv3/main.py:make_env` as `gta1_*`.
3. **State-only run** (vector obs, no image): the cheapest test of whether
   the objective loop works at all.
4. **Image + state**, then **image only**, to see how much the state
   carries.

## Measured on the GTX 1050 Ti (2026-11-28)

JAX 0.4.33 sees the GPU (about 2 TFLOP/s fp32 on a matmul). Smoke run on
the built-in dummy env, `size12m`, 64x64 images, batch 8 x 32 (the
default 16 x 64 has not been tried since the cuDNN fix):

| | value |
|---|---|
| trained frames/s | ~620 |
| env steps/s at `train_ratio 32` | ~19 |
| 1M env steps at `train_ratio 32` | ~15 h |
| 1M env steps at `train_ratio 512` (paper's Crafter setting) | ~10 days |

So the local GPU is fine for smoke tests and short state-only runs;
paper-scale runs go to Modal. The emulator (~70 steps/s per core) is not
the bottleneck either way.

## Status

- [x] vendored at `a851fa3`
- [x] JAX GPU check: runs, numbers above
- [ ] env adapter
- [ ] first run
