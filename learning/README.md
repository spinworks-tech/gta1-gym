# Training approaches

One folder per approach that has actually been run, each with the config,
the measured result and — more usefully — how it failed. The point is
that the reasoning behind the current defaults stays recoverable instead
of living in a chat log.

| approach | result | status |
|---|---|---|
| [naive/](naive/) | return 11.5 → 15.0 over 10M steps, but ~80% of the gain was the dense movement term rather than the game | baseline; do not build on |

Each folder holds its own `README.md`, the final `checkpoint.pt`, the
tensorboard event file and the console log, so any run can be re-evaluated
later:

```bash
.venv/bin/python tools/eval_policy.py --ckpt learning/<name>/checkpoint.pt \
    --envs 6 --steps 9200
```
