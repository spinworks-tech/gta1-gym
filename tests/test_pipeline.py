#!/usr/bin/env python3
"""End-to-end tests of the plumbing, against tests/mock_core.c.

These never touch the real BIOS or disc; they prove that the ctypes
frontend, the Gymnasium env, the shared-memory vector env, savestate
resets, determinism and the RAM scanner all work. Run:

    .venv/bin/python tests/test_pipeline.py
"""

import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from gta1env.env import GTA1Config, GTA1Env  # noqa: E402
from gta1env.libretro import LibretroCore  # noqa: E402
from gta1env.ram import RamMap  # noqa: E402
from gta1env.vec import SubprocVecEnv  # noqa: E402

MOCK = os.path.join(HERE, "mock_core_libretro.so")
DUMMY_ROM = os.path.join(ROOT, "scratch", "dummy.mock")
RAM_MAP = os.path.join(ROOT, "scratch", "test_ram_map.json")
MOCK_BOOT_STATE = os.path.join(ROOT, "scratch", "mock_boot.state")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" +
          (f" -- {detail}" if detail else ""))


def mock_cfg(**kw):
    # boot_state must point somewhere of our own: the default is
    # config/boot.state, which holds a real PlayStation savestate once the
    # game has been set up, and the mock core would load it as garbage RAM.
    base = dict(core_path=MOCK, rom_path=DUMMY_ROM, require_assets=False,
                ram_map_path=RAM_MAP, boot_state=MOCK_BOOT_STATE,
                boot_frames=10, noop_max=0,
                frameskip=4, episode_steps=200)
    base.update(kw)
    return base


def write_ram_map():
    rm = RamMap(path=RAM_MAP)
    rm.set("cash", 0x800A1234, size=4, note="mock cash")
    rm.set("health", 0x800B5678, size=1, note="mock health")
    rm.set("player_x", 0x800C0000, size=4, signed=True)
    rm.set("player_y", 0x800C0004, size=4, signed=True)
    rm.set("game_over", 0x800C0010, size=1)
    rm.save(RAM_MAP)


def test_core():
    print("\n== libretro frontend ==")
    open(DUMMY_ROM, "wb").close()
    core = LibretroCore(MOCK, os.path.join(ROOT, "bios"))
    av = core.load_game(DUMMY_ROM)
    check("load_game", av["base_width"] == 320 and av["base_height"] == 240,
          str(av))
    frame = core.run()
    check("frame shape", frame is not None and frame.shape == (240, 320, 3),
          None if frame is None else str(frame.shape))
    check("XRGB8888 decode", frame is not None and
          tuple(frame[0, 0]) == (0x20, 0x20, 0x20), str(tuple(frame[0, 0])))

    ram = core.memory()
    check("system RAM view", ram is not None and ram.size == 2 * 1024 * 1024)

    # Input must actually reach the core.
    x0 = int(np.frombuffer(ram[0xC0000:0xC0004].tobytes(), np.int32)[0])
    core.set_buttons(["right"])
    for _ in range(10):
        core.run()
    x1 = int(np.frombuffer(ram[0xC0000:0xC0004].tobytes(), np.int32)[0])
    check("input reaches core", x1 > x0, f"x {x0} -> {x1}")

    # Savestates must round-trip exactly.
    blob = core.serialize()
    core.set_buttons(["left"])
    for _ in range(20):
        core.run()
    x2 = int(np.frombuffer(ram[0xC0000:0xC0004].tobytes(), np.int32)[0])
    core.unserialize(blob)
    x3 = int(np.frombuffer(ram[0xC0000:0xC0004].tobytes(), np.int32)[0])
    check("savestate restores state", x2 != x1 and x3 == x1,
          f"{x1} -> {x2} -> restored {x3}")
    core.close()


def test_env():
    print("\n== GTA1Env ==")
    write_ram_map()
    env = GTA1Env(GTA1Config(**mock_cfg()))
    obs, info = env.reset()
    check("obs shape/dtype", obs.shape == (4, 84, 84) and obs.dtype == np.uint8,
          f"{obs.shape} {obs.dtype}")
    check("ram fields in info", "cash" in info and "health" in info,
          str(sorted(info)))

    # Reaching the pad pays cash and running out of health kills you; the
    # shaper must produce a positive and a negative event respectively.
    rng = np.random.default_rng(0)
    events = []
    term = trunc = False
    for _ in range(env.cfg.episode_steps + 5):
        obs, r, term, trunc, info = env.step(int(rng.integers(
            env.action_space.n)))
        if r != 0.0:
            events.append(r)
        if term or trunc:
            break
    check("reward credits cash gains", any(e > 0 for e in events),
          f"events {events}, cash {info.get('cash')}")
    check("reward penalises death", any(e < 0 for e in events) or not term,
          f"events {events}, health {info.get('health')}")
    check("frame stack advances", obs[0].tobytes() != obs[-1].tobytes())

    check("truncates at limit", trunc or term, f"term={term} trunc={trunc}")
    env.close()


def test_determinism():
    print("\n== determinism ==")
    actions = [(i * 7) % 14 for i in range(60)]
    traces = []
    for _ in range(2):
        env = GTA1Env(GTA1Config(**mock_cfg()))
        env.reset(seed=3)
        trace = []
        for a in actions:
            obs, r, _, _, info = env.step(a)
            trace.append((float(r), int(info.get("player_x", 0)),
                          obs.sum()))
        traces.append(trace)
        env.close()
    check("same inputs -> same trajectory", traces[0] == traces[1],
          "first divergence: " + str(next((i for i, (a, b) in enumerate(
              zip(*traces)) if a != b), None)))


def test_gym_api():
    print("\n== Gymnasium API ==")
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    import gta1env  # registers GTA1-v0

    env = gym.make("GTA1-v0", **mock_cfg(noop_max=5))
    check("gym.make resolves GTA1-v0", env is not None)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        check("passes gymnasium env_checker", True)
    except Exception as exc:
        check("passes gymnasium env_checker", False, str(exc)[:160])
    env.close()

    # Same seed must give the same first observation.
    seeds = []
    for _ in range(2):
        e = gym.make("GTA1-v0", **mock_cfg(noop_max=5))
        obs, _ = e.reset(seed=123)
        seeds.append(obs.tobytes())
        e.close()
    check("reset(seed) is reproducible", seeds[0] == seeds[1])

    e = gta1env.make(**mock_cfg())
    check("gta1env.make() returns a single env", hasattr(e, "action_space"))
    e.close()


def test_vec():
    print("\n== SubprocVecEnv ==")
    n = 4
    vec = SubprocVecEnv(n, cfg_kwargs=mock_cfg(), base_seed=7)
    obs, infos = vec.reset()
    check("vec obs block", obs.shape == (n, 4, 84, 84), str(obs.shape))
    check("workers report ram fields",
          all("cash" in s["ram_fields"] for s in vec.specs))

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    steps = 60
    for _ in range(steps):
        obs, rewards, terms, truncs, infos = vec.step(
            rng.integers(vec.num_actions, size=n))
    dt = time.perf_counter() - t0
    check("shared memory differs per env",
          len({obs[i].tobytes() for i in range(n)}) > 1)
    check("rewards vector", rewards.shape == (n,) and rewards.dtype ==
          np.float32)
    print(f"       mock throughput: {steps*n/dt:.0f} steps/s over {n} procs")

    # Autoreset path
    long_vec_steps = 0
    saw_reset = False
    while long_vec_steps < 250 and not saw_reset:
        obs, rewards, terms, truncs, infos = vec.step(
            rng.integers(vec.num_actions, size=n))
        saw_reset = any(i.get("episode_ended") for i in infos)
        long_vec_steps += 1
    check("autoreset on episode end", saw_reset,
          f"after {long_vec_steps} steps")

    # Group-wise stepping: the two halves must advance independently, which
    # is what lets the trainer overlap emulation with GPU work.
    a, b = slice(0, 2), slice(2, 4)
    vec.step_async([0, 0], a)
    vec.step_async([1, 1], b)
    oa, ra, _, _, ia = vec.step_wait(a)
    ob, rb, _, _, ib = vec.step_wait(b)
    check("group step shapes", oa.shape == (2,) + vec.single_obs_shape
          and ra.shape == (2,), f"{oa.shape} {ra.shape}")
    check("groups return their own infos", len(ia) == 2 and len(ib) == 2)
    try:
        vec.step_async([0, 0], a)
        vec.step_async([0, 0], a)
        double = False
    except RuntimeError:
        double = True
    finally:
        vec.step_wait(a)
    check("double step_async on a group is refused", double)
    vec.close()


def test_scanner():
    print("\n== ram_scan ==")
    snap_dir = os.path.join(ROOT, "scratch", "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    core = LibretroCore(MOCK, os.path.join(ROOT, "bios"))
    core.load_game(DUMMY_ROM)
    ram = core.memory()

    # The memory view is zero-copy, so writing through it is the same as
    # the game writing: a good way to plant known values for the scanner.
    names = []
    for target in (100, 250, 900):
        ram[0xA1234:0xA1238] = np.frombuffer(
            target.to_bytes(4, "little"), dtype=np.uint8)
        for _ in range(3):
            core.run()                      # let unrelated RAM churn
        readback = int(np.frombuffer(ram[0xA1234:0xA1238].tobytes(),
                                     np.uint32)[0])
        if readback != target:
            check("RAM view is writable", False,
                  f"wrote {target}, read {readback}")
            core.close()
            return
        path = os.path.join(snap_dir, f"selftest_{target}.npy")
        np.save(path, ram.copy())
        names.append((f"selftest_{target}", target))
    check("RAM view is writable", True)
    core.close()

    cmd = [sys.executable, os.path.join(ROOT, "tools", "ram_scan.py"),
           "known"] + sum((["--at", f"{n}={v}"] for n, v in names), [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    check("scanner finds the planted address", "0x800A1234" in res.stdout,
          (res.stdout or res.stderr).strip().splitlines()[0]
          if (res.stdout or res.stderr) else "no output")
    for n, _ in names:
        os.remove(os.path.join(snap_dir, f"{n}.npy"))


def test_boot_states():
    """A directory of start states must actually be sampled."""
    print("\n== boot state diversity ==")
    import shutil
    import tempfile
    from gta1env.env import GTA1Config, GTA1Env

    tmp = tempfile.mkdtemp(prefix="bootstates_")
    try:
        # Build three distinct states by walking the mock core apart.
        env = GTA1Env(GTA1Config(**mock_cfg()))
        try:
            env.reset(seed=0)
            # The mock only moves on left/right, so steer deliberately
            # rather than trusting an arbitrary action index.
            right = next(i for i, a in enumerate(env.actions)
                         if "turn_right" in a)
            left = next(i for i, a in enumerate(env.actions)
                        if "turn_left" in a)
            # px clamps at both edges, so pin to the left one first: the
            # snapshots are then spaced the same way wherever we began.
            for _ in range(120):
                env.step(left)
            marks = []
            for i in range(3):
                for _ in range(12):
                    env.step(right)
                blob = env.core.serialize()
                with open(os.path.join(tmp, f"{i:02d}.state"), "wb") as fh:
                    fh.write(blob)
                marks.append(env._read_state().get("player_x"))
        finally:
            env.close()
        check("distinct states written", len(set(marks)) == 3, str(marks))

        env = GTA1Env(GTA1Config(**mock_cfg(boot_state=tmp)))
        try:
            check("directory of states loaded", len(env._boot_blobs) == 3,
                  f"{len(env._boot_blobs)} blobs")
            seen = set()
            starts = set()
            for seed in range(25):
                _, info = env.reset(seed=seed)
                seen.add(info["boot_index"])
                starts.add(info.get("player_x"))
            check("resets sample several start states", len(seen) > 1,
                  f"indices seen: {sorted(seen)}")
            check("sampled states differ in RAM", len(starts) > 1,
                  f"{len(starts)} distinct player_x")
        finally:
            env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reward_reweighting():
    """set_reward_scales must reach the shaper, including over the pipe."""
    print("\n== reward reweighting ==")
    from gta1env.env import GTA1Config, GTA1Env
    from gta1env.vec import SubprocVecEnv

    env = GTA1Env(GTA1Config(**mock_cfg(distance_scale=0.01)))
    try:
        env.reset(seed=0)
        out = env.set_reward_scales(distance_scale=0.0, score_scale=0.5)
        check("scales applied to shaper",
              env.shaper.distance_scale == 0.0
              and env.shaper.score_scale == 0.5, str(out))
        bad = False
        try:
            env.set_reward_scales(nonsense=1.0)
        except TypeError:
            bad = True
        check("unknown scale is refused", bad)
    finally:
        env.close()

    vec = SubprocVecEnv(2, cfg_kwargs=mock_cfg(distance_scale=0.01),
                        base_seed=0, pin_cpus=False)
    try:
        vec.reset()
        res = vec.call("set_reward_scales", distance_scale=0.0)
        check("broadcast reaches every worker",
              len(res) == 2 and all(r["distance_scale"] == 0.0 for r in res),
              str(res))

        # A call() while a step is in flight used to read that step's
        # reply and leave its own for the next step_wait, which then
        # unpacked an info dict: "too many values to unpack". Silent
        # stream corruption -- it must refuse instead.
        vec.step_async([0, 0])
        refused = False
        try:
            vec.call("set_reward_scales", distance_scale=0.5)
        except RuntimeError:
            refused = True
        check("call() with a step in flight is refused", refused)
        vec.step_wait()
        # ...and the stream still works afterwards.
        res = vec.call("set_reward_scales", distance_scale=0.25)
        check("pipe still usable after the refusal",
              all(r["distance_scale"] == 0.25 for r in res), str(res))

        # The per-group form is the one the trainer uses while pipelined.
        g = slice(0, 1)
        vec.step_async([0], group=g)
        vec.step_wait(g)
        res = vec.call("set_reward_scales", distance_scale=0.1, group=g)
        check("per-group call on an idle group works",
              len(res) == 1 and res[0]["distance_scale"] == 0.1, str(res))
    finally:
        vec.close()


if __name__ == "__main__":
    if not os.path.exists(MOCK):
        raise SystemExit("build the mock core first: "
                         "gcc -O2 -fPIC -shared -o tests/mock_core_libretro.so "
                         "tests/mock_core.c")
    test_core()
    test_env()
    test_determinism()
    test_gym_api()
    test_vec()
    test_boot_states()
    test_reward_reweighting()
    test_scanner()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
