"""Parallel GTA1 envs: one process per emulator, shared memory for frames.

A libretro core keeps global state, so N instances means N processes --
never N objects in one process. Observations are written by the workers
straight into one shared (N, C, H, W) uint8 block, so the parent never
unpickles a frame; it just reads the block and pushes it to the GPU
through a pinned staging buffer.
"""

import multiprocessing as mp
import os
from multiprocessing import shared_memory

import numpy as np

from .env import GTA1Config, GTA1Env

CMD_RESET, CMD_STEP, CMD_CLOSE, CMD_CALL = range(4)


def _worker(index, pipe, shm_name, obs_shape, obs_dtype, cfg_kwargs, seed,
            cpu):
    """Runs in its own process: owns exactly one emulator."""
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError:
            pass
    # Each emulator is single-threaded; stop BLAS from oversubscribing.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    shm = shared_memory.SharedMemory(name=shm_name)
    obs_all = np.ndarray(obs_shape, dtype=np.dtype(obs_dtype), buffer=shm.buf)
    slot = obs_all[index]

    env = None
    try:
        env = GTA1Env(GTA1Config(**cfg_kwargs), seed=seed)
        pipe.send(("ready", {"action_n": env.action_space.n,
                             "obs_shape": env.observation_space.shape,
                             "av": env.av,
                             "ram_fields": sorted(env.ram_map.fields)}))
        while True:
            cmd, payload = pipe.recv()
            if cmd == CMD_STEP:
                obs, reward, term, trunc, info = env.step(payload)
                if term or trunc:
                    info = dict(info)
                    # The slot now holds the RESET observation, not the
                    # final one; callers bootstrap using term/trunc.
                    info["episode_ended"] = ("terminated" if term
                                             else "truncated")
                    # Kept because the slot is about to be overwritten:
                    # world-model training needs the terminal frame as a
                    # prediction target. One copy per episode end, so cheap.
                    info["final_obs"] = obs
                    obs, reset_info = env.reset()
                    info["reset_steps"] = reset_info.get("steps", 0)
                slot[...] = obs
                pipe.send((float(reward), bool(term), bool(trunc), info))
            elif cmd == CMD_RESET:
                obs, info = env.reset(seed=payload)
                slot[...] = obs
                pipe.send(info)
            elif cmd == CMD_CALL:
                name, args, kwargs = payload
                attr = getattr(env, name)
                pipe.send(attr(*args, **kwargs) if callable(attr) else attr)
            elif cmd == CMD_CLOSE:
                pipe.send(True)
                break
    except Exception as exc:                       # surface, don't hang
        import traceback
        pipe.send(("error", f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}"))
    finally:
        if env is not None:
            env.close()
        shm.close()


class SubprocVecEnv:
    """Synchronous vectorised GTA1, backed by one process per env."""

    def __init__(self, num_envs, cfg_kwargs=None, base_seed=0,
                 pin_cpus=True, start_method="spawn"):
        self.num_envs = num_envs
        self.cfg_kwargs = dict(cfg_kwargs or {})
        probe = GTA1Config(**self.cfg_kwargs)
        chans = probe.frame_stack * (1 if probe.grayscale else 3)
        self.obs_shape = (num_envs, chans, probe.obs_height, probe.obs_width)
        self.single_obs_shape = self.obs_shape[1:]

        nbytes = int(np.prod(self.obs_shape))
        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self.obs = np.ndarray(self.obs_shape, dtype=np.uint8,
                              buffer=self.shm.buf)
        self.obs[:] = 0

        ctx = mp.get_context(start_method)
        ncpu = os.cpu_count() or 1
        self._pipes, self._procs = [], []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            cpu = (i % ncpu) if pin_cpus else None
            proc = ctx.Process(
                target=_worker,
                args=(i, child, self.shm.name, self.obs_shape, "uint8",
                      self.cfg_kwargs, base_seed + i, cpu),
                daemon=True)
            proc.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(proc)

        self.specs = []
        for i, pipe in enumerate(self._pipes):
            msg = pipe.recv()
            if msg[0] == "error":
                self.close()
                raise RuntimeError(f"env {i} failed to start:\n{msg[1]}")
            self.specs.append(msg[1])
        self.num_actions = self.specs[0]["action_n"]
        self._waiting = np.zeros(num_envs, dtype=bool)
        self._staging = None

    # ------------------------------------------------------------- stepping

    def _as_slice(self, group):
        """Groups are contiguous slices so shared memory stays a view."""
        if group is None:
            return slice(0, self.num_envs)
        if isinstance(group, slice):
            return group
        raise TypeError("group must be a slice or None")

    def reset(self, seeds=None):
        for i, pipe in enumerate(self._pipes):
            pipe.send((CMD_RESET, None if seeds is None else seeds[i]))
        infos = [self._recv(i) for i in range(self.num_envs)]
        return self.obs, infos

    def step_async(self, actions, group=None):
        """Start a step on a subset of workers without blocking.

        Stepping two halves out of phase lets the policy run on the GPU for
        one half while the other half's emulators are busy.
        """
        group = self._as_slice(group)
        if self._waiting[group].any():
            raise RuntimeError("step_async called twice for the same envs")
        for i, action in zip(range(group.start, group.stop), actions):
            self._pipes[i].send((CMD_STEP, int(action)))
        self._waiting[group] = True

    def step_wait(self, group=None):
        group = self._as_slice(group)
        if not self._waiting[group].all():
            raise RuntimeError("step_wait without a matching step_async")
        n = group.stop - group.start
        rewards = np.empty(n, dtype=np.float32)
        terms = np.zeros(n, dtype=bool)
        truncs = np.zeros(n, dtype=bool)
        infos = []
        for k, i in enumerate(range(group.start, group.stop)):
            reward, term, trunc, info = self._recv(i)
            rewards[k], terms[k], truncs[k] = reward, term, trunc
            infos.append(info)
        self._waiting[group] = False
        return self.obs[group], rewards, terms, truncs, infos

    def step(self, actions, group=None):
        self.step_async(actions, group)
        return self.step_wait(group)

    def _recv(self, i):
        msg = self._pipes[i].recv()
        if isinstance(msg, tuple) and msg and msg[0] == "error":
            raise RuntimeError(f"env {i} crashed:\n{msg[1]}")
        return msg

    # ------------------------------------------------------------ transfers

    def obs_to_torch(self, device, normalize=True, non_blocking=True,
                     group=None):
        """Shared memory -> pinned host buffer -> CUDA, in one hop.

        Each group stages through its own slice of the pinned buffer, so an
        in-flight async copy for one group is never overwritten by another.
        """
        import torch
        group = self._as_slice(group)
        if self._staging is None:
            self._staging = torch.empty(self.obs_shape, dtype=torch.uint8,
                                        pin_memory=(str(device) != "cpu"))
        staged = self._staging[group]
        staged.copy_(torch.from_numpy(self.obs[group]))
        out = staged.to(device, non_blocking=non_blocking)
        return out.float().div_(255.0) if normalize else out

    # ------------------------------------------------------------- plumbing

    def call(self, name, *args, group=None, **kwargs):
        """Call a method on each worker's env and collect the results.

        Each pipe is FIFO, so a worker with a step still in flight would
        answer with that step's result and leave the call's reply to be
        read by the next step_wait -- silent desynchronisation. Refuse
        instead; a pipelined caller should use `group` and call between a
        step_wait and the next step_async for that group.
        """
        group = self._as_slice(group)
        if self._waiting[group].any():
            raise RuntimeError(
                "call() with a step in flight would desynchronise the "
                "pipe; call between step_wait and step_async for this "
                "group, or pass the group that is idle")
        for i in range(group.start, group.stop):
            self._pipes[i].send((CMD_CALL, (name, args, kwargs)))
        return [self._recv(i) for i in range(group.start, group.stop)]

    def close(self):
        # A pipelined caller may still have steps in flight; drain them so
        # the workers are not killed mid-command.
        for i in np.flatnonzero(self._waiting) if len(self._pipes) else []:
            try:
                self._pipes[int(i)].recv()
            except (EOFError, OSError):
                pass
        self._waiting[:] = False
        for pipe in self._pipes:
            try:
                pipe.send((CMD_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for pipe in self._pipes:
            try:
                pipe.close()
            except OSError:
                pass
        self._pipes, self._procs = [], []
        if getattr(self, "shm", None) is not None:
            self.obs = None
            self.shm.close()
            self.shm.unlink()
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
