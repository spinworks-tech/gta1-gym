"""Gymnasium environment wrapping the original 1997 GTA (PlayStation).

One env == one process == one libretro core instance. The emulator is
stepped in lockstep: nothing is driven by wall-clock time, so N copies
run as fast as the CPU allows and the result is reproducible.
"""

import os
from dataclasses import dataclass, field, fields, replace

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .libretro import LibretroCore
from .ram import CONTROLS, DEFAULT_ACTIONS, RamMap, RewardShaper
from .text import MISSION_COMPLETE, MISSION_FAILED, MessageReader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Option values chosen for determinism first, speed second. Every threaded
# path in the core is switched off: a background DRC/GPU/SPU thread makes
# the same input sequence produce different frames run to run.
DETERMINISTIC_CORE_OPTIONS = {
    "pcsx_rearmed_drc": "enabled",
    "pcsx_rearmed_drc_thread": "disabled",
    "pcsx_rearmed_gpu_thread_rendering": "disabled",
    "pcsx_rearmed_spu_thread": "disabled",
    "pcsx_rearmed_frameskip": "disabled",
    "pcsx_rearmed_frameskip_type": "disabled",
    "pcsx_rearmed_bios": "auto",
    "pcsx_rearmed_region": "auto",
    "pcsx_rearmed_rgb32_output": "enabled",
    "pcsx_rearmed_dithering": "disabled",
    "pcsx_rearmed_show_bios_bootlogo": "disabled",
    "pcsx_rearmed_display_internal_fps": "disabled",
    "pcsx_rearmed_spu_reverb": "disabled",
    "pcsx_rearmed_nocdaudio": "enabled",
    "pcsx_rearmed_cd_turbo": "enabled",
    "pcsx_rearmed_cd_readahead": "0",
    "pcsx_rearmed_vibration": "disabled",
}


DISC_EXT = (".cue", ".chd", ".m3u", ".pbp", ".img", ".iso")


def find_disc(rom_dir=os.path.join(HERE, "roms")):
    """Pick the disc image in roms/ when no explicit path is configured."""
    if not os.path.isdir(rom_dir):
        return ""
    # A .cue/.chd is the whole disc; bare .bin tracks are referenced by it.
    discs = sorted(f for f in os.listdir(rom_dir)
                   if f.lower().endswith(DISC_EXT))
    if len(discs) == 1:
        return os.path.join(rom_dir, discs[0])
    if len(discs) > 1:
        raise FileNotFoundError(
            f"{len(discs)} disc images in {rom_dir}; set rom_path explicitly: "
            + ", ".join(discs))
    return ""


@dataclass
class GTA1Config:
    rom_path: str = ""          # "" = auto-discover the disc in roms/
    core_path: str = os.path.join(
        HERE, "third_party", "pcsx_rearmed", "pcsx_rearmed_libretro.so")
    bios_dir: str = os.path.join(HERE, "bios")
    save_dir: str = os.path.join(HERE, "scratch", "saves")
    boot_state: str = os.path.join(HERE, "config", "boot.state")
    ram_map_path: str = os.path.join(HERE, "config", "ram_map.json")

    # observation
    obs_width: int = 84
    obs_height: int = 84
    grayscale: bool = True
    frame_stack: int = 4

    # timing
    frameskip: int = 4            # emulated frames per agent step
    # Named episode_steps, not max_episode_steps: the latter is a
    # reserved gym.make() kwarg and would be swallowed by Gymnasium's
    # TimeLimit wrapper instead of reaching this config.
    episode_steps: int = 4500
    boot_frames: int = 3600       # only used when no boot state exists
    noop_max: int = 30            # random no-op steps at reset, for diversity
    sticky_prob: float = 0.0      # action-repeat stochasticity

    # reward
    cash_scale: float = 0.01
    score_scale: float = 0.01
    death_penalty: float = 1.0
    step_penalty: float = 0.0
    reward_clip: float = 5.0
    distance_scale: float = 0.0      # dense movement reward; try 0.002
    # Pays once per newly entered world cell per episode. Unlike
    # distance_scale this cannot be farmed by jittering in place.
    novelty_scale: float = 0.0
    novelty_cell: float = 256.0
    respawn_distance: float = 1500.0  # position jump read as a death; 0 off
    # On-screen messages (gta1env/text.py). Ids need the decoded text
    # table from tools/gta_text.py; without it, text is still reported.
    read_messages: bool = True
    text_table: str = os.path.join(HERE, "scratch", "text", "english.json")
    # Paid once when the message appears. Default off: not yet seen live.
    mission_complete_reward: float = 0.0
    mission_failed_penalty: float = 0.0

    actions: list = field(default_factory=lambda: list(DEFAULT_ACTIONS))
    enable_audio: bool = False
    verbose: bool = False
    # "auto" uses a real BIOS image from bios_dir; "HLE" uses the core's
    # built-in high-level emulation and needs no BIOS file at all. HLE costs
    # compatibility and changes boot timing, so prefer a real dump once you
    # have one -- but it lets you start today.
    bios_mode: str = "auto"
    # Only the mock-core tests set this: real runs must have both files.
    require_assets: bool = True


class GTA1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config=None, render_mode=None, seed=None,
                 **overrides):
        """config: a GTA1Config, or leave it out and pass its fields as
        keyword arguments -- which is what gym.make("GTA1-v0", ...) does."""
        super().__init__()
        self.cfg = config or GTA1Config()
        if overrides:
            unknown = set(overrides) - {f.name for f in fields(GTA1Config)}
            if unknown:
                raise TypeError(
                    f"unknown GTA1Config field(s): {', '.join(sorted(unknown))}")
            self.cfg = replace(self.cfg, **overrides)
        self.render_mode = render_mode
        # Applied at the first reset() that is not given a seed.
        self._init_seed = seed

        if not self.cfg.rom_path:
            self.cfg.rom_path = find_disc()
        self._check_assets()
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        options = dict(DETERMINISTIC_CORE_OPTIONS)
        options["pcsx_rearmed_bios"] = self.cfg.bios_mode
        self.core = LibretroCore(
            self.cfg.core_path, self.cfg.bios_dir, self.cfg.save_dir,
            options=options,
            enable_audio=self.cfg.enable_audio, verbose=self.cfg.verbose)
        self.av = self.core.load_game(self.cfg.rom_path)
        self.ram = self.core.memory()
        if self.ram is None:
            raise RuntimeError("core exposed no SYSTEM_RAM region")

        self.ram_map = RamMap.load(self.cfg.ram_map_path)
        self.shaper = RewardShaper(
            cash_scale=self.cfg.cash_scale, score_scale=self.cfg.score_scale,
            death_penalty=self.cfg.death_penalty,
            step_penalty=self.cfg.step_penalty, clip=self.cfg.reward_clip,
            distance_scale=self.cfg.distance_scale,
            respawn_distance=self.cfg.respawn_distance,
            novelty_scale=self.cfg.novelty_scale,
            novelty_cell=self.cfg.novelty_cell)

        self.messages = (MessageReader(self.cfg.text_table)
                         if self.cfg.read_messages else None)
        self._msg = {}

        self.actions = [tuple(a) for a in self.cfg.actions]
        self._button_sets = [
            tuple(CONTROLS[i] for i in combo) for combo in self.actions]
        self.action_space = spaces.Discrete(len(self.actions))

        chans = self.cfg.frame_stack * (1 if self.cfg.grayscale else 3)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(chans, self.cfg.obs_height, self.cfg.obs_width),
            dtype=np.uint8)

        self._stack = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._steps = 0
        self._last_action = 0
        self.boot_state_names = []
        self._boot_blobs = self._load_or_make_boot_state()
        self._boot_index = 0

    # ------------------------------------------------------------- startup

    def _check_assets(self):
        if not self.cfg.require_assets:
            return
        missing = []
        if not os.path.exists(self.cfg.core_path):
            missing.append(f"libretro core: {self.cfg.core_path}")
        if not os.path.exists(self.cfg.rom_path):
            missing.append(f"GTA PS1 disc image: {self.cfg.rom_path}")
        if self.cfg.bios_mode.lower() != "hle":
            bios = []
            if os.path.isdir(self.cfg.bios_dir):
                bios = [f for f in os.listdir(self.cfg.bios_dir)
                        if f.lower().endswith((".bin", ".rom"))]
            if not bios:
                missing.append(
                    f"PS1 BIOS image in: {self.cfg.bios_dir}"
                    "  (or set bios_mode='HLE' to run without one)")
        if missing:
            raise FileNotFoundError(
                "Cannot start GTA1Env; provide your own dumps of:\n  - "
                + "\n  - ".join(missing))

    def _load_or_make_boot_state(self):
        """Savestates taken past the boot/menus are what make resets cheap.

        cfg.boot_state may name one file or a directory of ``*.state``
        files. A directory is sampled per reset, which is the cheapest way
        to stop every episode beginning from the identical frame -- with a
        single state, all envs see the same opening and explore the same
        corner of the map. Returns a list of blobs.
        """
        path = self.cfg.boot_state
        if os.path.isdir(path):
            names = sorted(f for f in os.listdir(path)
                           if f.endswith(".state"))
            if not names:
                raise FileNotFoundError(
                    f"boot_state directory holds no *.state files: {path}")
            blobs = []
            for name in names:
                with open(os.path.join(path, name), "rb") as fh:
                    blobs.append(fh.read())
            self.boot_state_names = names
            return blobs
        if os.path.exists(path):
            with open(path, "rb") as fh:
                self.boot_state_names = [os.path.basename(path)]
                return [fh.read()]
        # No state yet: boot once now and keep the result in memory so that
        # at least resets within this process are fast.
        for _ in range(self.cfg.boot_frames):
            self.core.run()
        self.boot_state_names = ["<booted>"]
        return [self.core.serialize()]

    def set_reward_scales(self, **scales):
        """Retune the shaper mid-training (reached via SubprocVecEnv.call).

        Used to wean the agent off the dense movement bonus: distance gets
        it driving early, but left at full weight it is what the policy
        ends up maximising, so it is annealed down while the sparse game
        signals keep their weight.
        """
        allowed = {"cash_scale", "score_scale", "distance_scale",
                   "death_penalty", "step_penalty", "health_scale",
                   "novelty_scale"}
        unknown = set(scales) - allowed
        if unknown:
            raise TypeError(f"unknown reward scale(s): "
                            f"{', '.join(sorted(unknown))}")
        for key, value in scales.items():
            if value is not None:
                setattr(self.shaper, key, float(value))
                setattr(self.cfg, key, float(value))
        return {k: getattr(self.shaper, k) for k in sorted(allowed)}

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        if seed is None and self._init_seed is not None:
            seed, self._init_seed = self._init_seed, None
        super().reset(seed=seed)       # seeds self.np_random
        self._boot_index = int(self.np_random.integers(
            0, len(self._boot_blobs)))
        self.core.unserialize(self._boot_blobs[self._boot_index])
        self.core.set_buttons(())
        self._stack[:] = 0
        self._steps = 0
        self._last_action = 0

        for _ in range(int(self.np_random.integers(
                0, self.cfg.noop_max + 1))):
            self._advance(())
        frame = self._advance(())
        obs = self._push(frame)
        self.shaper.reset(self._read_state())
        if self.messages is not None:
            # Prime, so a message already on screen at reset is not an event.
            self.messages.reset()
            self.messages.read(self.ram)
            self._msg = self.messages.read(self.ram)
        return obs, self._info()

    def step(self, action):
        action = int(action)
        if (self.cfg.sticky_prob
                and self.np_random.random() < self.cfg.sticky_prob):
            action = self._last_action
        self._last_action = action

        frame = self._advance(self._button_sets[action])
        obs = self._push(frame)
        self._steps += 1

        state = self._read_state()
        reward, dead = self.shaper(state)
        if self.messages is not None:
            self._msg = self.messages.read(self.ram)
            for mid in self._msg["new_message_ids"]:
                if mid == MISSION_COMPLETE:
                    reward += self.cfg.mission_complete_reward
                elif mid == MISSION_FAILED:
                    reward -= self.cfg.mission_failed_penalty
        terminated = bool(dead)
        truncated = self._steps >= self.cfg.episode_steps
        return obs, reward, terminated, truncated, self._info(state)

    def _advance(self, buttons):
        """Hold `buttons` for frameskip emulated frames; return last frame."""
        self.core.set_buttons(buttons)
        last = self.cfg.frameskip - 1
        for i in range(self.cfg.frameskip):
            # Only the final frame of the window is ever observed, so the
            # others skip RGB conversion.
            self.core.run(convert=(i == last))
        return self.core.frame

    def _push(self, frame):
        proc = self._preprocess(frame)
        step = 1 if self.cfg.grayscale else 3
        self._stack[:-step] = self._stack[step:]
        self._stack[-step:] = proc
        return self._stack.copy()

    def _preprocess(self, frame):
        h, w = self.cfg.obs_height, self.cfg.obs_width
        if frame is None:
            return np.zeros((1 if self.cfg.grayscale else 3, h, w),
                            dtype=np.uint8)
        if self.cfg.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            return small[None, :, :]
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return np.transpose(small, (2, 0, 1))

    def _read_state(self):
        if self.ram_map.is_empty:
            return {}
        return self.ram_map.read_all(self.ram)

    def _info(self, state=None):
        info = {"steps": self._steps,
                "boot_index": self._boot_index,
                "cells": len(self.shaper.visited),
                "frame_shape": (self.core.height, self.core.width)}
        info.update(state if state is not None else self._read_state())
        info.update(self._msg)
        return info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.core.frame
        return None

    # ------------------------------------------------- savestates & cleanup

    def save_state(self, path):
        blob = self.core.serialize()
        with open(path, "wb") as fh:
            fh.write(blob)
        return path

    def load_state(self, path):
        with open(path, "rb") as fh:
            self.core.unserialize(fh.read())

    def set_boot_state(self, path):
        with open(path, "rb") as fh:
            self._boot_blob = fh.read()

    def close(self):
        if getattr(self, "core", None) is not None:
            self.core.close()
            self.core = None
