"""GTA1Sim — the original 1997 Grand Theft Auto as a Gymnasium environment."""

from gymnasium.envs.registration import register

from .env import GTA1Config, GTA1Env, find_disc
from .libretro import LibretroCore
from .ram import CONTROLS, DEFAULT_ACTIONS, RamMap, RewardShaper

__all__ = ["GTA1Env", "GTA1Config", "LibretroCore", "RamMap", "RewardShaper",
           "CONTROLS", "DEFAULT_ACTIONS", "find_disc", "make"]

# The env truncates on its own `max_episode_steps`, so no TimeLimit wrapper
# here -- adding one would truncate twice at different horizons.
register(
    id="GTA1-v0",
    entry_point="gta1env.env:GTA1Env",
    max_episode_steps=None,
    nondeterministic=False,
)


def make(num_envs=None, **kwargs):
    """One env, or a vectorised bundle of them.

        env = gta1env.make(frameskip=4)
        vec = gta1env.make(num_envs=8, frameskip=4)

    Keyword arguments are GTA1Config fields. The vectorised form returns a
    SubprocVecEnv, because a libretro core can only be driven safely by one
    process at a time.
    """
    if num_envs is None:
        import gymnasium as gym
        return gym.make("GTA1-v0", **kwargs)
    from .vec import SubprocVecEnv
    return SubprocVecEnv(num_envs, cfg_kwargs=kwargs)
