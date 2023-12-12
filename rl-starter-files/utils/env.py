import gymnasium as gym
from minigrid.wrappers import PositionBonus
from envs.four_rooms import FourRoomsEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.wrappers import ViewSizeWrapper
from envs.register import *
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX


def make_env(env_key, seed=None, render_mode=None, obs_size=7):
    '''
    obs_size: 0 for partial and fully observations at the same time, positive integers for partial observation. obs['full_obs'] is the full observation.
    obs_size: -1 for fully observation only
    '''
    if render_mode:
        env = gym.make(env_key, render_mode=render_mode)
    else:
        env = gym.make(env_key)
    # env = FullyObsWrapper(env)
    env.reset(seed=seed)
    if obs_size == -1:
        env = FullyObsWrapper(env)
    elif obs_size == 0:
        # default agent view size is 11
        env = PartialFullWrapper(env, agent_view_size=11)
    else:
        env = ViewSizeWrapper(env, agent_view_size=obs_size)  #default is 7
    env.reset(seed=seed)
    return env


def make_env_pos_bonus(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    env = PositionBonus(env)
    env.reset(seed=seed)
    return env


def make_four_rooms_env(seed=42, size=19):
    env = FourRoomsEnv(size=size)
    env.reset(seed=seed)
    return env


class PartialFullWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8")

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict({**self.observation_space.spaces, "image": new_image_space})

    def observation(self, obs):
        env = self.unwrapped

        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir])
        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image, "full_obs": full_grid}


if __name__ == "__main__":
    env = gym.make("BabyAI-OpenTwoDoors-v0")
    obs, _ = env.reset()
    print(obs['image'].shape)
    env = PartialFullWrapper(env, agent_view_size=11)
    obs, _ = env.reset()
    print(obs['full_obs'].shape)
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    print(obs['image'].shape)