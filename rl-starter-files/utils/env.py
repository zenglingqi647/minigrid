import gymnasium as gym
from minigrid.wrappers import PositionBonus
from envs.four_rooms import FourRoomsEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.wrappers import ViewSizeWrapper
from envs.register import *

def make_env(env_key, seed=None, render_mode=None, obs_size=7):
    '''
    obs_size: -1 for fully observable
    '''
    if render_mode:
        env = gym.make(env_key, render_mode=render_mode)
    else:
        env = gym.make(env_key)
    # env = FullyObsWrapper(env)
    env.reset(seed=seed)
    if obs_size == -1:
        env = FullyObsWrapper(env)
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