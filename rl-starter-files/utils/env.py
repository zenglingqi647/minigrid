import gymnasium as gym
from minigrid.wrappers import PositionBonus
from customized_env.four_rooms import FourRoomsEnv

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
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