import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from minigrid.wrappers import ViewSizeWrapper

def make_env(env_key, seed=None, render_mode=None, obs_size=7):
    env = gym.make(env_key, render_mode=render_mode)
    # env = FullyObsWrapper(env)
    env = ViewSizeWrapper(env, agent_view_size=obs_size) #default is 7
    return env
#7 11 13 9 fourroom