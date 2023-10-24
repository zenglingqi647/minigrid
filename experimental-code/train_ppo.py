# The code below trains a PPO agent.

import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import gymnasium as gym
import torch
import torch.nn as nn
from textual_minigrid import gpt_reward_func
import numpy as np

class GPTShapedEnv(gym.Env):
    def __init__(self, query_gpt_prob=0.2, gpt_prob_decay=0.99):
        super(GPTShapedEnv, self).__init__()
        self.env = gym.make('BabyAI-GoToImpUnlock-v0')
        self.query_gpt_prob = query_gpt_prob
        self.gpt_prob_decay = gpt_prob_decay

    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping logic here
        if np.random.rand() < self.query_gpt_prob:
            shaped_reward = original_reward + gpt_reward_func(observation)
        else:
            shaped_reward = original_reward
        self.query_gpt_prob *= self.gpt_prob_decay

        return observation, shaped_reward, terminated, truncated, info

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = GPTShapedEnv()
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(2e5, progress_bar=True)