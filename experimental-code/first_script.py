# This script displays the MiniGrid Empty 5x5 environment.
# It just verifies if your things are installed correctly.
import gymnasium as gym

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()