# This script generates textual prompt. The prompt could then be fed to GPT.

import gymnasium as gym
from minigrid.core.constants import *
import matplotlib.pyplot as plt
from gpt_interface import *

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}

def img_to_str(img):
    result = ""
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            obj = IDX_TO_OBJECT[img[i, j, 0]]
            color = IDX_TO_COLOR[img[i, j, 1]]
            state = IDX_TO_STATE[img[i, j, 2]]
            if obj in ["unseen", "empty", "wall"]:
                result += f"{obj}, "
            elif obj in ["ball", "box"]:
                result += f"{color} {obj}, "
            else:
                result += f"{color} {state} {obj}, "
        result += '\n'
    return result

def get_prompt_str(obs):
    image, direction, mission = img_to_str(obs['image']), DIR_TO_STR[obs['direction']], obs['mission']
    return f'''
You are an agent in a Minigrid environment. Your mission is {mission}. The objective location may be in a room different from the one you are in. Rooms are connected with doors. Some doors are unlocked and can be interacted with toggles, however, other doors are locked and you need to find the key of the matching color to open it. You want to do this in as few steps as possible.
Your agent's direction is currently {direction}.
Your agent can only see in front of itself. It cannot see blocked objects. Here is the vision of your agent. Assume your agent is at the center of the last row. The first row is the furthest in front of you:
{image}

Evaluate how this state is helpful for achieving the goal, using a number between -1 and 1. Please only return that single number, and do not return anything else. Do not explain your reasoning, just provide a reward.
'''

def gpt_reward_func(obs):
    prompt = get_prompt_str(obs)
    return float(generate_reward_from_gpt(prompt))


# The things below are just test code.
if __name__ == "__main__":
    env = gym.make("BabyAI-GoToImpUnlock-v0", render_mode='rgb_array')
    # Reset the environment to get the initial state
    obs = env.reset()
    # Take some actions and continue displaying the state
    for _ in range(1):
        action = env.action_space.sample()  # Replace with your desired action
        obs, reward, terminated, truncated, info = env.step(action)
        plt.figure()
        plt.imshow(env.render())
        print(get_prompt_str(obs))
    plt.show()

