# This script generates textual prompt. The prompt could then be fed to GPT.

import gymnasium as gym
from minigrid.core.constants import *
import matplotlib.pyplot as plt
from .gpt_interface import *
import random

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}
ACTION_TO_STR = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up the object in front", 4: "drop the object in front", 5: "toggle the object in front", 6: "finish the environment"}

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

def get_prompt_str(obs, action):
    image, direction, mission = img_to_str(obs['image']), DIR_TO_STR[obs['direction']], obs['mission']
    act = ACTION_TO_STR[action.item()]
    return f'''
You are an agent in a Minigrid environment. Your mission is {mission}. The objective location may be in a room different from the one you are in. Rooms are connected with doors. Some doors are unlocked and can be interacted with toggles, however, other doors are locked and you need to find the key of the matching color to open it. You want to do this in as few steps as possible.
Your agent's direction is currently {direction}.
Your agent can only see in front of itself. It cannot see blocked objects. Here is the vision of your agent. Assume your agent is at the center of the last row. The first row is the furthest in front of you:
{image}

Your agent would like to {act}. Evaluate how this state and action is helpful for achieving the goal, using a number between -1 and 1. Please only return that single number, and do not return anything else. Do not explain your reasoning, just provide a reward.
'''

class GPTRewardFunction():
    def __init__(self, query_gpt_prob, ask_every, gpt_prob_decay=1):
        self.query_gpt_prob = query_gpt_prob
        self.ask_interval = ask_every
        self.gpt_prob_decay = gpt_prob_decay
        self.counter = 0

    def should_ask_gpt(self):
        if self.query_gpt_prob == -1:
            if self.counter <= 0:
                self.counter = self.ask_interval
                return True
            else:
                self.counter -= 1
                return False
        else:
            return random.random() < self.query_gpt_prob

    def reshape_reward(self, observation, action, reward, done):
        if self.should_ask_gpt():
            gpt_reward = gpt_reward_func(observation, action)
            print(f"gpt reward is {gpt_reward}")
            shaped_reward = reward + gpt_reward
        else:
            shaped_reward = reward
        # self.query_gpt_prob *= self.gpt_prob_decay
        return shaped_reward
            
def gpt_reward_func(obs, action):
    prompt = get_prompt_str(obs, action)
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

