# This script generates textual prompt. The prompt could then be fed to GPT.

import gymnasium as gym
from minigrid.core.constants import *
import matplotlib.pyplot as plt
from utils.env import make_env
from utils.gpt_interface import *
from utils.llama_interface import *
from utils.human_interface import *
from utils.format import Vocabulary
from minigrid.wrappers import FullyObsWrapper
import random
import re
import json

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}
ACTION_TO_STR = {0: "turn left", 1: "turn right", 2: "move forward", 3: "pick up the object in front", 4: "drop the object in front", 5: "toggle the object in front", 6: "finish the environment"}
ROOM_SIZE = 7

def get_relative_position(i, j, img):
    center = img.shape[0] // 2
    dist_h = img.shape[1] - j - 1
    dist_v = i - center
    if dist_v < 0:
        return f"{dist_h} steps in front, {-dist_v} steps to the left"
    elif dist_v > 0:
        return f"{dist_h} steps in front, {dist_v} steps to the right"
    else:
        return f"{dist_h} steps directly in front"

def get_absolute_position(i, j, img):
    return f"on row {j} column {i}"

def get_object_name(r, c, img):
    obj = IDX_TO_OBJECT[img[c, r, 0]]
    if obj == "agent":
        return None
    color = IDX_TO_COLOR[img[c, r, 1]]
    state = IDX_TO_STATE[img[c, r, 2]]
    obj_info = None
    if obj in ["ball", "box", "key"]:
        obj_info = f"{color} {obj}"
    elif obj in ["door"]:
        obj_info = f"{color} {state} {obj}"
    return obj_info

def process_room(room_row, room_col, img):
    result = f"Room ({room_row}, {room_col}): \n"

    room_r, room_c = room_row * ROOM_SIZE, room_col * ROOM_SIZE
    # Process the walls
    for i in range(ROOM_SIZE):
        # Top
        door = get_object_name(room_r, room_c + i, img)
        if door:
            result += f"Connected to ({room_row - 1}, {room_col}) via a {door}. \n"
        
        
        door = get_object_name(room_r + ROOM_SIZE, room_c + i, img)
        if door:
            result += f"Connected to room ({room_row + 1}, {room_col}) via a {door}. \n"
        
        # Left and right
        door = get_object_name(room_r + i, room_c, img)
        if door:
            result += f"Connected to room ({room_row}, {room_col - 1}) via a {door}. \n"
        
        door = get_object_name(room_r + i, room_c + ROOM_SIZE, img)
        if door:
            result += f"Connected to room ({room_row}, {room_col + 1}) via a {door}. \n"
    
    # Now process the interior
    for r in range(1, 7):
        for c in range(1, 7):
            abs_r, abs_c = room_row * 7 + r, room_col * 7 + c
            o = get_object_name(abs_r, abs_c, img)
            if o:
                result += f"{o} at row {r} and column {c} \n"
    
    result += "\n"
    return result

def img_to_str(img):
    result = ""
    for room_row in range(3):
        for room_col in range(3):
            result += process_room(room_row, room_col, img)
    return result

def get_agent_position(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if IDX_TO_OBJECT[img[j, i, 0]] == "agent":
                return i // ROOM_SIZE, j // ROOM_SIZE
    raise Exception("This shouldn't have happened...")

def get_reward_prompt_str(obs, action):
    image, direction, mission = img_to_str(obs['image']), DIR_TO_STR[obs['direction']], obs['mission']
    act = ACTION_TO_STR[action.item()]
    return f'''
You are an agent in a Minigrid environment. Your mission is {mission}. Your agent's direction is currently {direction}.
Your agent can only see in front of itself. It cannot see blocked objects.
{image}

Your agent would like to {act}. Evaluate how this state and action is helpful for achieving the goal, using a number between -1 and 1. Please only return that single number, and do not return anything else. Do not explain your reasoning, just provide a reward.
'''

def get_planning_prompt_str(obs_img, mission_txt):
    room_r, room_c = get_agent_position(obs_img)
    image_as_str, mission = img_to_str(obs_img), mission_txt
    json_example = "{ skill : 0, goal : 'go to the red box' }"
    return f'''You are an agent in a Minigrid environment. Your agent is in Room {(room_r, room_c)}. Rooms are 7 by 7, and there are 9 of them, connected by doors. Your mission is {mission}. Here is the state of the entire environment: 
    
    {image_as_str}

You have the following skills and their allowed goal grammar below:
Skill 0: Go to Object
    "go to the [color] [type]"
    [color]: the color of the object. Allowed values are "red", "green", "blue", "purple", "yellow" or "grey".
    [type]: the type of the object. Allowed values are "ball", "box", "key".
Skill 1: Open door
    "open the [color] door"
    [color]: the color of door. Allowed values are "red", "green", "blue", "purple", "yellow" or "grey".
Skill 2: Pickup an item
    "pick up the [color] [type]"
    [color]: the color of the object. Allowed values are "red", "green", "blue", "purple", "yellow" or "grey".
    [type]: the type of the object. Allowed values are "ball", "box", or "key".
Skill 3: Unlock a door
    "unlock the [color] door"
    [color]: the color of the object. Allowed values are "red", "green", "blue", "purple", "yellow" or "grey".

Based on the current state of the agent, what is the first skill it should use, and what would be the short-term goal for that skill (most likely different from the long-term) ? Format your answer in json format. As an example, you can return {json_example}.
'''

def gpt_skill_planning(obs_img, mission_txt):
    prompt = get_planning_prompt_str(obs_img, mission_txt)
    response = interact_with_gpt(prompt)
    answer = json.loads(response)
    return answer['skill'], answer['goal']

def llama_skill_planning(obs, mission_txt):
    prompt = get_planning_prompt_str(obs, mission_txt)
    response = interact_with_llama(prompt)
    match = re.search(r'Answer: Skill (\d)', response)
    if match:
        skill = int(match.group(1))
    return skill

def human_skill_planning():
    skill_num, goal_text = interact_with_human()
    return skill_num, goal_text

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
    prompt = get_reward_prompt_str(obs, action)
    return float(interact_with_gpt(prompt))


# The things below are dry run test code
if __name__ == "__main__":
    env = make_env("BabyAI-GoToImpUnlock-v0", seed=1, render_mode="human", obs_size=-1)
    # Reset the environment to get the initial state
    obs, _ = env.reset()
    # Take some actions and continue displaying the state
    for _ in range(1):
        # action = env.action_space.sample()  # Replace with your desired action
        # obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(obs)
        print(get_planning_prompt_str(obs['image'], obs['mission']))
    while True:
        pass

