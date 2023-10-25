import gymnasium as gym
from minigrid.core.constants import *
import matplotlib.pyplot as plt
import random
import openai
import json

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}
ACTION_TO_STR = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up the object",
    4: "drop the object",
    5: "toggle the object",
    6: "finish the environment"
}


class GPTRewardFunction():

    def __init__(self, query_gpt_interval=100, decay=0.7):
        self.query_gpt_interval = query_gpt_interval
        self.trajectory = []
        self.steps_since_last_query = 0
        self.decay = decay

    def reshape_reward(self, observation, action, reward, done):
        # If it's time to query GPT or the trajectory is empty
        if self.steps_since_last_query >= self.query_gpt_interval or not self.trajectory:
            self.trajectory = self.get_gpt_trajectory(observation)
            self.steps_since_last_query = 0

        # Check if the agent's action aligns with the expected trajectory
        expected_action = self.trajectory.pop(0)
        if action.item() == expected_action:
            shaped_reward = reward + self.decay  # Positive reward for following the trajectory
            self.decay *= self.decay
        else:
            shaped_reward = reward  # Neutral or negative reward for deviating

        self.steps_since_last_query += 1
        return shaped_reward

    def get_gpt_trajectory(self, obs):
        prompt = get_prompt_str(obs)
        trajectory_json = trajectory_gen(obs)
        trajectory_str = json.loads(trajectory_json)
        # Convert the trajectory string to a list of actions
        trajectory = [trajectory_str[f'{i}'] for i in range(10)]
        return trajectory


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
    return f'''Your mission is to {mission}. You're currently facing {direction}. Here's what you see:
{image}
What are the best actions to take in the next 10 steps to achieve your mission?
Answer in json format. Each action is represented by an integer from 0 to 6. It should be like:
'{{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 0, "8": 1, "9": 2}}'
'''


def trajectory_gen(obs):
    prompt = get_prompt_str(obs)
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":
                    "system",
                "content":
                    '''You are now a route planning assistant in the Minigrid game. Each action is represented by an integer from 0 to 6, and here's the mapping:
0: "turn left",
1: "turn right",
2: "move forward",
3: "pick up the object",
4: "drop the object",
5: "toggle the object",
6: "finish the environment"
'''
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.3,
    )
    return output.choices[0].message['content']


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
    # plt.savefig("test.png")
    print('llm response: ')
    print(trajectory_gen(obs))
    print('Here is the trajectory: ')
    print(GPTRewardFunction().get_gpt_trajectory(obs))