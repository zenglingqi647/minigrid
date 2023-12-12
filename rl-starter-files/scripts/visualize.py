import argparse
import numpy
import pygame

import utils
from utils import device

import time
from utils.textual_minigrid import ACTION_TO_STR


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--llm-planner-variant", type=str, default=None, 
                    help="Specify llm variant, leave empty for no llm")
parser.add_argument("--ask-every", type=int, default=500, help="how often to ask the planner")
parser.add_argument("--obs-size",
                    type=int,
                    default=7,
                    help="size of observation for skill agents, should be an odd number (default: 7)")


args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment
env = utils.make_env(args.env, args.seed, render_mode="human", obs_size=0)
for _ in range(args.shift + 1):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text, planner_variant=args.llm_planner_variant, ask_every=args.ask_every, skill_obs_size=args.obs_size, procs=1)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()
    running = True

    while running:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)
        print(ACTION_TO_STR[action])
        time.sleep(0.5)
        if done:
            print(f"reward is {reward}")
            break

        for event in pygame.event.get():
            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    print("skipped, reward is 0.")
                    running = False
                    break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
