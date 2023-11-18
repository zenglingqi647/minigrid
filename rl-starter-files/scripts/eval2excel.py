import argparse
import time
import torch
import pandas as pd  # Import pandas for Excel file handling
from torch_ac.utils.penv import ParallelEnv
import utils
from utils import device
import os
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--res-dir", type=str, default='/data1/lzengaf/cs285/proj/minigrid/rl-starter-files/storage/',
                    help="random seed (default: 0)")
parser.add_argument("--excel-dir", type=str, default='/data1/lzengaf/cs285/proj/minigrid/rl-starter-files',
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=True,
                    help="add a GRU to the model")

args = parser.parse_args()

# Create a DataFrame to store results
df = pd.DataFrame(columns=["env", "F", "FPS", "D", "R:μ", "R:σ", "R:m", "R:M", "F:μ", "F:σ", "F:m", "F:M"])

for model in tqdm(os.listdir(args.res_dir)):
    args.env = model.split('_')[0]
    args.model = model

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    print(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()
    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Store results in the DataFrame
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    results = {
        "model": args.model,
        "env": args.env,
        "F": num_frames,
        "FPS": fps,
        "D": duration,
        **dict(zip(["R:μ", "R:σ", "R:m", "R:M"], return_per_episode.values())),
        **dict(zip(["F:μ", "F:σ", "F:m", "F:M"], num_frames_per_episode.values()))
    }
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)


# Save the DataFrame to an Excel file
df.set_index("model", inplace=True)  # Set 'env' as the row index
df.to_excel(f"{args.excel_dir}/results.xlsx")
