import argparse
import time
import datetime
import torch_ac
from torch_ac import DictList
import tensorboardX
import sys
from tqdm import tqdm

import utils
import numpy as np
import torch
from utils import device
from model import ACModel
import utils.pytorch_util as ptu
from utils.trajectory_reward import LLMRewardFunction
from utils.textual_minigrid import GPTRewardFunction
from utils.planner_policy import PlannerPolicy
from utils.q_planner_policy import QPlannerPolicy
from minigrid.wrappers import PositionBonus

from torch_ac.algos.replay_buffer import ReplayBuffer

from utils.prompt_validation import similarity

# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True, help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True, help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval",
                    type=int,
                    default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16, help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7, help="number of frames of training (default: 1e7)")
parser.add_argument("--obs-size",
                    type=int,
                    default=7,
                    help="size of observation for environment, should be an odd number (default: 7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256, help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc",
                    type=int,
                    default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99, help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda",
                    type=float,
                    default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8, help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99, help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2, help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument(
    "--recurrence",
    type=int,
    default=1,
    help=
    "number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory."
)
parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model to handle text input")

# Parameters for LLM
parser.add_argument("--llm-reward-variant", default=None, help="Use LLM to shape rewards. Possible values: 'gpt-3.5-turbo', 'gpt-4'")
parser.add_argument("--use-trajectory",
                    action="store_true",
                    default=False,
                    help="Use GPT to predict the trajectory for reward")
parser.add_argument('--traj-r-decay', type=float, default=0.7, help='Decay factor for trajectory reward')
parser.add_argument('--llm-temperature', type=float, default=0.3, help='Temperature for LLM reward')
parser.add_argument("--ask-gpt-prob", type=float, default=-1, help="Probability of Asking GPT")
parser.add_argument("--ask-every", type=float, default=20, help="Number of update iterations between asking GPT")
parser.add_argument("--llm-planner-variant", type=str, default=None, help="LLM Planner Variant")
parser.add_argument("--use-position-bonus",
                    action="store_true",
                    default=False,
                    help="uses a high level planner network")
parser.add_argument("--custom-hw",
                    default=19,
                    type=int,
                    help="customize the height and width of the Minigrid-FourRooms gridworld, h==w")

# Parameter for DQN Planner
parser.add_argument("--use-dqn",
                    action="store_true",
                    default=False,
                    help="if dqn planner is used")
parser.add_argument("--llm-augmented",
                    action="store_true",
                    default=False,
                    help="if dqn planner is going to be llm augmented")


def similarity_bonus(llm_rsp_skill, llm_rsp_goal, dqn_rsp_skill, dqn_rsp_goal):
    return torch.tensor([similarity(llm_rsp_skill[i], llm_rsp_goal[i], dqn_rsp_skill[i], dqn_rsp_goal[i]) for i in range(len(llm_rsp_skill))])

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    askevery_str = f"_askevery{args.ask_every}" if args.llm_reward_variant or args.llm_planner_variant else ""
    append = f'_hw{args.custom_hw}' if args.env == 'FourRooms' else ''
    llm_reward_variant = f'_llm{args.llm_reward_variant}' if args.llm_reward_variant else ''
    llm_planner_variant = f'_llmplanner{args.llm_planner_variant}' if args.llm_planner_variant else ''

    default_model_name = f"{args.env}_{args.algo}_rec{args.recurrence}_f{args.frames}_fp{args.frames_per_proc}_seed{args.seed}{llm_reward_variant}{llm_planner_variant}{askevery_str}{append}_{date}"

    print(
        f"llm{args.llm_reward_variant}_{llm_planner_variant}_traj{args.use_trajectory}_trajrdecay{args.traj_r_decay}_llmtemp{args.llm_temperature}_askgptprob{args.ask_gpt_prob}{askevery_str}_useposbonus{args.use_position_bonus}"
    )

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")
    ptu.init_gpu()

    # Load environments
    envs = []
    for i in range(args.procs):
        if args.use_position_bonus:
            envs.append(utils.make_env_pos_bonus(args.env, args.seed + 10000 * i))
        if args.env == 'FourRooms':
            envs.append(utils.make_four_rooms_env(args.seed + 10000 * i, size=9))
        else:
            envs.append(utils.make_env(args.env, args.seed + 10000 * i, obs_size=args.obs_size))
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    if args.use_dqn:
        acmodel = QPlannerPolicy(obs_space, envs[0].action_space, preprocess_obss.vocab, llm_variant=args.llm_planner_variant, ask_cooldown=args.ask_every, num_procs=args.procs, use_memory=args.mem, use_text=args.text, num_skills=4, llm_augmented=args.llm_augmented)
        if args.llm_augmented:
            llm_model = PlannerPolicy(obs_space, envs[0].action_space, preprocess_obss.vocab, llm_variant=args.llm_planner_variant, ask_cooldown=args.ask_every, num_procs=args.procs, use_memory=args.mem, use_text=args.text)
    elif args.llm_planner_variant is not None:
        acmodel = PlannerPolicy(obs_space, envs[0].action_space, preprocess_obss.vocab, llm_variant=args.llm_planner_variant, ask_cooldown=args.ask_every, num_procs=args.procs, use_memory=args.mem, use_text=args.text)
    else:
        acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        if not args.frames_per_proc:
            args.frames_per_proc = 5
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        if not args.frames_per_proc:
            args.frames_per_proc = 128
        if args.llm_reward_variant is not None and args.use_trajectory:
            reshape_reward = LLMRewardFunction(query_interval=args.ask_every,
                                               decay=args.traj_r_decay,
                                               llm_temperature=args.llm_temperature,
                                               llm=args.llm_reward_variant).reshape_reward
        elif args.llm_reward_variant is not None:
            reshape_reward = GPTRewardFunction(query_gpt_prob=args.ask_gpt_prob,
                                               ask_every=args.ask_every).reshape_reward
        else:
            reshape_reward = None
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
        reshape_reward)
    elif args.algo == "base":
        algo = torch_ac.BaseAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence, preprocess_obss, None)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    total_frames = num_frames + args.frames

    if args.use_dqn:
        replay_buffer = ReplayBuffer()


    with tqdm(initial=num_frames, total=total_frames) as pbar:
        while num_frames < total_frames:
            # Update model parameters
            update_start_time = time.time()
            if args.use_dqn:
                exps, logs1 = algo.collect_experiences(replay_buffer=replay_buffer)
            else:
                exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            pbar.update(logs["num_frames"])
            update += 1

            # Update Planner
            if args.use_dqn:
                batch = replay_buffer.sample(args.batch_size)

                full_obs, text = [], []
                next_full_obs, next_text = [], []
                for i in range(args.batch_size):
                    obs_dict, next_obs_dict = batch["observations"][i], batch["next_observations"][i]
                    # Replay buffer does not store DictList, but just store dicts.
                    full_obs.append(obs_dict['full_obs'])
                    text.append(obs_dict['text'])
                    next_full_obs.append(next_obs_dict['full_obs'])
                    next_text.append(next_obs_dict['text'])
                
                obs = DictList({ "full_obs" : ptu.from_numpy(np.stack(full_obs)), "text" : ptu.from_numpy(np.stack(text)) })
                next_obs = DictList({"full_obs" : ptu.from_numpy(np.stack(next_full_obs)), "text" : ptu.from_numpy(np.stack(next_text)) })
                rewards = ptu.from_numpy(batch["rewards"])
                dones = ptu.from_numpy(batch["dones"])
                actions = ptu.from_numpy(batch["actions"])

                if args.llm_augmented:
                    llm_rsp_skill, _, llm_rsp_goal = llm_model.get_skills_and_goals(obs)
                    dqn_rsp_skill, _, dqn_rsp_goal = acmodel.get_skills_and_goals(obs)
                    bonus =  similarity_bonus(llm_rsp_skill, llm_rsp_goal, dqn_rsp_skill, dqn_rsp_goal)
                    assert(bonus.shape == rewards.shape)
                    rewards = rewards + bonus

                # TODO: when using the critic network, we directly pass in the embeddings (processed already by the CNN and GRU and concatenated together)
                # When doing the target critic, we probably also want the same thing?
                obs_embedded, next_obs_embedded = acmodel.get_embeddings(obs), acmodel.get_embeddings(next_obs)
                dqn_log = acmodel.dqn_agent.update(            
                    obs_embedded,
                    actions,
                    rewards,
                    next_obs_embedded,
                    dones,
                    update,
                )
                logs.update(dqn_log)

            if isinstance(acmodel, PlannerPolicy):
                acmodel.decrease_cooldown()
                print(f"Asking LLM in {acmodel.timer} updates.")


            # Print logs
            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                
                logs = "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} ".format(*data)

                if "entropy" in logs:
                    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                    new_data = [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                    logs += "| H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(*new_data)
                    data += new_data
                    
                if args.use_dqn:
                    header += ["critic_loss", "q_values", "target_values"]
                    new_data = [logs["critic_loss"], logs["q_values"], logs["target_values"]]
                    logs += "| criticL {:.3f} | Q {:.3f} | targetQ {:.3f}".format(*new_data)
                    data += new_data

                txt_logger.info(logs)

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status
            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {
                    "num_frames": num_frames,
                    "update": update,
                    "model_state": acmodel.state_dict(),
                }
                if hasattr(algo, "optimizer"):
                    status["optimizer_state"] = algo.optimizer.state_dict()
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")
