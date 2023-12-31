import torch.nn as nn
import torch
from pathlib import Path
from model import ACModel
from .textual_minigrid import gpt_skill_planning, llama_skill_planning, human_skill_planning
from .format import Vocabulary
from torch_ac.utils.dictlist import DictList
import torch_ac
from torch.distributions import Categorical
from .other import device

from torch_ac.algos.replay_buffer import ReplayBuffer
from utils.dqn_agent import DQNAgent
from utils.constants import *


class QPlannerPolicy(nn.Module, torch_ac.RecurrentACModel):
    '''ask_cooldown: how many steps to wait before asking GPT again. For synchronization.'''

    def __init__(self,
                 skill_obs_space,
                 action_space,
                 vocab,
                 llm_variant,
                 ask_cooldown,
                 num_procs,
                 use_memory=False,
                 use_text=False,
                 num_skills=4,
                 llm_augmented=False):
        super().__init__()
        # adapted from ACModel
        self.use_memory = use_memory
        self.use_text = use_text

        n = FULL_OBSERVED_SIZE
        m = FULL_OBSERVED_SIZE
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        n_s, m_s = skill_obs_space['image'][0], skill_obs_space['image'][1]
        self.skill_image_embedding_size = ((n_s-1)//2-2)*((m_s-1)//2-2)*64

        # Define image embedding
        self.image_conv = nn.Sequential(nn.Conv2d(3, 16, (2, 2)), nn.ReLU(), nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(16, 32, (2, 2)), nn.ReLU(), nn.Conv2d(32, 64, (2, 2)), nn.ReLU())
        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(skill_obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
        # Resize image embedding
        self.embedding_size = self.image_embedding_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size


        self.skill_obs_space = skill_obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.num_envs = num_procs

        self.current_skills: list[int] = [0] * self.num_envs
        self.current_goals: list[int] = [None] * self.num_envs
        self.skill_vocabs: list[Vocabulary] = [None] * self.num_skills
        self.vocab: Vocabulary = vocab.vocab
        self.invert_vocab: dict = {v: k for k, v in self.vocab.items()}

        self.llm_variant = llm_variant
        # load skill mmodel
        for i in range(num_skills):
            self.ac_models.append(self.load_model(i))

        self.llm_augmented = llm_augmented
        self.dqn_agent = DQNAgent(observation_shape=self.embedding_size,
                                  num_actions=48,
                                  discount=0.99,
                                  target_update_period=1000,
                                  use_double_q=True)

        # self.lock = threading.Lock()

    @property
    def memory_size(self):
        return 2 * self.skill_image_embedding_size

    def load_model(self, index):
        mdl = ACModel(self.skill_obs_space, self.action_space, self.use_memory, self.use_text)
        p = Path(SKILL_MDL_PATH[index], "status.pt")
        with open(p, "rb") as f:
            status = torch.load(f)
            model_state = status['model_state']
        mdl.load_state_dict(model_state)
        vocab = Vocabulary(100)
        vocab.load_vocab(status['vocab'])
        self.skill_vocabs[index] = vocab
        for p in mdl.parameters():
            p.requires_grad = True
        return mdl

    def parse_action(self, action_num):
        if action_num < 18:
            return 0, SKILL_LIST[0][action_num]
        if action_num < 24:
            return 1, SKILL_LIST[1][action_num]
        if action_num < 42:
            return 2, SKILL_LIST[2][action_num]
        if action_num < 48:
            return 3, SKILL_LIST[3][action_num]
        assert False, "Action number out of range"

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    
    def get_embeddings(self, obs):
        '''
            Runs observation through image and text feature (if specified) extractors, resulting in a concatenated feature vector.
        '''
        obs_img : torch.Tensor = obs.full_obs

        x = obs_img.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        return embedding


    def get_skills_and_goals(self, obs):
        '''
            Get the skill numbers and goals for an observation. Must ensure observation batch size is the same as the number of parallel environments
        '''
        # Here, we enforce that the batch size of this obs is the same as the number of parallel environments
        # assert (obs.full_obs.shape[0] == self.num_envs)
        # assert (obs.text.shape[0] == self.num_envs)
        assert(obs.full_obs.shape[0] == obs.text.shape[0])
        current_skills: list[int] = [0] * obs.full_obs.shape[0]
        current_goals: list[int] = [None] * obs.full_obs.shape[0]
        current_goal_texts: list[str] = [None] * obs.full_obs.shape[0]

        embedding = self.get_embeddings(obs)

        action_nums = self.dqn_agent.get_action(embedding)

        for idx in range(obs.full_obs.shape[0]):
            skill_num, goal_text = self.parse_action(action_nums[idx])

            goal_tokens = []
            for s in goal_text.split():
                if s not in self.skill_vocabs[skill_num].vocab:
                    print(f"Warning: unknown word {s} in mission text {goal_text}")
                goal_tokens.append(self.skill_vocabs[skill_num][s])
            goal_tokens = torch.IntTensor(goal_tokens).to(device)

            current_skills[idx] = skill_num
            current_goals[idx] = goal_tokens
            current_goal_texts[idx] = goal_text

        return current_skills, current_goals, current_goal_texts

    def forward(self, obs: DictList, memory):
        # here, obs is a dictionary of batched images and batched text. The batch size is a integer multiple of the number of parallel environments.

        dist_logits, values, memories = [], [], []
        # In each iteration of this loop, we need to extract one step of observations from all parallel environments, and ask get_skill.
        for i in range(0, len(obs), self.num_envs):
            obs_one_step = obs[i:i + self.num_envs]
            current_skills, current_goals, _ = self.get_skills_and_goals(obs_one_step)
            self.current_skills, self.current_goals = current_skills, current_goals
            # Iterate over skill and goal token pairs
            # Need to gather the dist, value, and memory
            for j in range(self.num_envs):
                skill_num, goal = current_skills[j], current_goals[j]
                obs_one_env = obs_one_step[j:j + 1]
                memory_one_step = memory[i + j:i + j + 1]

                # Use the same image observation but change the goal
                new_obs = DictList({"image" : obs_one_env.image, "text" : goal.unsqueeze(0)})
                d, v, m = self.ac_models[skill_num](new_obs, memory_one_step)

                dist_logits.append(d.logits)
                values.append(v)
                memories.append(m)

        dist_logits = torch.cat(dist_logits)
        values = torch.cat(values)
        memories = torch.cat(memories)

        return Categorical(logits=dist_logits), values, memories
