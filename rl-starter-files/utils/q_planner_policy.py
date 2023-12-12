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

from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent


SKILL_MDL_PATH = [
    "storage/skill-model-v1-curriculum/GoToObj",
    "storage/skill-model-v1-curriculum/OpenDoor",
    "storage/skill-model-v1-curriculum/Pickup",
    "storage/skill-model-v1-curriculum/Unlock",
    # "storage/skill-model-v1-curriculum/PutNext"
]

colors = ["red", "green", "blue", "purple", "yellow", "grey"]
objects = ["ball", "box", "key"]

GO_TO = {i + j * 6: f"go to the {colors[i]} {objects[j]}" for j in range(len(objects)) for i in range(len(colors))}

OPEN = {18: "open the red door", 19: "open the green door", 20: "open the blue door", 21: "open the purple door", 22: "open the yellow door", 23: "open the grey door"}

PICK_UP = {24: "pick up a red ball", 25: "pick up a green ball", 26: "pick up a blue ball", 27: "pick up a purple ball", 28: "pick up a yellow ball", 29: "pick up a grey ball", 
30: "pick up a red box", 31: "pick up a green box", 32: "pick up a blue box", 33: "pick up a purple box", 34: "pick up a yellow box", 35: "pick up a grey box", 
36: "pick up a red key", 37: "pick up a green key", 38: "pick up a blue key", 39: "pick up a purple key", 40: "pick up a yellow key", 41: "pick up a grey key"}

UNLOCK = {42: "open the red door", 43: "open the green door", 44: "open the blue door", 45: "open the purple door", 46: "open the yellow door", 47: "open the grey door"}

# PUT_NEXT = {48: "put the red ball next to the red ball", 49: "put the red ball next to the green ball", 50: "put the red ball next to the blue ball", 51: "put the red ball next to the purple ball", 52: "put the red ball next to the yellow ball", 53: "put the red ball next to the grey ball", 
# 54: "put the green ball next to the red ball", 55: "put the green ball next to the green ball", 56: "put the green ball next to the blue ball", 57: "put the green ball next to the purple ball", 58: "put the green ball next to the yellow ball", 59: "put the green ball next to the grey ball",

# }

PUT_NEXT = f"put the {0} {1} next to the {2} {3}"

# In total, there are 18 + 6 + 18 + 6 + 6 * 3 * 6 * 3 = 366 configurations.

SKILL_LIST = [GO_TO, OPEN, PICK_UP, UNLOCK, PUT_NEXT]

class QPlannerPolicy(nn.Module, torch_ac.RecurrentACModel):
    '''ask_cooldown: how many steps to wait before asking GPT again. For synchronization.'''
    
    def __init__(self, obs_space, action_space, vocab, llm_variant, ask_cooldown, num_procs, use_memory=False, use_text=False, num_skills=5, llm_augmented=False):
        super().__init__()
        # adapted from ACModel
        self.use_memory = use_memory
        self.use_text = use_text

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size


        self.obs_space = obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.num_envs = num_procs

        self.current_skills : list[int] = [0] * self.num_envs
        self.current_goals : list[int] = [None] * self.num_envs
        self.skill_vocabs : list[Vocabulary] = [None] * self.num_skills
        self.vocab : Vocabulary = vocab.vocab
        self.invert_vocab : dict = {v: k for k, v in self.vocab.items()}
        
        self.llm_variant = llm_variant
        # load skill mmodel 
        for i in range(num_skills):
            self.ac_models.append(self.load_model(i))

        self.llm_augmented = llm_augmented
        self.dqn_agent = DQNAgent(observation_shape=self.embedding_size, num_actions=48, discount=0.99, target_update_period=1000, use_double_q=True)

        # self.lock = threading.Lock()

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def load_model(self, index):
        mdl = ACModel(self.obs_space, self.action_space, self.use_memory, self.use_text)
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
            return SKILL_LIST[0][action_num]
        if action_num < 24:
            return SKILL_LIST[1][action_num]
        if action_num < 42:
            return SKILL_LIST[2][action_num]
        if action_num < 48:
            return SKILL_LIST[3][action_num]
        assert False


    def get_skills_and_goals(self, obs):
        '''
            Get the skill numbers and goals for an observation. Must ensure observation batch size is the same as the number of parallel environments
        '''
        # Here, we enforce that the batch size of this obs is the same as the number of parallel environments
        assert (obs.full_obs.shape[0] == self.num_envs)
        assert (obs.text.shape[0] == self.num_envs)

        current_skills : list[int] = [0] * self.num_envs
        current_goals : list[int] = [None] * self.num_envs

        for idx in range(obs.full_obs.shape[0]):
            # Extract the individual image and mission texts
            obs_img : torch.Tensor = obs.full_obs[idx]
            mission_txt = " ".join([self.invert_vocab[s.item()] for s in obs.text[idx]])
            print(f"Mission text sent is {mission_txt}")
            x = obs_img.transpose(1, 3).transpose(2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)
            embedding = x
            if self.use_text:
                embed_text = self._get_embed_text(obs.text)
                embedding = torch.cat((embedding, embed_text), dim=1)

            action_num = self.dqn_agent.get_action(embedding)
            skill_num, goal_text = parse_action(action_num)

            goal_tokens = []
            for s in goal_text.split():
                if s not in self.skill_vocabs[skill_num].vocab:
                    print(f"Warning: unknown word {s} in mission text {goal_text}")
                goal_tokens.append(self.skill_vocabs[skill_num][s])
            goal_tokens = torch.IntTensor(goal_tokens).to(device)

            current_skills[idx] = skill_num
            current_goals[idx] = goal_tokens
        
        return current_skills, current_goals


    def forward(self, obs : DictList, memory):
        # here, obs is a dictionary of batched images and batched text. The batch size is a integer multiple of the number of parallel environments.

        dist_logits, values, memories = [], [], []
        # In each iteration of this loop, we need to extract one step of observations from all parallel environments, and ask get_skill.
        for i in range(0, len(obs), self.num_envs):
            obs_one_step = obs[i:i + self.num_envs]
            current_skills, current_goals = self.get_skills_and_goals(obs_one_step)
            self.current_skills, self.current_goals = current_skills, current_goals
            # Iterate over skill and goal token pairs
            # Need to gather the dist, value, and memory
            for j in range(self.num_envs):
                skill_num, goal = current_skills[j], current_goals[j]
                obs_one_step = obs_one_step[j:j + 1]
                memory_one_step = memory[i + j:i + j + 1]

                # Use the same image observation but change the goal
                new_obs = DictList({"image" : obs_one_step.image, "text" : goal.unsqueeze(0)})
                d, v, m = self.ac_models[skill_num](new_obs, memory_one_step)

                dist_logits.append(d.logits)
                values.append(v)
                memories.append(m)
            
        dist_logits = torch.cat(dist_logits)
        values = torch.cat(values)
        memories = torch.cat(memories)

        return Categorical(logits=dist_logits), values, memories


    
