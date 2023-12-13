from typing import Any, Mapping
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
from utils.prompt_validation import validate_goal


SKILL_MDL_PATH = [
    "storage/skill-model-v2/Goto-Finetune",
    "storage/skill-model-v2/Open",
    "storage/skill-model-v2/PickUp",
    "storage/skill-model-v2/Unlock-Finetune",
    # "storage/skill-model-v1-curriculum/PutNext"
]


class PlannerPolicy(nn.Module, torch_ac.RecurrentACModel):
    '''ask_cooldown: how many steps to wait before asking GPT again. For synchronization.'''
    
    def __init__(self, skill_obs_space, action_space, vocab, llm_variant, ask_cooldown, num_procs, use_memory=False, use_text=False, num_skills=4):
        super().__init__()
        # adapted from ACModel
        self.use_memory = use_memory
        self.use_text = use_text

        n = skill_obs_space["image"][0]
        m = skill_obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        if self.use_text:
            self.word_embedding_size = 32
            self.text_embedding_size = 128

        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.skill_obs_space = skill_obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.num_envs = num_procs

        self.current_skills : list[int] = [0] * self.num_envs
        self.current_goals : list[int] = [torch.zeros(1) for _ in range(self.num_envs)]
        self.current_goals_text : list[int] = ["" for _ in range(self.num_envs)]
        self.skill_vocabs : list[Vocabulary] = [None] * self.num_skills
        self.vocab : Vocabulary = vocab.vocab
        self.invert_vocab : dict = {v: k for k, v in self.vocab.items()}
        
        self.llm_variant = llm_variant
        # load skill mmodel 
        for i in range(num_skills):
            self.ac_models.append(self.load_model(i))

        # self.lock = threading.Lock()

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return
    
    def state_dict(self):
        ''' Planner policy does not need state. '''
        return {}

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
    
    def decrease_cooldown(self):
        if self.timer > 0:
            self.timer -= 1

    def get_skills_and_goals(self, obs : DictList):
        '''
            Get the skill numbers and goals for an observation. Must ensure observation batch size is the same as the number of parallel environments
        '''
        # Here, we enforce that the batch size of this obs is the same as the number of parallel environments
        # assert (obs.full_obs.shape[0] == self.num_envs)
        # assert (obs.text.shape[0] == self.num_envs)
        assert(obs.full_obs.shape[0] == obs.text.shape[0])
        current_skills = [None] * obs.text.shape[0]
        current_goals = [None] * obs.text.shape[0]
        current_goals_text = [None] * obs.text.shape[0]
        if self.timer == 0:
            # Iterate over batches
            for idx in range(obs.full_obs.shape[0]):

                # Extract the individual image and mission texts
                obs_img : torch.Tensor = obs.full_obs[idx]
                self.invert_vocab : dict = {v: k for k, v in self.vocab.items()}
                mission_txt = " ".join([self.invert_vocab[s.item()] for s in obs.text[idx]])
                print(f"Mission text sent is {mission_txt}")

                planning_success = False
                while not planning_success:
                    # Ask the LLM planner
                    try:
                        if self.llm_variant == "gpt":
                            skill_num, goal_text = gpt_skill_planning(obs_img.int().cpu().numpy(), mission_txt)
                        elif self.llm_variant == "llama":
                            skill_num, goal_text = llama_skill_planning(obs_img.int().cpu().numpy(), mission_txt)
                        elif self.llm_variant == "human":
                            skill_num, goal_text = human_skill_planning()
                        
                        validate_goal(skill_num, goal_text)
                        # validate_goal_text = self.skill_vocabs[skill_num].decode(self.current_goals[idx])
                        print(f"Skill planning outcome: {skill_num}. Goal: {goal_text}")
                        goal_tokens = []
                        for s in goal_text.split():
                            if s not in self.skill_vocabs[skill_num].vocab:
                                print(f"Warning: unknown word {s} in mission text {goal_text}")
                            goal_tokens.append(self.skill_vocabs[skill_num][s])
                        goal_tokens = torch.IntTensor(goal_tokens).to(device)
                        planning_success = True

                    except Exception as e:
                        print("Planning failed, using the old goal and current skill. Replanning...")
                        print(e)

                # Store the skill numbers and goal tokens returned by the planner
                current_skills[idx] = skill_num
                current_goals[idx] = goal_tokens
                current_goals_text[idx] = goal_text
            self.timer = self.ask_cooldown
            self.current_skills = current_skills
            self.current_goals = current_goals
            self.current_goals_text = current_goals_text
        return self.current_skills, self.current_goals, self.current_goals_text

    def forward(self, obs : DictList, memory):
        '''
            If you need to update the ask_cooldown (collecting new experiences rather than learning from the collected experience), set advance_time to True.
        '''
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


    
