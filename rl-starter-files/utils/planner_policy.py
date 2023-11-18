import torch.nn as nn
import torch
from pathlib import Path
from model import ACModel
from .textual_minigrid import gpt_planning_prob
from .format import Vocabulary


SKILL_MDL_PATH = [
    "storage\BabyAI-GoToObj-v0_a2c_Nollm_seed1_memory_4",
    "storage\BabyAI-OpenDoor-v0_a2c_Nollm_seed1_memory_4",
    "storage\BabyAI-PickupDist-v0_a2c_Nollm_seed1_memory_4",
    "storage\BabyAI-PutNextLocal-v0_a2c_Nollm_seed1_memory_4",
    "storage\BabyAI-UnlockLocal-v0_a2c_Nollm_seed1_memory_4",
    "storage\BabyAI-FindObjS5-v0_a2c_Nollm_seed1_memory_4",
    "storage\MiniGrid-FourRooms-v0_a2c_Nollm_seed1_memory_4"
]

class PlannerPolicy(ACModel):
    def __init__(self, obs_space, action_space, vocab, use_memory=False, use_text=False, num_skills=7, ask_cooldown=200):
        super().__init__(obs_space, action_space, use_memory, use_text)
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.distr = torch.distributions.Categorical(torch.ones(num_skills) / num_skills)
        self.vocab : Vocabulary = vocab
        for i in range(num_skills):
            self.ac_models.append(self.load_model(i))

    def load_model(self, index):
        mdl = ACModel(self.obs_space, self.action_space, self.use_memory, self.use_text)
        p = Path(SKILL_MDL_PATH[index], "status.pt")
        with open(p, "rb") as f:
            model_state = torch.load(f)['model_state']
        mdl.load_state_dict(model_state)
        return mdl

    def get_skill_distr(self, obs, memory):
        if self.timer == 0:
            invert_vocab = {v: k for k, v in self.vocab.vocab.items()}
            idx = torch.randint(low=0, high=obs.image.shape[0], size=(1,)).item()
            obs_img : torch.Tensor = obs.image[idx]
            mission_txt = " ".join([invert_vocab[s.item()] for s in obs.text[idx]])
            try:
                p = torch.Tensor(gpt_planning_prob(obs_img.cpu().numpy(), mission_txt))
            except Exception as e:
                p = torch.ones(size=(self.num_skills,))
            print(f"Skill planning outcome: {p} ")
            self.distr = torch.distributions.Categorical(probs=p)
            self.timer = self.ask_cooldown
        else:
            self.timer -= 1
        return self.distr

    def forward(self, obs, memory):
        # for network in self.ac_models:
        #     network.zero_grad()
        skill_network_idx = self.get_skill_distr(obs, memory).sample()
        # for j in range(self.num_skills):
        #     if j != skill_network_idx:
        #         for p in self.ac_models[j].parameters():
        #             p.grad = torch.zeros_like(p)
        return self.ac_models[skill_network_idx](obs, memory)


    