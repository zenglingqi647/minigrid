import torch.nn as nn
import torch
from pathlib import Path
from model import ACModel
from .textual_minigrid import gpt_planning_prob


SKILL_MDL_PATH = [
    "storage\BabyAI-GoToObj-v0_a2c_Nollm_seed1_23-11-04-12-45-27",
    "storage\BabyAI-OpenDoor-v0_a2c_Nollm_seed1_23-11-04-12-51-14",
    "storage\BabyAI-PickupDist-v0_a2c_Nollm_seed1_23-11-04-14-00-02",
    "storage\BabyAI-PutNextLocal-v0_a2c_Nollm_seed1_23-11-04-14-06-44",
    "storage\BabyAI-UnlockLocal-v0_a2c_Nollm_seed1_23-11-04-14-21-11",
    "storage\BabyAI-FindObjS5-v0_a2c_Nollm_seed1_23-11-04-14-26-17",
    "storage\MiniGrid-FourRooms-v0_a2c_Nollm_seed1_23-11-04-14-37-08"
]

class PlannerPolicy(ACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, num_skills=7, ask_cooldown=200):
        super().__init__(obs_space, action_space, use_memory, use_text)
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.distr = torch.distributions.Categorical(torch.ones(num_skills) / num_skills)
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
            p = torch.Tensor(gpt_planning_prob(obs))
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


    