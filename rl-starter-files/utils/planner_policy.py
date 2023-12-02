import torch.nn as nn
import torch
from pathlib import Path
from model import ACModel
from .textual_minigrid import gpt_skill_planning, llama_skill_planning
from .format import Vocabulary
import torch_ac


SKILL_MDL_PATH = [
    "storage\skill-model-v1\BabyAI-GoToObj-v0_ppo_Nollm_seed1_23-11-26-20-13-04",
    "storage\skill-model-v1\BabyAI-OpenDoor-v0_ppo_Nollm_seed1_23-11-26-20-38-31",
    "storage\skill-model-v1\BabyAI-PickupDist-v0_ppo_Nollm_seed1_23-11-26-21-02-46",
    "storage\skill-model-v1\BabyAI-PutNextLocalS5N3-v0_ppo_Nollm_seed1_23-11-26-22-15-24",
    "storage\skill-model-v1\BabyAI-UnlockLocal-v0_ppo_Nollm_seed2_23-11-27-03-01-30",
    "storage\skill-model-v1\BabyAI-FindObjS5-v0_ppo_Nollm_seed1_23-11-27-02-54-00",
    "storage\skill-model-v1\MiniGrid-FourRooms-v0_ppo_Nollm_seed1_23-11-27-04-41-55"
]

class PlannerPolicy(nn.Module, torch_ac.RecurrentACModel):

    
    def __init__(self, obs_space, action_space, vocab, llm_variant, ask_cooldown,use_memory=False, use_text=False, num_skills=7):
        super().__init__()
        # adapted from ACModel
        self.use_memory = use_memory
        self.use_text = use_text

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        if self.use_text:
            self.word_embedding_size = 32
            self.text_embedding_size = 128

        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_skills = num_skills
        self.ac_models = nn.ModuleList()
        self.timer = 0
        self.ask_cooldown = ask_cooldown
        self.current_skill : int = 0
        self.vocab : Vocabulary = vocab
        self.llm_variant = "gpt" if llm_variant is None else llm_variant
        for i in range(num_skills):
            self.ac_models.append(self.load_model(i))

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
            model_state = torch.load(f)['model_state']
        mdl.load_state_dict(model_state)
        for p in mdl.parameters():
            p.requires_grad = True
        return mdl

    def get_skill_distr(self, obs, memory):
        if self.timer == 0:
            invert_vocab = {v: k for k, v in self.vocab.vocab.items()}
            idx = torch.randint(low=0, high=obs.image.shape[0], size=(1,)).item()
            obs_img : torch.Tensor = obs.image[idx]
            mission_txt = " ".join([invert_vocab[s.item()] for s in obs.text[idx]])
            try:
                if self.llm_variant == "gpt":
                    skill_num = gpt_skill_planning(obs_img.cpu().numpy(), mission_txt)
                elif self.llm_variant == "llama":
                    skill_num = llama_skill_planning(obs_img.cpu().numpy(), mission_txt)
                print(f"Skill planning outcome: {skill_num} ")
            except Exception as e:
                skill_num = torch.randint(0, len(self.ac_models), size=(1,)).item()
                print(f"Planning failed, randomly generated {skill_num} ")
            self.current_skill = skill_num
            self.timer = self.ask_cooldown
        else:
            self.timer -= 1
        return self.current_skill

    def forward(self, obs, memory):
        # for network in self.ac_models:
        #     network.zero_grad()
        skill_network_idx = self.get_skill_distr(obs, memory)
        result = self.ac_models[skill_network_idx](obs, memory)
        for j in range(len(self.ac_models)):
            if j != skill_network_idx:
                model = self.ac_models[j]
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
        return result


    