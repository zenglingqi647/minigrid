# Final Project for COMPSCI 285
## Extended Abstract

In the dynamic field of Reinforcement Learning (RL) research, mastering long-horizon sparse-reward tasks remains a formidable challenge. Traditional RL agents, lacking prior knowledge, rely solely on sparse environmental rewards to discern effective actions. In contrast, humans leverage past knowledge to efficiently adapt and accomplish new tasks, a capability that current RL methodologies often lack.

Recent advancements in Large Language Models (LLMs) like GPT and BERT have showcased their remarkable ability to encode vast world knowledge and perform contextual reasoning. However, LLMs are not inherently grounded in specific tasks or environments, and directly using them as primary agents can lead to uncertainty and instability. Moreover, the computational demands of these pre-trained models, with billions or trillions of parameters, make them impractical for local deployment. Concurrently, Hierarchical Reinforcement Learning (HRL) methods have shown promise in managing complex tasks by exploiting their hierarchical nature, central to which is an effective high-level planning policy that can reason about composing skills.

Our work leverages the planning capabilities of LLMs to augment a Deep Q-Network (DQN) planner within an HRL framework. We first train a set of basic skill agents using curriculum learning in various BabyAI environments. These skills are then composed using a DQN planner, forming the basis of our hierarchical approach. The DQN planner is further augmented with an LLM-matching reward bonus, based on the similarity between its decisions and those suggested by the LLM during training. This integration offers three key advantages: 
1. It eliminates the need for an LLM during evaluation, which is beneficial in environments with limited internet access, reduces LLM API costs and avoids rate-limiting issues.
2. The DQN's learning process is accelerated by providing additional signals that would typically require manual specification.
3. With sufficient experience, the system retains the potential to outperform pure LLM-based planners.

We benchmark our method against various baseline models in multiple complex test environments. These include models trained using Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), a standalone DQN planner, and a pure LLM planner. Our approach demonstrates improved performance over these baselines in certain scenarios and can even surpass a pure LLM-based planner due to the DQN's additional optimization. Both the LLM and DQN are shown to contribute to the model's performance through ablation studies. However, our method's performance is not consistently optimal, highlighting future research directions such as further training and benchmarking against state-of-the-art models in the BabyAI environment.


# Directory structures:
experimental-code:
    Draft code, not really used

rl-starter-files:
    Cloned from the original repository, and I modified to include a GPT interface.
    I also wrote a GPT-based reward shaping function to ask GPT about the reward.

# Running the code:
Setting up the repository:
    After creating your conda environment:
```
cd rl-starter-files
pip install -r requirements.txt
```