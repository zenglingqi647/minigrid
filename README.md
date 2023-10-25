experimental-code:
    Draft code, not really used

rl-starter-files:
    Cloned from the original repository, and I modified to include a GPT interface.
    I also wrote a GPT-based reward shaping function to ask GPT about the reward.

Setting up the repository:
    After creating your conda environment:
```
cd rl-starter-files
pip install -r requirements.txt
```

Right now my training script:
```
cd rl-starter-files/

python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.0005Ask --text --save-interval 10 --frames 250000 --gpt
```
The problem is, an ask probability of 0.0005 is still very bad...It takes a really long time to train.