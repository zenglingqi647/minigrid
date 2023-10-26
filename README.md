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

Right now my training script:
```
cd rl-starter-files/

python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.0005Ask --text --save-interval 10 --frames 250000 --gpt
```
The problem is, an ask probability of 0.0005 is still very bad...It takes a really long time to train.

### **Update**
- Bash script of experiments of different babyai and minigrid environments can be found as `babyai.sh` and `minigrid.sh`.

- The reshaped reward with gpt predicting for a single action and for the next few actions (currently hardcoded as 10) are implemented and merged in the `train.py` and the `utils` folder.

- Add `eval2excel.py` for evaluation and convert the results to excel files.