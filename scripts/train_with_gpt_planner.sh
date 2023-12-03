# This is for training on windows with a small gpu.
cd ../rl-starter-files
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --text --use-planner --frames 1000000 --recurrence 20 --obs-size 11 --frames-per-proc 40 --procs 1 --batch-size 200 --ask-every 500 --llm gpt