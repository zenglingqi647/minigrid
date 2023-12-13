cd ../rl-starter-files

python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt