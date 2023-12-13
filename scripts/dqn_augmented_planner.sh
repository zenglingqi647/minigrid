cd ../rl-starter-files

python -m scripts.train --algo base --env BabyAI-GoToImpUnlock-v0  --text --frames 3000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 1 --dqn-batch-size 64

python -m scripts.train --algo base --env BabyAI-GoToSeq-v0  --text --frames 3000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 1 --dqn-batch-size 64

python -m scripts.train --algo base --env BabyAI-KeyCorridorS4R3-v0 --text --frames 3000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 1 --dqn-batch-size 64

python -m scripts.train --algo base --env BabyAI-UnblockPickup-v0  --text --frames 3000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 1 --dqn-batch-size 64