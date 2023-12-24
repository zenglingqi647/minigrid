cd ../rl-starter-files

python -m scripts.train --algo base --env BabyAI-GoToObjMazeOpen-v0  --text --frames 15000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 200 --procs 16 --frames-per-proc 40 --obs-size 0 --llm-planner-variant gpt --ask-every 4

# python -m scripts.train --algo base --env BabyAI-UnblockPickup-v0  --text --frames 15000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 200 --procs 16 --frames-per-proc 40 --obs-size 0 --llm-planner-variant gpt --ask-every 4

python -m scripts.train --algo base --env BabyAI-GoToSeq-v0 --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 4

python -m scripts.train --algo base --env BabyAI-GoToImpUnlock-v0 --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 4

python -m scripts.train --algo base --env BabyAI-UnblockPickup-v0 --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 4

python -m scripts.train --algo base --env BabyAI-Pickup-v0 --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 4

python -m scripts.train --algo base --env BabyAI-GoTo-v0 --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size 0 --use-dqn --llm-augmented --llm-planner-variant gpt --ask-every 4 
