# w. llm
# a2c
python3 -m scripts.train --algo a2c --env BabyAI-GoToImpUnlock-v0 --text --frames 250000 --use-trajectory --llm gpt-3.5-turbo

# ppo
python3 -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --text --frames 250000 --use-trajectory --llm gpt-3.5-turbo


# w/o. llm
# a2c
python3 -m scripts.train --algo a2c --env BabyAI-GoToImpUnlock-v0 --text --frames 250000

# ppo
python3 -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --text --frames 250000





# visualize
python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey

# evaluate
python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey






Train with llama
cd ../rl-starter-files
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --text --frames 1000000 --recurrence 20 --obs-size 11 --frames-per-proc 40 --procs 64 --batch-size 200 --ask-every 500 