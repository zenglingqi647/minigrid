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
