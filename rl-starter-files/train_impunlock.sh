python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlockNoGPT --text --save-interval 10 --frames 250000
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock --text --save-interval 10 --frames 250000 --gpt
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.02Ask --text --save-interval 10 --frames 250000 --gpt
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.002Ask --text --save-interval 10 --frames 250000 --gpt
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.0005Ask --text --save-interval 10 --frames 250000 --gpt