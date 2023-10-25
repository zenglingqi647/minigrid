python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlockNoGPT --text --save-interval 10  --frames 1000000 &
# python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.002Ask --text --save-interval 10 --frames 250000 --gpt --ask_gpt_prob 0.002
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.0005Ask --text --save-interval 10 --gpt --ask_gpt_prob 0.0005 --frames 1000000
# python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock0.001Ask --text --save-interval 10 --frames 250000 --gpt --ask_gpt_prob 0.001
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlockAskEvery2000 --text --save-interval 10 --gpt --ask-every 2000 --frames 250000