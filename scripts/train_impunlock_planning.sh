python -m scripts.train --algo a2c --env BabyAI-GoToImpUnlock-v0 --text --use-planner --frames 500000

python -m scripts.train --algo a2c --env BabyAI-GoToImpUnlock-v0 --text --use-planner --frames 500000 --recurrence 4 --frames-per-proc 16