python -m scripts.visualize --env BabyAI-GoToImpUnlock-v0 --model GoToImpUnlock --text --memory

python -m scripts.visualize --env BabyAI-GoToImpUnlock-v0 --model BabyAI-GoToImpUnlock-v0_ppo_rec20_f1000000_fp40_seed1_llmplannergpt_askevery2000_23-12-07-15-27-46 --text --memory --llm-planner-variant gpt --ask-every 500  --obs-size 11

python -m scripts.visualize --env BabyAI-GoToImpUnlock-v0 --model BabyAI-GoToImpUnlock-v0_ppo_rec20_f1000000_fp40_seed1_llmplannergpt_askevery2000_23-12-07-15-27-46 --text --memory --llm-planner-variant human --ask-every 25  --obs-size 11