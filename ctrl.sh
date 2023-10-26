#!/bin/bash

environments=(
    'BabyAI-GoToImpUnlock-v0_a2c_gpt-3.5-turbo_seed1_23-10-26-07-48-03'
    'BabyAI-GoToImpUnlock-v0_a2c_None_seed1_23-10-26-07-48-08'
    'BabyAI-GoToImpUnlock-v0_ppo_None_seed1_23-10-26-07-48-12'
    'MiniGrid-BlockedUnlockPickup-v0_a2c_None_seed1_23-10-26-07-54-52'
    'MiniGrid-BlockedUnlockPickup-v0_a2c_None_seed1_23-10-26-07-54-56'
    'MiniGrid-BlockedUnlockPickup-v0_a2c_gpt-3.5-turbo_seed1_23-10-26-07-55-31'
    'MiniGrid-BlockedUnlockPickup-v0_ppo_gpt-3.5-turbo_seed1_23-10-26-07-56-24'
)

# Define the list as an array
environments=(
    'MiniGrid-LavaCrossingS9N1-v0_a2c_None_askevery2000_seed1_23-10-26-08-00-15'
    'MiniGrid-LavaCrossingS9N1-v0_a2c_None_askevery2000_seed1_23-10-26-08-00-20'
    'MiniGrid-LavaCrossingS9N1-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-01-05'
    'BabyAI-GoToRedBallGrey-v0_a2c_None_askevery2000_seed1_23-10-26-08-01-11'
    'BabyAI-GoToRedBallGrey-v0_a2c_None_askevery2000_seed1_23-10-26-08-01-17'
    'BabyAI-GoToRedBallGrey-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-01-22'
    'BabyAI-GoToRedBallGrey-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-01-27'
)

environments=(
    'MiniGrid-LavaCrossingS9N1-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-05-41'
    'MiniGrid-LavaCrossingS9N2-v0_a2c_None__seed1_23-10-26-08-09-16'
    'MiniGrid-LavaCrossingS9N3-v0_a2c_None__seed1_23-10-26-08-09-19'
    'MiniGrid-LavaCrossingS9N2-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-10-32'
    'BabyAI-GoToRedBall-v0_a2c_None__seed1_23-10-26-08-13-10'
    'BabyAI-GoToRedBall-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-13-12'
    'BabyAI-GoToObj-v0_a2c_None__seed1_23-10-26-08-13-14'
)

for env in "${environments[@]}"; do
    env_name="${env%%_*}"
    echo "Processing environment: $env_name"
    python3 -m scripts.evaluate --env $env_name --model $env >/data1/lzengaf/cs285/proj/minigrid/rl-starter-files/evaluate/$env.log --text
done

environments=(
    'MiniGrid-SimpleCrossingS9N1-v0_a2c_None__seed1_23-10-26-08-29-13'
    'MiniGrid-LavaCrossingS11N5-v0_a2c_None__seed1_23-10-26-08-29-14'
    'MiniGrid-LavaCrossingS11N5-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-30-49'
    'BabyAI-GoToObj-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-30-50'
    'BabyAI-GoToObjS4-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-36-18'
    'BabyAI-GoToObjS4-v0_a2c_None__seed1_23-10-26-08-36-53'
    'BabyAI-GoToObjS6-v0_a2c_None__seed1_23-10-26-08-38-41'
    'MiniGrid-DistShift1-v0_a2c_None__seed1_23-10-26-08-38-56'
    'MiniGrid-SimpleCrossingS9N1-v0_a2c_None__seed1_23-10-26-08-39-21'
)

environments=(
    'MiniGrid-SimpleCrossingS9N1-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-41-00'
    'MiniGrid-DistShift2-v0_a2c_None__seed1_23-10-26-08-46-35'
    'BabyAI-GoToImpUnlock-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-47-10'
    'MiniGrid-DistShift1-v0_a2c_None__seed1_23-10-26-08-47-35'
    'BabyAI-GoToObjS6-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-48-28'
    'MiniGrid-LavaCrossingS11N5-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-48-40'
    'BabyAI-GoToObjS6-v0_a2c_None__seed1_23-10-26-08-49-00'
    'MiniGrid-DistShift1-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-49-20'
)

environments=(
    'BabyAI-GoToLocal-v0_a2c_None__seed1_23-10-26-08-50-57'
    'MiniGrid-DistShift2-v0_a2c_None__seed1_23-10-26-08-56-21'
    'MiniGrid-DoorKey-5x5-v0_a2c_None__seed1_23-10-26-08-57-05'
    'BabyAI-GoToLocalS6N3-v0_ppo_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-58-21'
    'MiniGrid-DistShift2-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-08-58-22'
    'MiniGrid-Dynamic-Obstacles-16x16-v0_a2c_None__seed1_23-10-26-08-59-52'
    'BabyAI-GoToLocal-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-09-02-28'
    'MiniGrid-DoorKey-5x5-v0_a2c_gpt-3.5-turbo_askevery10000.0_seed1_23-10-26-09-06-24'
)