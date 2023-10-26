#!/bin/bash

# Define the list as an array
envs=(
MiniGrid-BlockedUnlockPickup-v0
MiniGrid-LavaCrossingS9N1-v0
MiniGrid-LavaCrossingS9N2-v0
MiniGrid-LavaCrossingS9N3-v0
MiniGrid-LavaCrossingS11N5-v0
MiniGrid-SimpleCrossingS9N1-v0
MiniGrid-DistShift1-v0
MiniGrid-DistShift2-v0
MiniGrid-DoorKey-5x5-v0
MiniGrid-DoorKey-6x6-v0
MiniGrid-DoorKey-8x8-v0
MiniGrid-DoorKey-16x16-v0
MiniGrid-Dynamic-Obstacles-5x5-v0
MiniGrid-Dynamic-Obstacles-Random-5x5-v0
MiniGrid-Dynamic-Obstacles-6x6-v0
MiniGrid-Dynamic-Obstacles-Random-6x6-v0
MiniGrid-Dynamic-Obstacles-8x8-v0
MiniGrid-Dynamic-Obstacles-16x16-v0
MiniGrid-Empty-5x5-v0
MiniGrid-Empty-Random-5x5-v0
MiniGrid-Empty-6x6-v0
MiniGrid-Empty-Random-6x6-v0
MiniGrid-Empty-8x8-v0
MiniGrid-Empty-16x16-v0
MiniGrid-Fetch-5x5-N2-v0
MiniGrid-Fetch-6x6-N2-v0
MiniGrid-Fetch-8x8-N3-v0
MiniGrid-FourRooms-v0
MiniGrid-GoToDoor-5x5-v0
MiniGrid-GoToDoor-6x6-v0
MiniGrid-GoToDoor-8x8-v0
MiniGrid-GoToObject-6x6-N2-v0
MiniGrid-GoToObject-8x8-N2-v0
MiniGrid-KeyCorridorS3R1-v0
MiniGrid-KeyCorridorS3R2-v0
MiniGrid-KeyCorridorS3R3-v0
MiniGrid-KeyCorridorS4R3-v0
MiniGrid-KeyCorridorS5R3-v0
MiniGrid-KeyCorridorS6R3-v0
MiniGrid-LavaGapS5-v0
MiniGrid-LavaGapS6-v0
MiniGrid-LavaGapS7-v0
MiniGrid-LockedRoom-v0
MiniGrid-MemoryS17Random-v0
MiniGrid-MemoryS13Random-v0
MiniGrid-MemoryS13-v0
MiniGrid-MemoryS11-v0
MiniGrid-MultiRoom-N2-S4-v0
MiniGrid-MultiRoom-N4-S5-v0
MiniGrid-MultiRoom-N6-v0
MiniGrid-ObstructedMaze-1Dlhb-v0
MiniGrid-ObstructedMaze-Full-v0
MiniGrid-Playground-v0
MiniGrid-PutNear-6x6-N2-v0
MiniGrid-PutNear-8x8-N3-v0
MiniGrid-RedBlueDoors-6x6-v0
MiniGrid-RedBlueDoors-8x8-v0
MiniGrid-Unlock-v0
MiniGrid-UnlockPickup-v0
)

# a2c w/o. llm
for env in "${envs[@]}"; do
    echo "Processing environment: $env"
    python3 -m scripts.train --algo a2c --env $env --text --frames 250000
done

# ppo w/o. llm
for env in "${envs[@]}"; do
    python3 -m scripts.train --algo a2c --env $env --text --frames 250000
done


# a2c w. llm
for env in "${envs[@]}"; do
    python3 -m scripts.train --algo a2c --env $env --text --frames 250000 --use-trajectory --llm gpt-3.5-turbo --ask-every 10000
done

# ppo w. llm
for env in "${envs[@]}"; do
    python3 -m scripts.train --algo ppo --env $env --text --frames 250000 --use-trajectory --llm gpt-3.5-turbo --ask-every 10000
done