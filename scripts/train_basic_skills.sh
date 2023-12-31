conda activate bl
cd /data1/lzengaf/cs285/proj/minigrid/rl-starter-files
export CUDA_VISIBLE_DEVICES=
# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo ppo --env BabyAI-GoToObj-v0 --text --frames 500000 --log-interval 10

# Skill 2: Open door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-OpenDoor-v0 --text --frames 500000 --log-interval 10

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PickupDist-v0 --text --frames 500000 --log-interval 10

# Skill 4: Put an item next to an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PutNextLocal-v0 --text --frames 500000 --log-interval 10

# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-UnlockLocal-v0 --text --frames 500000 --log-interval 10

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo ppo --env BabyAI-FindObjS5-v0 --text --frames 500000 --log-interval 10

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v0 --text --frames 500000 --log-interval 10


# Need at least one skill that enables the agent to go to a different room.
# Skill 6 or 7?



