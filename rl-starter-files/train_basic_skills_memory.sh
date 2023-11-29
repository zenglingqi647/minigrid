# Experiment 1: Recurrence 5, training A2C on all environments
tmux
conda activate bl
cd /data1/lzengaf/cs285/proj/minigrid/rl-starter-files
export CUDA_VISIBLE_DEVICES=

# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo a2c --env BabyAI-GoToObj-v0 --text --frames 10000000 --log-interval 10 --recurrence 5

# Skill 2: Open door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-OpenDoor-v0 --text --frames 10000000 --log-interval 10 --recurrence 5

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo a2c --env BabyAI-PickupDist-v0 --text --frames 10000000 --log-interval 10 --recurrence 5

# Skill 4: Put an item next to an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PutNextLocalS5N3-v0 --text --batch-size 1280 --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15 --procs 64 --frames-per-proc 40 --seed 2


# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-UnlockLocal-v0 --text --frames 10000000 --log-interval 10 --recurrence 5

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo a2c --env BabyAI-FindObjS5-v0 --text --frames 10000000 --log-interval 10 --recurrence 5

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --text --frames 10000000 --log-interval 10 --recurrence 5 

python -m scripts.train --algo a2c --env FourRooms --text --frames 10000000 --log-interval 10 --recurrence 5 --custom-hw 15


# Experiment 2: Recurrence 4, training A2C on all environments

# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo ppo --env BabyAI-GoToObj-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128 --batch-size 1280

# Skill 2: Open door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-OpenDoor-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PickupDist-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

# Skill 4: Put an item next to an item (in the same room)
# python -m scripts.train --algo ppo --env BabyAI-PutNextLocal-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-UnlockLocal-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo ppo --env BabyAI-FindObjS5-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N4-S5-v0 --text --frames 10000000 --log-interval 10 --recurrence 16 --frames-per-proc=128

python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N4-S5-v0 --text --batch-size 1280 --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15 --procs 64 --frames-per-proc 40

# hw: 9, 11, 13
python -m scripts.train --algo ppo --env FourRooms --text --batch-size 1280 --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15 --procs 64 --frames-per-proc 40 --custom-hw 11 --seed 2