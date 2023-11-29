# Experiment 1: Recurrence 5, training A2C on all environments

# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo a2c --env BabyAI-GoToObj-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 2: Open door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-OpenDoor-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo a2c --env BabyAI-PickupDist-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 4: Put an item next to an item (in the same room)
python -m scripts.train --algo a2c --env BabyAI-PutNextLocal-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-UnlockLocal-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo a2c --env BabyAI-FindObjS5-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --text --frames 500000 --log-interval 10 --recurrence 5

# Experiment 2: Recurrence 4, training A2C on all environments

# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo a2c --env BabyAI-GoToObj-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 2: Open door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-OpenDoor-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo a2c --env BabyAI-PickupDist-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 4: Put an item next to an item (in the same room)
python -m scripts.train --algo a2c --env BabyAI-PutNextLocal-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo a2c --env BabyAI-UnlockLocal-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo a2c --env BabyAI-FindObjS5-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --text --frames 500000 --log-interval 10 --recurrence 4 --frames-per-proc 16