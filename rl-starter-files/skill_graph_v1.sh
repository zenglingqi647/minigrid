

# Skill 1: Go to Object (in the same room)
python -m scripts.train --algo ppo --env BabyAI-GoToObj-v0 --text --frames 10000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 2: Open door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-OpenDoor-v0 --text --frames 10000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 3: Pickup an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PickupDist-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 4: Put an item next to an item (in the same room)
python -m scripts.train --algo ppo --env BabyAI-PutNextLocalS5N3-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 5: Unlock a door (in the same room)
python -m scripts.train --algo ppo --env BabyAI-UnlockLocal-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 6: Find an object (in a random room)
python -m scripts.train --algo ppo --env BabyAI-FindObjS5-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 7: Go to the green object (in a random room)
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11

# Skill 8: Also Four Rooms
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N4-S5-v0 --text --frames 30000000 --log-interval 10 --recurrence 20 --save-interval 15   --procs 64 --frames-per-proc 40 --obs-size 11