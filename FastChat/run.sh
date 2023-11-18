conda activate fastchat
cd /data1/lzengaf/cs285/proj/minigrid/FastChat
export CUDA_VISIBLE_DEVICES=

python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5


"You are an agent in a Minigrid environment. Your mission is go to a green key in as few steps aspossible. Your agent's direction is currently right. Your agent can only see in front of itself. It cannot see blocked objects. Here is the vision of youagent. Assume your agent is at the center of the last row. The first row is the furthest from yourfront: unseen, unseen, unseen, unseen, unseen, unseen, unseenunseen, unseen, unseen, unseen, unseen, unseen, unseenunseen, wall, wall, wall, wall, wall, wall. unseen, yellow closed door, empty, empty, empty, empty, empty,unseen, wall, empty, empty, empty, empty, empty,unseen, wall, empty, empty, empty, grey box, emptyunseen, wall, empty, empty, empty, empty, empty. You have a set of skills at your disposal. They are listed in the following: Skill 1: Go to Obiect (in the same room) Skill 2: Open door (in the same room) Skill 3: Pickup an item (in the same room) Skill 4. Put an item next to an item (in the same room) Skill 5. Unlock a door (in the same room) Skill 6: Find an object (in a random room) Skill 7: Go to the green object (in a random n). Generate a probability vector for using each of the skills given the circumstance, in a comma:"