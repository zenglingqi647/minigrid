cd ../rl-starter-files

# python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
# python -m scripts.train --algo a2c --env BabyAI-GoToImpUnlock-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1


# python -m scripts.train --algo ppo --env BabyAI-GoToSeq-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
# python -m scripts.train --algo a2c --env BabyAI-GoToSeq-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1

# python -m scripts.train --algo ppo --env BabyAI-GoToObjMazeS7-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
# python -m scripts.train --algo a2c --env BabyAI-GoToObjMazeS7-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1

# python -m scripts.train --algo ppo --env BabyAI-UnblockPickup-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
# python -m scripts.train --algo a2c --env BabyAI-UnblockPickup-v0  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1

for i in BabyAI-GoTo-v0
do
python -m scripts.train --algo a2c --env $i --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
done

for i in BabyAI-GoTo-v0
do
python -m scripts.train --algo ppo --env $i  --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --batch-size 1280 --procs 64 --frames-per-proc 40 --obs-size -1
done
