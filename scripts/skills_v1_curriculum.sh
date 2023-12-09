# If On Local Computer, change the number of processes to be 16

# Use these if you are on savio (more processses)

# for name in S5N2, S6N2, S6N3, S6N4, S7N4, S7N5;
# do
#     python -m scripts.train --algo ppo --env BabyAI-GoToLocal$name-v0 --model skill-model-v1-curriculum/GoToObj --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --procs 64 --batch-size 1280 --frames-per-proc 40 --obs-size 11
# done

cd ../rl-starter-files
for i in $(seq 2 7)
do python -m scripts.train --algo ppo --env BabyAI-GoToLocalS8N$i-v0 --model skill-model-v1-curriculum/GoToObj --text --frames 1000000 --log-interval 10 --recurrence 20 --save-interval 15 --procs 64 --batch-size 1280 --frames-per-proc 40 --obs-size 11
done