conda activate bl
cd "/data1/lzengaf/cs285/proj/minigrid/rl-starter-files"
export PYTHONPATH="/data1/lzengaf/cs285/proj/minigrid/rl-starter-files":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=

tmux attach -t 
tmux kill-session
clear

python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Goto --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Open --skill open
# debug
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --skill open --repeat-threshold 1 --update-interval 1

python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PickUp --skill pickup
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PutNext --skill putnext
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Unlock --skill unlock
# finetune
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Unlock-Finetune --skill unlock
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Goto-Finetune --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Open-Finetune --skill open
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PickUp-Finetune --skill pickup
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PutNext-Finetune --skill putnext


# baseline ppo, debugging required
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Unlock-Baseline --obs-size -1 --skill unlock
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Goto-Baseline --obs-size -1 --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Open-Baseline --obs-size -1 --skill open
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PickUp-Baseline --obs-size -1 --skill pickup
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PutNext-Baseline --obs-size -1 --skill putnext