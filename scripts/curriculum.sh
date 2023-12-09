conda activate bl
cd "/data1/lzengaf/cs285/proj/minigrid/rl-starter-files"
export PYTHONPATH="/data1/lzengaf/cs285/proj/minigrid/rl-starter-files":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=

python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model GoTo --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Open --skill open
# debug
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --skill open --repeat-threshold 1 --update-interval 1

python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PickUp --skill pickup
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PutNext --skill putnext
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Unlock --skill unlock
# finetune
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model UnlockFinetune --skill unlock
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model GoToFinetune --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model OpenFinetune --skill open
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PickUpFinetune --skill pickup
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model PutNextFinetune --skill putnext