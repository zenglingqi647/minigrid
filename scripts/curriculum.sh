conda activate bl
cd "/data1/lzengaf/cs285/proj/minigrid/rl-starter-files"
export PYTHONPATH="/data1/lzengaf/cs285/proj/minigrid/rl-starter-files":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=

python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model GoTo --skill goto
python /data1/lzengaf/cs285/proj/minigrid/rl-starter-files/scripts/skill_trainer.py --model Open --skill open