cd /data1/lzengaf/cs285/proj/minigrid/FastChat
conda activate fastchat
export CUDA_VISIBLE_DEVICES=0
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5