export CUDA_VISIBLE_DEVICES=0
cd /data1/lzengaf/cs285/proj/minigrid/FastChat
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5