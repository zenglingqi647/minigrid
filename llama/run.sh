# find -s somedir -type f -exec md5sum {} \; | md5sum

cd /data1/lzengaf/cs285/proj/llama
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 --master_port=29501 example_chat_completion.py    --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6