# find -s somedir -type f -exec md5sum {} \; | md5sum

conda activate llama
cd /data1/lzengaf/cs285/proj/minigrid/llama
export CUDA_VISIBLE_DEVICES=

# reproduce
torchrun --nproc_per_node 1 --master_port=29501 example_chat_completion.py    --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6

# dry run
torchrun --nproc_per_node 1 --master_port=29501 dry_run.py    --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6

torchrun --nproc_per_node 2 --master_port=29501 dry_run.py    --ckpt_dir llama-2-13b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512 --max_batch_size 6