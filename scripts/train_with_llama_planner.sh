#!/bin/bash
# Job name:
#SBATCH --job-name=cs285_proj
#
# Account:
#SBATCH --account=pc_dsdisc
#
# Partition:
#SBATCH --partition=savio3_gpu
#
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#SBATCH --qos=v100_gpu3_normal
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=4
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:V100:1
#
# Wall clock limit:
#SBATCH --time=24:00:00

## Command(s) to run:
cd ../rl-starter-files
python -m scripts.train --algo ppo --env BabyAI-GoToImpUnlock-v0 --text --use-planner --llm llama --ask-every 5 --frames 1000000 --recurrence 20 --obs-size 11 --frames-per-proc 40 --procs 64 --batch-size 200 