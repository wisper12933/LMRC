#!/bin/bash
#SBATCH --job-name=qwen2,5_7B_docred_dev_20
#SBATCH -p q_intel_share_L20
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/progressing/qwen2,5_7B_docred_dev_20.out
#SBATCH -e ./logs/progressing/qwen2,5_7B_docred_dev_20.err
module add anaconda3/2023.3
module add cuda/12.9
cd /mnt/home/user28/LMRC/RC
source activate lmrc_rc

python -m infer.run_eval \
    --config ./infer/eval_config.yaml \
    --test_file /mnt/home/user28/LMRC/RC/data/progressing/docred_dev_all_20.json \
    --result_file /mnt/home/user28/LMRC/RC/results/progressing/qwen2,5_7B_docred_dev_20.json