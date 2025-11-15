#! /bin/bash
model_name="qwen-2.5-7b"
dataset=("math" "mmlu_pro" "aime24" "aime25")
seeds=(41 42 43 44 45)
gpus=0,1

for dataset in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
        python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seed --self_consistency --num_reasoning_paths 6 --gpu_id $gpus --parallel
    done
done