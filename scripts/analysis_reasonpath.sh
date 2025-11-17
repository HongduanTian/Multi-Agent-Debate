#! /bin/bash
model_name="qwen-2.5-7b"
dataset=("math" "mmlu_pro" "aime24" "aime25")
seeds=42
gpus=0,1

num_reasoning_paths=(4 10 16 32 48 64 96)

for dataset in "${dataset[@]}"; do
    for num_reasoning_paths in "${num_reasoning_paths[@]}"; do
        python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seeds --self_consistency --num_reasoning_paths $num_reasoning_paths --gpu_id $gpus --parallel --exp_name "analysis_reasonpath/${num_reasoning_paths}"
    done
done

