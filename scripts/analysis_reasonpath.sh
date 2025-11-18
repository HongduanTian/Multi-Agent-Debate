#! /bin/bash
model_name="qwen2.5-7b"
datasets=("math" "mmlu_pro" "aime24" "aime25")
seeds=45
gpus=0,1

num_reasoning_paths=(4 10 16 32 48 64 96)

for ds in "${datasets[@]}"; do
    for num_paths in "${num_reasoning_paths[@]}"; do
        python chain_of_thoughts.py --model_name $model_name --dataset $ds --seed $seeds --self_consistency --num_reasoning_paths $num_paths --gpu_id $gpus --parallel --exp_name "analysis_reasonpath/reasonpath_${num_paths}"
    done
done
