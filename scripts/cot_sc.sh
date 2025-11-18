#! /bin/bash
model_name="qwen2.5-7b"
dataset=("math" "mmlu_pro")
seeds=(41 42 43 44 45)
gpus=0,1

for dataset in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
        python chain_of_thoughts.py --model_name $model_name --dataset $dataset --seed $seed --self_consistency --num_reasoning_paths 6 --gpu_id $gpus --parallel
    done
done

python chain_of_thoughts.py --model_name $model_name --dataset "aime24" --seed 42 --self_consistency --num_reasoning_paths 6 --gpu_id $gpus --parallel

python chain_of_thoughts.py --model_name $model_name --dataset "aime25" --seed 42 --self_consistency --num_reasoning_paths 6 --gpu_id $gpus --parallel