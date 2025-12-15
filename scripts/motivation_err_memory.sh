#! /bin/bash
model_name=("qwen2.5-3b" "qwen2.5-7b" "qwen2.5-math-7b")
num_erroneous_memories=(0 1 2 3)
seed=42
gpus=0,1

for model in "${model_name[@]}"; do
    for num_erroneous_memory in "${num_erroneous_memories[@]}"; do
        python motivation_err_mem.py --model_name $model --seed $seed --gpu_id $gpus --dataset "math" --num_wrong_ans $num_erroneous_memory --num_agents 3 --parallel
    done
done