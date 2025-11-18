#! /bin/bash
model_name="qwen2.5-7b"
dataset=("math" "mmlu_pro" "aime24" "aime25")
seeds=(41 42 43 44 45)
gpus=0,1

for dataset in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seed --num_agents 3 --max_round 2 --prune_strategy "none" --gpu_id $gpus --parallel
    done
done

python multi_agent_debate.py --model_name $model_name --dataset "aime24" --seed 42 --num_agents 3 --max_round 2 --prune_strategy "none" --gpu_id $gpus --parallel

python multi_agent_debate.py --model_name $model_name --dataset "aime25" --seed 42 --num_agents 3 --max_round 2 --prune_strategy "none" --gpu_id $gpus --parallel