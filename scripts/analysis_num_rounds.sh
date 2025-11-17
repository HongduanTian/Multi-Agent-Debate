#! /bin/bash
model_name="qwen-2.5-7b"
dataset=("math" "mmlu_pro" "aime24" "aime25")
seeds=42
gpus=0,1

num_rounds=(3 5 6 8 10)

for dataset in "${dataset[@]}"; do
    for num_rounds in "${num_rounds[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "none" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad/${num_rounds}"
    done
done

for dataset in "${dataset[@]}"; do
    for num_agents in "${num_agents[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad-naive-prune/${num_rounds}"
    done
done

for dataset in "${dataset[@]}"; do
    for num_agents in "${num_agents[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $dataset --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "ppl" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad-ppl-prune/${num_rounds}"
    done
done