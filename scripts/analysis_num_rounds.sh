#! /bin/bash
model_name="qwen2.5-7b"
datasets=("math" "mmlu_pro" "aime24" "aime25")
seeds=45
gpus=0,1

num_agents=3
num_rounds_arr=(3 5 6 8 10)

for ds in "${datasets[@]}"; do
    for num_rounds in "${num_rounds_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "none" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad/num_rounds_${num_rounds}"
    done
done

for ds in "${datasets[@]}"; do
    for num_rounds in "${num_rounds_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad-naive-prune/num_rounds_${num_rounds}"
    done
done

for ds in "${datasets[@]}"; do
    for num_rounds in "${num_rounds_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round $num_rounds --prune_strategy "ppl" --gpu_id $gpus --parallel --exp_name "analysis_num_rounds_mad-ppl-prune/num_rounds_${num_rounds}"
    done
done