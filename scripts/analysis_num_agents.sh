#! /bin/bash
model_name="qwen2.5-7b"
datasets=("math" "mmlu_pro" "aime24" "aime25")
seeds=45
gpus=0,1

num_agents_arr=(4 5 6 8 10)

for ds in "${datasets[@]}"; do
    for num_agents in "${num_agents_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round 2 --prune_strategy "none" --gpu_id $gpus --parallel --exp_name "analysis_num_agents_mad/num_agents_${num_agents}"
    done
done

for ds in "${datasets[@]}"; do
    for num_agents in "${num_agents_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round 2 --prune_strategy "naive" --gpu_id $gpus --parallel --exp_name "analysis_num_agents_mad-naive-prune/num_agents_${num_agents}"
    done
done

for ds in "${datasets[@]}"; do
    for num_agents in "${num_agents_arr[@]}"; do
        python multi_agent_debate.py --model_name $model_name --dataset $ds --seed $seeds --num_agents $num_agents --max_round 2 --prune_strategy "ppl" --gpu_id $gpus --parallel --exp_name "analysis_num_agents_mad-ppl-prune/num_agents_${num_agents}"
    done
done