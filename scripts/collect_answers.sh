#! /bin/bash
model_name=("qwen2.5-3b" "qwen2.5-7b" "qwen2.5-math-7b")
#model_name=("qwen2.5-3b")
seed=1996
gpus=0,1

for model in "${model_name[@]}"; do
    python generate_wrong_ans.py --model_name $model --seed $seed --gpu_id $gpus --dataset "math" --num_reasoning_paths 100 --self_consistency --parallel
done