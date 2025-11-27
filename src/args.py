import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Load arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--parallel", action="store_true", default=False, help="Whether to use parallel mode.")
    parser.add_argument("--gpu_id", type=str, default="0", help="The GPU ID.")
    parser.add_argument("--seed", type=int, default=42, help="The seed of the experiment.")
    parser.add_argument("--model_name", type=str, default="qwen-2.5-7b", help="The name of the experiment.")
    parser.add_argument("--dataset", type=str, default="math", choices=["gsm8k", "math", "mmlu_pro", "aime24", "aime25"], help="The name of the dataset.")
    parser.add_argument("--num_agents", type=int, default=3, help="The number of agents.")
    parser.add_argument("--prune_strategy", type=str, default="naive", choices=["none","naive", "ppl"], help="The prune mode.")
    parser.add_argument("--strict", action="store_true", default=False, help="Whether to use strict mode.")
    parser.add_argument("--max_round", type=int, default=2, help="Maximum number of rounds.")
    parser.add_argument("--mode", type=str, default="mad", choices=["mad", "cot"], help="The mode of the experiment.")
    parser.add_argument("--num_reasoning_paths", type=int, default=6, help="The number of reasoning paths. Only for CoT.")
    parser.add_argument("--save_path", type=str, default="results", help="The path to save the results.")
    parser.add_argument("--self_consistency", action="store_true", default=False, help="Whether to use self-consitency mode.")
    
    parser.add_argument("--exp_name", type=str, default="main", help="The name of the experiment.")
    args = parser.parse_args()
    return args