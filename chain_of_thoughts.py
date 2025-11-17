import os
import yaml
import asyncio
import sys
import torch

import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)
logging.getLogger("vllm.logger").setLevel(logging.ERROR)

from src.args import parse_args
from src.config_utils import LLMConfig, load_configs_from_yaml
from src.models import LanguageModel
from src.reasoning_models import ChainOfThought
from src.evaluator import MATHEval, MMLUProEval, AIMEEval, GSM8KEval


def main():
    args = parse_args()
    
    # GPU settings
    if args.parallel:
        gpus = args.gpu_id.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        tensor_parallel_size = len(gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        tensor_parallel_size = 1
    
    # load configurations
    configs = load_configs_from_yaml("configs.yaml")   
    llm_configs = LLMConfig(configs["llm_configs"]["general_configs"], configs["llm_configs"][args.model_name])
    llm_configs.tensor_parallel_size = tensor_parallel_size
    
    # dataset
    dataset_path = f"./processed_data/{args.dataset}/{args.dataset}_test.jsonl"
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    print(f">>>>>> Loading {args.dataset} dataset from {dataset_path}...")
    
    save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    if args.dataset == "math":
        samples = 100
        evaluator = MATHEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "gsm8k":
        samples = 100
        evaluator = GSM8KEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "mmlu_pro":
        samples = 100
        evaluator = MMLUProEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "aime24":
        samples = None 
        evaluator = AIMEEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "aime25":
        samples = None 
        evaluator = AIMEEval(dataset_path, save_path, samples, args.seed)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Initialize an LLM as an agent
    agent = LanguageModel(llm_configs)
    cot = ChainOfThought(agent, dataset_name=args.dataset, num_reasoning_paths=args.num_reasoning_paths, seed=args.seed, self_consistency=args.self_consistency)
    
    evaluator.eval(cot, args)
    
if __name__ == "__main__":
    main()