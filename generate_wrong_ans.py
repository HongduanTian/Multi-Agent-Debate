import os
import json
import logging
import gc
import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)
logging.getLogger("vllm.logger").setLevel(logging.ERROR)

from typing import List, Dict

from src.args import parse_args
from src.config_utils import LLMConfig, load_configs_from_yaml
from src.models import LanguageModel
from src.reasoning_models import ChainOfThought
from src.evaluator import MATHEval, MMLUProEval, AIMEEval, GSM8KEval


class MATHWrongAnsGenerator(MATHEval):
    def __init__(self, dataset_path:str, save_path:str, samples:int=100, seed:int=42):
        super().__init__(dataset_path, save_path, samples, seed)

    def eval(self, agent_pipeline, args = None):
        
        # evaluating models of datasets
        print(f"Evaluating on {len(self.data)} test data.")
        results = agent_pipeline(self.data)
        
        result_dict, sc_flag = results
        case_list = self._collect_answers(result_dict)

        save_path = f"./motivations/answer_collection/"
        os.makedirs(save_path, exist_ok=True)
        file_name = f"{save_path}/{args.model_name}_{args.dataset}_seed{args.seed}"
        self.save_results(case_list, file_name+".json")
    
    def _collect_answers(self, ans_list:List) -> List:
        all_items = []
        for ans_dict in ans_list:
            cur_cor_list = []
            cur_err_list = []
            gt = self._extract_model_answer(ans_dict["ground_truth"])
            for resp in ans_dict["response"]:
                pred = self._extract_model_answer(resp["answer"])
                if resp["answer"] == "":
                    continue
                elif self._math_equal(pred, gt) or gt in pred:
                    cur_cor_list.append(resp)
                else:
                    cur_err_list.append(resp)
            if len(cur_cor_list) <4 or len(cur_err_list) <4:
                continue
            item_dict = {
                "id": ans_dict["id"],
                "question": ans_dict["question"],
                "ground_truth": ans_dict["ground_truth"],
                "correct_answers": cur_cor_list,
                "wrong_answers": cur_err_list,
            }
            all_items.append(item_dict)
        return all_items
    
    def save_results(self, results, file_name):
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

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
    
    dataset_path = f"./processed_data/{args.dataset}/{args.dataset}_test.jsonl"
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    print(f">>>>>> Loading {args.dataset} dataset from {dataset_path}...")
    
    save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # dataset
    if args.dataset == "math":
        samples = 1000
        evaluator = MATHWrongAnsGenerator(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "gsm8k":
        samples = 100
        evaluator = GSM8KEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "mmlu_pro":
        samples = 100
        evaluator = MMLUProEval(dataset_path, save_path, samples, args.seed)
    elif args.dataset == "aime24":
        samples = 100
        evaluator = AIMEEval(dataset_path, save_path, samples, args.seed)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Initialize an LLM as agent
    agent = LanguageModel(llm_configs)
    cot_agent = ChainOfThought(agent, dataset_name=args.dataset, num_reasoning_paths=args.num_reasoning_paths, seed=args.seed, self_consistency=args.self_consistency)
    
    evaluator.eval(cot_agent, args)
    del agent
    del cot_agent
    gc.collect()


if __name__ == "__main__":
    main()