import os
import json
import yaml
import asyncio
import sys
import torch
import random
import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)
logging.getLogger("vllm.logger").setLevel(logging.ERROR)

from src.args import parse_args
from src.config_utils import LLMConfig, load_configs_from_yaml
from src.models import LanguageModel
from src.reasoning_models import MultiAgentDebate
from src.evaluator import MATHEval, MMLUProEval, AIMEEval, GSM8KEval
from src.reasoning_models import prompts_with_format, get_response_from_agent
from src.utils import if_reach_consensus, dataset_2_process_fn
from typing import List


class VariantMATHEval(MATHEval):
    def __init__(self, num_agents:int=3, num_wrong_ans:int=0):
        self.num_agents = num_agents
        self.num_wrong_ans = num_wrong_ans
        self.num_correct_ans = num_agents - num_wrong_ans
        self.dataset_2_process_fn = dataset_2_process_fn("math")
    
    def eval(self, agent_pipeline, data:List, args = None):
        idList = []
        question_list = []
        ground_truth_list = []
        context_list = []
        
        # prepare data
        for item in data:   # each item is a dictionary
            correct_mem = item["correct_answers"]
            wrong_mem = item["wrong_answers"]
            
            if len(wrong_mem) < self.num_wrong_ans or len(correct_mem) < self.num_correct_ans:
                continue
            
            idList.append(item["id"])
            question_list.append(item["question"])
            ground_truth_list.append(item["ground_truth"])
            
            cur_context_list = []
            if self.num_wrong_ans > 0:
                cur_context_list.extend(random.sample(wrong_mem, self.num_wrong_ans))
            
            if self.num_correct_ans > 0:
                cur_context_list.extend(random.sample(correct_mem, self.num_correct_ans))

            context_list.append(cur_context_list)
        
        # simulate the debate round
        formatted_prompts = prompts_with_format(question_list, context_list, reasoning_mode="debate")
        assert len(formatted_prompts) == len(question_list), "The number of formatted prompts should be the same as the number of questions"
        
        agent_response_list = []
        perplexity_list = []
        for _ in range(self.num_agents):
            responsesList, perplexityList = get_response_from_agent(agent_pipeline, formatted_prompts, answer_process=True)
            agent_response_list.append(responsesList)
            perplexity_list.append(perplexityList)
        
        common_answers = []
        for id, prompt, all_resps, all_ppls in zip(idList, formatted_prompts, list(zip(*agent_response_list)), list(zip(*perplexity_list))):
            answer_list = []
            for resp, ppl in zip(all_resps, all_ppls):
                answer_list.append(resp["answer"])
            
            flag, consensus_answer = if_reach_consensus(answer_list, self.dataset_2_process_fn)
            common_answers.append(consensus_answer)
        
        final_results = []
        for id, question, pred, ground_truth, debate_history in zip(idList, question_list, common_answers, ground_truth_list, context_list):
            final_results.append({
                "id": id,
                "question": question,
                "pred": pred,
                "ground_truth": ground_truth,
                "debate_history": debate_history,
            })
        
        # score
        score = np.mean([self.calculate_score(pred, ground_truth) for pred, ground_truth in zip(common_answers, ground_truth_list)])
        
        summary = {
            "accuracy": score,
            "results": final_results,
        }
        
        file_name = f"./motivations/err_mem_test/{args.model_name}_math_{self.num_wrong_ans}-erroneous-memories.json"
        self.save_results(summary, file_name)


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
    data_file = f"./motivations/answer_collection/{args.model_name}_math_seed1996.json"
    
    with open(data_file, "r") as f:
        dataList = json.load(f)
    
    evaluator = VariantMATHEval(args.num_agents, args.num_wrong_ans)
    agent = LanguageModel(llm_configs)
    evaluator.eval(agent, dataList, args)
    
if __name__ == "__main__":
    main()