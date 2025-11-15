import json

import numpy as np

from src.models import LanguageModel
from typing import List, Tuple, Dict, Any
from src.prompts import *
from src.utils import if_reach_consensus, extract_answers, dataset_2_process_fn, extract_with_label


class MultiAgentDebate:
    
    def __init__(self, agent:LanguageModel, dataset_name:str, num_agents:int, max_round:int, prune_strategy:str) -> None:
        self.agent = agent
        self.num_agents = num_agents
        self.max_round = max_round
        self.prune_strategy = prune_strategy
        self.dataset_2_process_fn = dataset_2_process_fn(dataset_name)
        
        self.debate_log = {}
    
    
    def _initial_round(self, prompts:List[str], idList:List) -> Tuple[List, List, List]:
        """The initial round of debate is performed in the CoT way."""
        print(f"\n\n>>>>>> The initial round of debate (Only CoT)\n")
        
        formatted_prompts = prompts_with_format(prompts, contextList=None, reasoning_mode="cot")
        
        agent_response_list = []    # num_agents x batch_size
        perplexity_list = []
        for _ in range(self.num_agents):
            responsesList, perplexityList = get_response_from_agent(self.agent, formatted_prompts, answer_process=True)
            agent_response_list.append(responsesList)
            perplexity_list.append(perplexityList)
        
        # log all results
        agent_masks = []
        for id, prompt, all_resps, all_ppls in zip(idList, formatted_prompts, list(zip(*agent_response_list)), list(zip(*perplexity_list))):
            agents = []
            answer_list = []
            mask = []
            for resp, ppl in zip(all_resps, all_ppls):
                agents.append({
                    "prompt": prompt,
                    "response": resp,
                    "perplexity": ppl
                })
                
                mask.append([])

                answer_list.append(resp["answer"])
            
            agent_masks.append(mask)
            flag, consensus_answer = if_reach_consensus(answer_list, self.dataset_2_process_fn)
            
            self.debate_log[id]["rounds"].append({
                "round": 1,
                "agents": agents,
                "common_answer": consensus_answer,
                "consensus": flag
            })
        return agent_response_list, perplexity_list, agent_masks
    
    
    def _debate_with_contexts(self, round:int,questionsList:List[str], contexts:List, idList:List[str]):
        print(f"\n\n>>>>>> The {round}th round of debate (MAD with Contexts)\n")
        formatted_prompts = prompts_with_format(questionsList, contexts, reasoning_mode="debate")
        assert len(formatted_prompts) == len(questionsList), "The number of formatted prompts should be the same as the number of questions"
        
        agent_response_list = []    # num_agents x batch_size
        perplexity_list = []
        for _ in range(self.num_agents):
            responsesList, perplexityList = get_response_from_agent(self.agent, formatted_prompts, answer_process=True)
            agent_response_list.append(responsesList)
            perplexity_list.append(perplexityList)
        
        common_answers, consensus_flags = [], []
        # log all results
        for id, prompt, all_resps, all_ppls in zip(idList, formatted_prompts, list(zip(*agent_response_list)), list(zip(*perplexity_list))):
            agents = []
            answer_list = []
            mask = []
            for resp, ppl in zip(all_resps, all_ppls):
                agents.append({
                    "prompt": prompt,
                    "response": resp,
                    "perplexity": ppl
                })
                mask.append([])
                answer_list.append(resp["answer"])

            flag, consensus_answer = if_reach_consensus(answer_list, self.dataset_2_process_fn)
            
            self.debate_log[id]["rounds"].append({
                "round": round,
                "agents": agents,
                "common_answer": consensus_answer,
                "consensus": flag
            })
            
            common_answers.append(consensus_answer)
            consensus_flags.append(flag)
    
        return agent_response_list, perplexity_list, common_answers, consensus_flags
    
    def __call__(self, prompts:Dict):
        
        idsList, questionsList, answersList = preprocess_data(prompts)
        for id, question in zip(idsList, questionsList):
            self.debate_log[id] = {
                "question": question,
                "rounds": []
            }
        
        history = []
        perplexity_history = []
        mask_history = []   
        
        # The first debate round
        initial_resps_list, initial_ppls_list, initial_masks = self._initial_round(questionsList, idsList)
        contexts = [list(item) for item in zip(*initial_resps_list)]
        history.append(contexts)
        ppls = [list(item) for item in zip(*initial_ppls_list)]
        perplexity_history.append(ppls)
        mask_history.append(initial_masks)
        
        # Later round of debate
        for round in range(1, self.max_round):
            if self.prune_strategy == "naive":
                new_contexts, masks = self._subjective_prune(questionsList, history[-1])
            elif self.prune_strategy == "ppl":
                new_contexts, masks = self._objective_prune(history[-1], perplexity_history[-1])
            else:
                new_contexts = history[-1]
                masks = [[True, True, True]] * len(questionsList)
            
            resps_list, ppls_list, common_answers, consensus_flags = self._debate_with_contexts(round+1, questionsList, new_contexts, idsList)
            contexts = [list(item) for item in zip(*resps_list)]
            history.append(contexts)
            ppls = [list(item) for item in zip(*ppls_list)]
            perplexity_history.append(ppls)
            mask_history.append(masks)
        
        final_results = []
        for id, quetion, pred, ground_truth, debate_history, consensus_flag, perplexity_history, mask_history in zip(idsList, questionsList, common_answers, answersList, list(zip(*history)), consensus_flags, list(zip(*perplexity_history)), list(zip(*mask_history))):
            final_results.append({
                "id": id,
                "question": quetion,
                "pred": pred,
                "ground_truth": ground_truth,
                "debate_history": debate_history,
                "consensus": consensus_flag,
                "mask_history": mask_history,
                "perplexity_history": perplexity_history,
            })
        return final_results

    def _subjective_prune(self, questionList:List[str], contextList:List[List[str]]) -> List[str]:

        def merge_context_prompts(question:str,ctxList:List[str]) -> List[str]:
            merged_ctx_list = [PRUNE_PROMPT.format(question=question, solution=ctx) for ctx in ctxList]
            return merged_ctx_list
        
        new_contexts = []
        masks = []
        print(f">>>>>> Performing Memory Masking via Subjective Pruning...")
        for question, contexts in zip(questionList, contextList):
            cur_selected_contexts = []
            cur_mask = []
            memory_mask_prompts = merge_context_prompts(question, contexts)
            responses, _ = get_response_from_agent(self.agent, memory_mask_prompts, answer_process=False)
            for resp, context in zip(responses, contexts):
                opinion = extract_with_label(resp, "label")
                if opinion in ["YES", "yes", "Yes", "Y", "y", "Y", "NOT SURE", "not sure", "Not Sure", "Not sure"]:
                    cur_selected_contexts.append(context)
                    cur_mask.append(True)
                else:
                    cur_mask.append(False)
                    continue
            
            new_contexts.append(cur_selected_contexts)
            masks.append(cur_mask)
        return new_contexts, masks
    
    def _objective_prune(self, contextList:List[List[str]], perplexity_contexts:List) -> List[str]:
        new_contexts = []
        masks = []
        
        for ctxList, pplList in zip(contextList, perplexity_contexts):
            cur_selected_contexts = []
            sorted_ppl_list = sorted(pplList, reverse=True)
            threshold = sorted_ppl_list[len(pplList)//2 - 1]
            cur_masks = [ppl < threshold for ppl in pplList]
            for ctx, mask in zip(ctxList, cur_masks):
                if mask:
                    cur_selected_contexts.append(ctx)
                else:
                    continue
            masks.append(cur_masks)
            new_contexts.append(cur_selected_contexts)
        return new_contexts, masks


    def get_summary(self):
        return self.agent.get_token_usage_summary()

    def save_debate_log(self, filename:str):
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.debate_log, f, ensure_ascii=False, indent=2)


class ChainOfThought:
    
    def __init__(self, agent:LanguageModel, dataset_name:str, num_reasoning_paths:int, seed:int=42, self_consistency:bool=False):
        self.agent = agent
        self.dataset_name = dataset_name
        self.num_reasoning_paths = num_reasoning_paths
        self.self_consistency = self_consistency
        self.process_fn = dataset_2_process_fn(dataset_name)
        
        self.cot_log = None
    
    def __call__(self, prompts:Dict):
        idsList, questionsList, answersList = preprocess_data(prompts)
        
        formatted_prompts = prompts_with_format(questionsList, reasoning_mode="cot")
        final_results = []
        if self.self_consistency:
            reasoning_paths = []
            perplexities = []
            for idx, prompt in enumerate(formatted_prompts):
                sc_prompts = [prompt] * self.num_reasoning_paths
                response_list, perplexity_list = get_response_from_agent(self.agent, sc_prompts, answer_process=True)
                
                all_answers = [response["answer"] for response in response_list]
                flag, consensus_answer = if_reach_consensus(all_answers, self.process_fn)
                
                final_results.append({
                    "id": idsList[idx],
                    "question": questionsList[idx],
                    "pred": consensus_answer,
                    "ground_truth": answersList[idx],
                    "response": response_list,
                    "reasoning_paths": reasoning_paths,
                    "perplexities": perplexity_list,
                })
        else:
            responses, perplexities = get_response_from_agent(self.agent, formatted_prompts, answer_process=True)
            
            for id, question, response, perplexity, answer in zip(idsList, questionsList, responses, perplexities, answersList):
                final_results.append({
                    "id": id,
                    "question": question,
                    "pred": self.process_fn(response["answer"]),
                    "ground_truth": answer,
                    "response": response,
                    "perplexity": perplexity,
                })
        self.cot_log = final_results
        return final_results, self.self_consistency
    
    def get_summary(self):
        return self.agent.get_token_usage_summary()
    
    def save_cot_log(self, filename:str):
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.cot_log, f, ensure_ascii=False, indent=2)



def preprocess_data(prompts:Dict):
    ids = []
    questions = []
    answers = []
    for p in prompts:
        ids.append(p["id"])
        questions.append(p["query"])
        answers.append(p["answer"])
    return ids, questions, answers


def prompts_with_format(promptList:List[str], contextList:List[str]=None, reasoning_mode:str="cot") -> List[str]:
    """reasoning_mode: naive, cot, debate"""
    updated_prompts = []
    # if contextList is not provided, use cot mode
    if reasoning_mode == "debate" and contextList is None:
        reasoning_mode = "cot"
    
    for idx, prompt in enumerate(promptList):
        if reasoning_mode == "cot" or contextList is None or len(contextList[idx]) == 0:
            new_prompt = f"{prompt}\n{COT_PROMPT}"
        elif reasoning_mode == "debate":
            assert contextList is not None and len(contextList[idx]) > 0, "Context list is required for debate mode"
            new_prompt = DEBATE_PROMPT.format(question=prompt, context="\n".join(str(ctx) for ctx in contextList[idx]))
        else:
            raise ValueError(f"Invalid reasoning mode: {reasoning_mode}")
        
        new_prompt = prompt + f"\n\n### Response format (must be strictly followed) (do not include any other formats except for the given XML format): \n<think></think>\n<answer></answer>."
        
        updated_prompts.append(new_prompt)
    return updated_prompts


def get_response_from_agent(agent:LanguageModel, prompts:List[str], answer_process:bool=True):
    resultsDict = agent(prompts, answer_process=answer_process)
    return resultsDict["results"], resultsDict["perplexities"]