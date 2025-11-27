import math
from re import escape
import torch
from vllm import LLM, SamplingParams
from src.config_utils import LLMConfig
from src.model_utils import TokenUsageTracker
from typing import List
from src.utils import extract_answers, extract_answers_with_box


class LanguageModel:
    """LLM class for the LLM model."""
    def __init__(self, llm_config: LLMConfig, system_msg:str=None):
        self.llm_config = llm_config
        
        if self.llm_config.model_path is not None and self.llm_config.model_path != "None":
            llm_name_path = self.llm_config.model_path
        else:
            llm_name_path = self.llm_config.model
        
        print(f">>>>>> LLM name path: {llm_name_path}")

        self.llm = LLM(
            model=llm_name_path,
            tensor_parallel_size=getattr(self.llm_config, "tensor_parallel_size", 1),
            dtype=torch.bfloat16,
            max_model_len=getattr(self.llm_config, "max_token_length", 24064),
        )
        
        self.system_msg = system_msg
        self.token_usage_tracker = TokenUsageTracker()
        
    def __call__(self, prompts:List[str], answer_process:bool=True):
        # construct message
        batch_message = []
        for prompt in prompts:
            message = []
            if self.system_msg is not None:
                message.append({
                    "role": "system",
                    "content": self.system_msg
                })
            message.append({
                "role": "user",
                "content": prompt
            })
            batch_message.append(message)
        
        tokenized_batch_message = self.llm.get_tokenizer().apply_chat_template(batch_message, tokenize=False, add_generation_prompt=True)
        
        outputs = self.llm.generate(
            tokenized_batch_message,
            sampling_params=SamplingParams(
                temperature=getattr(self.llm_config, "temperature", 1),
                top_p=getattr(self.llm_config, "top_p", 1),
                max_tokens=getattr(self.llm_config, "max_tokens", 24064),
                stop_token_ids=[self.llm.get_tokenizer().eos_token_id],
                logprobs=True
            )
        )
        
        results_tuple = self._answer_process(outputs, answer_process)
        results_list, perplexity_list, original_logprobs, total_input_tokens, total_output_tokens = results_tuple
        
        results = {
            "prompts": prompts,
            "results": results_list,
            "perplexities": perplexity_list,
            "original_logprobs": original_logprobs,
            "cur_batch_input_tokens": total_input_tokens,
            "cur_batch_output_tokens": total_output_tokens
        }
        
        self.token_usage_tracker.update_usage(self.llm_config.model, total_input_tokens, total_output_tokens)
        
        print(f">>>>>> Token usage in this batch: Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}, Total tokens: {total_input_tokens + total_output_tokens}")
        return results
        
    def _answer_process(self, outputs:List[str], answer_process:bool=True):
        results_list = []
        perplexity_list = []
        original_logprobs = []
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        for output in outputs:
            response = output.outputs[0].text
            if answer_process:
                try:
                    result = extract_answers(response)
                except ValueError as e:
                    try:
                        answer = extract_answers_with_box(response)
                        result = {"think": response, "answer": answer}
                    except:
                        result = {"think": "", "answer": ""}
                        print(f">>>>>> Error: Fail to parse answers.")
                results_list.append(result)
            else:
                results_list.append(response)
            
            # count the token usage
            total_input_tokens += len(output.prompt_token_ids)
            total_output_tokens += len(output.outputs[0].token_ids)
            
            # perplexity calculation
            logprobs = []
            token_logprobs = output.outputs[0].logprobs
            for tk_logprob in token_logprobs:
                for key, value in tk_logprob.items():
                    if value.rank == 1:
                        logprobs.append(value.logprob)
                        break
            
            original_logprobs.append(logprobs)
            
            if logprobs:
                valid_logprobs = [prob for prob in logprobs if prob != float("-inf") and prob is not None]
                if valid_logprobs:
                    perplexity = math.exp(sum(valid_logprobs) / len(valid_logprobs))
                else:
                    perplexity = float("inf")
            else:
                perplexity = float("inf")
                
            perplexity_list.append(perplexity)
            
        return results_list, perplexity_list, original_logprobs, total_input_tokens, total_output_tokens
    
    def get_token_usage_summary(self):
        return self.token_usage_tracker.get_summary()