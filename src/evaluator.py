import os
import json
import time
import numpy as np
import datetime
import asyncio
import re
import regex
import sys
import inspect
from math import isclose
from typing import Any, Callable, List, Literal, Tuple
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import OrderedDict

try:
    from sympy import simplify, parse_expr
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    def __init__(self, data_path:str, save_path:str, num_samples:int=None, seed:int=42):
        self.data_path = data_path
        self.save_path = save_path
        
        self.data = self.load_data(num_samples, seed)

    def load_data(self, num_samples:int=None, seed:int=42):
        # load query data from the json files.
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
        # prepare data
        if num_samples is not None:
            np.random.seed(seed)
            data = np.random.choice(data, size=num_samples, replace=False)
        else:
            data = data
        return data
    
    @abstractmethod
    def calculate_score(self, prediction, expected_output) -> float:
        pass
    
    def prepare_data_item(self, data_item):
        """Prepare data item for evaluation. Override in subclasses if needed."""
        return data_item
    
    def eval(self, agent_pipeline, args = None):
        
        # evaluating models of datasets
        time_start = time.time()
        print(f"Evaluating on {len(self.data)} test data.")
        results = agent_pipeline(self.data)
        
        if agent_pipeline.__class__.__name__ == "MultiAgentDebate":
            score = np.mean([self.calculate_score(result["pred"], result["ground_truth"]) for result in results])
            token_usage_summary = agent_pipeline.get_summary()
            summary = {
                "accuracy": score,
                "token_usage_summary": token_usage_summary,
                "results": results,
            }
            eval_time = time.time() - time_start
            print(f"Evaluation time: {eval_time} seconds")
            print(f"Average time per sample: {eval_time / len(self.data)} seconds")
            summary["time_consumption"] = eval_time
            summary["average_time_per_sample"] = eval_time / len(self.data)
            
            method_name = "mad_naive" if args.prune_strategy == "naive" else "mad_ppl" if args.prune_strategy == "ppl" else "mad"
            file_name = f"{self.save_path}/{args.model_name}/{args.dataset}/{method_name}_{args.num_agents}agents_{args.max_round}rounds_seed{args.seed}"
            agent_pipeline.save_debate_log(file_name+"_debate_log.json")
            self.save_results(summary, file_name+".json")
            
            print("================================================")
            print(f"MAD ({args.prune_strategy} prune) with {args.num_agents} agents and {args.max_round} rounds")
            print(f"Accuracy: {score}")
            print(f"Total Tokens: {token_usage_summary['total_tokens']}")
            print("================================================")

        elif agent_pipeline.__class__.__name__ == "ChainOfThought":
            result_dict, sc_flag = results
            if sc_flag:
                score = np.mean([self.calculate_score(result["pred"], result["ground_truth"]) for result in result_dict])
                token_usage_summary = agent_pipeline.get_summary()
                summary = {
                    "accuracy": score,
                    "token_usage_summary": token_usage_summary,
                    "results": result_dict,
                }
                eval_time = time.time() - time_start
                print(f"Evaluation time: {eval_time} seconds")
                print(f"Average time per sample: {eval_time / len(self.data)} seconds")
                summary["time_consumption"] = eval_time
                summary["average_time_per_sample"] = eval_time / len(self.data)
                
                method_name = "cot_sc"
                file_name = f"{self.save_path}/{args.model_name}/{args.dataset}/{method_name}_{args.num_reasoning_paths}paths_seed{args.seed}"
                agent_pipeline.save_cot_log(file_name+"_cot_log.json")
                self.save_results(summary, file_name+".json")
                
                print("================================================")
                print(f"Self-Consitency: {args.num_reasoning_paths} paths")
                print(f"Accuracy: {score}")
                print(f"Total Tokens: {token_usage_summary['total_tokens']}")
                print("================================================")
            else:
                score = np.mean([self.calculate_score(result["pred"], result["ground_truth"]) for result in result_dict])
                token_usage_summary = agent_pipeline.get_summary()
                summary = {
                    "accuracy": score,
                    "token_usage_summary": token_usage_summary,
                    "results": result_dict,
                }
                eval_time = time.time() - time_start
                print(f"Evaluation time: {eval_time} seconds")
                print(f"Average time per sample: {eval_time / len(self.data)} seconds")
                summary["time_consumption"] = eval_time
                summary["average_time_per_sample"] = eval_time / len(self.data)
                
                method_name = "cot"
                file_name = f"{self.save_path}/{args.model_name}/{args.dataset}/{method_name}_seed{args.seed}"
                agent_pipeline.save_cot_log(file_name+"_cot_log.json")
                self.save_results(summary, file_name+".json")
                
                print("================================================")
                print(f"CoT")
                print(f"Accuracy: {score}")
                print(f"Total Tokens: {token_usage_summary['total_tokens']}")
                print("================================================")
        else:
            raise ValueError(f"Invalid agent pipeline: {agent_pipeline.__class__.__name__}")
   
    def save_results(self, results, file_name):
        # Ensure the directory exists before saving
        import os
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

# --------------------------------------- Evaluators ---------------------------------------
class MMLUProEval(BaseEvaluator):
    #@classmethod
    def calculate_score(self, prediction: str, expected_output: str) -> float:
        try:
            extracted_answer = self._extract_model_answer(prediction)
            pred_char = self.normalize_char(extracted_answer)
        except:
            pred_match = re.search(r'([Ａ-ＺA-Za-z])', prediction)
            if not pred_match:
                return 0.0
            pred_char = self.normalize_char(pred_match.group(1))
        correct_char = self.normalize_char(expected_output.strip())
        return 1.0 if pred_char == correct_char else 0.0
    
    def _extract_model_answer(self, text):
        pattern = r"answer is \(?([A-J])\)?"
        try:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                # print("1st answer extract failed\n" + text)
                return self._extract_again(text)
        except Exception as e:
            print(text)
            raise Exception(f"Error in extracting model answer: {e}")
        

    def _extract_again(self, text):
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return self._extract_final(text)


    def _extract_final(self, text):
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None
    
    def normalize_char(self, char: str) -> str:
        if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
            char = chr(ord(char) - 0xFEE0)
        
        char = char.replace("：", ":")
        
        return char.strip().lower()[0] if char else ""
    
    def prepare_data_item(self, data_item):
        data_item["query"] = f"### Please DIRECTLY provide the option letter in your final answer. ###\n{data_item['query']}"
        data_item["id"] = data_item.get("id", "unknown-id")
        return data_item


class MATHEval(BaseEvaluator):
    #@classmethod
    def calculate_score(self, prediction:str, expected_output:str) -> float:
        try:
            # print("prediction: ", prediction)
            pred_answer = self._extract_model_answer(prediction)
            is_correct = self._math_equal(pred_answer, self._extract_model_answer(str(expected_output)))
            return 1.0 if is_correct else 0.0
        except Exception as e:
            print(f"Error in evaluation: {e}")
            print(f"Prediction: {prediction}")
            print(f"Ground Truth: {expected_output}")
    
    def _extract_model_answer(self, prediction: str) -> str:
        
        def _find_box(string:str) -> str:
            res = regex.findall(r"\\boxed\{(.*)\}", string)
            if not res:
                res = regex.findall(r"\\fbox\{(.*)\}", string)
            if not res:
                return None
            return res[-1]
        
        if prediction is None:
            return ""

        box_format_ans = _find_box(prediction)
        if box_format_ans:
            return box_format_ans
        
        gsm8k_format_ans = re.findall(r"####\s*(-?[0-9\.\,]+)", prediction)
        if gsm8k_format_ans:
            return gsm8k_format_ans[-1].replace(",", "").replace("$", "")
        
        # Then try to find "The answer is X" format
        answer_pattern = re.findall(r'(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)', prediction)
        if answer_pattern:
            return answer_pattern[-1]

        ans_line = prediction.split("\n")[-1]
        return ans_line
        
    def _math_equal(self, pred: str, reference: str) -> bool:
        if pred is None or reference is None:
            return False
        
        pred_raw = str(pred)
        reference_raw = str(reference)
        
        pm_pattern = r"(.+?)\s*\\pm\s*(.+)"
        
        def _check_pm_match(pm_str, list_str):
            pm_match = re.match(pm_pattern, pm_str.strip())
            if not pm_match:
                return False
            
            parts = re.split(r'[,;]', list_str)
            if len(parts) != 2:
                return False
            
            A = pm_match.group(1).strip()
            B = pm_match.group(2).strip()
            
            pm_set = {strip_string(f"{A}+{B}"), strip_string(f"{A}-{B}")}
            list_set = {strip_string(parts[0]), strip_string(parts[1])}
            
            return pm_set == list_set
    
        if r"\pm" in reference_raw and _check_pm_match(reference_raw, pred_raw):
            return True
        if r"\pm" in pred_raw and _check_pm_match(pred_raw, reference_raw):
            return True

        # Apply enhanced normalization to both prediction and reference
        prediction_str = strip_string(pred_raw)
        reference_str = strip_string(reference_raw)
    
        # Path 0: Direct string equality after normalization
        if prediction_str == reference_str:
            return True

        # Path 1: Numerical comparison
        if is_digit(prediction_str) and is_digit(reference_str):
            pred_float = parse_digits(prediction_str)
            ref_float = parse_digits(reference_str)
            if pred_float is not None and ref_float is not None and isclose(pred_float, ref_float, rel_tol=1e-4):
                return True

        # Path 2: Unordered Tuple/Vector comparison
        pred_parts = [p.strip() for p in re.split(r'[,;]', prediction_str.strip('()[]{}')) if p.strip()]
        ref_parts = [r.strip() for r in re.split(r'[,;]', reference_str.strip('()[]{}')) if r.strip()]
        if len(pred_parts) > 1 and len(pred_parts) == len(ref_parts):
            # Sort the parts before comparing to handle unordered lists
            if sorted(pred_parts) == sorted(ref_parts):
                return True

        # Path 3: Symbolic comparison
        if symbolic_equal(prediction_str, reference_str):
            return True

        return False
    
    def prepare_data_item(self, data_item):
        data_item["query"] = "### Please provide your final answer in the format of \\boxed{} format. ###" + f"\n{data_item['query']}"
        return data_item
        

from src.utils import extract_number
class AIMEEval(BaseEvaluator):
    #@classmethod
    def calculate_score(self, prediction:str, expected_output:str) -> float:
        try:
            pred_answer = self._extract_model_answer(str(prediction))
            # print("prediction: ", prediction)
            # print("pred_answer: ", pred_answer)
            # print("expected_output: ", expected_output)

            is_correct = self._math_equal(pred_answer, str(expected_output))
            return 1.0 if is_correct else 0.0
        except Exception as e:
            print(prediction)
            print(f"Error in evaluation: {e}")
            print(f"Prediction: {prediction}")
            print(f"Ground Truth: {expected_output}")
    
    def _extract_model_answer(self, prediction: str) -> str:
        
        def _find_box(string:str) -> str:
            res = regex.findall(r"\\boxed\{(.*)\}", string)
            if not res:
                res = regex.findall(r"\\fbox\{(.*)\}", string)
            if not res:
                return None
            return res[-1]
        
        if prediction is None:
            return ""

        box_format_ans = _find_box(prediction)
        if box_format_ans:
            return box_format_ans

        gsm8k_format_ans = re.findall(r"####\s*(-?[0-9\.\,]+)", prediction)
        if gsm8k_format_ans:
            return gsm8k_format_ans[-1].replace(",", "").replace("$", "")
        
        # Then try to find "The answer is X" format
        answer_pattern = re.findall(r'(?:[Tt]he(?:\s+final)?(?:\s+answer)?(?:\s+is)?:?)\s*([^\n]+)', prediction)
        if answer_pattern:
            return answer_pattern[-1]

        answer_extracted = extract_number(prediction)
        if answer_extracted:
            return answer_extracted

        ans_line = prediction.split("\n")[-1]
        return ans_line
        
    def _math_equal(self, pred: str, reference: str) -> bool:
        if pred is None or reference is None:
            return False
        
        pred_raw = str(pred)
        reference_raw = str(reference)
        
        pm_pattern = r"(.+?)\s*\\pm\s*(.+)"
        
        def _check_pm_match(pm_str, list_str):
            pm_match = re.match(pm_pattern, pm_str.strip())
            if not pm_match:
                return False
            
            parts = re.split(r'[,;]', list_str)
            if len(parts) != 2:
                return False
            
            A = pm_match.group(1).strip()
            B = pm_match.group(2).strip()
            
            pm_set = {strip_string(f"{A}+{B}"), strip_string(f"{A}-{B}")}
            list_set = {strip_string(parts[0]), strip_string(parts[1])}
            
            return pm_set == list_set
    
        if r"\pm" in reference_raw and _check_pm_match(reference_raw, pred_raw):
            return True
        if r"\pm" in pred_raw and _check_pm_match(pred_raw, reference_raw):
            return True

        # Apply enhanced normalization to both prediction and reference
        prediction_str = strip_string(pred_raw)
        reference_str = strip_string(reference_raw)
    
        # Path 0: Direct string equality after normalization
        if prediction_str == reference_str:
            return True

        # Path 1: Numerical comparison
        if is_digit(prediction_str) and is_digit(reference_str):
            pred_float = parse_digits(prediction_str)
            ref_float = parse_digits(reference_str)
            if pred_float is not None and ref_float is not None and isclose(pred_float, ref_float, rel_tol=1e-4):
                return True

        # Path 2: Unordered Tuple/Vector comparison
        pred_parts = [p.strip() for p in re.split(r'[,;]', prediction_str.strip('()[]{}')) if p.strip()]
        ref_parts = [r.strip() for r in re.split(r'[,;]', reference_str.strip('()[]{}')) if r.strip()]
        if len(pred_parts) > 1 and len(pred_parts) == len(ref_parts):
            # Sort the parts before comparing to handle unordered lists
            if sorted(pred_parts) == sorted(ref_parts):
                return True

        # Path 3: Symbolic comparison
        if symbolic_equal(prediction_str, reference_str):
            return True

        return False 
    
    def prepare_data_item(self, data_item):
        data_item["query"] = "### Please provide your final answer in the format of \\boxed{} format. ###" + f"\n{data_item['query']}"
        return data_item


from src.utils import extract_number
class GSM8KEval(BaseEvaluator):
    def calculate_score(self, prediction, expected_output) -> float:
        expected_output = extract_number(expected_output)
        prediction = extract_number(prediction)

        if prediction is None:
            return 0.0
        
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0
    
    def prepare_data_item(self, data_item):
        # GSM8K doesn't need special preparation
        data_item["query"] = data_item["question"]
        data_item["id"] = data_item.get("id", "unknown-id")
        data_item["answer"] = data_item["answer"]
        return data_item


from src.utils import f1_score
class HotpotQAEval(BaseEvaluator):
    def calculate_score(self, prediction, expected_output) -> float:
        return f1_score(prediction, expected_output)
    
    def prepare_data_item(self, data_item):
        paragraphs = [item[1] for item in data_item["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        
        # Store original query for reporting
        #data_item["question"] = data_item["query"]
        # Add context to query
        #data_item["query"] = f"{data_item['query']}\n\nRevelant Context:{context_str}"
        # Ensure ID field is standardized
        #data_item["id"] = data_item.get("_id", data_item.get("id", "unknown"))
        
        return data_item


class MMLUEval(BaseEvaluator):
    def calculate_score(self, prediction: str, expected_output: str) -> float:

        pred_match = re.search(r'([Ａ-ＤA-Da-d])', prediction)
        if not pred_match:
            return 0.0
        
        pred_char = self.normalize_char(pred_match.group(1))
        correct_char = self.normalize_char(expected_output.strip())
        
        return 1.0 if pred_char == correct_char else 0.0
    
    def normalize_char(self, char: str) -> str:

        if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
            char = chr(ord(char) - 0xFEE0)
        
        char = char.replace("：", ":")
        
        return char.strip().lower()[0] if char else ""
    
    def prepare_data_item(self, data_item):
        data_item["query"] = f"**ONLY include option letter in your final answer**\n{data_item['query']}"
        data_item["id"] = data_item.get("id", "unknown-id")
        return data_item

def strip_string(string: str) -> str:
    """Enhanced string normalization to handle units, LaTeX commands, and other inconsistencies."""
    string = str(string).strip()
    
    # Remove LaTeX wrappers for text/units
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{(.*?)\}", r"\1", string)

    # Normalize and remove common symbols and units
    string = string.replace("^{\\circ}", "")
    string = string.replace("\\circ", "")
    string = string.replace("°", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = re.sub(r"\s*degrees", "", string, flags=re.IGNORECASE)
    
    # Remove common textual units (can be expanded)
    string = re.sub(r"\s*inches\^2", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*square inches", "", string, flags=re.IGNORECASE)

    # Standard replacements
    string = string.replace("\n", "")
    if string.endswith("."):
        string = string[:-1]
    string = string.replace("\\!", "")
    string = string.replace("\\ ", " ")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\n", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # Remove dollar signs
    string = string.replace("$", "")
    
    # Remove spaces and special characters last for clean comparison
    string = re.sub(r" ", "", string)
    string = re.sub("\u200b", "", string)  # Zero-width space

    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    
    return string.strip()


def _fix_fracs(string):
    """Fix fraction formatting in LaTeX strings."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_sqrt(string):
    """Fix square root formatting in LaTeX strings."""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def strip_string(string: str) -> str:
    """Enhanced string normalization to handle units, LaTeX commands, and other inconsistencies."""
    string = str(string).strip()
    
    # Remove LaTeX wrappers for text/units
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{(.*?)\}", r"\1", string)

    # Normalize and remove common symbols and units
    string = string.replace("^{\\circ}", "")
    string = string.replace("\\circ", "")
    string = string.replace("°", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = re.sub(r"\s*degrees", "", string, flags=re.IGNORECASE)
    
    # Remove common textual units (can be expanded)
    string = re.sub(r"\s*inches\^2", "", string, flags=re.IGNORECASE)
    string = re.sub(r"\s*square inches", "", string, flags=re.IGNORECASE)

    # Standard replacements
    string = string.replace("\n", "")
    if string.endswith("."):
        string = string[:-1]
    string = string.replace("\\!", "")
    string = string.replace("\\ ", " ")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\n", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # Remove dollar signs
    string = string.replace("$", "")
    
    # Remove spaces and special characters last for clean comparison
    string = re.sub(r" ", "", string)
    string = re.sub("\u200b", "", string)  # Zero-width space

    string = _fix_fracs(string)
    string = _fix_sqrt(string)
    
    return string.strip()

def parse_digits(num):
    """Parse numeric values from strings."""
    num_str = strip_string(str(num))
    num_str = regex.sub(",", "", num_str)
    try:
        return float(num_str)
    except:
        return None


def is_digit(num):
    """Check if a value can be parsed as a number."""
    return parse_digits(num) is not None

def symbolic_equal(a, b):
    """Check if two expressions are symbolically equal."""
    if not LATEX2SYMPY_AVAILABLE:
        try:
            return simplify(a) == simplify(b)
        except:
            return False

    def _parse(s):
        try:
            return latex2sympy(s)
        except:
            try:
                return parse_expr(s)
            except:
                return s
    
    try:
        if simplify(_parse(a) - _parse(b)) == 0:
            return True
    except:
        pass
    return False


# class MATHEval(BaseEvaluator):
    
#     def extract_model_answer(self, text: str) -> str:
#         pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
#         boxed_matches = re.findall(pattern, text, re.DOTALL)
#         if boxed_matches:
#             return boxed_matches[-1].strip()

#         sentence_end_pattern = r"(?<!\d)[.!?]\s+"
#         sentences = re.split(sentence_end_pattern, text)
#         sentences = [s.strip() for s in sentences if s.strip()]
#         return sentences[-1] if sentences else ""

#     def calculate_score(self, prediction: str, expected_output: str) -> float:
#         expected_answer = self.extract_model_answer(expected_output)
#         predicted_answer = prediction.strip()

#         if self.math_equal(predicted_answer, expected_answer):
#             return 1.0
#         else:
#             return 0.0

#     def math_equal(self, prediction: Any, reference: Any) -> bool:
#         if str(prediction) == str(reference):
#             return True

#         try:
#             if self.is_digit(prediction) and self.is_digit(reference):
#                 prediction = self.parse_digits(prediction)
#                 reference = self.parse_digits(reference)
#                 return isclose(prediction, reference, abs_tol=1e-3)
#         except:
#             pass

#         if SYMPY_AVAILABLE:
#             try:
#                 return self.symbolic_equal(prediction, reference)
#             except:
#                 pass

#         return False

#     def is_digit(self, num):
#         return self.parse_digits(num) is not None

#     def parse_digits(self, num):
#         if not regex or not SYMPY_AVAILABLE:
#             return None
            
#         num = regex.sub(",", "", str(num))
#         try:
#             return float(num)
#         except:
#             if num.endswith("%"):
#                 num = num[:-1]
#                 if num.endswith("\\"):
#                     num = num[:-1]
#                 try:
#                     return float(num) / 100
#                 except:
#                     pass
#         return None

#     def symbolic_equal(self, a, b):
#         if not SYMPY_AVAILABLE:
#             return False
            
#         def _parse(s):
#             for f in [parse_latex, parse_expr]:
#                 try:
#                     return f(s)
#                 except:
#                     pass
#             return s

#         a = _parse(a)
#         b = _parse(b)

#         try:
#             if simplify(a - b) == 0:
#                 return True
#         except:
#             pass

#         try:
#             if isclose(N(a), N(b), abs_tol=1e-3):
#                 return True
#         except:
#             pass
#         return False
    
#     def log_mismatch(self, input_text, expected_output, output, extracted_output, extract_answer_code=None):
#         """Log mismatched answers for debugging"""
#         print("\n==== MATH MISMATCH ====")
#         print(f"Input: {input_text[:100]}...")
#         print(f"Expected: {expected_output[:100]}...")
#         print(f"Output: {output[:100]}...")
#         print(f"Extraction: {extracted_output}")
#         print("=======================\n")