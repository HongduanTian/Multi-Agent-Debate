import re
from collections import Counter
import random


def extract_answers_with_box(text:str):
    # pattern = r"\\boxed{(.*?)}"
    # match = re.search(pattern, text, re.DOTALL)
    # if match:
    #     return match.group(1).strip()
    # else:
    #     raise ValueError("Answer Parsing Failure: 'answer' is missing or empty.")
    """
    Extract the answer from the text using the \\boxed{} format.
    Use balanced bracket matching to handle nested curly braces.
    """
    pattern = r"\\boxed{"
    start_pos = text.find(pattern)
    if start_pos == -1:
        raise ValueError("Answer Parsing Failure: 'answer' is missing or empty.")
    
    # find the position of the first {
    brace_start = start_pos + len(pattern)
    brace_count = 0
    i = brace_start
    
    # find the matching closing brace
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            if brace_count == 0:
                # find the matching closing brace
                answer = text[brace_start:i].strip()
                if not answer:
                    raise ValueError("Answer Parsing Failure: 'answer' is empty inside \\boxed{}.")
                return answer
            brace_count -= 1
        i += 1
    
    # if the matching closing brace is not found
    raise ValueError("Answer Parsing Failure: 'answer' is missing or empty.")


def extract_answers(text:str):
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    found_fields = {match[0]: match[1].strip() for match in matches}
    if "think" not in found_fields:
        raise ValueError("Info Parsing Failure: 'think' is missing or empty.")
    if "answer" not in found_fields:
        raise ValueError("Info Parsing Failure: 'answer' is missing or empty.")
    return found_fields

def extract_with_label(text: str, pattern: str = "answer"):
    pattern = rf'<{pattern}>(.*?)</{pattern}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def if_reach_consensus(answers: list, process_fn = lambda x: x):
    if not answers:
        return False, None
        
    answers = [process_fn(answer) for answer in answers]
    # Count occurrences of each answer
    answer_counts = {}
    for answer in answers:
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1
    
    # Find the most frequent answer
    most_common = max(answer_counts.items(), key=lambda x: x[1])
    most_common_answer, count = most_common
    
    # Check if it appears more than half the time
    if count >= len(answers) / 2:
        return True, most_common_answer
    else:
        return False, most_common_answer

def extract_number(text: str):
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
    if matches:
        last_number = matches[-1].replace(",", "")
        try:
            return float(last_number)
        except ValueError:
            return None
    else:
        return None

def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation by:
    1. Converting to lowercase
    2. Removing parentheses, brackets around options
    3. Removing whitespace
    """
    # Remove various forms of option markers: (A), [A], A), A.
    s = re.sub(r'[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\)]', r'\1\2', str(s))
    return s.lower().strip()

def f1_score(prediction, ground_truth):
    """
    Compute the F1 score between prediction and ground truth answers.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_char(char: str) -> str:

    if len(char) == 1 and 0xFF01 <= ord(char) <= 0xFF5E:
        char = chr(ord(char) - 0xFEE0)
    
    char = char.replace("：", ":")
    
    return char.strip().lower()[0] if char else ""

def dataset_2_process_fn(dataset: str):
    if dataset in ["gsm8k", "aime24", "aime25"]:
        return lambda x: extract_number(x)
    elif dataset == "hotpotqa":
        return lambda x: normalize_answer(x)
    elif dataset == "math":
        return lambda x: x.strip()
    elif dataset in ["mmlu", "mmlu_pro"]:
        return lambda x: normalize_char(x)
    else:
        return None