import os
import json
import string
import argparse


ORIGINAL_DATA_DIR = "./data"
TARGET_DATA_DIR = "./processed_data"


def load_data_from_json(file_path:str):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def dump_data_to_json(data:list, file_path:str):
    with open(file_path, "w") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


def gsm8k_processor():
    dataset_path = os.path.join(ORIGINAL_DATA_DIR, "GSM8K", "test.json")
    all_data = load_data_from_json(dataset_path)
    
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "gsm8k")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "gsm8k"))
        
    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "gsm8k", "gsm8k_test.jsonl"))
    
    
def math_processor():
    filenames = [f for f in os.listdir(os.path.join(ORIGINAL_DATA_DIR, "MATH")) if "test" in f and f.endswith(".json")]
    
    all_data = []
    id = 0
    for filename in filenames:
        with open(os.path.join(ORIGINAL_DATA_DIR, "MATH", filename), "r") as f:
            for line in f:
                entry = json.loads(line)
                if "problem" in entry:
                    entry["query"] = entry.pop("problem")
                if "solution" in entry:
                    entry["answer"] = entry.pop("solution")
                entry["id"] = id
                id += 1
                all_data.append(entry)
    
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "math")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "math"))

    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "math", "math_test.jsonl"))


def mmlu_processor():
    filenames = [f for f in os.listdir(os.path.join(ORIGINAL_DATA_DIR, "MMLU")) if "test" in f and f.endswith(".json")]
    
    all_data = []
    for filename in filenames:
        with open(os.path.join(ORIGINAL_DATA_DIR, "MMLU", filename), "r") as f:
            for line in f:
                entry = json.loads(line)
                if "question" in entry:
                    entry["query"] = entry.pop("question")
                if "answer" in entry:
                    entry["answer"] = entry.pop("answer")
                all_data.append(entry)
                
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "mmlu")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "mmlu"))
        
    print(len(all_data))
    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "mmlu", "mmlu_test.jsonl"))
    

def mmlu_pro_processor():
    dataset_path = os.path.join(ORIGINAL_DATA_DIR, "MMLU-PRO", "mmlu_pro_test.json")
    alphabet_list = string.ascii_uppercase
    all_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            option_letters = alphabet_list[:len(entry["options"])]
            if "question" in entry:
                entry["query"] = entry.pop("question")
            if "answer" in entry:
                entry["answer"] = entry.pop("answer")
            if "question_id" in entry:
                entry["id"] = entry.pop("question_id")
            
            for i in range(len(option_letters)):
                entry["query"] += f"\n{option_letters[i]}. {entry['options'][i]}"
            entry.pop("options")
            all_data.append(entry)
    
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "mmlu_pro")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "mmlu_pro"))
        
    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "mmlu_pro", "mmlu_pro_test.jsonl"))
    

def aime24_processor():
    dataset_path = os.path.join(ORIGINAL_DATA_DIR, "AIME24", "aime24_test.json")
    all_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "problem" in entry:
                entry["query"] = entry.pop("problem")
            if "answer" in entry:
                entry["answer"] = entry.pop("answer")
            all_data.append(entry)
    
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "aime24")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "aime24"))
        
    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "aime24", "aime24_test.jsonl"))


def aime25_processor():
    filenames = [f for f in os.listdir(os.path.join(ORIGINAL_DATA_DIR, "AIME25")) if "test" in f and f.endswith(".json")]
    all_data = []
    for filename in filenames:
        with open(os.path.join(ORIGINAL_DATA_DIR, "AIME25", filename), "r") as f:
            for line in f:
                entry = json.loads(line)
                if "problem" in entry:
                    entry["query"] = entry.pop("problem")
                if "answer" in entry:
                    entry["answer"] = entry.pop("answer")
                all_data.append(entry)
    
    if not os.path.exists(os.path.join(TARGET_DATA_DIR, "aime25")):
        os.makedirs(os.path.join(TARGET_DATA_DIR, "aime25"))
        
    dump_data_to_json(all_data, os.path.join(TARGET_DATA_DIR, "aime25", "aime25_test.jsonl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="gsm8k")
    args = parser.parse_args()
    
    if args.dataset_name == "gsm8k":
        gsm8k_processor()
        print(f"Processed {args.dataset_name} dataset")
    elif args.dataset_name == "math":
        math_processor()
        print(f"Processed {args.dataset_name} dataset")
    elif args.dataset_name == "mmlu":
        mmlu_processor()
        print(f"Processed {args.dataset_name} dataset")
    elif args.dataset_name == "mmlu_pro":
        mmlu_pro_processor()
        print(f"Processed {args.dataset_name} dataset")
    elif args.dataset_name == "aime24":
        aime24_processor()
        print(f"Processed {args.dataset_name} dataset")
    elif args.dataset_name == "aime25":
        aime25_processor()
        print(f"Processed {args.dataset_name} dataset")
        
        