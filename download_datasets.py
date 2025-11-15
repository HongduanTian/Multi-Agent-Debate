"This file is used to download necessary datasets for the project."

import os
import sys
import argparse
import shutil
import time
import random
import json

from datasets import load_dataset


DATASET_NAMES = ["gsm8k", "math", "mmlu", "arithmetic", "mmlu_pro", "aime24", "aime25"]

def gsm8k(target_dir:str):
    """
    Download GSM8K dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    dataset_id = "openai/gsm8k"
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=target_dir)
    print(f"DATASET INFO: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")
    
    train_set = dataset["train"]
    test_set = dataset["test"]
    
    target_dataset_dir = os.path.join(target_dir, "GSM8K")

    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    train_set.to_json(os.path.join(target_dataset_dir, "train.json"))
    test_set.to_json(os.path.join(target_dataset_dir, "test.json"))
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"GSM8K dataset downloaded to {target_dataset_dir}")
    
    
def math(target_dir:str):
    """Download Math dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    dataset_id = "EleutherAI/hendrycks_math"
    for subset_name in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
        dataset = load_dataset("EleutherAI/hendrycks_math", subset_name, cache_dir=target_dir)
        print(f"DATASET ({subset_name}) INFO: {len(dataset['train'])} training samples, {len(dataset['test'])} test samples")
        
        train_set = dataset["train"]
        test_set = dataset["test"]
        
        target_dataset_dir = os.path.join(target_dir, "MATH")

        if not os.path.exists(target_dataset_dir):
            os.makedirs(target_dataset_dir)
            
        train_set.to_json(os.path.join(target_dataset_dir, f"{subset_name}_train.json"))
        test_set.to_json(os.path.join(target_dataset_dir, f"{subset_name}_test.json"))
        
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"Math dataset downloaded to {target_dataset_dir}")


def mmlu(target_dir:str):
    """Download MMLU dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    subset_names = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    target_dataset_dir = os.path.join(target_dir, "MMLU")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    failed_subsets = []
    dataset_id = "cais/mmlu"
    for subset_name in subset_names:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Downloading {subset_name} (attempt {retry_count + 1}/{max_retries})...")
                dataset = load_dataset("cais/mmlu", subset_name, cache_dir=target_dir)
                print(f"DATASET ({subset_name}) INFO: {len(dataset['test'])} test samples")
                
                test_set = dataset["test"]
                test_set.to_json(os.path.join(target_dataset_dir, f"{subset_name}_test.json"))
                
                print(f"Successfully downloaded {subset_name}")
                break
                
            except Exception as e:
                retry_count += 1
                print(f"Error downloading {subset_name} (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # exponential backoff
                    wait_time = min(30, 2 ** retry_count + random.uniform(0, 1))
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to download {subset_name} after {max_retries} attempts")
                    failed_subsets.append(subset_name)
    
    # clean up cache
    try:
        tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
        shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    except:
        pass
    
    # clean up lock files
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        try:
            os.remove(os.path.join(target_dir, lockfile))
        except:
            pass
    
    if failed_subsets:
        print(f"Warning: Failed to download the following subsets: {failed_subsets}")
        print("You may need to retry the download later or check your network connection.")
    
    print(f"MMLU dataset downloaded to {target_dataset_dir}")
    print(f"Successfully downloaded {len(subset_names) - len(failed_subsets)}/{len(subset_names)} subsets")


def arithmetic(target_dir:str):
    """Download Arithmetic dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    dataset_id = "EleutherAI/arithmetic"
    subset_names = ['arithmetic_1dc', 'arithmetic_2da', 'arithmetic_2dm', 'arithmetic_2ds', 'arithmetic_3da', 'arithmetic_3ds', 'arithmetic_4da', 'arithmetic_4ds', 'arithmetic_5da', 'arithmetic_5ds']
    
    target_dataset_dir = os.path.join(target_dir, "ARITHMETIC")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    failed_subsets = []
    
    for subset_name in subset_names:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Downloading {subset_name} (attempt {retry_count + 1}/{max_retries})...")
                dataset = load_dataset("EleutherAI/arithmetic", subset_name, cache_dir=target_dir)
                print(f"DATASET ({subset_name}) INFO: {len(dataset['validation'])} validation samples")
                
                validation_set = dataset["validation"]
                validation_set.to_json(os.path.join(target_dataset_dir, f"{subset_name}_test.json"))
                
                print(f"Successfully downloaded {subset_name}")
                break
                
            except Exception as e:
                retry_count += 1
                print(f"Error downloading {subset_name} (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # exponential backoff
                    wait_time = min(30, 2 ** retry_count + random.uniform(0, 1))
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to download {subset_name} after {max_retries} attempts")
                    failed_subsets.append(subset_name)
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        try:
            os.remove(os.path.join(target_dir, lockfile))
        except:
            pass
    
    if failed_subsets:
        print(f"Warning: Failed to download the following subsets: {failed_subsets}")
        print("You may need to retry the download later or check your network connection.")
    
    print(f"Arithmetic dataset downloaded to {target_dataset_dir}")
    print(f"Successfully downloaded {len(subset_names) - len(failed_subsets)}/{len(subset_names)} subsets")
    
def mmlu_pro(target_dir:str):
    """Download MMLU-Pro dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    dataset_id = "TIGER-Lab/MMLU-Pro"
    target_dataset_dir = os.path.join(target_dir, "MMLU-PRO")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default", cache_dir=target_dir)
    test_set = dataset["test"]
    print(f"DATASET INFO: {len(test_set)} test samples")
    test_set.to_json(os.path.join(target_dataset_dir, "mmlu_pro_test.json"))
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"MMLU-Pro dataset downloaded to {target_dataset_dir}")
    

def aime24(target_dir:str):
    """Download AIME24 dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    target_dataset_dir = os.path.join(target_dir, "AIME24")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    dataset_id = "HuggingFaceH4/aime_2024"
    dataset = load_dataset("HuggingFaceH4/aime_2024", "default", cache_dir=target_dir)
    train_set = dataset["train"]
    print(f"DATASET INFO: {len(train_set)} train samples")
    train_set.to_json(os.path.join(target_dataset_dir, "aime24_test.json"))
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"AIME24 dataset downloaded to {target_dataset_dir}")
    

def aime25(target_dir:str):
    """Download AIME25 dataset to the target directory.
    
    Args:
        target_dir: str, the directory to save the dataset.
        
    Returns:
        None
    """
    target_dataset_dir = os.path.join(target_dir, "AIME25")
    if not os.path.exists(target_dataset_dir):
        os.makedirs(target_dataset_dir)
    
    dataset_id = "yentinglin/aime_2025"
    dataset = load_dataset("yentinglin/aime_2025", "default", cache_dir=target_dir)
    print(f"DATASET INFO: {len(dataset['train'])} test samples")
    dataset["train"].to_json(os.path.join(target_dataset_dir, "aime25_test.json"))
    
    tmp_dir_name = f"{dataset_id.split('/')[0]}___{dataset_id.split('/')[1].lower()}"
    shutil.rmtree(os.path.join(target_dir, tmp_dir_name))
    
    lockfiles = [file for file in os.listdir(target_dir) if file.endswith(".lock")]
    for lockfile in lockfiles:
        os.remove(os.path.join(target_dir, lockfile))
    
    print(f"AIME25 dataset downloaded to {target_dataset_dir}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="gsm8k")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    args = parser.parse_args()
    
    assert args.dataset_name in DATASET_NAMES, f"Invalid dataset name: {args.dataset_name}"
    
    dataset_target_dir = args.dataset_dir
    
    if not os.path.exists(dataset_target_dir):
        os.makedirs(dataset_target_dir)
    
    if args.dataset_name == "gsm8k":
        if not os.path.exists(os.path.join(dataset_target_dir, "GSM8K")):
            print(f"Downloading GSM8K dataset to {dataset_target_dir}...")
            gsm8k(dataset_target_dir)
        else:
            print(f"GSM8K dataset already exists in {dataset_target_dir}")
    
    elif args.dataset_name == "math":
        if not os.path.exists(os.path.join(dataset_target_dir, "MATH")):
            print(f"Downloading Math dataset to {dataset_target_dir}...")
            math(dataset_target_dir)
        else:
            print(f"Math dataset already exists in {dataset_target_dir}")
    
    elif args.dataset_name == "mmlu":
        if not os.path.exists(os.path.join(dataset_target_dir, "MMLU")):
            print(f"Downloading MMLU dataset to {dataset_target_dir}...")
            mmlu(dataset_target_dir)
        else:
            print(f"MMLU dataset already exists in {dataset_target_dir}")
    
    elif args.dataset_name == "arithmetic":
        if not os.path.exists(os.path.join(dataset_target_dir, "ARITHMETIC")):
            print(f"Downloading Arithmetic dataset to {dataset_target_dir}...")
            arithmetic(dataset_target_dir)
        else:
            print(f"Arithmetic dataset already exists in {dataset_target_dir}")
    
    elif args.dataset_name == "mmlu_pro":
        if not os.path.exists(os.path.join(dataset_target_dir, "MMLU-PRO")):
            print(f"Downloading MMLU-Pro dataset to {dataset_target_dir}...")
            mmlu_pro(dataset_target_dir)
        else:
            print(f"MMLU-Pro dataset already exists in {dataset_target_dir}")
        
    elif args.dataset_name == "aime24":
        if not os.path.exists(os.path.join(dataset_target_dir, "AIME24")):
            print(f"Downloading AIME24 dataset to {dataset_target_dir}...")
            aime24(dataset_target_dir)
        else:
            print(f"AIME24 dataset already exists in {dataset_target_dir}")
            
    elif args.dataset_name == "aime25":
        if not os.path.exists(os.path.join(dataset_target_dir, "AIME25")):
            print(f"Downloading AIME25 dataset to {dataset_target_dir}...")
            aime25(dataset_target_dir)
        else:
            print(f"AIME25 dataset already exists in {dataset_target_dir}")