import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import accelerate

from model_merging_methods.merging_methods import *
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp


import psutil

def print_cpu_memory_usage():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f'CPU Usage: {cpu_percent}%')

    # Memory usage
    mem = psutil.virtual_memory()
    mem_total = mem.total / (1024 ** 3)  # Convert bytes to gigabytes
    mem_used = mem.used / (1024 ** 3)
    mem_percent = mem.percent
    print(f'Memory Usage: {mem_used:.2f}GB / {mem_total:.2f}GB ({mem_percent}%)')


merging_method = MergingMethod(merging_method_name="ties_merging")
finetuned_model_names = ["../../WizardMath-13B-V1.0", "../../WizardLM-13B-V1.2"]


models_to_merge = []
finetuned_tokenizers = []
for finetuned_model_name in finetuned_model_names:
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, device_map="cpu", torch_dtype = torch.float16)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name, device_map="cpu")
    models_to_merge.append(finetuned_model)
    finetuned_tokenizers.append(finetuned_tokenizer)
print_cpu_memory_usage()   
print("one")


pretrained_model = AutoModelForCausalLM.from_pretrained("../../Llama-2-13b-hf", device_map="cpu", torch_dtype = torch.float16)
pretrained_tokenizer = AutoTokenizer.from_pretrained("../../Llama-2-13b-hf", device_map="cpu")
print_cpu_memory_usage()
print("two")

smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=pretrained_model,
            tokenizer=pretrained_tokenizer,
        )
for finetuned_model, finetuned_tokenizer in zip(models_to_merge, finetuned_tokenizers):
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
    )
    
print("three")
merged_model = merging_method.get_merged_model(merged_model=pretrained_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[])


merged_model.save_pretrained(save_directory="test_ties")
print_cpu_memory_usage()
print("three")

