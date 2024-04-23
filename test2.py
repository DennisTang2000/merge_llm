"""
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

from model_merging_methods.merging_methods import *
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp

from model_merging_methods.task_vector import *
import time
"""
import psutil

import gc

def print_cpu_memory_usage():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f'CPU Usage: {cpu_percent}%')

    # Memory usage
    mem = psutil.virtual_memory()
    mem_total = mem.total / (1024 ** 3)  # Convert bytes to gigabytes
    mem_used = mem.used / (1024 ** 3)
    mem_percent = mem.percent
    print(f'Memory Usage: {mem_used}GB / {mem_total:.2f}GB ({mem_percent}%)')
    
   
    
import torch
    
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        #self.lm = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="cpu")
    
    def area(self):
        return self.length * self.width
    
    
rect = torch.rand(90000, 90000)


def alter(rect):
    rect2 = rect*5
    
    return rect

rect2 = alter(rect)
#del rect

print_cpu_memory_usage() 
#rect[3] = 999999

print(rect2)
print(rect)    
    



