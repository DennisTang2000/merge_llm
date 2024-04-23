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
import copy

from model_merging_methods.merging_methods import *
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp

from model_merging_methods.task_vector import *
import time


# GOAL - assumed the MERGED Task Vector has already been created
# We want to load it in, merge it with the pre-trained model at various different coefficient values, and save it with the 
# correct tokenizer

parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument('--merge_option', type=int, default=0)
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

starting = args.start

# LOAD IN PRE-TRAINED MERGE
all_possible_names = ["LM_CODE", "LM_MATH", "CODE_MATH", "LM_CODE_MATH"]
name = all_possible_names[args.merge_option]

if "CODE" in name:
    tokenizer_name = "../../llama-2-13b-code-alpaca"
else:
    tokenizer_name = "../../WizardMath-13B-V1.0"

finetuned_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device_map="cpu") 
merged_task_vector = torch.load("merged_task_vector_" + name + ".pt") # Task Vector object

### LOAD IN PRETRAINED MODEL HERE ###
pretrained_model = AutoModelForCausalLM.from_pretrained("../../Llama-2-13b-hf", device_map="cpu", torch_dtype = torch.float16)
pretrained_tokenizer = AutoTokenizer.from_pretrained("../../Llama-2-13b-hf", device_map="cpu")

smart_tokenizer_and_embedding_resize(
special_tokens_dict=dict(pad_token="[PAD]"),
model=pretrained_model,
tokenizer=pretrained_tokenizer)


def copy_params_to_model(params: dict, model: nn.Module):
        """
        copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])
            
           
params = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 3.0]            
for i in range(len(params)):
    
    deep_copied_model = copy.deepcopy(pretrained_model)
    merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=deep_copied_model, scaling_coefficient=params[i])
    
    copy_params_to_model(params=merged_params, model=deep_copied_model)
    
    deep_copied_model.save_pretrained(save_directory=str(i + starting) + "_model")
    finetuned_tokenizer.save_pretrained(save_directory=str(i + starting) + "_model")
    
    del deep_copied_model
    del merged_params
    
    
    
    

