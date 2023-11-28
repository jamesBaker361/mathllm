from transformers import AutoModelForCausalLM
from datasets import Dataset,load_dataset,concatenate_datasets
from string_globals import *
import torch
import re

SEED=1234

def reward_function(decoded_response:str, answer:float, mae:bool=True)->float:
    decoded_response=re.sub(",","", decoded_response) #get rid of commas so 9,000 -> 9000
    nums=re.findall(r'\d+\.\d+|\d+', decoded_response)
    if len(nums)==0:
        return -100000.0
    guess=float(nums[0])
    if mae:
        return -1.0 * abs(guess-answer)
    else:
        return -1.0 * (guess-answer)**2
    
def download_datasets(task_list: list[str], number_type_list:list[str],prefix:str):
    src_dict={
        TEXT:[],
        INPUT:[],
        OUTPUT:[]
    }
    big_train=Dataset.from_dict(src_dict)
    big_test=Dataset.from_dict(src_dict)
    for task in task_list:
        for number_type in number_type_list:
            if len(prefix)>0:
                split_dict=load_dataset(f"jlbaker361/{prefix}_{task}_{number_type}")
            else:
                split_dict=load_dataset(f"jlbaker361/{task}_{number_type}")
            train=split_dict["train"]
            test=split_dict["test"]
            big_train=concatenate_datasets([big_train, train])
            big_test=concatenate_datasets([big_test, test])
    return big_train, big_test

def get_run_name(training_type:str, task_list:list,number_type_list:list,prefix:str )->str:
    task_list.sort()
    number_type_list.sort()
    all_tasks='_'.join(task_list)
    all_number_types='_'.join(number_type_list)
    run_name=f"{training_type}_{all_tasks}_{all_number_types}"
    if len(prefix)>0:
        run_name=f"{prefix}_{run_name}"
    return run_name

def expand_embedding_vocab_size(new_tokens:int, model: AutoModelForCausalLM):
    init_embedding_tensor=model.get_parameter('transformer.wte.weight').detach()
    embedding_dim=init_embedding_tensor.size()[1]
    new_words=torch.randn((new_tokens, embedding_dim))
    new_wte_weight=torch.cat([init_embedding_tensor, new_words ])
    model.transformer.wte.weight=torch.nn.Parameter(new_wte_weight)

    init_head_tensor=model.get_parameter('lm_head.weight').detach() #lm_head.weight
    new_words=torch.randn((new_tokens, embedding_dim))
    new_head_weight=torch.cat([init_head_tensor, new_words ])
    model.lm_head.weight=torch.nn.Parameter(new_head_weight)

    return model