from peft import LoraConfig, TaskType
from datasets import Dataset,load_dataset,concatenate_datasets
from string_globals import *
import re

SEED=1234

def reward_function(decoded_response:str, answer:float, mae:bool=True)->float:
    nums=re.findall(r'\d+\.\d+|\d+', decoded_response)
    if len(nums)==0:
        return -1000
    guess=float(nums[0])
    if mae:
        return abs(guess-answer)
    else:
        return (guess-answer)**2
    
def download_datasets(task_list, number_type_list):
    src_dict={
        TEXT:[],
        INPUT:[],
        OUTPUT:[]
    }
    big_train=Dataset.from_dict(src_dict)
    big_test=Dataset.from_dict(src_dict)
    for task in task_list:
        for number_type in number_type_list:
            hf_dataset=load_dataset(f"jlbaker361/{task}_{number_type}")
            split_dict=hf_dataset.train_test_split(.10,seed=SEED)
            train=split_dict["train"]
            test=split_dict["test"]
            big_train=concatenate_datasets([big_train, train])
            big_test=concatenate_datasets([big_test, test])
    return big_train, big_test
