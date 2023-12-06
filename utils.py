from transformers import AutoModelForCausalLM
from datasets import Dataset,load_dataset,concatenate_datasets
from string_globals import *
import torch
import re

SEED=1234
PENALTY=-10.0

def reward_function(decoded_response:str, answer:float, threshold:float, penalty:float, mae:bool=True)->float:
    '''The `reward_function` takes in a decoded response, an answer, a threshold, a penalty, and a flag
    indicating whether to use mean absolute error (MAE) and returns a reward value based on the
    difference between the guess and the answer.
    
    Parameters
    ----------
    decoded_response : str
        The decoded response is a string that represents the output of a model or system. It is assumed to
    contain a numerical value that is being guessed or predicted.
    answer : float
        The `answer` parameter represents the correct answer or target value that the model is trying to
    predict.
    threshold : float
        The threshold is the maximum allowable difference between the decoded response and the answer. If
    the difference is within this threshold, the reward will be maximized. If the difference exceeds the
    threshold, a penalty will be applied.
    penalty : float
        The penalty parameter is a float value that represents the penalty to be applied when the guess is
    outside the threshold.
    mae : bool, optional
        The parameter `mae` stands for Mean Absolute Error. It is a boolean flag that determines whether
    the reward function should use Mean Absolute Error (MAE) or not. If `mae` is set to `True`, the
    reward function will use MAE to calculate the difference between the guess
    
    Returns
    -------
        a float value, which represents the reward.
    
    '''
    decoded_response=re.sub(",","", decoded_response) #get rid of commas so 9,000 -> 9000
    nums=re.findall(r'\d+\.\d+|\d+', decoded_response)
    if len(nums)==0:
        return PENALTY
    guess=float(nums[0])
    difference=abs(guess-answer)
    if difference<=threshold:
        difference=0
    return max(1.0-difference, penalty)
    
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