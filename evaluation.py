from utils import get_run_name, download_datasets, reward_function
from transformers import AutoTokenizer, AutoModelForCausalLM
from string_globals import *
from trl.core import respond_to_batch

def evaluate_trained_model(
        training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str],
                  prefix:str,
                  model=None):
    columns=["training_type", "avg_mae", "correct pct"]
    _,test_dataset=download_datasets(task_list, number_type_list,prefix)
    for training_type in training_type_list:
        run_name=get_run_name(training_type, task_list, number_type_list,prefix)
        print(run_name)
        if model is None:
            model=AutoModelForCausalLM.from_pretrained(f"jlbaker361/{run_name}")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        reward_list=[]
        correct_count=0

        for row in test_dataset:
            input =row[INPUT]
            output=row[OUTPUT]
            encoded_input=tokenizer.encode(input, return_tensors="pt",padding="max_length", max_length=64)
            response=respond_to_batch(model, encoded_input, txt_len=64)
            reward=reward_function(tokenizer.decode(response[0]),output)
            reward_list.append(reward)
            if reward < 0.00001:
                correct_count+=1
        