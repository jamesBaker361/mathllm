from utils import get_run_name, download_datasets, reward_function
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from string_globals import *
from constant_globals import *
from trl.core import respond_to_batch
from training import generation_kwargs, tokenizer
import numpy as np
import wandb
import re

def evaluate_trained_model(
        training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str],
                  prefix:str,
                  model=None):
    columns=["training_type", "avg_mae", "correct_fraction"]
    table_data=[]
    _,test_dataset=download_datasets(task_list, number_type_list,prefix)
    for training_type in training_type_list:
        if model is None:
            run_name=get_run_name(training_type, task_list, number_type_list,prefix)
            print(run_name)
            model=AutoModelForCausalLMWithValueHead.from_pretrained(f"jlbaker361/{run_name}")


        ppo_config = PPOConfig(
            batch_size=1,
        )
        ppo_trainer = PPOTrainer(ppo_config, model, model, tokenizer)
        reward_list=[]
        correct_count=0.0

        for row in test_dataset:
            input =row[INPUT]
            output=row[OUTPUT]
            query_tensor = [tokenizer.encode(q, return_tensors="pt")[0] for q in input]
            response_tensor  = ppo_trainer.generate(query_tensor,**generation_kwargs )
            decoded_response=tokenizer.decode(response_tensor[0])
            decoded_response=re.sub(input, "", decoded_response)
            #print('$',input,tokenizer.decode(response_tensor[0]), '$')
            reward=reward_function(decoded_response, output, THRESHOLD, TEST_PENALTY)
            reward_list.append(reward)
            if abs(reward) <=0.0001:
                correct_count+=1.0

        table_row=[
            training_type, np.mean(reward_list), correct_count/len(reward_list)
        ]
        table_data.append(table_row)
        print(table_row)
    run = wandb.init()
    eval_table=wandb.Table(columns=columns, data=table_data)
    run.log({"eval_table":eval_table})
    wandb.finish()
            

        
if __name__=='__main__':
    model=AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    task_list=[ADDITION]
    training_type_list=[RL]
    number_type_list=[WHOLE]
    prefix=""
    evaluate_trained_model(training_type_list, task_list, number_type_list, prefix, model)