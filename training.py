from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments
from trl import SFTTrainer
from string_globals import *
from datasets import Dataset
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
import numpy as np
import wandb
import os
from utils import download_datasets, reward_function, get_run_name

os.environ["WANDB_API_KEY"]="004792fd620af032a735920a6cd036486b182519"
os.environ["WANDB_NOTEBOOK_NAME"]="math-notebook"
os.environ["WANDB_PROJECT"]="math-llm"

def training_loop(epochs:int,
                  training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str],
                  prefix:str):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    for training_type in training_type_list:
        run_name=get_run_name(training_type, task_list, number_type_list,prefix)
        print(run_name)
        model=AutoModelForCausalLM.from_pretrained('gpt2')
        model = get_peft_model(model, peft_config)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if training_type==FT:
            ft_epochs=epochs
            rl_epochs=0
        elif training_type == RL:
            ft_epochs=0
            rl_epochs=epochs
        elif training_type == MIXED:
            ft_epochs=epochs//2
            rl_epochs=epochs//2
        os.environ["WANDB_PROJECT"]=run_name
        batch_size=8
        run = wandb.init(
            # Set the project where this run will be logged
            project=run_name,
            # Track hyperparameters and run metadata
            config={
                "rl_epochs": rl_epochs,
                "ft_epochs":ft_epochs,
                "rl_batch_size":batch_size
            })

        args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=ft_epochs,
            # other args and kwargs here
            run_name=run_name,
            report_to="wandb",  # enable logging to W&B
            logging_steps=64  # how often to log to W&B
        )

        train_dataset,_=download_datasets(task_list, number_type_list,prefix)

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=128,
            args=args
        )
        if ft_epochs>0:
            trainer.train()
            print("fine tune training complete!")
        
        ppo_config = PPOConfig(
            batch_size=batch_size,
        )
        model_2=AutoModelForCausalLMWithValueHead.from_pretrained(get_peft_model(trainer.model, peft_config))
        model_ref = create_reference_model(model_2)
        ppo_trainer = PPOTrainer(ppo_config, model_2, model_ref, tokenizer)

        batched_dataset=[t for t in train_dataset]
        batched_dataset=batched_dataset[:len(batched_dataset)-  (len(batched_dataset) %batch_size)]
        batched_dataset=np.reshape(batched_dataset, (len(batched_dataset)//batch_size, batch_size))
        for e in range(ft_epochs,ft_epochs+ rl_epochs):
            mean_scores=[]
            for batch in batched_dataset:
                batch={key: [i[key] for i in batch] for key in batch[0]}
                query_tensor = torch.cat([tokenizer.encode(q, return_tensors="pt",padding="max_length", max_length=64) for q in batch[INPUT]])
                try:
                    response_tensor  = respond_to_batch(model_2, query_tensor)
                except IndexError as index_error:
                    print("index error! batch input:\n")
                    for i,o in zip(batch[INPUT], batch[OUTPUT]):
                        print(i,o)
                        exit()
                reward = [torch.tensor(reward_function(tokenizer.decode(response),answer)) for response, answer in zip(response_tensor, batch[OUTPUT])]
                train_stats = ppo_trainer.step([t for t in query_tensor], [t for t in response_tensor], reward)
                mean_scores.append(train_stats['ppo/mean_scores'])
            wandb.log({"ppo/mean_scores":np.mean(mean_scores)})

        wandb.finish()
        print("rl training complete")
        model_2.push_to_hub(run_name)

        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--training_type_list", nargs = '*', help="training types", default=TRAINING_TYPE_LIST)
parser.add_argument("--task_list", nargs = '*', help="task types", default=TASK_LIST)
parser.add_argument("--number_type_list", nargs = '*', help="number types", default=NUMBER_TYPE_LIST)
parser.add_argument("--epochs", type=int, help="total epochs to train for")
parser.add_argument("--prefix", type=str,default="")

args = parser.parse_args()
if __name__=='__main__':
    training_loop(
        args.epochs,
        args.training_type_list,
        args.task_list,
        args.number_type_list,
        args.prefix
    )
    print("done :)")
