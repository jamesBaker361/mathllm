from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments
from trl import SFTTrainer
from string_globals import *
from constant_globals import *
from datasets import Dataset
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import re
import numpy as np
import wandb
import os
from utils import download_datasets, reward_function, get_run_name, expand_embedding_vocab_size
import time
#pdb.set_trace()

os.environ["WANDB_API_KEY"]="004792fd620af032a735920a6cd036486b182519"
os.environ["WANDB_NOTEBOOK_NAME"]="math-notebook"
os.environ["WANDB_PROJECT"]="math-llm"
np.random.seed(1234)


tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token=tokenizer.eos_token
batch_size=8
top_k=4
top_p=1.0
temperature=1.5

generation_kwargs = { #here? https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    "min_length": -1,
    "top_k": top_k,
    "top_p": top_p,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_length": 30,
    "eos_token_id": -1,
    "temperature":temperature
}

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

def training_loop(epochs:int,
                  training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str],
                  prefix:str,
                  threshold:float, 
                  penalty:float,
                  kl_penalty:str,
                  init_kl_coef:float):
    ppo_config = PPOConfig(
        batch_size=batch_size,
        init_kl_coef=init_kl_coef,
        kl_penalty=kl_penalty,
        ratio_threshold=1000000.0
    )
    for training_type in training_type_list:
        run_name=get_run_name(training_type, task_list, number_type_list,prefix)
        print(run_name)
        model=AutoModelForCausalLM.from_pretrained('gpt2')
        #model=expand_embedding_vocab_size(1,model)
        model = get_peft_model(model, peft_config)

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


        run = wandb.init(
            # Set the project where this run will be logged
            project=run_name,
            # Track hyperparameters and run metadata
            config={
                "rl_epochs": rl_epochs,
                "ft_epochs":ft_epochs,
                "rl_batch_size":batch_size,
                "top_k":top_k,
                "top_p":top_p,
                "temperature": temperature,
                "threshold": threshold,
                "penalty": penalty,
                "kl_penalty":kl_penalty
            })

        args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=ft_epochs,
            # other args and kwargs here
            run_name=run_name,
            report_to="wandb",  # enable logging to W&B
            logging_steps=1000  # how often to log to W&B
        )

        train_dataset,_=download_datasets(task_list, number_type_list,prefix)

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=30,
            args=args
        )
        if ft_epochs>0:
            start=time.time()
            trainer.train()
            print("fine tune training complete! time elapsed: ",time.time()-start)
        
        model_2=AutoModelForCausalLMWithValueHead.from_pretrained(get_peft_model(trainer.model, peft_config))
        model_ref = create_reference_model(model_2)
        ppo_trainer = PPOTrainer(ppo_config, model_2, model_ref, tokenizer)

        batched_dataset=[t for t in train_dataset]
        batched_dataset=batched_dataset[:len(batched_dataset)-  (len(batched_dataset) %batch_size)]
        batched_dataset=np.reshape(batched_dataset, (len(batched_dataset)//batch_size, batch_size))
        start=time.time()
        for e in range(ft_epochs,ft_epochs+ rl_epochs):
            mean_scores=[]
            for i,batch in enumerate(batched_dataset):
                batch={key: [i[key] for i in batch] for key in batch[0]}
                query_tensor = [tokenizer.encode(q, return_tensors="pt")[0] for q in batch[INPUT]]
                try:
                    response_tensor  = ppo_trainer.generate(query_tensor,**generation_kwargs )
                except RuntimeError as exc:
                    print(batch[INPUT])
                    print(query_tensor)
                    print(f"runtime error at epoch {e} batch {i}")
                    raise exc
                #print([(input, tokenizer.decode(response)) for input,response in zip(batch[INPUT],response_tensor)])
                decoded_response_list=[]
                for response,query in zip(response_tensor, batch[INPUT]):
                    decoded_response=tokenizer.decode(response)
                    decoded_response=re.sub(query, "", decoded_response)
                    decoded_response_list.append(decoded_response)
                
                reward=[torch.tensor(reward_function(decoded_response,answer,threshold, penalty)) for decoded_response, answer in zip(decoded_response_list, batch[OUTPUT])]
                #print("reward", reward)
                train_stats = ppo_trainer.step([t for t in query_tensor], [t for t in response_tensor], reward)
                mean_scores.append(train_stats['ppo/mean_scores'])
            wandb.log({"ppo/mean_scores":np.mean(mean_scores), "epoch":e})
            print(f"ended epoch {e} with mean score {np.mean(mean_scores)}")

        wandb.finish()
        print("rl training complete! time elapsed: ", time.time()-start)
        model_2.push_to_hub(f"jlbaker361/{run_name}")

        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--training_type_list", nargs = '*', help="training types", default=TRAINING_TYPE_LIST)
parser.add_argument("--task_list", nargs = '*', help="task types", default=TASK_LIST)
parser.add_argument("--number_type_list", nargs = '*', help="number types", default=NUMBER_TYPE_LIST)
parser.add_argument("--epochs", type=int, help="total epochs to train for")
parser.add_argument("--prefix", type=str,default="")
parser.add_argument("--threshold",type=float, default=THRESHOLD, help="threshold for equality")
parser.add_argument("--penalty", type=float, default=PENALTY, help="minimum penalty")
parser.add_argument("--kl_penalty", type=str,default="full", help="kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution")
parser.add_argument('--init_kl_coef', type=float, default=0.0,help="beta on kl cost for loss")

args = parser.parse_args()
if __name__=='__main__':
    print(args)
    training_loop(
        args.epochs,
        args.training_type_list,
        args.task_list,
        args.number_type_list,
        args.prefix,
        args.threshold,
        args.penalty,
        args.kl_penalty,
        args.init_kl_coef
    )
    print("done :)")