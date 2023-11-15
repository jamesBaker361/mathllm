from peft import LoraConfig, TaskType, get_peft_model
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM,TrainingArguments
from trl import SFTTrainer
from string_globals import *
from datasets import Dataset
import time
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
import wandb
import os
from utils import download_datasets, reward_function

os.environ["WANDB_API_KEY"]="004792fd620af032a735920a6cd036486b182519"
os.environ["WANDB_NOTEBOOK_NAME"]="math-notebook"
os.environ["WANDB_PROJECT"]="math-llm"

def training_loop(epochs:int,
                  training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str]):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    for training_type in training_type_list:
        task='+'.join(task_list)
        number_type='+'.join(number_type)
        run_name=f"{training_type}_{task}_{number_type}"
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

        args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=ft_epochs,
            # other args and kwargs here
            run_name=run_name,
            report_to="wandb",  # enable logging to W&B
            logging_steps=1  # how often to log to W&B
        )

        train_dataset,_=download_datasets(task_list, number_type_list)

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=128,
            args=args
        )

        trainer.train()



