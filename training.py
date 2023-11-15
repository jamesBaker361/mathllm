from peft import LoraConfig, TaskType, get_peft_model
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2Model,AutoTokenizer, AutoConfig, AutoModelForCausalLM,TrainingArguments
from trl import SFTTrainer
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

def training_loop(training_type_list:list[str],
                  task_list:list[str],
                  number_type_list:list[str]):
    return

