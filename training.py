from peft import LoraConfig, TaskType
import re

def reward_function(decoded_response:str, answer:float, mae:bool=True):
    nums=re.findall(r'\d+\.\d+|\d+', decoded_response)
    if len(nums)==0:
        return -1000
    guess=float(nums[0])
    if mae:
        return abs(guess-answer)
    else:
        return (guess-answer)**2