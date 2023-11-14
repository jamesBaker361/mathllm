from datasets import Dataset
from string_globals import *

INPUT="input"
OUTPUT="output"
TEXT="text"

big_num=100

def multiplication_dataset(number_type:str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,big_num):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} * {b} = ")
            answer=a*b
            src_dict[TEXT].append(f"{a} * {b} = {answer}")
            src_dict[OUTPUT].append(answer)
    return Dataset.from_dict(src_dict)

def division_dataset(number_type: str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(1,big_num):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} / {b} = ")
            answer=a/b
            src_dict[TEXT].append(f"{a} / {b} = {answer}")
            src_dict[OUTPUT].append(answer)

def addition_dataset(number_type:str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,big_num):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} + {b} = ")
            answer=a+b
            src_dict[TEXT].append(f"{a} + {b} = {answer}")
            src_dict[OUTPUT].append(answer)
    return Dataset.from_dict(src_dict)

def subtraction_dataset(number_type:str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,a):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} - {b} = ")
            answer=a-b
            src_dict[TEXT].append(f"{a} - {b} = {answer}")
            src_dict[OUTPUT].append(answer)
    return Dataset.from_dict(src_dict)

def dummy_datset(number_type:str=None):
    src_dict={
        INPUT:["a","b","c"],
        OUTPUT:["e","f","g"],
        TEXT:["ae","bf","cg"]
    }
    return Dataset.from_dict(src_dict)

if __name__ == "__main__":
    for number_type in NUMBER_TYPE_LIST:
        mult_dataset_name=f"{MULTIPLICATION}_{number_type}"
        multiplication_dataset(number_type).push_to_hub(f"jlbaker361/{mult_dataset_name}")
        divi_dataset_name=f"{DIVISION}_{number_type}"
        division_dataset(number_type).push_to_hub(f"jlbaker361/{divi_dataset_name}")
        subt_dataset_name=f"{SUBTRACTION}_{number_type}"
        subtraction_dataset(number_type).push_to_hub(f"jlbaker361/{subt_dataset_name}")
        addi_dataset_name=f"{ADDITION}_{number_type}"
        addition_dataset(number_type).push_to_hub(f"jlbaker361/{addi_dataset_name}")
        dum_dataset_name=f"dumb_{number_type}"
        dummy_datset(number_type).push_to_hub(f"jlbaker361/{dum_dataset_name}")