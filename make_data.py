from datasets import Dataset
from string_globals import *

INPUT="input"
OUTPUT="output"

big_num=100

def multiplication_dataset(number_type:str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[]
    }
    for a in range(0,big_num):
        for b in range(0,big_num):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} * {b} =")
            answer=a*b
            src_dict[OUTPUT].append(f"{answer}")
    return Dataset.from_dict(src_dict)

def division_dataset(number_type: str):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[]
    }
    for a in range(0,big_num):
        for b in range(1,big_num):
            if number_type==DECIMAL:
                a=float(a)/big_num
                b=float(b)/big_num
            src_dict[INPUT].append(f"{a} / {b} =")
            answer=a/b
            src_dict[OUTPUT].append(f"{answer}")
