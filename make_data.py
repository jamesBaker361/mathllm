from datasets import Dataset
from string_globals import *

SPLIT_FRACTION=0.1

def multiplication_dataset(number_type:str,big_num:int=100):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,a):
            if number_type==WHOLE:
                src_dict[INPUT].append(f"{a} * {b} = ")
                answer=a*b
                src_dict[TEXT].append(f"{a} * {b} = {answer}")
            elif number_type==DECIMAL:
                dec_a=float(a)/big_num
                dec_b=float(b)/big_num
                src_dict[INPUT].append(f"{dec_a} * {dec_b} = ")
                answer=dec_a*dec_b
                src_dict[TEXT].append(f"{dec_a} * {dec_b} = {answer}")
            src_dict[OUTPUT].append(float(answer))
    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

def division_dataset(number_type: str,big_num:int=100):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(1,a):
            if number_type==WHOLE:
                src_dict[INPUT].append(f"{a} / {b} = ")
                answer=a/b
                src_dict[TEXT].append(f"{a} / {b} = {answer}")
            elif number_type==DECIMAL:
                dec_a=float(a)/big_num
                dec_b=float(b)/big_num
                src_dict[INPUT].append(f"{dec_a} / {dec_b} = ")
                answer=dec_a/dec_b
                src_dict[TEXT].append(f"{dec_a} / {dec_b} = {answer}")
            src_dict[OUTPUT].append(float(answer))
    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

def addition_dataset(number_type:str,big_num:int=100):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,a):
            if number_type==WHOLE:
                src_dict[INPUT].append(f"{a} + {b} = ")
                answer=a+b
                src_dict[TEXT].append(f"{a} + {b} = {answer}")
            elif number_type==DECIMAL:
                dec_a=float(a)/big_num
                dec_b=float(b)/big_num
                src_dict[INPUT].append(f"{dec_a} + {dec_b} = ")
                answer=dec_a+dec_b
                src_dict[TEXT].append(f"{dec_a} + {dec_b} = {answer}")
            src_dict[OUTPUT].append(float(answer))
    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

def subtraction_dataset(number_type:str,big_num:int=100):
    if number_type not in NUMBER_TYPE_LIST:
        raise Exception("invalid number type")
    src_dict={
        INPUT:[],
        OUTPUT:[],
        TEXT:[]
    }
    for a in range(0,big_num):
        for b in range(0,a):
            if number_type==WHOLE:
                src_dict[INPUT].append(f"{a} - {b} = ")
                answer=a-b
                src_dict[TEXT].append(f"{a} - {b} = {answer}")
            elif number_type==DECIMAL:
                dec_a=float(a)/big_num
                dec_b=float(b)/big_num
                src_dict[INPUT].append(f"{dec_a} - {dec_b} = ")
                answer=dec_a-dec_b
                src_dict[TEXT].append(f"{dec_a} - {dec_b} = {answer}")
            src_dict[OUTPUT].append(float(answer))
    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)

def dummy_datset(number_type:str=None,big_num:int=100):
    src_dict={
        INPUT:["a" for _ in range(10)],
        OUTPUT:[0.1 for _ in range(10)],
        TEXT:["aa bb cc" for _ in range(10)]
    }
    return Dataset.from_dict(src_dict).train_test_split(SPLIT_FRACTION)