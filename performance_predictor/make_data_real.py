import sys
import os
import json
from utils import *
from process_plans import *
from parse_plan import *
from process_plans import *

import pandas as pd
import numpy as np

TYPE_SIZE = {
    "int": 4,
    "integer": 4,
    "int32": 4,
    "int64": 8,
    "float": 4,
    "double": 8,
    "decimal": 16,
    "date": 4,
    "timestamp": 8,
    "boolean": 1 
}

import re

def decimal_storage_bytes_from_string(type_str: str) -> int:
    match = re.match(r'\s*decimal\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*',
                     type_str, re.IGNORECASE)
    if not match:
        raise ValueError("输入格式错误，应类似 Decimal(15, 2)")

    M = int(match.group(1))
    D = int(match.group(2))

    if M <= 0 or D < 0 or D > M:
        raise ValueError("非法的 DECIMAL(M, D) 定义")

    def bytes_for_digits(digits: int) -> int:
        full_groups = digits // 9
        remainder = digits % 9

        bytes_ = full_groups * 4
        if remainder == 0:
            return bytes_
        elif remainder <= 2:
            return bytes_ + 1
        elif remainder <= 4:
            return bytes_ + 2
        elif remainder <= 6:
            return bytes_ + 3
        else:
            return bytes_ + 4

    int_digits = M - D
    frac_digits = D

    return bytes_for_digits(int_digits) + bytes_for_digits(frac_digits)


def read_col_to_width():
    with open("../predicate_tuning/histogram_data/tpch1g-column-minmax.json", "r") as f:
        data = json.load(f)
    col_to_width = defaultdict()
    for item in data:
        if item["col_type"] == "String" or item["col_type"] == "Nullable(String)":
            name = item["table_name"] + "." + item["col_name"]
            col_to_width[name] = 6 + (item["min"] + item["max"]) // 2
        else:
            type = item["col_type"].lower()
            name = item["table_name"] + "." + item["col_name"]
            if "decimal" in type:
                col_to_width[name] = decimal_storage_bytes_from_string(type)
                continue
            col_to_width[name] = TYPE_SIZE[type]
    return col_to_width

def read_plans(record_file):
    with open(record_file, "r") as f:
        return json.load(f)    
    
def get_col_info(real_plan):
    s = '\n'.join(real_plan)
    root_node = parse_query_plan(s)
    _, _, card_list, cols_list = get_estimation(root_node)
    return card_list, cols_list

def calc_width(cols_list, col_to_width):
    ans = []
    for cols_node in cols_list:
        tmp = []
        for cols in cols_node:
            tmp_width = 0
            for col in cols:
                tmp_width += col_to_width[col]
            tmp.append(tmp_width)
        ans.append(tmp)
    return ans