from utils import *
from process_plans import *
from parse_plan import *
from process_plans import *

import pandas as pd
import numpy as np
from make_data_real import *
from hash_join.hash_join_predictor import *
from eval.eval_predictor import *
from sort.sort_predictor import *

col_to_width = read_col_to_width()

def get_col_info(real_plan):
    s = '\n'.join(real_plan)
    root_node = parse_query_plan(s)
    results = parse_operator_info(root_node)
    return results
def cal_cpu(logic_plan, agg_metrics):
    res = get_col_info(logic_plan)
    cols_list = res["join_cols"]
    width_list = calc_width(cols_list, col_to_width)
    idx = 0
    join_node_info = []
    for node in agg_metrics:
        if "HashJoin" in node["name"]:
            involve_rows = node["involved_rows"]

            t = [
                    involve_rows[1], 
                    involve_rows[2], 
                    involve_rows[1] / (involve_rows[2] + 1),
                    np.log1p(involve_rows[1]),
                ]
            
            width = width_list[idx]
            idx += 1
            t.append(width[1] * involve_rows[1])
            t.append(width[0] * involve_rows[0])

            join_node_info.append(t)
            
    cpu_hash_join = hash_join_predict(join_node_info)
    cpu_eval = eval_predict(res, agg_metrics)
    output_width = width_list[0][0]
    cpu_sort = sort_predict(output_width, res, agg_metrics)
    
    return (cpu_hash_join + cpu_eval + cpu_sort) / 1000

def cal_cpu_from_plan(logic_plan):
    # get all information from logic plan
    res = get_col_info(logic_plan)
    cols_list = res["join_cols"]
    join_cost = res["join_cost"]
    width_list = calc_width(cols_list, col_to_width)
    join_node_info = []
    for i, costs in enumerate(join_cost):
        for j in range(len(costs)):
            costs[j] = int(float(costs[j]))
        t = [
                # costs[0], 
                costs[1], 
                costs[2], 
                costs[1] / (costs[2] + 1),
                np.log1p(costs[1]),
            ]
        
        width = width_list[i]
        t.append(width[1] * costs[1])
        t.append(width[0] * costs[0])

        join_node_info.append(t)
    if not join_node_info:
        return 0
    
    cpu_hash_join = hash_join_predict(join_node_info)
    
    return cpu_hash_join / 1000