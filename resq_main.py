from general_agent import *
import sys
import os
import yaml
import json
import time
from process_plans import *
from utils import *
from statistic_retrieve.statistic_prun import *
from predicate_tuning.tuning_function import *

GET_ALL_NAME = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '{}'
    ORDER BY table_name, ordinal_position
"""

def formulate_col_input(column_names, targets):
    schema_json = json.dumps(column_names, indent=2)
    targets = json.dumps(targets, indent=2)
    
    template = f"""
    Below is the database schema you must strictly follow.
    You must generate a single SQL query that meets my description and only uses the listed tables and columns.

    ### Database schema
    {schema_json}

    ### Query requirement
    {targets}
    """
    return template.strip()
def gen_template(column_names, database, targets):
    key_path = "./configs/qwen_keys.txt"
    key = get_key(key_path)
    
    col_gen_prompt = get_template_agent_prompt()
    
    # gen_agent
    gen_agent = ChatAgent(
        provider="qwen",
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        enable_thinking=False,
        system_prompt=col_gen_prompt
    )
    
    database = config["database_choose"]
    
    database_choosing = get_database_scale(database)
    gen_agent.add_system_message(database_choosing)
    
    table_schema = get_table_schema(database)
    gen_agent.add_system_message(table_schema)
    
    col_input = formulate_col_input(column_names, targets)
    template_sql = gen_agent.chat(col_input)
    
    token_usage = gen_agent.token_tracker.summary()
    
    return template_sql, token_usage

def get_agg_input(temp_sql):
    res = f"""
    Below is the SQL you need to rewrite:
    {temp_sql}
    """
    
    return res

def gen_agg_sql(temp_sql):
    key_path = "./configs/qwen_keys.txt"
    key = get_key(key_path)
    
    agg_gen_prompt = get_agg_agent_prompt()
    
    # gen_agent
    gen_agent = ChatAgent(
        provider="qwen",
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-flash",
        enable_thinking=False,
        system_prompt=agg_gen_prompt
    )
    
    rewrite_input = get_agg_input(temp_sql)
    rewrite_sql = gen_agent.chat(rewrite_input)
    
    token_usage = gen_agent.token_tracker.summary()
    
    return rewrite_sql, token_usage

def save_res(config, 
             idx, 
             out_sql, 
             query_plan,
             total_cputime,
             total_scan,
             duration,
             turn,
             token_use,
             latency
             ):
    model_name = config["model_name"]
    workload_name = config["workload_name"]
    output_path = f"./outputs/{model_name}-{workload_name}-sql-metrics.json"
    if duration == -1:
        res = {
            "query": out_sql,
            "index": idx,
            "turn": turn,
            "is_valid": 0
        }
    else:
        real_query_plan = query_plan
        
        plan = '\n'.join(real_query_plan)
        
        ops = parse_plan(plan)
        res = {
            "query": out_sql,
            "real_query_plan": real_query_plan,
            "avg_cpu_time": total_cputime,
            "avg_scan_bytes": total_scan,
            "filter": ops["filter"],
            "join": ops["join"],
            "agg": ops["agg"],
            "sort": ops["sort"],
            "index": idx,
            "token_use": token_use,
            "latency": latency,
            "turn": turn,
            "is_valid": 1
        }
    if not os.path.exists(
            os.path.join(
                output_path
            )
        ):
            with open(
                os.path.join(
                    output_path
                ),
                "w",
            ) as f:
                json.dump([res], f, indent=4)
    else:
        with open(
                    output_path,
                    "r+",
                ) as f:
                    data = json.load(f)
                    data.append(res)
                    # clear the file
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=4)
                    
def save_his(history, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def update_his(history, info, tar_cpu, tar_scan):
    query_hash = info["query_hash"]
    
    # if query_hash not in history:
    history[query_hash].append(info)
    
    return history
                    
def date_agg_func(col_name):
    return f"DATE_TRUNC('month', {col_name}) AS {col_name}"

def string_agg_func(col_name):
    # LENGTH(str) low
    # UPPER(str)
    # return f"SUBSTRING(UPPER({col_name}), 1, 3)"
    return f"MD5({col_name}) AS {col_name}"

def num_agg_func(col_name):
    # return f"{col_name} / (SUM({col_name}) OVER() + 1)"
    return f"sin({col_name}) AS {col_name}"
                
def add_cpu(columns, cpu_diff, card, name_to_type):
    cpu_diff = cpu_diff * 1000
    num_weight = num_predict_from_scalar(1, card)
    date_weight = date_predict_from_scalar(1, card)
    str_weight = str_predict_from_scalar(1, card)
    
    tar_width = cpu_diff * 2
    
    num_cols, date_cols, str_cols = [], [], []
    
    for table_name in columns.keys():
            col_list = columns[table_name]
            for col in col_list:
                dtype = name_to_type[col].lower()
                if any(t in dtype for t in ["int", "decimal", "float", "double"]):
                    num_cols.append(col)
                elif dtype == "date":
                    date_cols.append(col)
                elif dtype == "string":
                    str_cols.append(col)
    res = []
    total_weight = 0
    for col in str_cols:
        if total_weight + str_weight > tar_width:
            break
        res.append(string_agg_func(col))
        total_weight += str_weight
    for col in date_cols:
        if total_weight + date_weight > tar_width:
            break
        res.append(date_agg_func(col))
        total_weight += date_weight
    for col in num_cols:
        if total_weight + num_weight > tar_width:
            break
        res.append(num_agg_func(col))
        total_weight += num_weight
        
    if not res:
        if date_cols:
            res.append(date_agg_func(date_cols[0]))
        elif str_cols:
            res.append(string_agg_func(str_cols[0]))
        elif num_cols:
            res.append(num_agg_func(num_cols[0]))
    
    return res

def get_card(agg_sql, config, database):
    res = execute_sys_query(config, agg_sql, database)
    
    return res[0][0]

def read_meta(database):
    path = f"./schema/{database}/{database}_distinct_samples.json"
    with open(path, "r") as f:
        meta = json.load(f)
        
    return meta

def read_his(database, dataset_name, model_name):
    from pathlib import Path
    path = Path(f"./history/{database}/{dataset_name}_{model_name}_history.json")
    if not path.exists():
        return defaultdict(lambda: list())
    with open(path, "r") as f:
        his = json.load(f)
        
    return defaultdict(list, his)
    
def add_operation_to_sql(config, sql, cols):
    key_path = "./configs/qwen_keys.txt"
    key = get_key(key_path)
    
    op_gen_prompt = get_operation_agent_prompt()
    gen_agent = ChatAgent(
        provider="qwen",
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        enable_thinking=False,
        system_prompt=op_gen_prompt
    )
    
    database = config["database_choose"]
    
    table_schema = get_table_schema(database)
    gen_agent.add_system_message(table_schema)
    
    user_input = formulate_add_operation_input(sql, cols)
    sql = gen_agent.chat(user_input)
    
    token_usage = gen_agent.token_tracker.summary()
    
    return sql, token_usage
    
def gen_predicate(temp_sql, cols):
    key_path = "./configs/qwen_keys.txt"
    key = get_key(key_path)
    
    predicate_gen_prompt = get_predicate_agent_prompt()
    
    # gen_agent
    gen_agent = ChatAgent(
        provider="qwen",
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        enable_thinking=False,
        system_prompt=predicate_gen_prompt
    )
    
    database = config["database_choose"]
    
    table_schema = get_table_schema(database)
    gen_agent.add_system_message(table_schema)
        
def gen_predicate_loc(temp_sql, cols):
    temp_sql = temp_sql.split(";")[0]
    all_pre = []
    for col in cols:
        pre = col + "<= {}"
        all_pre.append(pre)
    predicates = " and ".join(all_pre)
    return temp_sql + "\nWHERE " + predicates

def choose_predicate(column, name_to_type, meta_info, predicate_num=3):
    col_chosen = []
    cols = []
    for table_name, col_names in column.items():
        col_list = []
        for col_name in col_names:
            if name_to_type[col_name] == 'String' or name_to_type[col_name] == 'varchar':
                continue
            tmp = table_name + "." + col_name
            info = meta_info[tmp]
            cols.append([info["all_cnt"], info["all_cnt"] / info["distinct_cnt"], tmp])
    cols.sort(key=lambda x: (-x[0], x[1]))
    predicate_num = min(predicate_num, len(cols))
    if not cols:
        return []
    idx = 0
    while True:
        col_chosen.append(cols[idx][2])
        idx += 1
        if idx == predicate_num:
           return col_chosen
    
def main(config):
    query_set = config["query"]
    database_list = config["db"]
    k = config["turns"]
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    sql_metrics = read_workload(query_set, dataset_name)
    
    meta_info = read_meta(database)
    
    query = GET_ALL_NAME.format(database)
    columns = execute_sys_query(config, query, database)
    name_to_type = defaultdict()
    column_to_table = defaultdict()
    for table_name, col_name, col_type in columns:
        name_to_type[col_name] = col_type
        column_to_table[col_name] = table_name
        
    history = read_his(database, dataset_name, model_name)
    his_path = f"./history/{database}/{dataset_name}_{model_name}_history.json"
    
    for idx, metrics in enumerate(sql_metrics):
        use_history_template = False
        main_key, sub_key = '', ''
        if "bendset" in dataset_name:
            main_key = metrics["query_parameterized_hash"]
            sub_key = metrics["query_hash"]
        elif "redset" in dataset_name:
            main_key = metrics["feature_fingerprint"]
            
        no_need_to_run = False
        if main_key in history:
            token_use = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            infos = history[main_key]
            mx_cpu = 0
            mx_idx = -1
            for j, info in enumerate(infos):
                real_query_plan = info["query_plan"]
                ans = info["query"]
                total_cputime = info["cputime"]
                total_scan = info["scanbytes"]
                duration = info["duration"]
                
                if total_cputime > mx_cpu:
                    mx_cpu = total_cputime
                    mx_idx = j
                mx_cpu = max(mx_cpu, total_cputime)
                
                if abs(total_cputime - metrics["cputime"]) / metrics["cputime"] < 0.2 or abs(total_cputime - metrics["cputime"]) <= 0.1:
                    save_res(config, idx, ans, real_query_plan, total_cputime, total_scan, duration, 0, token_use, 0)
                    no_need_to_run = True
                    break

            if mx_cpu > metrics["cputime"] and "col_chosen" in infos[mx_idx]:
                col_chosen = infos[mx_idx]["col_chosen"]
                temp_sql = infos[mx_idx]["template"]
                agg_sql = infos[mx_idx]["agg_sql"]
                base_cpu = infos[mx_idx]["base_cpu"]
                use_history_template = True
        
        if no_need_to_run:
            continue
        cpu_time = metrics["cputime"]
        scan_bytes = metrics["scanbytes"] * (1024 ** 3)
        join, agg, sort = metrics["hash_join_num"], metrics["agg_num"], metrics["sort_num"]

        database = choose_database(scan_bytes, join, database_list)

        targets = {
            "Join": f"{join}",
            "Agg": f"{agg}",
            "Sort": f"{sort}"
        }
        ops = {
            "join": join,
            "agg": agg,
            "sort": sort
        }
        
        if use_history_template:
            start_time = time.time()
            res_sql = two_stage_predicate_bo(
                                            config,
                                            database,
                                            col_chosen,
                                            temp_sql,
                                            agg_sql,
                                            name_to_type,
                                            cpu_time,
                                            base_cpu
                                        )
            end_time = time.time()
            latency = end_time - start_time
            for ans in res_sql:
                if not ans:
                    continue
                real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                    config,
                    ans,
                    database=database,
                    real_run = True
                )
                info = {
                    "query_hash": main_key,
                    "query_param_hash": sub_key,
                    "query": ans,
                    "template": temp_sql,
                    "agg_sql": agg_sql,
                    "query_plan": real_query_plan,
                    "cputime": total_cputime,
                    "scanbytes": total_scan,
                    "duration": duration,
                    "col_chosen": col_chosen,
                    "base_cpu": base_cpu,
                }
                history = update_his(history, info, cpu_time, scan_bytes)
                save_res(config, idx, ans, real_query_plan, total_cputime, total_scan, duration, 0, token_use, latency)
            save_his(config, history, his_path)
        else:
            columns = select_table(scan_bytes, cpu_time, database, ops, k)
            for turn, (base_scan, base_cpu, column) in enumerate(columns):
                token_use = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
                start_time = time.time()
                temp_sql, token_usage = gen_template(column, database, targets) 
                token_use["prompt_tokens"] += token_usage["prompt_tokens"]
                token_use["completion_tokens"] += token_usage["completion_tokens"]
                token_use["total_tokens"] += token_usage["total_tokens"] 

                if join == 0:
                    test_res = 0
                else:
                    test_res = test_query(config, temp_sql, database)
                
                if test_res == -1:
                    continue
                elif test_res >= 40:
                    total_cputime, total_scan = test_res + base_cpu, base_scan * 1024
                else:
                    real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                        config,
                        temp_sql,
                        database=database,
                        real_run = True
                    )
                
                agg_sql, token_usage = gen_agg_sql(temp_sql)
                token_use["prompt_tokens"] += token_usage["prompt_tokens"]
                token_use["completion_tokens"] += token_usage["completion_tokens"]
                token_use["total_tokens"] += token_usage["total_tokens"]
                
                # add sort operation part
                if sort > 0:
                    temp_sql = add_sort_operation(config, temp_sql, column)
                
                # Tuning stage
                cpu_diff = total_cputime - cpu_time
                scan_diff = total_scan - scan_bytes
                
                # If cpu time is not enough, we need to add CPU_compute
                if cpu_diff / cpu_time < -0.1:
                    card = get_card(agg_sql, config, database)
                    col_ops = add_cpu(column, -cpu_diff, card, name_to_type)
                    temp_sql, token_usage = add_operation_to_sql(config, temp_sql, col_ops)
                    token_use["prompt_tokens"] += token_usage["prompt_tokens"]
                    token_use["completion_tokens"] += token_usage["completion_tokens"]
                    token_use["total_tokens"] += token_usage["total_tokens"]
                    
                    real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                        config,
                        temp_sql,
                        database=database,
                        real_run = True
                    )
                    if duration == -1:
                        continue
                    
                # Predicate tuning
                # temporarily choose k columns
                if (total_cputime - cpu_time) / cpu_time > 0.1:
                    col_chosen = choose_predicate(column, name_to_type, meta_info, predicate_num=3)
                    use_bo = True
                    if not col_chosen:
                        use_bo = False
                    if use_bo:
                        temp_sql = gen_predicate_loc(temp_sql, col_chosen)
                        agg_sql = gen_predicate_loc(agg_sql, col_chosen)
                        
                        res_sql = two_stage_predicate_bo(
                                                    config,
                                                    database,
                                                    col_chosen,
                                                    temp_sql,
                                                    agg_sql,
                                                    name_to_type,
                                                    cpu_time,
                                                    base_cpu
                                                )
                    else:
                        res_sql = [temp_sql]
                    end_time = time.time()
                    latency = end_time - start_time
                    
                    for ans in res_sql:
                        if not ans:
                            continue
                        real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                            config,
                            ans,
                            database=database,
                            real_run = True
                        )
                        info = {
                            "query_hash": main_key,
                            "query": ans,
                            "template": temp_sql,
                            "agg_sql": agg_sql,
                            "query_plan": real_query_plan,
                            "cputime": total_cputime,
                            "scanbytes": total_scan,
                            "duration": duration,
                            "col_chosen": col_chosen,
                            "base_cpu": base_cpu,
                        }
                        history = update_his(history, info, cpu_time, scan_bytes)
                        save_res(config, idx, ans, real_query_plan, total_cputime, total_scan, duration, turn, token_use, latency)
                        save_his(config, history, his_path)
                else:
                    end_time = time.time()
                    latency = end_time - start_time
                    info = {
                            "query_hash": main_key,
                            "query": temp_sql,
                            "template": temp_sql,
                            "agg_sql": agg_sql,
                            "query_plan": real_query_plan,
                            "cputime": total_cputime,
                            "scanbytes": total_scan,
                            "duration": duration,
                        }
                    history = update_his(history, info, cpu_time, scan_bytes)
                    save_his(config, history, his_path)
                    save_res(config, idx, temp_sql, real_query_plan, total_cputime, total_scan, duration, turn, token_use, latency)
                    
                if abs(total_cputime - cpu_time) / cpu_time <= 0.1:
                    break
        
        

if __name__ == "__main__":
    directory = './configs'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"processing {file}")
                    print(config)
                    
                    main(config)