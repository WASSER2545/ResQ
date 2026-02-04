import re
import csv
import sys
sys.path.append("/Users/zsy/Documents/codespace/python/FlexBench_original/Demo1/models")
from utils import *
import json
from decimal import Decimal

# === 需要 profile 的 schema ===
SCHEMA = "tpch1g"

# === 获取所有列信息 ===
GET_ALL_NAME = f"""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '{SCHEMA}'
    ORDER BY table_name, ordinal_position
"""

# === 辅助函数：选择聚合函数 ===
def agg_func(dtype: str, col: str):
    dtype = dtype.lower()
    if any(t in dtype for t in ["int", "decimal", "float", "double"]):
        return f"MIN({col}), MAX({col})"
    elif "date" in dtype:
        return f"MIN({col}), MAX({col})"
    else:
        return f"MIN(LENGTH({col})), MAX(LENGTH({col}))"

@func_set_timeout(80)
def execute_sys_query(config, query, database):
    # if there are more than one query in the file, only execute them one by one
    query = query.split(";")
    # make sure the last query is not empty
    if query[-1] == "":
        query = query[:-1]
    # add the last semicolon to each query
    query = [q + ";" for q in query]
    ret = []
    for q in query:
        password = config["PASSWORD"]
        host = config["HOST"]
        warehouse = config["WAREHOUSE_NAME"]
        client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

        cursor = client.cursor()
        cursor.execute(q)
        # consider the case that there are multiple queries in tpc-ds
        rows = cursor.fetchall()
        
        # 打印结果
        res = [row.values() for row in rows]
    return res

def execute_normal_query(config, query, database):
    # if there are more than one query in the file, only execute them one by one
    query = query.split(";")
    # make sure the last query is not empty
    if query[-1] == "":
        query = query[:-1]
    # add the last semicolon to each query
    query = [q + ";" for q in query]
    ret = []
    for q in query:
        password = config["PASSWORD"]
        host = config["HOST"]
        warehouse = config["WAREHOUSE_NAME"]
        client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

        cursor = client.cursor()
        cursor.execute(q)
        # consider the case that there are multiple queries in tpc-ds
        rows = cursor.fetchall()
        
        # 打印结果
        res = [row.values() for row in rows]
    return res[0]

def save_data_to_file(data, record_file):
    """ Save data to a file. """
    with open(record_file, "w") as file:
        json.dump(data, file, indent=2)
def main(config):
    database = 'tpch1g'    
    record_file = f"/Users/zsy/Documents/codespace/python/FlexBench_original/Demo1/models/predicate_tuning/histogram_data/{database}-column-minmax.json"
    
    query = GET_ALL_NAME
    columns = execute_sys_query(config, query, database)
    data = []
    if os.path.exists(record_file):
            with open(record_file, "r") as file:
                data = json.load(file)
    start_index = len(data)
    for table_name, col_name, col_type in columns[start_index:]:
        sql = f"SELECT {agg_func(col_type, col_name)} FROM {SCHEMA}.{table_name}"
        mi, mx = execute_normal_query(config, sql, database)
        if isinstance(mi, Decimal):
            mi = float(mi)
            mx = float(mx)
        if col_type == "Date":
            mi = mi.isoformat()
            mx = mx.isoformat()
        data.append({
                "table_name": table_name,
                "col_name": col_name,
                "col_type": col_type,
                "min": mi,
                "max": mx
            })
        save_data_to_file(data, record_file)
        

if __name__ == "__main__":
    directory = '/Users/zsy/Documents/codespace/python/FlexBench_original/Demo1/models/statistic_retrieve/config'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"processing {file}")
                    print(config)
                    main(config)