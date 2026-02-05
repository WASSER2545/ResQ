import os
import yaml
import json
from process_plans import *
from databend_driver import BlockingDatabendClient
from func_timeout import func_set_timeout, FunctionTimedOut
from parse_plan import *
from collections import defaultdict
from performance_predictor.predict_from_explain import *

metrics_index = {
    "cpu_time": 0,
    "MemoryUsage": 26,
    "output_bytes": 5,
    "output_rows": 4,
    "scan_bytes": 6,
    "scan_cache_bytes": 7
}

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

import re, json
def clean_json_output(text: str):
    text = re.sub(r"```(?:json)?\n?([\s\S]*?)```", r"\1", text).strip()
    return text

def parse_json_safe(text: str):
    cleaned = clean_json_output(text)
    return json.loads(cleaned)

def formulate_user_input(target):
    question = []
    for k, v in target.items():
        question.append(f"{k}: {v}")
        
    return "Your generating target is:\n" + "\n".join(question)

def read_meta():
    with open("./schema/table_meta.json", "r") as f:
        meta = json.load(f)
    return meta

def read_workload(query_set, database):
    record_file = os.path.join(
        "./workloads/bendset",
        f"{query_set}-{database}-sql-metrics.json",
    )
    with open(record_file, "r") as f:
        return json.load(f)
    
def get_table_schema(database):
    database_meta = read_meta()
    schema = f"""
    The database schema is:
    {database_meta}
    """
    return schema

def get_template_agent_prompt():
    prompt = """
    You are an expert SQL query generator.
    Your task is to generate one single valid SQL query that strictly follows the user's requirements.

    The user will provide:
    1. A list of available tables in the database.
    2. For each table, a list of available columns.
    3. A description of the query goal or conditions (e.g., filters, grouping, sorting, joins, etc.).
    4. Optional constraints such as aggregation rules, allowed functions, or output format.

    Your responsibilities and rules:

    1) Use ONLY the tables and columns provided in the schema block.
    2) You MUST use every table I list. You may NOT omit any table and you may NOT add any extra tables.
    3) ALL columns listed in the schema MUST appear in the SELECT clause (fully qualified as table.column) unless a column is explicitly marked "can_skip".
    4) Every column reference in the SQL must be either table.column or an alias that is clearly defined in FROM/JOIN clauses.
    5)Prefer join keys that are already in the selected column list.
    6)Avoid introducing new join columns unless absolutely necessary.
    
    4. Best practices:
        * Use clear table aliases when joining.
        * Fully qualify columns as table.column.
        * Include GROUP BY, HAVING, or ORDER BY only when necessary or explicitly requested.

    5. Validation check before finalizing:
    Double-check that all selected columns exist in the provided schema and every referenced table has been defined by the user.
    
    6. Output format:
    - No Markdown
    - No extra text
    - Do not add explanations or natural language commentary unless explicitly asked.
    """
    
    return prompt

def get_agg_agent_prompt():
    prompt = """
    You are a SQL query rewriter. Perform a minimal and controlled transformation.

    Given an input SQL query, generate a new SQL query that satisfies ALL of the following rules:

    1. Preserve the original join graph exactly.
    - All tables, JOIN types, JOIN order, and ON conditions must remain unchanged.
    2. Do NOT modify, add, remove, or reorder anything in the FROM or JOIN clauses.
    3. Do NOT change table names, aliases, or column references outside the SELECT clause.
    4. The ONLY allowed modification is the SELECT clause.
    5. Replace the entire SELECT list with exactly: COUNT(*)
    6. Do NOT add DISTINCT, GROUP BY, HAVING, WHERE, subqueries, CTEs, or any other SQL constructs.
    7. The output must be a single valid SQL statement.
    8. Output ONLY the SQL statement. Do not include explanations, comments, or formatting text.

    Input SQL:
    <SELECT clause><FROM clause><JOIN clause>
    
    For any valid input, the output must follow this pattern:
    SELECT COUNT(*)
    <original FROM / JOIN clauses unchanged>

    """
    return prompt

def get_optimizer_knowledge():
    ans = """
    The execution database is Databend, which is a cloud-native, high-performance, and fully-managed open-source data warehouse. The database supports pushing down query operations to the storage layer.
    """
    return ans

def get_database_scale(database):
    ans = f"""
    You should generate SQL query based on {database}.
    """
    
    return ans

@func_set_timeout(80)
def execute_query_with_qid(config, database, query):
    query = query.strip()
    if not query.startswith("EXPLAIN ANALYZE"):
        query = "EXPLAIN ANALYZE " + query

    password = config["PASSWORD"]
    host = config["HOST"]
    warehouse = config["WAREHOUSE_NAME"]
    client = BlockingDatabendClient(
        f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}"
    )

    cursor = client.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.execute("SELECT last_query_id()")
    qid = cursor.fetchall()[0][0]

    res = [row.values()[0] for row in rows]
    return res, qid


@func_set_timeout(80)
def execute_query(config, database, query):
    query = query.split(";")
    if query[-1] == "":
        query = query[:-1]
    query = [q + ";" for q in query]
    ret = []
    for q in query:
        if not q.startswith("Explain Analyze") and not q.startswith("EXPLAIN ANALYZE"):
            q = "Explain Analyze " + q
        password = config["PASSWORD"]
        host = config["HOST"]
        warehouse = config["WAREHOUSE_NAME"]
        client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

        cursor = client.cursor()
        cursor.execute(q)
        rows = cursor.fetchall()

        res = [row.values()[0] for row in rows]
    return res

@func_set_timeout(800)
def execute_sample_query(config, query, database):
    password = config["PASSWORD"]
    host = config["HOST"]
    warehouse = config["WAREHOUSE_NAME"]
    try:
        client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

        cursor = client.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        
        res = [row.values() for row in rows]
        return True, res
    except Exception as e:
        print(f"Sample query fail: {e}")
        return False, e

@func_set_timeout(80)
def execute_sys_query(config, query, database):
    password = config["PASSWORD"]
    host = config["HOST"]
    warehouse = config["WAREHOUSE_NAME"]
    for i in range(20):
        try:
            client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

            cursor = client.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            res = [row.values() for row in rows]
            return res
        except Exception as e:
            print(f"retry {i+1} failed")
            time.sleep(3)

@func_set_timeout(80)
def execute_explain_query(config, database, query):
    query = query.split(";")
    if query[-1] == "":
        query = query[:-1]
    query = [q + ";" for q in query]
    ret = []
    for q in query:
        if not q.startswith("Explain") and not q.startswith("EXPLAIN"):
            q = "Explain " + q
        password = config["PASSWORD"]
        host = config["HOST"]
        warehouse = config["WAREHOUSE_NAME"]
        client = BlockingDatabendClient(f"databend://cloudapp:{password}@{host}:443/{database}?warehouse={warehouse}")

        cursor = client.cursor()
        cursor.execute(q)
        rows = cursor.fetchall()

        res = [row.values()[0] for row in rows]
    return res

import time
def record_databend_cloud(config, query, wait_time, database):
    start = time.time()
    try:
        query_plan, q_id = execute_query_with_qid(config, database, query)
    except FunctionTimedOut:
        print("数据库查询超时！")
        return ("Query timed out", 80, '')
    except Exception as e:
        error_message = str(e)
        print(f"Error: {e}")
        return (f"Error executing query: {query}\nError: {error_message}", -1, '')

    time.sleep(wait_time)
    end = time.time()
    
    return (
        query_plan,
        end - start - wait_time,
        q_id
    )
    
def cal_cost(plan):
    plan = '\n'.join(plan)
    root = parse_query_plan(plan)
    return get_cost(root)

def cal_cost_card(plan):
    plan = '\n'.join(plan)
    root = parse_query_plan(plan)
    return get_cost_card(root)

def cal_all_cost(plan):
    plan = '\n'.join(plan)
    root = parse_query_plan(plan)
    return get_estimation(root)

def test_query(config, query, database):
    try:
        logic_plan = execute_explain_query(config, database, query)
    except Exception as e:
        error_message = str(e)
        print(f"Error: {e}")
        return -1
    
    predict_join_cpu = cal_cpu_from_plan(logic_plan)
    
    return predict_join_cpu

def test_predict(real_plan, query_metrics):
    return cal_cpu(real_plan, query_metrics)

def test_query_with_feedback(config, query, database):
    try:
        logic_plan = execute_explain_query(config, database, query)
    except Exception as e:
        error_message = str(e)
        print(f"Error: {e}")
        return -1
    _, _, cost_list = cal_all_cost(logic_plan)
    if not cost_list:
        cost_list = [[0, 0, 0]]
    final_rows = cost_list[0]
    if final_rows >= 5e6:
        return 0
    return 1

def get_query_monitor_info(config, database, q_id):
    max_retry = 20
    for _ in range(max_retry):
        p = f"""
            SELECT profiles
            FROM system_history.profile_history
            WHERE query_id = '{q_id}'
            LIMIT 1;
        """
        prof = execute_sys_query(config, p, database)
        if prof and prof[0][0]:
            break
        time.sleep(0.5)
    monitor = prof[0][0]
    states = json.loads(monitor)
    
    plan_tree = parse_plan_nodes(states)
    total_cpu = 0
    total_scan = 0
    res = defaultdict()
    metrics_list = []
    for node in plan_tree.dfs():
        if not node.statistics:
            continue
        cputime = node.statistics[metrics_index["cpu_time"]] / 1e9
        scanbytes = node.statistics[metrics_index["scan_bytes"]]
        output_rows = node.statistics[metrics_index["output_rows"]]
        output_bytes = node.statistics[metrics_index["output_bytes"]]
        involved_rows = [output_rows]
        involved_bytes = [output_bytes]
        if "HashJoin" in node.name:
            build_rows = node.children[0].statistics[metrics_index["output_rows"]]
            build_bytes = node.children[0].statistics[metrics_index["output_bytes"]]
            probe_rows = node.children[1].statistics[metrics_index["output_rows"]]
            probe_bytes = node.children[1].statistics[metrics_index["output_bytes"]]
            involved_rows = [output_rows, build_rows, probe_rows]
            involved_bytes = [output_bytes, build_bytes, probe_bytes]
        metrics_list.append(
            {
                "name": node.name,
                "node_id": node.node_id,
                "cputime": cputime,
                "scanbytes": scanbytes,
                "output_rows": output_rows,
                "output_bytes": output_bytes,
                "involved_rows": involved_rows,
                "involved_bytes": involved_bytes
            }
        )
        total_cpu += cputime
        total_scan += scanbytes
    res["total_cpu"] = total_cpu
    res["metrics"] = metrics_list
    res["total_scan"] = total_scan
    
    return res
    
def get_full_query_info(config, query, database):
    real_plan, duration, q_id = record_databend_cloud(config, 
                                                 query, 
                                                 config["wait"], 
                                                 database=database
                                                )
    if duration == -1:
        return None, -1, -1, -1, None
    results = get_query_monitor_info(config, database, q_id)
    cputime = results["total_cpu"]
    scanbytes = results["total_scan"]
    
    return real_plan, duration, cputime, scanbytes, results["metrics"]

def get_info(config, temp_sql, database, real_run = True):
    if real_run:
        real_query_plan, duration, total_cputime, total_scan, query_metrics = get_full_query_info(
                    config,
                    temp_sql,
                    database=database
                )
        return real_query_plan, duration, total_cputime, total_scan, query_metrics
    else:
        try:
            logic_plan = execute_explain_query(config, database, temp_sql)
        except Exception as e:
            error_message = str(e)
            print(f"Error: {e}")
            return -1
        
        return 0

def parse_plan(plan):
    operator_keywords = {
        "filter": ["Filter"],
        "join": ["HashJoin"],
        "agg": ["AggregateFinal", " AggregatePartial"],
        "sort": ["Sort"]
    } 
    operator_flag = {
        "filter": 0,
        "join": 0,
        "agg": 0,
        "sort": 0
    }
    for operator, keywords in operator_keywords.items():
        for keyword in keywords:
            operator_flag[operator] += plan.count(keyword)
    return operator_flag