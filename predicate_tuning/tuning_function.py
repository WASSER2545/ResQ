from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import random
from datetime import datetime, timedelta
from general_agent import *
import os
import yaml
import json
import time
from process_plans import *
from utils import *
from collections import defaultdict

from performance_predictor.predict_from_explain import *

import warnings
from skopt.optimizer.optimizer import Optimizer

_original_ask = Optimizer._ask

def _ask_with_retry_limit(self, max_retry=20):
    if not hasattr(self, "_ask_retry_count"):
        self._ask_retry_count = 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _original_ask(self)

        triggered = False
        for warn in w:
            if "has been evaluated at point" in str(warn.message):
                self._ask_retry_count += 1
                triggered = True

                if self._ask_retry_count >= max_retry:
                    # 直接返回一个新的 random 点
                    x = self.space.rvs(random_state=self.rng)[0]
                    self._ask_retry_count = 0
                break

        if not triggered:
            self._ask_retry_count = 0

    return x

Optimizer._ask = _ask_with_retry_limit

NO_BUCKET_COLS = {}
def read_col_to_width():
    with open("./histogram_data/tpch1g-column-minmax.json", "r") as f:
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

col_to_width = read_col_to_width()

def random_date(start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    rand_days = random.randint(0, delta.days)
    return (start_dt + timedelta(days=rand_days)).strftime("%Y-%m-%d")

def day_to_date(start, day_offset):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    return (start_dt + timedelta(days=int(day_offset))).strftime("%Y-%m-%d")

def get_cpu_time(config, sql, count_sql, database, target_cpu, scan_cpu, from_local=False):
    if not from_local:
        real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                    config,
                    sql,
                    database=database,
                    real_run = True
                )
    else:
        logic_plan = execute_explain_query(config, database, sql)
        agg_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                        config,
                        count_sql,
                        database=database,
                        real_run = True
                    )
        total_cputime = cal_cpu(logic_plan, query_metrics)
        total_cputime += scan_cpu
        
        a, b = 1.2, 1.2
        if target_cpu / a < total_cputime < target_cpu * b:
            real_query_plan, duration, total_cputime, total_scan, query_metrics = get_info(
                    config,
                    sql,
                    database=database,
                    real_run = True
                )
            
    score = - (total_cputime - target_cpu) ** 2
    diff = total_cputime - target_cpu
    return score, diff

def get_date_diff(start, end):
    d1 = datetime.strptime(start, "%Y-%m-%d")
    d2 = datetime.strptime(end, "%Y-%m-%d")
    
    diff_days = abs((d2 - d1).days)
    return diff_days

def get_min_date(col_name, min_max_dict):
    return min_max_dict[col_name]["min"]

def get_column_min_max(col_names, min_max_dict):
    pbounds = {}
    for col_name in col_names:
        name = col_name.split('.')[1]
        tmp = min_max_dict[name]
        col_type = tmp["dtype"].lower()
        col_min, col_max = tmp["min"], tmp["max"]
        if col_type == 'date':
            dif = get_date_diff(col_min, col_max)
            pbounds[col_name] = (0, dif)
        elif col_type == 'string':
            pbounds[col_name] = (col_min, col_max)
        else:
            pbounds[col_name] = (col_min, col_max)
    return pbounds
        
def run_query(config, database, tar_cpu, sql_template, count_sql, name_to_type, min_max_dict, cpu_scan, **params):
    fixed_params = []
    for k, v in params.items():
        col_name = k.split('.')[1]
        dtype = name_to_type[col_name].lower()
        col_stat = min_max_dict[col_name]

        # ---------- bucket decode ----------
        if "buckets" in col_stat:
            bucket_id = int(v)
            l, r = col_stat["buckets"][bucket_id]

            real_v = int(r)

        else:
            real_v = v

        # ---------- type handling ----------
        if dtype == "date":
            min_date = get_min_date(col_name, min_max_dict)
            fixed_params.append(f"'{day_to_date(min_date, int(real_v))}'")
        else:
            fixed_params.append(int(real_v))
    
    sql = sql_template.format(
        *fixed_params
    )
    sql = sql.replace("\n", " ").replace("  ", " ").strip()
    
    count_sql = count_sql.format(
        *fixed_params
    )
    count_sql = count_sql.replace("\n", " ").replace("  ", " ").strip()
    
    from_local = True
    has_join = re.search(r'\bjoin\b', sql, re.IGNORECASE)
    if not has_join:
        from_local = False
    
    score, diff = get_cpu_time(config, sql, count_sql, database, tar_cpu, cpu_scan, from_local=from_local)
    return score, diff

def build_sql(params, sql_template, name_to_type, min_max_dict):
    fixed_params = []

    for k, v in params.items():
        col_name = k.split('.')[1]
        dtype = name_to_type[col_name].lower()
        col_stat = min_max_dict[col_name]

        # ---------- bucket decode ----------
        if "buckets" in col_stat:
            bucket_id = int(v)
            l, r = col_stat["buckets"][bucket_id]

            real_v = int(r)

        else:
            real_v = v

        # ---------- type handling ----------
        if dtype == "date":
            min_date = get_min_date(col_name, min_max_dict)
            fixed_params.append(f"'{day_to_date(min_date, int(real_v))}'")
        else:
            fixed_params.append(int(real_v))

    sql = sql_template.format(*fixed_params)
    return sql

def build_equal_width_buckets(min_v, max_v, num_buckets):    
    width = (max_v - min_v) / num_buckets
    buckets = []
    for i in range(num_buckets):
        l = min_v + i * width
        r = min_v + (i + 1) * width
        buckets.append((l, r))
    return buckets
    
def read_histogram_data(database, num_buckets=20):
    record_file = os.path.join(
        "./histogram_data",
        f"{database}-column-minmax.json",
    )

    with open(record_file, "r") as f:
        col_list = json.load(f)

    res = defaultdict()

    for col in col_list:
        col_name = col["col_name"]
        min_v = col["min"]
        max_v = col["max"]
        dtype = col["col_type"].lower()

        entry = {
            "table_name": col["table_name"],
            "dtype": dtype,
            "min": min_v,
            "max": max_v,
        }

        if any(t in dtype for t in ["int", "decimal", "float", "double"]) and col_name not in NO_BUCKET_COLS:
            entry["buckets"] = build_equal_width_buckets(
                min_v, max_v, num_buckets
            )

        res[col_name] = entry

    return res

def build_local_buckets(low, high, n_buckets):
    edges = np.linspace(low, high, n_buckets + 1)
    return edges

def get_local_range(center, col_min, col_max, is_neg=False, ratio=0.1):
    width = col_max - col_min
    if is_neg:
        low = max(col_min, center - width * 0.001)
        high = min(col_max, center + width * ratio)
    else:
        low = max(col_min, center - width * ratio)
        high = min(col_max, center + width * 0.001)
    return int(low), int(high)

def get_local_range_date(center, col_min, col_max, is_neg=False, ratio=0.1):
    mx, mi = get_date_diff(col_min, col_max), 0
    center_day = center
    width = mx
    
    if is_neg:
        low = max(mi, center_day - width * 0.001)
        high = min(mx, center_day + width * ratio)
    else:
        low = max(mi, center_day - width * ratio)
        high = min(mx, center_day + width * 0.001)
    
    return int(low), int(high)

def should_prune_by_monotone(
    params,
    X_train,
    y_train,
    diff_list,
    monotone_cols,
    alpha=1.5,
):
    if not y_train:
        return False

    best_score = max(y_train)
    bad_threshold = best_score * alpha

    for obs, score, diff in zip(X_train, y_train, diff_list):
        worse = True
        if diff < 0:
            for col in monotone_cols:
                if params[col] >= obs[col]:
                    worse = False
                    break

            if worse and score <= bad_threshold:
                return True
        else:
            for col in monotone_cols:
                if params[col] <= obs[col]:
                    worse = False
                    break

            if worse and score <= bad_threshold:
                return True

    return False


from sklearn.ensemble import GradientBoostingRegressor
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
def stage1_global_bo(
    config, 
    database, 
    col_names, 
    sql_template, 
    count_sql, 
    name_to_type, 
    tar_cpu, 
    scan_cpu, 
    n_calls=50, 
    n_random_starts=5
):
    min_max_dict = read_histogram_data(database)
            
    sk_space = []
    name_order = []
    bucketed_cols = set()

    for k in col_names:
        name_order.append(k)
        
        col_stat = min_max_dict[k.split('.')[1]]
        dtype = name_to_type[k.split('.')[1]].lower()

        if "buckets" in col_stat:
            B = len(col_stat["buckets"])
            sk_space.append(Integer(0, B - 1, name=k))
            bucketed_cols.add(k)
        else:
            low_day, high_day = col_stat["min"], col_stat["max"]
            low = get_date_diff(low_day, low_day)
            high = get_date_diff(low_day, high_day)
            if dtype in ("date", "int"):
                sk_space.append(Integer(int(low), int(high), name=k))
            else:
                sk_space.append(Real(low, high, name=k))

    X_train = []
    y_train = []
    diff_list = []

    opt = Optimizer(
        dimensions=sk_space,
        base_estimator="RF",
        n_initial_points=n_random_starts,
        acq_func="EI",  # Expected Improvement
        random_state=42
    )
    
    monotone_cols = name_order
    observed_params = []
    
    for _ in range(n_random_starts):
        next_point = opt.ask()
        params = {name_order[i]: next_point[i] for i in range(len(next_point))}
        
        score, diff = run_query(config, database, tar_cpu, sql_template, count_sql, name_to_type, min_max_dict, scan_cpu, **params)
        
        observed_params.append(params)
        X_train.append(next_point)
        y_train.append(score)
        diff_list.append(diff)
        opt.tell(next_point, score)
        print(f"[Random Start] score={score:.4f} | params={params}")

    # BO iterations
    for i in range(n_calls):
        while True:
            next_point = opt.ask()
            params = {name_order[j]: next_point[j] for j in range(len(next_point))}
            
            if should_prune_by_monotone(
                params,
                observed_params,
                y_train,   
                diff_list,          
                monotone_cols,
                alpha=1.5,
            ):
                worst_score = max(y_train) * 2.0
                opt.tell(next_point, worst_score)
                continue

            break
        
        score, diff = run_query(config, database, tar_cpu, sql_template, count_sql, name_to_type, min_max_dict, scan_cpu, **params)
        
        observed_params.append(params)
        X_train.append(next_point)
        y_train.append(score)
        diff_list.append(diff)
        opt.tell(next_point, score)
        print(f"[BO Stage 1 Iter {i+1:02d}] score={score:.4f} | params={params}")

    best_idx = np.argmax(y_train)
    best_diff = diff_list[best_idx]
    best_params = {name_order[i]: X_train[best_idx][i] for i in range(len(name_order))}
    
    return X_train, y_train, diff_list, min_max_dict, name_order

def adaptive_ratio(y_train, base=0.1, min_r=0.1, max_r=0.5):
    std = np.std(y_train)
    mean = abs(np.mean(y_train)) + 1e-6
    factor = std / mean
    ratio = base * factor
    return float(np.clip(ratio, min_r, max_r))

from skopt.space import Integer

from skopt.space import Integer, Real

def build_stage2_space(
    best_stage1_params,
    min_max_dict,
    name_to_type,
    is_neg=False,
    n_buckets=20,
    ratio=0.1,
):
    sk_space = []
    name_order = []
    bucket_info = {}

    for col, center in best_stage1_params.items():
        name = col.split('.')[-1]
        dtype = name_to_type[name].lower()

        col_min = min_max_dict[name]["min"]
        col_max = min_max_dict[name]["max"]
        if dtype == 'date':
            low, high = get_local_range_date(
                center,
                col_min,
                col_max,
                is_neg=is_neg,
                ratio=ratio,
            )
        else:
            low, high = get_local_range(
                center,
                col_min,
                col_max,
                is_neg=is_neg,
                ratio=ratio,
            )

        if any(t in dtype for t in ["int", "decimal", "float", "double"]) and col not in NO_BUCKET_COLS:
            edges = build_equal_width_buckets(low, high, n_buckets)

            bucket_info[col] = {
                "edges": edges,
                "low": low,
                "high": high,
            }

            sk_space.append(
                Integer(0, n_buckets - 1, name=col)
            )
            name_order.append(col)

        else:
            sk_space.append(
                Integer(int(low), int(high), name=col)
            )
            name_order.append(col)

    return sk_space, name_order, bucket_info

def build_sql_stage2(
    params,
    sql_template,
    count_sql,
    name_to_type,
    bucket_info,
    min_max_dict
):
    fixed_params = []

    for col, v in params.items():
        name = col.split('.')[-1]
        dtype = name_to_type[name].lower()

        # bucket 列
        if col in bucket_info:
            edges = bucket_info[col]["edges"]
            value = edges[int(v)][1]
            fixed_params.append(value)

        # date / 非 bucket 列
        else:
            if dtype == "date":
                min_date = get_min_date(name, min_max_dict)
                fixed_params.append(
                    f"'{day_to_date(min_date, int(v))}'"
                )
            else:
                fixed_params.append(int(v))

    return sql_template.format(*fixed_params), count_sql.format(*fixed_params)

def stage2_local_bo(
    config,
    database,
    sql_template,
    count_sql,
    name_to_type,
    tar_cpu,
    scan_cpu,
    sk_space,
    name_order,
    bucket_info,
    min_max_dict,
    n_calls=20,
    n_random_starts=5,
):
    opt = Optimizer(
        dimensions=sk_space,
        base_estimator="RF",
        n_initial_points=n_random_starts,
        acq_func="EI",
        acq_func_kwargs={"xi": 0.01},
        random_state=7,
    )

    X_train, y_train = [], []
    observed_params = []
    diff_list = []
    monotone_cols = name_order
    
    from_local = True
    if "join" not in sql_template or "JOIN" not in sql_template:
        from_local = False
        
    for _ in range(n_random_starts):
        next_point = opt.ask()
        
        params = {name_order[i]: next_point[i] for i in range(len(next_point))}
        sql, agg_sql = build_sql_stage2(params, sql_template, count_sql, name_to_type, bucket_info, min_max_dict)
        
        score, diff = get_cpu_time(config, sql, agg_sql, database, tar_cpu, scan_cpu, from_local=from_local)
        
        observed_params.append(params)
        X_train.append(next_point)
        y_train.append(score)
        diff_list.append(diff)
        opt.tell(next_point, score)
        print(f"[Random Start] score={score:.4f} | params={params}")
    
    for i in range(n_calls):
        while True:
            next_point = opt.ask()
            params = {name_order[j]: next_point[j] for j in range(len(next_point))}
            
            if should_prune_by_monotone(
                params,
                observed_params,
                y_train,   
                diff_list,          
                monotone_cols,
                alpha=1.5,
            ):
                worst_score = max(y_train) * 2.0
                opt.tell(next_point, worst_score)
                continue

            break
        
        sql, agg_sql = build_sql_stage2(params, sql_template, count_sql, name_to_type, bucket_info, min_max_dict)
        score, diff = get_cpu_time(config, sql, agg_sql, database, tar_cpu, scan_cpu, from_local=from_local)
        
        observed_params.append(params)
        X_train.append(next_point)
        y_train.append(score)
        diff_list.append(diff)
        opt.tell(next_point, score)
        print(f"[BO Stage 2 Iter {i+1:02d}] score={score:.4f} | params={params}")

    best_idx = np.argmax(y_train)
    return dict(zip(name_order, X_train[best_idx]))

def restore_param(params, name_to_type, min_max_dict):
    res = {}
    for k, v in params.items():
        col_name = k.split('.')[1]
        dtype = name_to_type[col_name].lower()
        col_stat = min_max_dict[col_name]

        # ---------- bucket decode ----------
        if "buckets" in col_stat:
            bucket_id = int(v)
            l, r = col_stat["buckets"][bucket_id]

            real_v = int(r)

        else:
            real_v = v

        res[k] = int(real_v)
    return res

def select_stage2_seeds(X_train, mse_list, diff_list, name_order):
    best_pos_idx = None
    best_neg_idx = None

    best_pos_mse = -float("inf")
    best_neg_mse = -float("inf")

    for i, (mse, diff) in enumerate(zip(mse_list, diff_list)):
        if diff > 0:
            if mse > best_pos_mse:
                best_pos_mse = mse
                best_pos_idx = i
        elif diff < 0:
            if mse > best_neg_mse:
                best_neg_mse = mse
                best_neg_idx = i

    best_pos = None
    best_neg = None

    if best_pos_idx is not None:
        best_pos = {
            name_order[j]: X_train[best_pos_idx][j]
            for j in range(len(name_order))
        }

    if best_neg_idx is not None:
        best_neg = {
            name_order[j]: X_train[best_neg_idx][j]
            for j in range(len(name_order))
        }

    return best_pos, best_neg

def two_stage_predicate_bo(
    config,
    database,
    col_names,
    sql_template,
    count_sql,
    name_to_type,
    tar_cpu,
    scan_cpu,
):
    # Stage 1
    X_train, y_train, diff_list, min_max_dict, name_order = stage1_global_bo(
        config, database, col_names,
        sql_template, count_sql,
        name_to_type, tar_cpu, scan_cpu
        ,n_calls=10, n_random_starts=10
    )
    
    # X_train, y_train, diff_list, min_max_dict, name_order = stage1_global_bo(
    #     config, database, col_names,
    #     sql_template, count_sql,
    #     name_to_type, tar_cpu, scan_cpu
    #     ,n_calls=0, n_random_starts=1
    # )
    
    best_pos, best_neg = select_stage2_seeds(X_train, y_train, diff_list, name_order)
    best_pos_sql = ""
    best_neg_sql = ""
    if best_pos:
        best_pos_sql = build_sql(best_pos, sql_template, name_to_type, min_max_dict).replace("\n", " ").replace("  ", " ").strip()
    if best_neg:
        best_neg_sql = build_sql(best_neg, sql_template, name_to_type, min_max_dict).replace("\n", " ").replace("  ", " ").strip()
                
    if min(list(map(abs, diff_list))) / tar_cpu < 0.05:
        return [best_pos_sql, best_neg_sql]
    
    if best_pos:
        best_stage1 = restore_param(best_pos, name_to_type, min_max_dict)
    else:
        best_stage1 = restore_param(best_neg, name_to_type, min_max_dict)

    # Stage 2 space
    sk_space, name_order, bucket_info = build_stage2_space(
        best_stage1,
        min_max_dict,
        name_to_type,
        is_neg=False,
        n_buckets=200,
        ratio=0.1,
    )

    # Stage 2
    best_bucket_params = stage2_local_bo(
        config, database,
        sql_template, count_sql,
        name_to_type, tar_cpu, scan_cpu,
        sk_space, name_order, bucket_info, min_max_dict
        ,n_calls=8, n_random_starts=7
    )

    # Final SQL
    final_sql, _ = build_sql_stage2(
        best_bucket_params,
        sql_template,
        count_sql,
        name_to_type,
        bucket_info,
        min_max_dict
    )
    
    final_sql = final_sql.replace("\n", " ").replace("  ", " ").strip()

    return [best_pos_sql, best_neg_sql, final_sql]
