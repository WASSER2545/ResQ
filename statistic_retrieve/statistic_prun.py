import os
import json
import sys
import yaml
from itertools import combinations
from collections import defaultdict, deque

SCALE = (1024 ** 1)

tpch_tables = ["region", "nation", "supplier", "customer", "orders", "lineitem", "part", "partsupp"]

# 定义合法 join 边
edges = {
    ("region", "nation"): ("r_regionkey", "n_regionkey"),
    ("nation", "supplier"): ("n_nationkey", "s_nationkey"),
    ("nation", "customer"): ("n_nationkey", "c_nationkey"),
    ("supplier", "partsupp"): ("s_suppkey", "ps_suppkey"),
    ("part", "partsupp"): ("p_partkey", "ps_partkey"),
    ("partsupp", "lineitem"): ("ps_partkey", "l_partkey"),
    ("orders", "lineitem"): ("o_orderkey", "l_orderkey"),
    ("customer", "orders"): ("c_custkey", "o_custkey"),
}

TPCH_JOIN_KEYS = {
    ("customer", "orders"): [("c_custkey", "o_custkey")],
    ("orders", "lineitem"): [("o_orderkey", "l_orderkey")],
    ("supplier", "lineitem"): [("s_suppkey", "l_suppkey")],
    ("part", "lineitem"): [("p_partkey", "l_partkey")],
    ("nation", "customer"): [("n_nationkey", "c_nationkey")],
    ("nation", "supplier"): [("n_nationkey", "s_nationkey")],
    ("region", "nation"): [("r_regionkey", "n_regionkey")],
    ("part", "partsupp"): [("p_partkey", "ps_partkey")],
    ("supplier", "partsupp"): [("s_suppkey", "ps_suppkey")],
}

# --- 转成邻接表 ---
graph = defaultdict(list)
for (a, b), keys in edges.items():
    graph[a].append(b)
    graph[b].append(a)

def is_connected(subset):
    """判断子集是否在 join 图中连通"""
    start = subset[0]
    visited = set()
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur not in visited:
            visited.add(cur)
            for nxt in graph[cur]:
                if nxt in subset and nxt not in visited:
                    q.append(nxt)
    return visited == set(subset)

def generate_join_combinations(num_joins: int):
    """
    生成所有 num_joins 个 join 的表组合。
    即包含 num_joins + 1 张表。
    """
    k = num_joins + 1
    combos = list(combinations(tpch_tables, k))
    res = []
    for combo in combos:
        if is_connected(combo):
            res.append(combo)
    return res
    
def read_database_state(database):
    record_file = os.path.join(
        "./state_metrics",
        f"{database}-column-metrics.json",
    )
    with open(record_file, "r") as f:
        return json.load(f)
    
def build_table_dict(states):
    tpch_tables.sort()
    idx = 0
    res = defaultdict()
    for i in range(len(tpch_tables)):
        table_name = tpch_tables[i]
        s_cpu, s_scan = 0, 0
        tmp = []
        cpu_list, scan_list = [], []
        while idx < len(states) and states[idx]["table_name"] == table_name:
            tmp.append(states[idx]["col_name"])
            tmp_scan = max(states[idx]["avg_scan_bytes"] // SCALE, 0.1)
            s_cpu += states[idx]["avg_cpu_time"]
            s_scan += tmp_scan
            cpu_list.append(states[idx]["avg_cpu_time"])
            scan_list.append(tmp_scan)
            idx += 1
        res[table_name] = {
            "col_name_list": tmp,
            "sum_cpu": s_cpu,
            "sum_scan": s_scan,
            "cpu_list": cpu_list,
            "scan_list": scan_list
        }
    return res

def group_sums_with_choice(group):
    """生成每组所有非空子集和及对应的选项"""
    result = []
    for r in range(1, len(group) + 1):
        for comb in combinations(range(len(group)), r):
            s = sum(group[i] for i in comb)
            result.append((s, list(comb)))
    return result

def grouped_subset_sum_with_path(groups, target, prune_limit_factor=2):
    """
    groups: List[List[int]] 各组的数值
    target: int 目标值
    返回 (最小和, 选择方案)
    """
    group_candidates = [group_sums_with_choice(g) for g in groups]

    # dp[sum] = path
    dp = {0: []}

    for g_idx, group in enumerate(group_candidates):
        new_dp = {}
        for base_sum, base_path in dp.items():
            for s, choice in group:
                new_sum = base_sum + s

                if new_sum > prune_limit_factor * target:
                    continue
                new_path = base_path + [choice]

                if new_sum not in new_dp or new_sum < min(new_dp):
                    new_dp[new_sum] = new_path

        keys = sorted(new_dp.keys())
        pruned = {}
        last = -float("inf")
        for k in keys:
            if k > last * 1.01:
                pruned[k] = new_dp[k]
                last = k
        dp = pruned

    feasible = [(s, p) for s, p in dp.items() if s >= target]
    if not feasible:
        return None, None

    best_sum, best_path = min(feasible, key=lambda x: x[0])
    return best_sum, best_path
def greedy_grouped_sum(groups, target, mandatory):
    """
    groups: List[List[int]]
    mandatory: List[Set[int]] 
    target: int

    return:
        total_sum: int
        choices: List[List[int]]
    """
    G = len(groups)
    choices = []
    total_sum = 0

    for g_idx, g in enumerate(groups):
        mand = mandatory[g_idx]
        choices.append(list(mand))
        total_sum += sum(g[i] for i in mand)

    avg_target = target / G
    for g_idx, g in enumerate(groups):
        if choices[g_idx]: 
            continue

        best_idx = min(
            range(len(g)),
            key=lambda i: abs(g[i] - avg_target)
        )
        choices[g_idx].append(best_idx)
        total_sum += g[best_idx]

    remaining = []
    for g_idx, g in enumerate(groups):
        chosen = set(choices[g_idx])
        for i, val in enumerate(g):
            if i not in chosen:
                remaining.append((val, g_idx, i))

    remaining.sort(key=lambda x: x[0])

    for val, g_idx, i in remaining:
        if total_sum >= target:
            break
        choices[g_idx].append(i)
        total_sum += val

    return total_sum, choices

def get_join_key(tables, name_to_colid):
    ban_set = {t: set() for t in tables}

    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            t1, t2 = tables[i], tables[j]

            key = (t1, t2)
            rev_key = (t2, t1)

            if key in TPCH_JOIN_KEYS:
                for c1, c2 in TPCH_JOIN_KEYS[key]:
                    ban_set[t1].add(c1)
                    ban_set[t2].add(c2)

            elif rev_key in TPCH_JOIN_KEYS:
                for c2, c1 in TPCH_JOIN_KEYS[rev_key]:
                    ban_set[t1].add(c1)
                    ban_set[t2].add(c2)
    res = defaultdict(list)
    for table, cols in ban_set.items():
        for col in cols:
            name = f"{table}.{col}"
            res[table].append(name_to_colid[name])
    return res

def select_join_tables(scan, cpu, join_num, table_dict, k):
    all_plans = generate_join_combinations(join_num)
    tar_scan = max(scan // SCALE, 0.1)
    
    name_to_colid = {}
    for tab_name, tab_info in table_dict.items():
        for idx, col_name in enumerate(tab_info["col_name_list"]):
            tmp_name = tab_name + "." + col_name
            name_to_colid[tmp_name] = idx

    all_ans = []
    for plan in all_plans:
        groups = []
        groups_name = []
        groups_cpu = []
        table_name_list = []
        
        for tab_name in plan:
            groups.append(table_dict[tab_name]["scan_list"])
            groups_cpu.append(table_dict[tab_name]["cpu_list"])
            groups_name.append(table_dict[tab_name]["col_name_list"])
            table_name_list.append(tab_name)
        
        ban_set = get_join_key(plan, name_to_colid)
        best_sum, best_path = greedy_grouped_sum(groups, tar_scan, ban_set)
        if best_sum < tar_scan * 0.95:
            continue
        solve_path = defaultdict()
        best_cpu = 0
        for i, path in enumerate(best_path):
            tmp = []
            for p in path:
                tmp.append(groups_name[i][p])
                best_cpu += groups_cpu[i][p]
            solve_path[table_name_list[i]] = tmp
        all_ans.append((best_sum, best_cpu, solve_path))
    all_ans = sorted(all_ans, key=lambda x: x[0])
    return all_ans[:k]
    
def select_table(scan, cpu, database, ops, k):
    # select table
    states = read_database_state(database)
    table_dict = build_table_dict(states)
    join_num = ops["join"]
    
    combine_tables = select_join_tables(scan, cpu, join_num, table_dict, k)
    
    return combine_tables