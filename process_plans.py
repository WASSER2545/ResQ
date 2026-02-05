import re
from collections import defaultdict
idx = 0

ban_set = set(["wait time", "pruning stats"])

class QueryPlanNode:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.children = []
        self.is_leaf = True

    def add_attribute(self, key, value):
        if key not in ban_set:
            self.attributes[key] = value

    def add_child(self, child_node):
        self.children.append(child_node)
        
def process_attribute(node):
    global idx
    node.name += f' (operator {idx}) '
    idx += 1
    ret = [node.name]
    for k, v in node.attributes.items():
        ret.append(f'{k}: {v}')
    return '\n  '.join(ret)
def merge_op(node):
    ret = [process_attribute(node), '\n father of:']
    for child in node.children:
        ret.append(child.name)
    return ' '.join(ret)
def DFS(cur):
    all_state = []
    if cur.is_leaf:
        return process_attribute(cur)
    else:
        for child in cur.children:
            all_state.append(DFS(child))
        all_state.append(merge_op(cur))
    return '\n'.join(all_state)
        
def parse_query_plan(text):
    lines = text.strip().split('\n')
    root = None
    stack = []
    current_indent = -1

    for line in lines:
        indent = len(re.match(r'^[\s│└├]*', line).group(0))
        line = line.strip().lstrip('├─└│ ')
        
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            if key == 'output columns':
                value = parse_out_cols(value)
        else:
            key, value = line, None

        if not root:
            root = QueryPlanNode(key)
            stack.append((root, indent))
            current_indent = indent
        else:
            while stack and stack[-1][1] >= indent:
                stack.pop()
            parent_node, _ = stack[-1]

            if value is not None:
                parent_node.add_attribute(key, value)
            else:
                child_node = QueryPlanNode(key)
                parent_node.add_child(child_node)
                stack.append((child_node, indent))
                parent_node.is_leaf = False

    return root

def extract_time_values(text):
    pattern = r'(\d+\.?\d*)\s*(ns|ms|µs|s)'
    match = re.search(pattern, text)
    if match:
        value = match.group(1)
        unit = match.group(2)
        return float(value), unit
    return None

def extract_storage_values(text):
    pattern = r'(\d+\.?\d*)\s*(B|KiB|MiB|GiB|TiB|PiB)'
    match = re.search(pattern, text)
    if match:
        value = match.group(1)
        unit = match.group(2) 
        return float(value), unit
    return None

def extract_card_values(text):
    pattern = r'(\d+\.?\d*)\s*(hundred|thousand|million|billion|trillion)?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value, unit
    return None
    
def get_cost(cur):
    cpu, scan = 0, 0
    if cur.is_leaf:
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        return cpu, scan
    else:
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        for child in cur.children:
            x, y = get_cost(child)
            cpu += x
            scan += y
    return cpu, scan

def parse_out_cols(text):
    cols = re.findall(r'([\w]+(?:\.[\w]+)?)\s*\(#\d+\)', text)
    return cols

def get_cost_card(cur):
    cpu, scan = 0, 0
    card = 0
    join_cost_list = []
    if cur.is_leaf:
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        return cpu, scan, join_cost_list
    else:
        node_name = cur.name.lower()
        if 'join' in node_name:
            val = cur.attributes.get('estimated rows')
            card += float(val)
            join_cost_list.append(card)            
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        for child in cur.children:
            x, y, cost_list = get_cost_card(child)
            cpu += x
            scan += y
            join_cost_list.extend(cost_list)
    return cpu, scan, join_cost_list

def get_estimation(cur):
    cpu, scan = 0, 0
    join_cost_list = []
    join_cols_list = []
    if cur.is_leaf:
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        return cpu, scan, join_cost_list, join_cols_list
    else:
        node_name = cur.name.lower()
        if 'join' in node_name:
            val = cur.attributes.get('estimated rows')
            
            build_side = cur.children[0]
            probe_side = cur.children[1]
            if 'Build' not in build_side.name:
                build_side, probe_side = probe_side, build_side
            
            build_rows = build_side.attributes.get('estimated rows')
            probe_rows = probe_side.attributes.get('estimated rows')
            join_cost_list.append([val, build_rows, probe_rows])
            
            out_cols = cur.attributes["output columns"]
            build_cols = build_side.attributes["output columns"]
            if "TableScan" in build_side.name:
                table_name = build_side.attributes["table"].split(".")[-1]
                for i in range(len(build_cols)):
                    build_cols[i] = table_name + "." + build_cols[i]
            
            join_cols_list.append([out_cols, build_cols])
            
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        for child in cur.children:
            x, y, cost_list, cols_list = get_estimation(child)
            cpu += x
            scan += y
            join_cost_list.extend(cost_list)
            join_cols_list.extend(cols_list)
    return cpu, scan, join_cost_list, join_cols_list

def parse_operator_info(cur):
    cpu, scan = 0, 0
    join_cost_list = []
    join_cols_list = []
    eval_types_cnt = []
    sort_key_list = []
    if cur.is_leaf:
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        res = {
            "cpu": cpu,
            "scan": scan,
            "join_cost": join_cost_list,
            "join_cols": join_cols_list,
            "eval_type_cnt": eval_types_cnt,
            "sort_key_list": sort_key_list
        }
        return res
    else:
        node_name = cur.name.lower()
        if 'join' in node_name:
            val = cur.attributes.get('estimated rows')
            
            build_side = cur.children[0]
            probe_side = cur.children[1]
            if 'Build' not in build_side.name:
                build_side, probe_side = probe_side, build_side
            
            build_rows = build_side.attributes.get('estimated rows')
            probe_rows = probe_side.attributes.get('estimated rows')
            join_cost_list.append([val, build_rows, probe_rows])
            
            out_cols = cur.attributes["output columns"]
            build_cols = build_side.attributes["output columns"]
            if "TableScan" in build_side.name:
                table_name = build_side.attributes["table"].split(".")[-1]
                for i in range(len(build_cols)):
                    build_cols[i] = table_name + "." + build_cols[i]
            
            join_cols_list.append([out_cols, build_cols])
        
        if 'eval' in node_name:
            op_cnt = defaultdict()
            exps = cur.attributes["expressions"]
            op_cnt["simple"] = exps.count("sin")
            op_cnt["date"] = exps.count("to_start_of_month")
            op_cnt["str"] = exps.count("md5")
            
            eval_types_cnt.append(op_cnt)
            
        if 'sort' in node_name:
            keys = cur.attributes["sort keys"][1:-1]
            key_list = [key.split()[0] for key in keys.split(',')]
            sort_key_list.append(key_list)
            
        for k, v in cur.attributes.items():
            if k == "cpu time":
                val, unit = extract_time_values(v)
                if unit == "ms":
                    cpu += float(val) / 1000
                elif unit == "µs":
                    cpu += float(val) / 1000000
                elif unit == "ns":
                    cpu += float(val) / 1000000000
                else:
                    cpu += float(val)
            if k == "bytes scanned":
                val, unit = extract_storage_values(v)
                if unit == "MiB":
                    scan += float(val) * (1024 ** 2)
                elif unit == "KiB":
                    scan += float(val) * 1024
                else:
                    scan += float(val)
        for child in cur.children:
            chi = parse_operator_info(child)
            cpu += chi["cpu"]
            scan += chi["scan"]
            join_cost_list.extend(chi["join_cost"])
            join_cols_list.extend(chi["join_cols"])
            eval_types_cnt.extend(chi["eval_type_cnt"])
            sort_key_list.extend(chi["sort_key_list"])
    res = {
            "cpu": cpu,
            "scan": scan,
            "join_cost": join_cost_list,
            "join_cols": join_cols_list,
            "eval_type_cnt": eval_types_cnt,
            "sort_key_list": sort_key_list
        }
    return res