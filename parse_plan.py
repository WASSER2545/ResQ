import json
from typing import Dict, List, Optional


class PlanNode:
    def __init__(
        self,
        node_id: int,
        parent_id: Optional[int],
        name: str,
        title: Optional[str] = None,
        statistics: Optional[List[dict]] = None,
        metrics: Optional[dict] = None,
        labels: Optional[List[dict]] = None,
    ):
        self.node_id = node_id
        self.parent_id = parent_id
        self.name = name
        self.title = title

        self.statistics = statistics or []
        self.metrics = metrics or {}
        self.labels = labels or []

        self.children: List["PlanNode"] = []

        # join-specific (optional)
        self.join_build_keys = None
        self.join_probe_keys = None
        self.join_type = None

        self._parse_join_labels()

    def _parse_join_labels(self):
        if self.name.lower().endswith("join"):
            for label in self.labels:
                if label["name"] == "Join Build Side Keys":
                    self.join_build_keys = label["value"]
                elif label["name"] == "Join Probe Side Keys":
                    self.join_probe_keys = label["value"]
                elif label["name"] == "Join Type":
                    self.join_type = label["value"][0]

    def add_child(self, child: "PlanNode"):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"{self.name}(id={self.node_id})"


class PlanTree:
    def __init__(self, root: PlanNode):
        self.root = root

    def dfs(self, node: Optional[PlanNode] = None):
        if node is None:
            node = self.root
        yield node
        for c in node.children:
            yield from self.dfs(c)

    def pretty_print(self):
        def _print(node: PlanNode, depth: int):
            indent = "  " * depth
            desc = node.name
            if node.title:
                desc += f" [{node.title}]"
            print(f"{indent}- {desc}")

            if node.name.lower().endswith("join"):
                print(f"{indent}  JoinType: {node.join_type}")
                print(f"{indent}  BuildKeys: {node.join_build_keys}")
                print(f"{indent}  ProbeKeys: {node.join_probe_keys}")

            for c in node.children:
                _print(c, depth + 1)

        _print(self.root, 0)


def parse_plan_nodes(plan_json: List[dict]) -> PlanTree:
    """
    Parse raw JSON plan log into a PlanTree
    """

    # 1. create nodes
    nodes: Dict[int, PlanNode] = {}
    for n in plan_json:
        node = PlanNode(
            node_id=n["id"],
            parent_id=n.get("parent_id"),
            name=n["name"],
            title=n.get("title"),
            statistics=n.get("statistics"),
            metrics=n.get("metrics"),
            labels=n.get("labels"),
        )
        nodes[node.node_id] = node

    # 2. link parent-child
    root = None
    for node in nodes.values():
        if node.parent_id is None:
            root = node
        else:
            parent = nodes[node.parent_id]
            parent.add_child(node)

    if root is None:
        raise ValueError("No root node (parent_id == None) found")

    # 3. normalize join children order (optional but recommended)
    _normalize_join_children(root)

    return PlanTree(root)


def _normalize_join_children(node: PlanNode):
    """
    Ensure join children order:
    - children[0] = build side
    - children[1] = probe side

    Heuristic:
    - smaller output_rows → build side
    """
    if node.name.lower().endswith("join") and len(node.children) == 2:
        left, right = node.children

        left_rows = _get_output_rows(left)
        right_rows = _get_output_rows(right)

        if left_rows is not None and right_rows is not None:
            if left_rows > right_rows:
                node.children = [right, left]

    for c in node.children:
        _normalize_join_children(c)

def _get_output_rows(node: PlanNode) -> Optional[float]:
    return node.statistics[4]