"""题目三：Minimax与Alpha-Beta；共享节点按每次递归访问计数，不做搜索缓存。

运行：py -m src.game_tree
"""
import json
import math
from dataclasses import dataclass


class TreeNode:
    # 大规模均匀树只保留必要字段，避免实例字典带来的额外内存开销。
    __slots__ = ("value", "children", "name", "is_maximizing_player")

    def __init__(self, value=None, children=None, name=None, is_maximizing_player=None):
        self.value = value
        self.children = children if children is not None else []
        self.name = name
        self.is_maximizing_player = is_maximizing_player


@dataclass
class GameStats:
    nodes: int = 0       # 递归访问次数，包含根、内部节点和叶子；共享节点可重复计入
    leaves: int = 0      # 实际评估叶子次数，按访问次数而非不同叶子对象数计数
    prunes: int = 0      # 确实跳过至少一个后续子节点的剪枝事件数
    shallow_prunes: int = 0   # 致剪界由直接父节点严格收紧
    deep_prunes: int = 0      # 致剪界由更高祖先严格收紧
    external_prunes: int = 0  # 调用方传入有限界但未提供来源，不能强行算作祖先


def node_role(node, maximizing):
    """JSON显式角色优先；手工构建节点默认按父节点角色交替。"""
    return maximizing if node.is_maximizing_player is None else node.is_maximizing_player


def _leaf_value(node):
    if not isinstance(node.value, (int, float)) or isinstance(node.value, bool) or not math.isfinite(node.value):
        raise ValueError("叶子必须有有限数值value，内部节点必须有非空children")
    return node.value


def from_json(obj):
    """读取dict或JSON文本，支持tree_structure外层；显式角色优先，缺省按父角色交替。"""
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        raise ValueError("树的JSON必须为对象")
    if "tree_structure" in obj:
        obj = obj["tree_structure"]

    def parse(data, maximizing, path):
        if not isinstance(data, dict):
            raise ValueError("每个节点必须为JSON对象")
        role = data.get("is_maximizing_player", maximizing)
        if not isinstance(role, bool):
            raise ValueError("is_maximizing_player必须为布尔值")
        children = data.get("children", [])
        if not isinstance(children, list):
            raise ValueError("children必须为列表")
        node = TreeNode(data.get("value"), name=data.get("name", path), is_maximizing_player=role)
        node.children = [parse(child, not role, f"{path}/{i}") for i, child in enumerate(children)]
        if not children:
            _leaf_value(node)
        return node

    return parse(obj, True, "root")


def build_test_tree():
    """严格保留题面中min1/min2共同引用的同一个max2对象。"""
    leaf1, leaf2, leaf3, leaf4, leaf5, leaf6 = [
        TreeNode(value=v, name=f"leaf{i}({v})") for i, v in enumerate((3, 5, 6, 9, 1, 2), 1)]
    max1 = TreeNode(children=[leaf1, leaf2], name="max1")
    max2 = TreeNode(children=[leaf3, leaf4], name="max2")
    max3 = TreeNode(children=[leaf5, leaf6], name="max3")
    min1 = TreeNode(children=[max1, max2], name="min1")
    min2 = TreeNode(children=[max2, max3], name="min2")
    return TreeNode(children=[min1, min2], name="root")


def minimax(node, maximizing, stats):
    """完整Minimax；不缓存共享子树，统计实际节点/叶子访问次数。"""
    stats.nodes += 1
    if not node.children:
        stats.leaves += 1
        return _leaf_value(node)
    maximizing = node_role(node, maximizing)
    values = (minimax(child, not maximizing, stats) for child in node.children)
    return max(values) if maximizing else min(values)


def alphabeta(node, alpha, beta, maximizing, stats, order=None, trace=None, *,
              depth=0, alpha_depth=None, beta_depth=None):
    """Fail-soft Alpha-Beta，alpha>=beta时剪枝，默认从左到右。

    order(node, maximizing)返回重排后的完整子节点序列，不应修改node.children。
    stats为GameStats。trace若为列表，记录剪枝节点、角色、alpha/beta及被跳过
    的直接子节点名称；末个孩子后即使窗口闭合也未跳过任何节点，不计剪枝。
    根为第0层；alpha_depth/beta_depth只在对应界严格收紧时改为当前层，等值
    保留旧来源。MAX剪枝追溯beta，MIN剪枝追溯alpha；来源为depth-1是浅剪枝，
    更小则为深剪枝。此跟踪不改变任何界值、排序、返回值或剪枝判定。
    """
    stats.nodes += 1
    if not node.children:
        stats.leaves += 1
        return _leaf_value(node)
    maximizing = node_role(node, maximizing)
    children = node.children if order is None else list(order(node, maximizing))
    value = -math.inf if maximizing else math.inf
    for i, child in enumerate(children):
        score = alphabeta(child, alpha, beta, not maximizing, stats, order, trace,
                          depth=depth+1, alpha_depth=alpha_depth, beta_depth=beta_depth)
        if maximizing:
            value = max(value, score)
            if value > alpha:
                alpha_depth = depth
            alpha = max(alpha, value)
        else:
            value = min(value, score)
            if value < beta:
                beta_depth = depth
            beta = min(beta, value)
        if alpha >= beta and i + 1 < len(children):
            stats.prunes += 1
            source_depth = beta_depth if maximizing else alpha_depth
            if source_depth == depth - 1:
                cutoff_type = "浅剪枝"
                stats.shallow_prunes += 1
            elif source_depth is not None and source_depth < depth - 1:
                cutoff_type = "深剪枝"
                stats.deep_prunes += 1
            else:
                cutoff_type = "外部界"
                stats.external_prunes += 1
            if trace is not None:
                trace.append({"节点": node.name or "未命名节点", "角色": "MAX" if maximizing else "MIN",
                              "alpha": alpha, "beta": beta,
                              "被剪子节点": [c.name if c.name is not None else f"child[{j}]"
                                            for j, c in enumerate(children[i+1:], i+1)],
                              "节点层数": depth, "alpha来源层": alpha_depth, "beta来源层": beta_depth,
                              "致剪界": "beta" if maximizing else "alpha",
                              "界来源层": source_depth, "剪枝类型": cutoff_type})
            break
    return value


def minimax_values(root, maximizing=True):
    """预计算真实值供理想排序使用；不计入搜索统计，缓存键包含有效角色。"""
    cache = {}

    def evaluate(node, role):
        role = node_role(node, role)
        key = (node, role)
        if key not in cache:
            if not node.children:
                cache[key] = _leaf_value(node)
            else:
                values = [evaluate(child, not role) for child in node.children]
                cache[key] = max(values) if role else min(values)
        return cache[key]

    evaluate(root, maximizing)
    # 递归闭包不再需要，断开自引用，避免大树的真实值缓存滞留到循环GC。
    evaluate = None
    return cache


def oracle_order(values, best=True):
    """MAX最优降序、MIN最优升序；反向排序取反，排序前提是真实值已预计算。"""
    def order(node, maximizing):
        return sorted(node.children,
                      key=lambda child: values[(child, node_role(child, not maximizing))],
                      reverse=maximizing if best else not maximizing)
    return order


def solve_game_tree(root) -> int:
    """按题面从左到右求MAX根的最佳得分；数值原样返回，不截断浮点实验得分。"""
    return alphabeta(root, -math.inf, math.inf, True, GameStats())


if __name__ == "__main__":
    root = build_test_tree()
    full, pruned, trace = GameStats(), GameStats(), []
    print("Minimax最佳得分:", minimax(root, True, full))
    print("Minimax统计:", full)
    print("Alpha-Beta最佳得分:", alphabeta(root, -math.inf, math.inf, True, pruned, trace=trace))
    print("Alpha-Beta从左到右统计:", pruned)
    print("剪枝记录:")
    for event in trace:
        print(event)
