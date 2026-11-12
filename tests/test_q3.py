"""题三：共享子树访问、JSON角色、随机树对拍与理想排序的精确叶子计数。"""
import math
import random

import pytest

from src.game_tree import (GameStats, TreeNode, alphabeta, build_test_tree, from_json,
                           minimax, minimax_values, oracle_order, solve_game_tree)


def test_q3_given_shared_tree():
    root = build_test_tree()
    assert root.children[0].children[1] is root.children[1].children[0]
    assert solve_game_tree(root) == 5
    full, pruned, trace = GameStats(), GameStats(), []
    assert minimax(root, True, full) == alphabeta(root, -math.inf, math.inf, True, pruned, trace=trace) == 5
    assert (full.leaves, full.nodes) == (8, 15)
    assert (pruned.leaves, pruned.nodes, pruned.prunes) == (7, 14, 1)
    assert trace == [{"节点": "max2", "角色": "MAX", "alpha": 6, "beta": 5, "被剪子节点": ["leaf4(9)"],
                      "节点层数": 2, "alpha来源层": 2, "beta来源层": 1,
                      "致剪界": "beta", "界来源层": 1, "剪枝类型": "浅剪枝"}]
    assert (pruned.shallow_prunes, pruned.deep_prunes, pruned.external_prunes) == (1, 0, 0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_q3_json(wrapped):
    max1 = {"children": [{"value": 3}, {"value": 5}]}
    max2 = {"children": [{"value": 6}, {"value": 9}]}
    max3 = {"children": [{"value": 1}, {"value": 2}]}
    tree = {"is_maximizing_player": True,
            "children": [{"children": [max1, max2]}, {"children": [max2, max3]}]}
    # JSON没有对象共享语义；重复展开max2与题面共享对象的数值、访问次数相同。
    root = from_json({"tree_structure": tree} if wrapped else tree)
    assert solve_game_tree(root) == 5


def test_q3_json_explicit_role_overrides_alternation():
    root = from_json({"children": [
        {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 9}]},
        {"children": [{"value": 4}, {"value": 5}]}]})
    assert solve_game_tree(root) == 9  # 显式MAX不能按层默认当作MIN。


def test_q3_prunes_on_equality_and_does_not_mutate_children():
    root = TreeNode(children=[TreeNode(children=[TreeNode(5)]),
                              TreeNode(children=[TreeNode(5), TreeNode(9)])])
    stats = GameStats()
    assert alphabeta(root, -math.inf, math.inf, True, stats) == 5
    assert stats.leaves == 2 and stats.prunes == 1
    original = tuple(root.children)
    alphabeta(root, -math.inf, math.inf, True, GameStats(), order=lambda n, m: reversed(n.children))
    assert tuple(root.children) == original


def test_q3_random_trees_against_minimax():
    rng = random.Random(2026)

    def generate(depth):
        if depth == 0 or rng.random() < .18:
            return TreeNode(rng.randint(-100, 100))
        return TreeNode(children=[generate(depth - 1) for _ in range(rng.randint(1, 4))])

    for _ in range(300):
        depth = rng.randint(1, 6)
        root = TreeNode(children=[generate(depth - 1) for _ in range(rng.randint(1, 4))])
        expected = minimax(root, True, GameStats())
        values = minimax_values(root)

        def shuffled(node, maximizing):
            children = node.children.copy()
            rng.shuffle(children)
            return children

        for order in (None, lambda n, m: reversed(n.children), shuffled,
                      oracle_order(values, True), oracle_order(values, False)):
            stats = GameStats()
            assert alphabeta(root, -math.inf, math.inf, True, stats, order=order) == expected
            assert stats.prunes == stats.shallow_prunes + stats.deep_prunes
            assert stats.external_prunes == 0


@pytest.mark.parametrize("maximizing", [True, False])
def test_q3_cutoff_depth_tracks_alpha_and_beta(maximizing):
    # 首个分支在第0层设界，另一分支第3层发生深剪枝、第1层发生浅剪枝。
    # 整树取负并交换MAX/MIN，分别覆盖alpha与beta作为致剪界。
    sign = 1 if maximizing else -1
    deep = TreeNode(children=[TreeNode(sign), TreeNode(2*sign)], name="deep")
    middle = TreeNode(children=[deep, TreeNode(3*sign)], name="middle")
    branch = TreeNode(children=[middle, TreeNode(4*sign)], name="branch")
    root = TreeNode(children=[TreeNode(5*sign), branch], name="root")
    stats, trace = GameStats(), []
    assert alphabeta(root, -math.inf, math.inf, maximizing, stats, trace=trace) == 5*sign
    assert (stats.shallow_prunes, stats.deep_prunes, stats.external_prunes) == (1, 1, 0)
    event = next(e for e in trace if e["节点"] == "deep")
    assert (event["节点层数"], event["界来源层"], event["剪枝类型"]) == (3, 0, "深剪枝")
    assert event["致剪界"] == ("alpha" if maximizing else "beta")


@pytest.mark.parametrize("maximizing", [True, False])
def test_q3_reverse_order_directions(maximizing):
    # 反向排序为MAX升序、MIN降序，不能只把根节点的顺序反转。
    root = TreeNode(children=[TreeNode(3), TreeNode(1), TreeNode(2)])
    values = minimax_values(root, maximizing)
    actual = [child.value for child in oracle_order(values, best=False)(root, maximizing)]
    assert actual == ([1, 2, 3] if maximizing else [3, 2, 1])


@pytest.mark.parametrize("b", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 6])
def test_q3_perfect_order_exact_leaf_count(b, d):
    rng = random.Random(b * 100 + d)
    leaves = list(range(b**d))
    rng.shuffle(leaves)
    iterator = iter(leaves)

    def build(depth):
        return TreeNode(next(iterator)) if depth == 0 else TreeNode(children=[build(depth - 1) for _ in range(b)])

    root = build(d)
    values = minimax_values(root)
    stats = GameStats()
    result = alphabeta(root, -math.inf, math.inf, True, stats, order=oracle_order(values))
    assert result == values[(root, True)]
    assert stats.leaves == b**((d+1)//2) + b**(d//2) - 1
