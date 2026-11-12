"""题一测试：给定用例、边界情况、随机地图上与 BFS 对拍。

运行：python -m pytest tests/test_q1.py -q
"""
import random

import pytest

from src.astar import TIE_BREAKS, astar, manhattan, zero_heuristic
from src.grid_utils import bfs_shortest_length, maze_map, pick_pair, random_map, u_trap_map
from src.landmarks import LandmarkHeuristic
from src.q1_maze import TEST_END, TEST_GRID, TEST_START, solve_maze


def is_valid_path(path, start, goal, grid):
    """路径合法：首尾正确、每步相邻、不踩障碍。"""
    if not path or path[0] != start or path[-1] != goal:
        return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    return all(grid[r][c] != "X" for r, c in path)


def test_given_case():
    path = solve_maze(TEST_START, TEST_END, TEST_GRID)
    # 最短路径可能不唯一，所以检查长度与合法性，而不是与预期输出逐项相等
    assert len(path) == 8
    assert is_valid_path(path, TEST_START, TEST_END, TEST_GRID)


def test_no_path():
    grid = [["S", "X", "."],
            ["X", "X", "."],
            [".", ".", "E"]]
    assert solve_maze((0, 0), (2, 2), grid) == []


def test_start_equals_goal():
    assert solve_maze((1, 1), (1, 1), [[".", "."], [".", "."]]) == [(1, 1)]


def test_start_or_goal_blocked():
    grid = [["X", "."], [".", "X"]]
    assert solve_maze((0, 0), (0, 1), grid) == []
    assert solve_maze((0, 1), (1, 1), grid) == []


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
@pytest.mark.parametrize("heuristic", [zero_heuristic, manhattan])
def test_against_bfs_random(heuristic, tie_break):
    """200 张随机地图上，所有启发函数 × 平局规则的最短路长度都必须与 BFS 一致。"""
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randint(5, 25)
        grid = random_map(n, rng.uniform(0.0, 0.4), rng)
        s = (rng.randrange(n), rng.randrange(n))
        g = (rng.randrange(n), rng.randrange(n))
        expect = bfs_shortest_length(grid, s, g)
        res = astar(s, g, grid, heuristic=heuristic, tie_break=tie_break, seed=rng.randrange(10**6))
        if expect < 0:
            assert res.path == []
        else:
            assert len(res.path) - 1 == expect
            assert is_valid_path(res.path, s, g, grid)


@pytest.mark.parametrize("method", ["random", "farthest"])
def test_landmarks_against_bfs(method):
    """地标启发函数在迷宫、U 型陷阱、随机地图上都必须给出最优解（验证可采纳性）。"""
    rng = random.Random(7)
    for i in range(40):
        kind = i % 3
        if kind == 0:
            grid = maze_map(31, rng)
            s, g = pick_pair(grid, rng, 10)
        elif kind == 1:
            grid, s, g = u_trap_map(31, rng)
        else:
            grid = random_map(31, 0.3, rng)
            pair = pick_pair(grid, rng, 10)
            if pair is None:
                continue
            s, g = pair
        h = LandmarkHeuristic(grid, k=rng.choice([1, 4, 8]), method=method, rng=rng)
        # 可采纳性的直接检查：起点处 h 不超过真实距离
        true_d = bfs_shortest_length(grid, s, g)
        assert h(s, g) <= true_d
        res = astar(s, g, grid, heuristic=h)
        assert len(res.path) - 1 == true_d


def test_spearman_average_rank():
    # 并列样本必须使用平均名次，避免 Spearman 相关随输入顺序改变。
    from experiments.q1_exp_b_maptypes import rank
    assert rank([1, 2, 2, 3]).tolist() == [0, 1.5, 1.5, 3]


@pytest.mark.parametrize("method", ["random", "farthest"])
def test_landmark_prefix(method):
    # 相同种子只改变 k 时，地标应取同一序列的前缀。
    grid = [["."] * 6 for _ in range(6)]
    h4 = LandmarkHeuristic(grid, 4, method, random.Random(42))
    h8 = LandmarkHeuristic(grid, 8, method, random.Random(42))
    assert h4.landmarks == h8.landmarks[:4]


@pytest.mark.parametrize("method", ["random", "farthest"])
@pytest.mark.parametrize("k,grid", [(0, [["."]]), (-1, [["."]]),
                                  (4, [["X"]]), (4, [])])
def test_landmark_empty_fallback(method, k, grid):
    # 非正 k 和无可通行格都应安全退化，不能对空候选集合随机取点。
    h = LandmarkHeuristic(grid, k, method, random.Random(42))
    assert h.landmarks == []
    assert h((0, 0), (2, 3)) == 5


@pytest.mark.parametrize("method", ["random", "farthest"])
def test_landmarks_in_largest_component(method):
    # 左侧 2 格封闭小区域与右侧 8 格区域断开，地标不能挤进小区域。
    grid = [list(".X...."), list(".X....")]
    largest = {(r, c) for r in range(2) for c in range(2, 6)}
    h = LandmarkHeuristic(grid, 20, method, random.Random(1))
    assert set(h.landmarks) == largest
    assert len(h.landmarks) == len(largest)
    assert h((0, 0), (1, 0)) == 1  # 区域外查询仍安全退化为曼哈顿。


@pytest.mark.parametrize("method", ["random", "farthest"])
def test_landmarks_unrestricted_can_select_small_component(method):
    # 种子2的最远点法从左侧小区域开始；random取满全部空格也必然覆盖小区域。
    grid = [list(".X...."), list(".X....")]
    h = LandmarkHeuristic(grid, 20, method, random.Random(2),
                          restrict_to_largest_component=False)
    assert any(c == 0 for r, c in h.landmarks)
    assert h.outside_largest_component_count == sum(c == 0 for r, c in h.landmarks)
    with pytest.raises(AttributeError):
        h.outside_largest_component_count = 0  # 统计属性不可写，避免消融记录被覆盖。


@pytest.mark.parametrize("method", ["random", "farthest"])
def test_unrestricted_landmark_prefix(method):
    # 不限制区域时仍只改变候选范围，不能同时破坏前缀嵌套这一控制条件。
    grid = [list(".X...."), list(".X....")]
    h4 = LandmarkHeuristic(grid, 4, method, random.Random(42),
                           restrict_to_largest_component=False)
    h8 = LandmarkHeuristic(grid, 8, method, random.Random(42),
                           restrict_to_largest_component=False)
    assert h4.landmarks == h8.landmarks[:4]
