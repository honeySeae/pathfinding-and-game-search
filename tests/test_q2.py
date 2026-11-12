"""题二：累计代价、方向性、启发函数及300张随机图五种平局规则对拍。"""
import random

import pytest

from src.astar import TIE_BREAKS, astar, manhattan, zero_heuristic
from src.grid_utils import all_distances, neighbors
from src.q2_terrain import (TEST_END, TEST_GRID, TEST_START, make_terrain_cost,
                            path_cost_trace, solve_terrain, terrain_heuristic)


def test_given_terrain():
    assert solve_terrain(TEST_START, TEST_END, TEST_GRID) == 7
    trace = path_cost_trace(TEST_START, TEST_END, TEST_GRID)
    assert trace[0] == (TEST_START, 0, 0)
    assert trace[-1] == (TEST_END, 0, 7)
    assert [point for point, _, _ in trace] == [(0, 0), (1, 0), (2, 0), (3, 0),
                                               (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]


def test_terrain_same_endpoint():
    assert solve_terrain((0, 0), (0, 0), [["S"]]) == 0
    assert path_cost_trace((0, 0), (0, 0), [["S"]]) == [((0, 0), 0, 0)]


def test_terrain_disconnected():
    grid = [["S", "X", "E"]]
    assert solve_terrain((0, 0), (0, 2), grid) == -1
    assert path_cost_trace((0, 0), (0, 2), grid) == []


@pytest.mark.parametrize("cell,expected", [("1", 1), ("3", 3), ("8", 8),
                                           ("E", 0), ("S", 2), ("X", None), (4, 4)])
def test_terrain_cost(cell, expected):
    assert make_terrain_cost([["S", "2", "8", "E"]])(cell) == expected


def assert_heuristic_bounds(grid, goal):
    cost = make_terrain_cost(grid)
    h = terrain_heuristic(grid)
    assert h(goal, goal) == 0
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "X":
                continue
            node = (r, c)
            # 进入代价有方向性：逐格正向Dijkstra求到目标的真实剩余代价，
            # 不能把从终点正向跑出的距离误当作剩余代价。
            true_remaining = all_distances(grid, node, cost).get(goal, float("inf"))
            assert 0 <= h(node, goal) <= true_remaining
            for nbr, weight in neighbors(grid, node, cost):
                assert h(node, goal) <= weight + h(nbr, goal)


def test_heuristic_includes_reentry_to_start():
    grid = [["2", "S", "2", "E"]]
    assert make_terrain_cost(grid)("S") == 2
    assert all_distances(grid, (0, 0), make_terrain_cost(grid))[(0, 3)] == 4
    assert_heuristic_bounds(grid, (0, 3))


def test_heuristic_admissibility_and_consistency():
    rng = random.Random(2026)
    for _ in range(35):
        n = rng.randint(3, 7)
        # 同时覆盖最小数字代价>1的地图，验证S采用w_min的必要性。
        choices = (2, 3, 8) if rng.random() < .5 else (1, 1, 2, 3, 8)
        grid = [[str(rng.choice(choices)) for _ in range(n)] for _ in range(n)]
        grid[0][0], grid[-1][-1] = "S", "E"
        assert_heuristic_bounds(grid, (n - 1, n - 1))


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_weighted_terrain_against_dijkstra(tie_break):
    rng = random.Random(2026)
    for _ in range(300):
        n = rng.randint(4, 12)
        grid = [[str(rng.choice((1, 1, 1, 2, 3, 8))) for _ in range(n)] for _ in range(n)]
        start, end = rng.sample([(r, c) for r in range(n) for c in range(n)], 2)
        grid[start[0]][start[1]], grid[end[0]][end[1]] = "S", "E"
        cost = make_terrain_cost(grid)
        seed = rng.randrange(2**32)
        expected = astar(start, end, grid, cost_fn=cost, heuristic=zero_heuristic,
                         tie_break=tie_break, seed=seed, goal_cost_override=0).cost
        for h in (terrain_heuristic(grid), manhattan):
            result = astar(start, end, grid, cost_fn=cost, heuristic=h,
                           tie_break=tie_break, seed=seed, goal_cost_override=0)
            assert result.cost == expected
