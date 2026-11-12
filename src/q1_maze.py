"""题一：迷宫寻路（A* + 曼哈顿距离）。

运行：python -m src.q1_maze
"""
from typing import List, Tuple

from .astar import astar, manhattan, zero_heuristic
from .grid_utils import unit_cost


def solve_maze(start: Tuple[int, int], end: Tuple[int, int],
               grid: List[List[str]]) -> List[Tuple[int, int]]:
    """按作业要求的接口：返回从起点到终点的最短路径坐标序列，无解返回 []。"""
    return astar(start, end, grid, cost_fn=unit_cost, heuristic=manhattan).path


TEST_START = (0, 0)
TEST_END = (3, 4)
TEST_GRID = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]


if __name__ == "__main__":
    path = solve_maze(TEST_START, TEST_END, TEST_GRID)
    print("最短路径:", path)
    print("步数:", len(path) - 1)
    for name, h in [("h=0", zero_heuristic), ("曼哈顿", manhattan)]:
        res = astar(TEST_START, TEST_END, TEST_GRID, heuristic=h)
        print(f"{name:6s} 扩展节点数={res.stats.expanded:3d}  入堆次数={res.stats.pushed:3d}")