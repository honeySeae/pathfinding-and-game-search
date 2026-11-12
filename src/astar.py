"""通用 A* 搜索（题一、题二共用）。

f(n) = g(n) + h(n)
- g(n)：从起点到 n 的已知最小代价
- h(n)：从 n 到终点的估计代价（启发函数）
- open 表：二叉堆，按 (f, 平局键, ...) 排序
- closed 表：已扩展节点集合；堆中的过期条目在弹出时跳过（惰性删除）

代价函数、启发函数、平局规则都是参数，方便做对照实验。
"""
import heapq
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .grid_utils import Coord, CostFn, Grid, neighbors, unit_cost

Heuristic = Callable[[Coord, Coord], float]


# ---------------------------------------------------------------- 启发函数

def zero_heuristic(node: Coord, goal: Coord) -> float:
    """h = 0：A* 退化为 Dijkstra（单位代价时等价于 BFS 的扩展顺序）。"""
    return 0


def manhattan(node: Coord, goal: Coord) -> float:
    """曼哈顿距离：四连通、单位代价下是可采纳且一致的。"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


# ---------------------------------------------------------------- 平局规则

TIE_BREAKS = ("h_low", "h_high", "fifo", "lifo", "random")


def _make_key(tie_break: str, rng: random.Random):
    """返回 key(f, h, counter) -> 堆排序键。f 相同时由第二项决定先后。

    h_low : 优先 h 小（等价于优先 g 大），倾向"冲向终点"
    h_high: 优先 h 大（等价于优先 g 小），倾向"横向铺开"
    fifo  : 先入先出
    lifo  : 后入先出
    random: 随机
    """
    if tie_break == "h_low":
        return lambda f, h, cnt: (f, h, cnt)
    if tie_break == "h_high":
        return lambda f, h, cnt: (f, -h, cnt)
    if tie_break == "fifo":
        return lambda f, h, cnt: (f, cnt)
    if tie_break == "lifo":
        return lambda f, h, cnt: (f, -cnt)
    if tie_break == "random":
        return lambda f, h, cnt: (f, rng.random(), cnt)
    raise ValueError(f"未知的平局规则: {tie_break}")


# ---------------------------------------------------------------- 结果结构

@dataclass
class SearchStats:
    expanded: int = 0            # 扩展（出堆并展开邻居）的节点数
    pushed: int = 0              # 入堆次数（含重复入堆）
    max_open: int = 0            # open 表（堆）的最大长度
    time_ms: float = 0.0
    expanded_order: List[Coord] = field(default_factory=list)   # 扩展顺序，用于热力图
    expanded_f: List[float] = field(default_factory=list)       # 每个被扩展节点当时的 f 值


@dataclass
class SearchResult:
    path: List[Coord]            # 无解时为空列表
    cost: float                  # 无解时为 -1
    stats: SearchStats


# ---------------------------------------------------------------- A* 主体

def astar(start: Coord, goal: Coord, grid: Grid,
          cost_fn: CostFn = unit_cost,
          heuristic: Heuristic = manhattan,
          tie_break: str = "h_low",
          seed: int = 0,
          goal_cost_override: Optional[float] = None) -> SearchResult:
    """在网格上运行 A*。

    cost_fn(cell)        : 进入某格的代价，None 表示不可通行
    heuristic(node, goal): 启发函数
    tie_break            : 见 TIE_BREAKS
    goal_cost_override   : 若给定，进入终点的代价改为该值（题二规定进入 E 代价为 0）
    """
    stats = SearchStats()
    t0 = time.perf_counter()

    def finish(path, cost):
        stats.time_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(path, cost, stats)

    # 边界情况：起点或终点本身不可通行
    if cost_fn(grid[start[0]][start[1]]) is None or cost_fn(grid[goal[0]][goal[1]]) is None:
        return finish([], -1)

    rng = random.Random(seed)
    key = _make_key(tie_break, rng)
    counter = itertools.count()

    g: Dict[Coord, float] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    closed = set()

    h0 = heuristic(start, goal)
    open_heap = [(key(h0, h0, next(counter)), start)]
    stats.pushed = 1
    stats.max_open = 1

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in closed:              # 惰性删除：跳过过期条目
            continue
        closed.add(node)
        stats.expanded += 1
        stats.expanded_order.append(node)
        stats.expanded_f.append(g[node] + heuristic(node, goal))

        if node == goal:                # 终点出堆时，g[goal] 即最优代价
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return finish(path[::-1], g[goal])

        for nbr, step_cost in neighbors(grid, node, cost_fn):
            if nbr in closed:
                continue
            if nbr == goal and goal_cost_override is not None:
                step_cost = goal_cost_override
            new_g = g[node] + step_cost
            if new_g < g.get(nbr, float("inf")):
                g[nbr] = new_g
                parent[nbr] = node
                h = heuristic(nbr, goal)
                heapq.heappush(open_heap, (key(new_g + h, h, next(counter)), nbr))
                stats.pushed += 1
                if len(open_heap) > stats.max_open:
                    stats.max_open = len(open_heap)

    return finish([], -1)