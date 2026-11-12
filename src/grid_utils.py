"""网格地图的通用工具：邻居、代价、全源最短距离、各类地图生成器。

所有坐标均为 (row, col)。地图是 List[List[str]]，'X' 表示障碍。
"""
import heapq
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

Coord = Tuple[int, int]
Grid = List[List[str]]
CostFn = Callable[[str], Optional[float]]  # 返回进入该格的代价；None 表示不可通行

# 固定的邻居顺序：上、下、左、右（影响相同长度路径中输出哪一条）
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def unit_cost(cell: str) -> Optional[float]:
    """题一的代价函数：障碍不可走，其余每步代价为 1。"""
    return None if cell == "X" else 1


def neighbors(grid: Grid, node: Coord, cost_fn: CostFn):
    """生成 (邻居坐标, 进入该邻居的代价)。"""
    rows, cols = len(grid), len(grid[0])
    r, c = node
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            cost = cost_fn(grid[nr][nc])
            if cost is not None:
                yield (nr, nc), cost


def all_distances(grid: Grid, source: Coord, cost_fn: CostFn = unit_cost) -> Dict[Coord, float]:
    """从 source 出发到所有可达格的最短距离（Dijkstra；单位代价时等价于 BFS）。

    用途：BFS 对拍、计算真实距离 h*、地标预处理。
    """
    dist = {source: 0}
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in neighbors(grid, u, cost_fn):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def bfs_shortest_length(grid: Grid, start: Coord, goal: Coord) -> int:
    """单位代价下的最短步数（不可达返回 -1），作为对拍的标准答案。"""
    if grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return -1
    seen = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            return seen[u]
        for v, _ in neighbors(grid, u, unit_cost):
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    return -1


def free_cells(grid: Grid) -> List[Coord]:
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != "X"]


def obstacle_density(grid: Grid) -> float:
    total = len(grid) * len(grid[0])
    return sum(ch == "X" for row in grid for ch in row) / total


# ---------------------------------------------------------------- 地图生成器

def open_map(n: int) -> Grid:
    """无障碍的 n×n 空旷地图。"""
    return [["." for _ in range(n)] for _ in range(n)]


def random_map(n: int, density: float, rng: random.Random) -> Grid:
    """每格以概率 density 独立地成为障碍。"""
    return [["X" if rng.random() < density else "." for _ in range(n)] for _ in range(n)]


def maze_map(n: int, rng: random.Random, corridor: int = 1) -> Grid:
    """递归回溯法生成的"完美迷宫"（任意两点间恰有一条通路，大量死胡同）。

    corridor 为走廊宽度，墙厚为 1。迷宫单元的边长 = corridor + 1。
    """
    step = corridor + 1
    cells_r = (n - 1) // step
    cells_c = (n - 1) // step
    grid = [["X" for _ in range(n)] for _ in range(n)]

    def carve_cell(cr, cc):
        r0, c0 = 1 + cr * step, 1 + cc * step
        for r in range(r0, r0 + corridor):
            for c in range(c0, c0 + corridor):
                grid[r][c] = "."

    def carve_between(a, b):
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:  # 水平相邻：打通两单元之间的竖墙
            wall_c = 1 + max(c1, c2) * step - 1
            r0 = 1 + r1 * step
            for r in range(r0, r0 + corridor):
                grid[r][wall_c] = "."
        else:  # 垂直相邻：打通横墙
            wall_r = 1 + max(r1, r2) * step - 1
            c0 = 1 + c1 * step
            for c in range(c0, c0 + corridor):
                grid[wall_r][c] = "."

    visited = [[False] * cells_c for _ in range(cells_r)]
    stack = [(0, 0)]
    visited[0][0] = True
    carve_cell(0, 0)
    while stack:
        cr, cc = stack[-1]
        options = []
        for dr, dc in DIRS:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cells_r and 0 <= nc < cells_c and not visited[nr][nc]:
                options.append((nr, nc))
        if not options:
            stack.pop()
            continue
        nxt = rng.choice(options)
        visited[nxt[0]][nxt[1]] = True
        carve_cell(*nxt)
        carve_between((cr, cc), nxt)
        stack.append(nxt)
    return grid


def u_trap_map(n: int, rng: random.Random) -> Tuple[Grid, Coord, Coord]:
    """U 型陷阱：起点在一个开口背向终点的 U 形墙内部，终点在墙外正前方。

    曼哈顿距离会把搜索"吸"进 U 形内部，直到填满后才绕出去。
    返回 (地图, 起点, 终点)。
    """
    grid = open_map(n)
    depth = rng.randint(n // 4, n // 2)      # U 的纵深
    half_w = rng.randint(n // 6, n // 3)     # U 的半宽
    cr = rng.randint(depth + 2, n - 3)       # U 底部所在行
    cc = rng.randint(half_w + 1, n - half_w - 2)
    top = cr - depth
    for c in range(cc - half_w, cc + half_w + 1):   # 底边（朝向终点）
        grid[cr][c] = "X"
    for r in range(top, cr + 1):                     # 两条侧边，开口朝上（背向终点）
        grid[r][cc - half_w] = "X"
        grid[r][cc + half_w] = "X"
    start = (cr - 1, cc)          # U 内部、紧贴底边
    goal = (n - 1, cc)            # 墙外正下方，曼哈顿距离直指底边
    return grid, start, goal


def shuffle_obstacles(grid: Grid, rng: random.Random, keep_free: List[Coord]) -> Grid:
    """置乱对照：障碍总数（密度）不变，位置随机打乱，从而只破坏"结构"。"""
    n_rows, n_cols = len(grid), len(grid[0])
    n_obs = sum(ch == "X" for row in grid for ch in row)
    candidates = [(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) not in set(keep_free)]
    chosen = set(rng.sample(candidates, n_obs))
    return [["X" if (r, c) in chosen else "." for c in range(n_cols)] for r in range(n_rows)]


def pick_pair(grid: Grid, rng: random.Random, min_manhattan: int, tries: int = 2000) -> Optional[Tuple[Coord, Coord]]:
    """随机选一对互相可达、曼哈顿距离不少于 min_manhattan 的起终点。"""
    cells = free_cells(grid)
    for _ in range(tries):
        s, g = rng.choice(cells), rng.choice(cells)
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) >= min_manhattan and bfs_shortest_length(grid, s, g) >= 0:
            return s, g
    return None