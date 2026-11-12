"""地标启发函数（ALT：A*, Landmarks, Triangle inequality）。

预处理：选 k 个地标 L，算出每个地标到所有格子的真实距离 d(L, ·)。
查询：由三角不等式，对任意 n 和终点 t 有
      d(n, t) >= |d(L, t) - d(L, n)|
因此 h(n) = max_L |d(L, t) - d(L, n)| 是可采纳的，并且是一致的。
再与曼哈顿取 max，仍可采纳且一致（一致启发函数的 max 仍一致）。

注意：这里依赖 d(a, b) = d(b, a)，即无向图。题一（单位代价）满足；
题二"进入格子的代价"一般不对称，那时需要同时预处理 d(L,·) 和 d(·,L)。
"""
import random
import time
from collections import deque
from typing import List

from .astar import manhattan
from .grid_utils import Coord, Grid, all_distances, free_cells, neighbors, unit_cost


def largest_component(grid: Grid) -> List[Coord]:
    """返回最大可通行连通区域；并列时取行列扫描遇到的第一个，顺序确定。"""
    cells = free_cells(grid)
    unseen = set(cells)
    largest = []
    for root in cells:
        if root not in unseen:
            continue
        component = []
        queue = deque([root])
        unseen.remove(root)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr, _ in neighbors(grid, node, unit_cost):
                if nbr in unseen:
                    unseen.remove(nbr)
                    queue.append(nbr)
        if len(component) > len(largest):
            largest = component
    return sorted(largest)


class LandmarkHeuristic:
    def __init__(self, grid: Grid, k: int, method: str = "farthest",
                 rng: random.Random = None, combine_manhattan: bool = True,
                 restrict_to_largest_component: bool = True):
        """
        method: "random"   随机选 k 个可通行格
                "farthest" 最远点采样：第一个取离随机种子点最远的格子，之后每次选
                           "到已有地标最近距离"最大的格子，使地标分散在边缘、角落和死胡同末端
        restrict_to_largest_component: 默认只在最大区域选点；False 用全部空格候选，
                                      最远点法仍保持原有“只在种子所属区域贪心”的行为。
        """
        self.rng = rng or random.Random(0)
        self.combine_manhattan = combine_manhattan
        # 消融只切换候选区域，默认行为及原有实验的地标顺序保持不变。
        self.restrict_to_largest_component = restrict_to_largest_component
        # 空地图也属于没有可通行格的情况，允许退化为曼哈顿。
        self.rows, self.cols = len(grid), len(grid[0]) if grid else 0
        t0 = time.perf_counter()
        self._cache = {}   # 选地标时算过的距离表直接复用，避免重复计算
        self._largest_component = frozenset()
        self.landmarks: List[Coord] = self._select(grid, k, method)
        # 每个地标一张二维距离表（不可达记为 None），用列表比字典查询更快
        self.tables = []
        for L in self.landmarks:
            dist = self._dist(grid, L)
            table = [[None] * self.cols for _ in range(self.rows)]
            for (r, c), d in dist.items():
                table[r][c] = d
            self.tables.append(table)
        self.preprocess_ms = (time.perf_counter() - t0) * 1000

    @property
    def outside_largest_component_count(self) -> int:
        """只读统计：实际选出的地标中，不在最大连通区域内的个数。"""
        return sum(point not in self._largest_component for point in self.landmarks)

    def _dist(self, grid: Grid, src: Coord):
        if src not in self._cache:
            self._cache[src] = all_distances(grid, src, unit_cost)
        return self._cache[src]

    def _select(self, grid: Grid, k: int, method: str) -> List[Coord]:
        cells = free_cells(grid)
        # 零/负地标数或全障碍地图不应随机取空列表，也不应偷偷选一个点。
        if k <= 0 or not cells:
            return []
        # 随机障碍可能形成封闭小区域；避免种子落入小区域而把全部地标挤在其中。
        # 实验关注最大连通区域的查询；其他区域跳过不可达表项，仍可退化为曼哈顿，
        # 不影响可采纳性。按行列顺序扫描、BFS 固定邻居顺序，保证候选次序可复现。
        largest = largest_component(grid)
        self._largest_component = frozenset(largest)
        if self.restrict_to_largest_component:
            cells = largest
        k = min(k, len(cells))
        if method == "random":
            # 同种子打乱完整候选副本后取前缀，保证不同 k 的地标严格嵌套。
            candidates = cells.copy()
            self.rng.shuffle(candidates)
            return candidates[:k]
        if method == "farthest":
            # 从一个随机种子点出发，第一个地标取离种子最远的格子（保证落在边缘），
            # 之后每次取"到已有地标的最近距离"最大的格子
            seed = self.rng.choice(cells)
            d_seed = self._dist(grid, seed)
            first = max(d_seed, key=d_seed.get)
            chosen = [first]
            min_d = dict(self._dist(grid, first))
            while len(chosen) < k:
                nxt = max(min_d, key=min_d.get)
                if min_d[nxt] == 0:
                    break
                chosen.append(nxt)
                d_new = self._dist(grid, nxt)
                for cell in min_d:
                    min_d[cell] = min(min_d[cell], d_new.get(cell, float("inf")))
            return chosen
        raise ValueError(f"未知的地标选取方式: {method}")

    def __call__(self, node: Coord, goal: Coord) -> float:
        # 无地标时按约定退化为曼哈顿，空图不再访问距离表。
        best = manhattan(node, goal) if self.combine_manhattan or not self.tables else 0
        nr, nc = node
        gr, gc = goal
        for table in self.tables:
            dn, dg = table[nr][nc], table[gr][gc]
            if dn is not None and dg is not None:
                v = dg - dn if dg > dn else dn - dg
                if v > best:
                    best = v
        return best
