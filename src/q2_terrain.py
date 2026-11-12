"""题目二：进入格子的加权代价，进入终点为0；复用题一的A*主体。

运行：py -m src.q2_terrain
"""
from .astar import astar, manhattan, zero_heuristic


def minimum_terrain_weight(grid):
    """数字格的最小代价；无数字格（例如仅S/E）时约定为1。"""
    weights = [int(cell) for row in grid for cell in row if cell not in ("S", "E", "X")]
    if any(weight < 0 for weight in weights):
        raise ValueError("A*要求非负代价，数字地形不能为负数")
    return min(weights, default=1)


def make_terrain_cost(grid):
    """由地图生成进入代价函数：数字=int，E=0，S=w_min，X=None。

    题面未规定重入S的代价。在正权图中最优路径不会再次进入起点；非负权下也
    总可删去回到起点的环。因此将进入S的代价设为数字格最小值，不影响从S出发
    的最优代价，并使除E以外所有可通行格的进入代价都至少为w_min。
    此设定同时保证启发函数在所有格子上的可采纳性和一致性。
    """
    w_min = minimum_terrain_weight(grid)

    def cost(cell):
        if cell == "X":
            return None
        if cell == "E":
            return 0
        if cell == "S":
            return w_min
        return int(cell)

    return cost


def terrain_heuristic(grid):
    """返回h(n)=max(0,Manhattan(n,goal)-1)*w_min，可采纳且一致。

    可采纳性：设到目标的曼哈顿距离为m。任一路径至少走m步；可以取不提前
    经过终点的简单路径，前面至少m-1次进入非终点格，每次代价>=w_min，最后
    进入终点的代价为0。因此真实剩余代价不小于max(0,m-1)*w_min。
    make_terrain_cost将进入S也定义为w_min，故证明涵盖经S的路径。

    一致性：普通四邻居移动时h最多减少w_min，且进入代价>=w_min；进入终点
    时出发格与终点相邻，两者h均为0，满足h(n)<=0+h(goal)。h(goal)=0。
    适用于地图中的E为本次查询终点；也支持没有E、通过override指定免费终点。
    """
    w_min = minimum_terrain_weight(grid)
    return lambda node, goal: max(0, manhattan(node, goal) - 1) * w_min


def solve_terrain(start, end, grid) -> int:
    """返回最小总代价，无解返回-1；起点初始代价为0。"""
    result = astar(start, end, grid, cost_fn=make_terrain_cost(grid),
                   heuristic=terrain_heuristic(grid), goal_cost_override=0)
    return int(result.cost)


def path_cost_trace(start, end, grid):
    """返回最优路径的(坐标,进入代价,累计代价)；首项起点为(起点,0,0)，无解为空。"""
    cost = make_terrain_cost(grid)
    result = astar(start, end, grid, cost_fn=cost,
                   heuristic=terrain_heuristic(grid), goal_cost_override=0)
    total = 0
    trace = []
    for index, node in enumerate(result.path):
        step = 0 if index == 0 or node == end else cost(grid[node[0]][node[1]])
        total += step
        trace.append((node, step, total))
    assert not trace or total == result.cost
    return trace


TEST_START = (0, 0)
TEST_END = (4, 4)
TEST_GRID = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]


if __name__ == "__main__":
    print("最小总代价:", solve_terrain(TEST_START, TEST_END, TEST_GRID))
    trace = path_cost_trace(TEST_START, TEST_END, TEST_GRID)
    print("路径:", [point for point, _, _ in trace])
    print("坐标 | 进入代价 | 累计代价")
    for point, step, total in trace:
        print(f"{point} | {step} | {total}")
    for name, h in (("h=0", zero_heuristic), ("可采纳下界", terrain_heuristic(TEST_GRID)), ("曼哈顿距离", manhattan)):
        result = astar(TEST_START, TEST_END, TEST_GRID, cost_fn=make_terrain_cost(TEST_GRID),
                       heuristic=h, goal_cost_override=0)
        print(f"{name}: 代价={result.cost}, 扩展节点数={result.stats.expanded}")
    print(f"起点曼哈顿距离={manhattan(TEST_START, TEST_END)} > 最优代价={trace[-1][2]}：高估但本例仍最优")
