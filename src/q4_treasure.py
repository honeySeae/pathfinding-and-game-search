"""竞争环境宝藏分配：A*距离与有限深度零和博弈。

寻路复用题一A*，不将途经宝藏自动收集，也不把另一玩家当作障碍。
"""
import math
from dataclasses import dataclass
from functools import lru_cache

from .astar import astar, manhattan

GRID = [row.split() for row in (
    'A 1 1 X 2 1 T1', '1 X 6 X 1 X 1', '1 1 T2 1 1 1 1',
    'X 1 X 8 X 1 X', '1 1 1 1 T3 1 1', '1 X 9 X 1 X 1', 'T4 1 1 X 1 1 B')]
POINTS = {'A': (0, 0), 'B': (6, 6), 'T1': (0, 6), 'T2': (2, 2), 'T3': (4, 4), 'T4': (6, 0)}
VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}


def terrain_cost(cell):
    """标记格代价1，数字严格为正，X不可通行。"""
    if cell == 'X':
        return None
    if cell in ('A', 'B') or cell.startswith('T'):
        return 1
    value = int(cell)
    if value < 1:
        raise ValueError('题四要求所有进入代价至少为1')
    return value


def distance_matrix(grid, points):
    """逐有向点对运行A*并检查对称性，不能预先镜像填表。

    一般的进入代价图是非对称的；本题所有标记端点代价相同，反转路径的
    代价差仅为两个端点代价之差，故这些端点之间距离对称。一般输入不对称时报错。
    不可达记为-1，博弈构造时拒绝不连通实例。
    """
    coords = list(points.values()) if hasattr(points, 'values') else list(points)
    matrix = [[int(astar(s, g, grid, cost_fn=terrain_cost, heuristic=manhattan).cost)
               for g in coords] for s in coords]
    if any(matrix[i][j] != matrix[j][i] for i in range(len(coords)) for j in range(len(coords))):
        raise ValueError('点间距离不对称：请确认标记端点的进入代价相同')
    return matrix


@dataclass(frozen=True, slots=True)
class TreasureGame:
    names: tuple
    distances: tuple
    values: tuple

    def distance(self, source, target):
        return self.distances[self.names.index(source)][self.names.index(target)]

    def value(self, target):
        return self.values[self.names.index(target)]


def make_game(grid, points, values):
    matrix = distance_matrix(grid, points)
    if any(d < 0 for row in matrix for d in row):
        raise ValueError('所有玩家和宝藏必须两两连通')
    return TreasureGame(tuple(points), tuple(map(tuple, matrix)), tuple(values.get(p, 0) for p in points))


@lru_cache(maxsize=1)
def original_game():
    return make_game(GRID, POINTS, VALUES)


@dataclass(frozen=True, slots=True)
class GameState:
    """不可变状态避免兄弟分支间污染；remaining使用frozenset。

    未来增量收益只取决于双方位置、剩余宝藏、当前玩家和剩余深度，与已积累
    分数无关；累计收益仅用于计算U。game保存只读地图距离/价值，允许独立随机实例。
    """
    pos_a: str = 'A'
    pos_b: str = 'B'
    remaining: frozenset = frozenset(VALUES)
    score_a: int = 0
    score_b: int = 0
    turn: str = 'A'
    depth_left: int = 4
    game: TreasureGame | None = None

    def __post_init__(self):
        object.__setattr__(self, 'remaining', frozenset(self.remaining))
        if self.depth_left < 0 or self.turn not in ('A', 'B'):
            raise ValueError('深度应非负，玩家应为A或B')
        if self.game is None:
            object.__setattr__(self, 'game', original_game())


@dataclass(slots=True)
class GameStats:
    nodes: int = 0
    leaves: int = 0
    prunes: int = 0


def target_key(target):
    return int(target[1:])


def immediate_gain(state, target):
    pos = state.pos_a if state.turn == 'A' else state.pos_b
    return state.game.value(target) - state.game.distance(pos, target)


def fixed(state):
    return sorted(state.remaining, key=target_key)


def reverse(state):
    return list(reversed(fixed(state)))


def gain(state):
    return sorted(state.remaining, key=lambda t: (-immediate_gain(state, t), target_key(t)))


def gain_asc(state):
    return sorted(state.remaining, key=lambda t: (immediate_gain(state, t), target_key(t)))


ORDERS = {'fixed': fixed, 'reverse': reverse, 'gain': gain, 'gain_asc': gain_asc}


def move(state, target):
    """一次移动只获取选定目标；即使收益为负，也不能放弃回合。"""
    if target not in state.remaining:
        raise ValueError('目标已被获取')
    reward = immediate_gain(state, target)
    a = state.turn == 'A'
    return GameState(target if a else state.pos_a, state.pos_b if a else target,
                     state.remaining - {target}, state.score_a + (reward if a else 0),
                     state.score_b + (0 if a else reward), 'B' if a else 'A',
                     max(0, state.depth_left-1), state.game)


def eval_score(state):
    return state.score_a - state.score_b


def eval_closer(state):
    """逐宝藏归给眼前收益较高方，不考虑轮次；相等归B，按题设对照定义。"""
    value = eval_score(state)
    for t in state.remaining:
        a = state.game.value(t) - state.game.distance(state.pos_a, t)
        b = state.game.value(t) - state.game.distance(state.pos_b, t)
        value += a if a > b else -b
    return value


def eval_rollout(state):
    """考虑轮流行动和每回合必须取一个宝藏，平局按编号贪心推演。

    不需要额外搜索分支，每个叶子的代价为O(剩余宝藏数²)；推演不受搜索剩余深度限制。
    """
    while state.remaining:
        state = move(state, gain(state)[0])
    return eval_score(state)


EVALUATORS = {'eval_score': eval_score, 'eval_closer': eval_closer, 'eval_rollout': eval_rollout}


def minimax(state, stats, evaluate=eval_score):
    stats.nodes += 1
    if not state.remaining or state.depth_left == 0:
        stats.leaves += 1
        return (eval_score(state) if not state.remaining else evaluate(state)), None
    best, target = (-math.inf if state.turn == 'A' else math.inf), None
    for t in fixed(state):
        value, _ = minimax(move(state, t), stats, evaluate)
        if target is None or (value > best if state.turn == 'A' else value < best):
            best, target = value, t
    return best, target


def alphabeta(state, alpha=-math.inf, beta=math.inf, stats=None, order=None, evaluate=eval_score):
    """返回(值, 首步)。根层并列最优统一取编号最小，必要时全窗口确认界值。

    Fail-soft返回值可能只是界，不能将等于根最优值的剪枝返回误当作并列最优。
    因而较小编号候选发生等值时，单独全窗口复核；复核工作计入统计与计时。
    """
    stats = GameStats() if stats is None else stats
    order = fixed if order is None else (ORDERS[order] if isinstance(order, str) else order)

    def search(s, lo, hi, root=False):
        stats.nodes += 1
        if not s.remaining or s.depth_left == 0:
            stats.leaves += 1
            return (eval_score(s) if not s.remaining else evaluate(s)), None
        maximizing = s.turn == 'A'
        best, target = (-math.inf if maximizing else math.inf), None
        choices = order(s)
        for i, t in enumerate(choices):
            child = move(s, t)
            value, _ = search(child, lo, hi)
            if root and target is not None and value == best and target_key(t) < target_key(target):
                value, _ = search(child, -math.inf, math.inf)
            if target is None or (value > best if maximizing else value < best) or (
                    root and value == best and target_key(t) < target_key(target)):
                best, target = value, t
            if maximizing:
                lo = max(lo, best)
            else:
                hi = min(hi, best)
            if lo >= hi and i+1 < len(choices):
                stats.prunes += 1
                break
        return best, target

    result = search(state, alpha, beta, True)
    # 断开递归闭包，及时释放大批实例搜索中的状态与统计引用。
    search = None
    return result


def first_step_values(state, evaluate=eval_score):
    return {t: alphabeta(move(state, t), stats=GameStats(), order='gain', evaluate=evaluate)[0]
            for t in fixed(state)} if state.remaining and state.depth_left else {}


def principal_variation(state, evaluate=eval_score):
    """以当前剩余深度精确选取每一步，平局编号优先，返回双方最优对局。"""
    steps = []
    while state.remaining and state.depth_left > 0:
        _, target = alphabeta(state, stats=GameStats(), evaluate=evaluate)
        reward, player = immediate_gain(state, target), state.turn
        state = move(state, target)
        steps.append({'玩家': player, '目标': target, '收益': reward, 'U': eval_score(state)})
    return steps


def solve(H, algorithm='alphabeta'):
    if algorithm not in ('alphabeta', 'minimax'):
        raise ValueError('未知算法')
    value, target = (alphabeta(GameState(depth_left=H), stats=GameStats()) if algorithm == 'alphabeta'
                     else minimax(GameState(depth_left=H), GameStats()))
    return {'best_target': target, 'best_value': value}


def main():
    game = original_game()
    print('距离矩阵', game.names)
    for name, row in zip(game.names, game.distances):
        print(name, row)
    for H in (2, 4):
        for algorithm in ('minimax', 'alphabeta'):
            print('H=', H, algorithm, solve(H, algorithm))
        state = GameState(depth_left=H)
        print('首步值', first_step_values(state))
        print('最优对局', principal_variation(state))


if __name__ == '__main__':
    main()
