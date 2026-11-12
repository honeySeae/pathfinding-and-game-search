"""题四：访问顺序与截断评价函数。

现象：原题H=2与H=4的最优首步不同，当前分差忽略了尚未行动的后果。
假设：考虑轮流行动的贪心推演比只看当前分差或忽略轮次的逐宝藏归属更能选对首步，
但增加叶子评价耗时；gain排序的剪枝效率也依赖实例，不能保证始终最好。
验证：Part1固定原题比较排序与评价。Part2固定种子0，n=6共500个、n=8共200个
11×11连通实例；每个实例复用完全相同的地图、距离、宝藏价值，比较有限深度决策与
完整博弈每个首步的精确值，多个首步并列最优均算正确。完整真值用无剪枝穷举并缓存
未来增量收益（不含累计U），只避免重复状态，仍搜索到宝藏取完；不计入有限搜索开销。
有限搜索为Alpha-Beta+gain，包含并列首步复核工作，每个配置5次独立搜索取耗时中位数，
不使用置换表。报告逐实例准确性、真实收益损失、节点数、时间和Wilson 95%区间。
运行：py -m experiments.q4_exp_order_and_eval。只写results/q4/，不改其他题目结果。
补测：加--supplement，仅从已保存的700个实例读取输入和真值，追加指定深度；
已有逐实例记录、5次计时和汇总数字原样保留，不重新测量。r=n-H为截断后的剩余回合数。
"""
import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from src.grid_utils import all_distances
from src.q4_treasure import (EVALUATORS, ORDERS, GameState, GameStats, alphabeta,
                             eval_score, first_step_values, make_game, minimax,
                             principal_variation, target_key, terrain_cost, TreasureGame)
from .common import out_dir, write_csv

OUT = Path(out_dir('q4'))
SEED = 0
SAMPLES = {6: 500, 8: 200}
REPEATS = 5
SUPPLEMENT = {6: {'eval_score': (5, 6), 'eval_rollout': (5,)},
              8: {'eval_score': (6, 7, 8), 'eval_rollout': (6, 7)}}


def random_instance(n, seed):
    """独立实例种子确定地图、放置、价值；拒绝不连通放置后重新生成整张地图。"""
    rng = random.Random(seed)
    names = ['A', 'B'] + [f'T{i+1}' for i in range(n)]
    attempts = 0
    while True:
        attempts += 1
        grid = [['X' if rng.random() < .15 else str(rng.choices(
            [1, 2, 3, 5, 8], weights=[60, 15, 10, 10, 5])[0]) for _ in range(11)] for _ in range(11)]
        free = [(r, c) for r in range(11) for c in range(11) if grid[r][c] != 'X']
        if len(free) < n+2:
            continue
        coords = rng.sample(free, n+2)
        for name, (r, c) in zip(names, coords):
            grid[r][c] = name  # 标记格统一代价1，替换原先数字。
        reachable = all_distances(grid, coords[0], terrain_cost)
        if all(p in reachable for p in coords):
            break
    points = dict(zip(names, coords))
    values = {name: rng.randint(5, 12) for name in names[2:]}
    game = make_game(grid, points, values)
    state = GameState(remaining=frozenset(values), depth_left=n, game=game)
    return state, {'n': n, 'seed': seed, 'generation_attempts': attempts,
                   'grid': json.dumps(grid, ensure_ascii=False), 'points': json.dumps(points),
                   'values': json.dumps(values), 'distances': json.dumps(game.distances)}


def exact_first_values(state):
    """无剪枝完整Minimax真值：缓存未来增量，不缓存与路径相关的累计分数。"""
    game = state.game
    names = game.names
    dist, values = game.distances, game.values
    targets = tuple(names.index(t) for t in sorted(state.remaining, key=target_key))

    @lru_cache(maxsize=None)
    def future(a, b, mask, turn):
        if mask == 0:
            return 0
        best = -math.inf if turn == 'A' else math.inf
        for i, t in enumerate(targets):
            if mask & (1 << i):
                reward = values[t] - dist[a if turn == 'A' else b][t]
                score = ((reward + future(t, b, mask ^ (1 << i), 'B')) if turn == 'A' else
                         (-reward + future(a, t, mask ^ (1 << i), 'A')))
                best = max(best, score) if turn == 'A' else min(best, score)
        return best

    a, b = names.index(state.pos_a), names.index(state.pos_b)
    mask = (1 << len(targets))-1
    result = {}
    for i, t in enumerate(targets):
        if state.turn == 'A':
            val = values[t] - dist[a][t] + future(t, b, mask ^ (1 << i), 'B')
        else:
            val = -(values[t] - dist[b][t]) + future(a, t, mask ^ (1 << i), 'A')
        result[names[t]] = eval_score(state) + val
    future.cache_clear()
    future = None  # 断开递归闭包，释放缓存及实例距离引用。
    return result


def part1():
    rows, first, pv, evaluations = [], [], [], []
    for H in (2, 4):
        state = GameState(depth_left=H)
        for algorithm, orders in (('Minimax', ['fixed']), ('Alpha-Beta', list(ORDERS))):
            for order in orders:
                stats = GameStats()
                value, target = (minimax(state, stats) if algorithm == 'Minimax' else
                                 alphabeta(state, stats=stats, order=order))
                row = {'H': H, 'algorithm': algorithm, 'order': order, 'best_target': target,
                       'best_value': value, 'nodes': stats.nodes, 'leaves': stats.leaves, 'cutoffs': stats.prunes}
                rows.append(row)
                print('Part1', row, flush=True)
        for target, value in first_step_values(state).items():
            first.append({'H': H, 'target': target, 'value': value})
        for i, step in enumerate(principal_variation(state), 1):
            pv.append({'H': H, 'ply': i, **step})
    for H in range(1, 5):
        for name, evaluate in EVALUATORS.items():
            value, target = alphabeta(GameState(depth_left=H), order='gain', evaluate=evaluate)
            evaluations.append({'H': H, 'evaluate': name, 'best_target': target, 'best_value': value})
    for name, data in (('exp_case', rows), ('exp_first_steps', first), ('exp_pv', pv),
                       ('exp_case_eval', evaluations)):
        write_csv(str(OUT / f'{name}.csv'), data)
    print('原题每首步值', first, flush=True)
    print('原题最优对局', pv, flush=True)
    print('原题评价首步', evaluations, flush=True)


def wilson(correct, total):
    """独立实例成功率的Wilson 95%区间，z取标准正态97.5%分位数。"""
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total))/denom
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return max(0., center-radius), min(1., center+radius)


def summarize(raw):
    groups = defaultdict(list)
    for r in raw:
        groups[(r['n'], r['H'], r['evaluate'])].append(r)
    rows = []
    for (n, H, name), samples in sorted(groups.items()):
        correct = sum(r['correct'] for r in samples)
        lo, hi = wilson(correct, len(samples))
        row = {'n': n, 'H': H, 'r': n-H, 'evaluate': name, 'instances': len(samples), 'correct': correct,
               'accuracy': correct/len(samples), 'ci95_low': lo, 'ci95_high': hi,
               'mean_loss': statistics.mean(r['loss'] for r in samples),
               'mean_nodes': statistics.mean(r['nodes'] for r in samples),
               'mean_leaves': statistics.mean(r['leaves'] for r in samples),
               'mean_cutoffs': statistics.mean(r['cutoffs'] for r in samples),
               'mean_median_ms': statistics.mean(r['median_ms'] for r in samples)}
        rows.append(row)
    return rows


def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def supplement():
    """仅补缺失配置；从实例表恢复只读博弈数据，避免重新生成地图或改动历史计时。"""
    instances = read_csv('exp_instances.csv')
    raw = read_csv('exp_order_and_eval_raw.csv')
    summary = read_csv('exp_order_and_eval_summary.csv')
    assert len(instances) == sum(SAMPLES.values())
    assert len({(r['n'], r['instance_id']) for r in instances}) == len(instances)
    for n, count in SAMPLES.items():
        assert sum(int(r['n']) == n for r in instances) == count
    existing = {(int(r['n']), int(r['instance_id']), int(r['H']), r['evaluate']) for r in raw}
    assert len(existing) == len(raw), '原始数据存在重复配置'
    added = []
    started = time.perf_counter()
    for i, item in enumerate(instances):
        n, index = int(item['n']), int(item['instance_id'])
        names = tuple(json.loads(item['points']))
        values = json.loads(item['values'])
        game = TreasureGame(names, tuple(map(tuple, json.loads(item['distances']))),
                            tuple(values.get(t, 0) for t in names))
        truth = json.loads(item['true_first_values'])
        optimum = max(truth.values())
        for name, depths in SUPPLEMENT[n].items():
            for H in depths:
                key = (n, index, H, name)
                if key in existing:
                    continue
                state = GameState(remaining=frozenset(values), depth_left=H, game=game)
                times, outputs = [], []
                for _ in range(REPEATS):
                    stats = GameStats()
                    t0 = time.perf_counter_ns()
                    value, target = alphabeta(state, stats=stats, order='gain', evaluate=EVALUATORS[name])
                    times.append((time.perf_counter_ns()-t0)/1e6)
                    outputs.append((value, target, stats.nodes, stats.leaves, stats.prunes))
                assert all(result == outputs[0] for result in outputs)
                loss = optimum-truth[target]
                assert loss >= 0
                if H == n:
                    assert loss == 0 and value == optimum, '完整搜索应与保存的真值一致'
                added.append({'n': n, 'instance_id': index, 'seed': item['seed'], 'H': H, 'evaluate': name,
                              'best_target': target, 'search_value': value, 'true_selected_value': truth[target],
                              'true_optimum': optimum, 'correct': int(loss == 0), 'loss': loss,
                              'nodes': stats.nodes, 'leaves': stats.leaves, 'cutoffs': stats.prunes,
                              'median_ms': statistics.median(times), 'repeats': REPEATS,
                              'times_ms': json.dumps(times)})
                existing.add(key)
        if (i+1) % 25 == 0:
            print(f'补测 {i+1}/{len(instances)}，新增{len(added)}条，耗时{time.perf_counter()-started:.1f}s', flush=True)
    # 原汇总只增加r；不重新计算或格式化历史数值，确保历史耗时逐字段保留。
    old_keys = {(int(r['n']), int(r['H']), r['evaluate']) for r in summary}
    new_summary = summarize(added)
    assert all((r['n'], r['H'], r['evaluate']) not in old_keys for r in new_summary)
    for row in summary:
        row['r'] = int(row['n'])-int(row['H'])
    if added:
        write_csv(str(OUT / 'exp_order_and_eval_raw.csv'), raw+added)
    write_csv(str(OUT / 'exp_order_and_eval_summary.csv'), summary+new_summary)
    for row in new_summary:
        print('补测汇总', row, flush=True)
    print(f'补测完成，新增{len(added)}条、{len(new_summary)}组；原记录未重跑。', flush=True)


def main():
    part1()
    raw, instances = [], []
    rng = random.Random(SEED)
    started = time.perf_counter()
    for n, count in SAMPLES.items():
        for i in range(count):
            seed = rng.getrandbits(64)
            state, instance = random_instance(n, seed)
            instance['instance_id'] = i
            truth = exact_first_values(state)
            instance['true_first_values'] = json.dumps(truth)
            optimum = max(truth.values())
            instances.append(instance)
            for H in range(1, 5 if n == 6 else 6):
                query = replace(state, depth_left=H)
                for name, evaluate in EVALUATORS.items():
                    times, outputs = [], []
                    for _ in range(REPEATS):
                        stats = GameStats()
                        t0 = time.perf_counter_ns()
                        value, target = alphabeta(query, stats=stats, order='gain', evaluate=evaluate)
                        times.append((time.perf_counter_ns()-t0)/1e6)
                        outputs.append((value, target, stats.nodes, stats.leaves, stats.prunes))
                    assert all(output == outputs[0] for output in outputs)
                    loss = optimum - truth[target]
                    assert loss >= 0
                    raw.append({'n': n, 'instance_id': i, 'seed': seed, 'H': H, 'evaluate': name,
                                'best_target': target, 'search_value': value, 'true_selected_value': truth[target],
                                'true_optimum': optimum, 'correct': int(loss == 0), 'loss': loss,
                                'nodes': stats.nodes, 'leaves': stats.leaves, 'cutoffs': stats.prunes,
                                'median_ms': statistics.median(times), 'repeats': REPEATS,
                                'times_ms': json.dumps(times)})
            if (i+1) % 25 == 0:
                print(f'n={n}: {i+1}/{count}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
        # 按宝藏数量保存检查点，长实验完成前一部分后不会丢失已有数据。
        write_csv(str(OUT / 'exp_instances.csv'), instances)
        write_csv(str(OUT / 'exp_order_and_eval_raw.csv'), raw)
    summary = summarize(raw)
    write_csv(str(OUT / 'exp_order_and_eval_summary.csv'), summary)
    for row in summary:
        print('Part2', row, flush=True)
    print(f'完成，{len(instances)}个实例，{len(raw)}条记录，耗时{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', action='store_true', help='只补测保存实例上的指定深度，保留历史数据')
    if parser.parse_args().supplement:
        supplement()
    else:
        main()
