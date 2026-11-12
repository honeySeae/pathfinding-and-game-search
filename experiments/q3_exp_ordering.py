"""题三：访问顺序如何影响Alpha-Beta剪枝。

现象：同一棵树、同一最优得分，访问顺序不同会改变叶子评估和剪枝次数。
假设：优先访问真实Minimax值更好的子节点能更早收紧alpha/beta；完整均匀树
最优排序叶子数恰为b^ceil(d/2)+b^floor(d/2)-1。排序噪声越大通常越接近随机。
验证：Part1保留题面共享max2结构；Part2用互不相同的[0,1)叶子随机值，
b=2/3/5/8、d=2/4/6/8，跳过b^d>400000，每组合30棵、主种子0；逐棵验证
最优叶子公式。Part3复用Part2的b=5,d=6同一批30棵树，分级加入高斯噪声，
另设随机顺序，不因结果偏离预期而换树。真实Minimax值和排序计划预先计算，
不计入搜索节点开销；本实验不使用计时比较。

准确度：实际访问的内部节点中，排序首个孩子具有最佳真实值的比例，每棵树
各算一个比例后等权平均；同时记录全树排序准确度，避免与访问分布混淆。
sigma各档共用同一组标准正态噪声，只改变其乘数；随机顺序使用独立种子。

运行：py -m experiments.q3_exp_ordering
输出：results/q3/exp_ordering_case.csv、exp_ordering_{raw,summary}.csv、
      exp_accuracy_{raw,summary}.csv。
      exp_cutoff_depth.csv汇总b=5,d=6的30棵树在三种顺序下的浅/深剪枝。
"""
import json
import math
import random
import statistics
import time
from collections import defaultdict

from src.game_tree import (GameStats, TreeNode, alphabeta, build_test_tree, minimax,
                           minimax_values, node_role, oracle_order)
from .common import out_dir, write_csv

SEED = 0
BRANCHES = (2, 3, 5, 8)
DEPTHS = (2, 4, 6, 8)
TREES_PER_COMBINATION = 30
MAX_LEAVES = 400000
SIGMAS = (0, .01, .02, .05, .1, .2, .5, 1, math.inf)
OUT = out_dir("q3")


def theoretical_leaves(b, d):
    return b**((d + 1)//2) + b**(d//2) - 1


def uniform_tree(b, d, seed):
    """生成完整b叉深度d的树；深度为边数，[0,1)浮点叶值用集合去重。"""
    rng = random.Random(seed)
    seen = set()

    def build(depth):
        if depth == 0:
            value = rng.random()
            while value in seen:
                value = rng.random()
            seen.add(value)
            return TreeNode(value=value)
        return TreeNode(children=[build(depth - 1) for _ in range(b)])

    root = build(d)
    assert len(seen) == b**d
    # 断开递归闭包的自引用，使叶值去重集合可在返回后及时释放。
    build = None
    return root


def random_order(seed):
    rng = random.Random(seed)

    def order(node, maximizing):
        children = node.children.copy()
        rng.shuffle(children)
        return children

    return order


def noisy_order(values, sigma, seed):
    """在全树上固定排序计划，然后只统计搜索实际调用的内部节点访问。"""
    rng = random.Random(seed)
    plans = {}
    correct = {}
    for (node, maximizing), _ in values.items():
        if not node.children:
            continue
        true_scores = {child: values[(child, node_role(child, not maximizing))] for child in node.children}
        if math.isinf(sigma):
            children = node.children.copy()
            rng.shuffle(children)
        else:
            # 每档sigma重置为相同seed，从而使用同一组z~N(0,1)，仅改变sigma。
            scores = {child: true_scores[child] + sigma * rng.gauss(0, 1) for child in node.children}
            children = sorted(node.children, key=scores.get, reverse=maximizing)
        best_score = (max if maximizing else min)(true_scores.values())
        key = (node, maximizing)
        plans[key] = children
        correct[key] = int(true_scores[children[0]] == best_score)
    counters = {"已访问内部节点": 0, "首选正确次数": 0}

    def order(node, maximizing):
        key = (node, maximizing)
        counters["已访问内部节点"] += 1
        counters["首选正确次数"] += correct[key]
        return plans[key]

    return order, counters, sum(correct.values()) / len(correct)


def part1():
    root = build_test_tree()
    values = minimax_values(root)
    full = GameStats()
    result = minimax(root, True, full)
    rows = [{"算法": "Minimax", "顺序": "从左到右", "结果": result,
             "叶子数": full.leaves, "访问节点数": full.nodes, "剪枝次数": 0, "剪枝记录": "[]"}]
    for name, order in (("从左到右", None), ("从右到左", lambda n, m: reversed(n.children)),
                        ("最优", oracle_order(values)), ("反向排序", oracle_order(values, best=False))):
        stats, trace = GameStats(), []
        score = alphabeta(root, -math.inf, math.inf, True, stats, order=order, trace=trace)
        assert score == result == 5
        rows.append({"算法": "Alpha-Beta", "顺序": name, "结果": score,
                     "叶子数": stats.leaves, "访问节点数": stats.nodes, "剪枝次数": stats.prunes,
                     "剪枝记录": json.dumps(trace, ensure_ascii=False)})
    write_csv(f"{OUT}/exp_ordering_case.csv", rows)
    for row in rows:
        print("Part1", row, flush=True)
    return rows


def evaluate_tree(b, d, tree_id, tree_seed, order_seed, noise_seed):
    """局部生命周期使大树和真实值缓存每次处理后释放，不同时保留30棵大树。"""
    root = uniform_tree(b, d, tree_seed)
    values = minimax_values(root)
    expected = values[(root, True)]
    baseline = b**d
    theory = theoretical_leaves(b, d)
    raw = []
    for name, order in (("随机顺序", random_order(order_seed)), ("最优", oracle_order(values)),
                        ("反向排序", oracle_order(values, best=False))):
        stats = GameStats()
        result = alphabeta(root, -math.inf, math.inf, True, stats, order=order)
        assert result == expected
        mismatch = int(stats.leaves != theory) if name == "最优" else ""
        if mismatch:
            print("理论计数不符，需要排查：", b, d, tree_id, stats.leaves, theory, flush=True)
        raw.append({"b": b, "d": d, "树编号": tree_id, "主种子": SEED,
                    "树种子": tree_seed, "随机排序种子": order_seed, "顺序": name,
                    "结果": result, "Minimax叶子数": baseline, "叶子数": stats.leaves,
                    "访问节点数": stats.nodes, "剪枝次数": stats.prunes,
                    "有效分支因子": stats.leaves**(1/d), "理论最优叶子数": theory,
                    "最优理论不符": mismatch,
                    "浅剪枝次数": stats.shallow_prunes, "深剪枝次数": stats.deep_prunes,
                    "外部界剪枝次数": stats.external_prunes})
        assert stats.prunes == stats.shallow_prunes + stats.deep_prunes
        assert stats.external_prunes == 0
    accuracy = []
    if (b, d) == (5, 6):
        for sigma in SIGMAS:
            # 随机档复用Part2随机种子，但计划为全树预生成，不按搜索访问情况抽样。
            ordering_seed = order_seed if math.isinf(sigma) else noise_seed
            order, counters, whole_accuracy = noisy_order(values, sigma, ordering_seed)
            stats = GameStats()
            result = alphabeta(root, -math.inf, math.inf, True, stats, order=order)
            assert result == expected
            internal = counters["已访问内部节点"]
            assert internal == stats.nodes - stats.leaves
            if sigma == 0:
                assert stats.leaves == theory
            accuracy.append({"b": b, "d": d, "树编号": tree_id, "主种子": SEED,
                             "树种子": tree_seed, "排序种子": ordering_seed,
                             "σ": sigma, "顺序": "随机" if math.isinf(sigma) else "真实值+高斯噪声",
                             "结果": result, "Minimax叶子数": baseline, "叶子数": stats.leaves,
                             "叶子比例": stats.leaves / baseline, "访问节点数": stats.nodes,
                             "剪枝次数": stats.prunes, "已访问内部节点": internal,
                             "首选正确次数": counters["首选正确次数"],
                             "首选准确率": counters["首选正确次数"] / internal,
                             "全树首选准确率": whole_accuracy})
    return raw, accuracy


def summarize_ordering(raw):
    groups = defaultdict(list)
    for row in raw:
        groups[(row["b"], row["d"], row["顺序"])].append(row)
    rows = []
    for (b, d, order), samples in groups.items():
        rows.append({"b": b, "d": d, "顺序": order, "树数": len(samples),
                     "Minimax叶子数": b**d, "理论最优叶子数": theoretical_leaves(b, d),
                     "叶子数均值": statistics.mean(r["叶子数"] for r in samples),
                     "叶子数标准差": statistics.stdev(r["叶子数"] for r in samples),
                     "有效分支因子均值": statistics.mean(r["有效分支因子"] for r in samples),
                     "有效分支因子标准差": statistics.stdev(r["有效分支因子"] for r in samples),
                     "最优理论不符次数": sum(r["最优理论不符"] for r in samples) if order == "最优" else ""})
    return rows


def summarize_accuracy(raw):
    groups = defaultdict(list)
    for row in raw:
        groups[row["σ"]].append(row)
    rows = []
    for sigma, samples in groups.items():
        first_correct = sum(r["首选正确次数"] for r in samples)
        internal = sum(r["已访问内部节点"] for r in samples)
        rows.append({"b": 5, "d": 6, "σ": sigma, "树数": len(samples), "Minimax叶子数": 5**6,
                     "叶子数均值": statistics.mean(r["叶子数"] for r in samples),
                     "叶子数标准差": statistics.stdev(r["叶子数"] for r in samples),
                     "叶子比例均值": statistics.mean(r["叶子比例"] for r in samples),
                     "叶子比例标准差": statistics.stdev(r["叶子比例"] for r in samples),
                     "首选准确率均值": statistics.mean(r["首选准确率"] for r in samples),
                     "首选准确率标准差": statistics.stdev(r["首选准确率"] for r in samples),
                     "首选正确次数": first_correct, "已访问内部节点": internal,
                     "首选准确率(汇总)": first_correct / internal,
                     "全树首选准确率均值": statistics.mean(r["全树首选准确率"] for r in samples)})
    return rows


def summarize_cutoff_depth(raw):
    """只统计既有b=5,d=6树，不增加搜索；浅/深沿用严格收紧的祖先来源定义。"""
    result = []
    for order in ("随机顺序", "最优", "反向排序"):
        rows = [r for r in raw if r["b"] == 5 and r["d"] == 6 and r["顺序"] == order]
        assert len(rows) == TREES_PER_COMBINATION
        shallow = sum(r["浅剪枝次数"] for r in rows)
        deep = sum(r["深剪枝次数"] for r in rows)
        total = sum(r["剪枝次数"] for r in rows)
        assert shallow + deep == total
        result.append({"b": 5, "d": 6, "顺序": order, "树数": len(rows),
                       "浅剪枝次数": shallow, "深剪枝次数": deep, "总剪枝次数": total,
                       "每树浅剪枝均值": statistics.mean(r["浅剪枝次数"] for r in rows),
                       "每树深剪枝均值": statistics.mean(r["深剪枝次数"] for r in rows)})
    return result


def main(seed=SEED):
    if seed != SEED:
        raise ValueError("本次实验主种子固定为0，修改配置需同步更新元数据")
    started = time.perf_counter()
    part1()
    rng = random.Random(seed)
    # 每树单独种子，顺序条件和sigma档数不会改变下一棵树。
    cases = [(b, d, i, rng.randrange(2**63), rng.randrange(2**63), rng.randrange(2**63))
             for b in BRANCHES for d in DEPTHS if b**d <= MAX_LEAVES
             for i in range(TREES_PER_COMBINATION)]
    ordering_raw, accuracy_raw = [], []
    for b, d, i, tree_seed, order_seed, noise_seed in cases:
        rows, accuracy = evaluate_tree(b, d, i, tree_seed, order_seed, noise_seed)
        ordering_raw.extend(rows)
        accuracy_raw.extend(accuracy)
        if (i+1) % 10 == 0:
            print(f"Part2 b={b}, d={d}：{i+1}/30，累计{time.perf_counter()-started:.1f}s", flush=True)
    ordering_summary = summarize_ordering(ordering_raw)
    accuracy_summary = summarize_accuracy(accuracy_raw)
    write_csv(f"{OUT}/exp_ordering_raw.csv", ordering_raw)
    write_csv(f"{OUT}/exp_ordering_summary.csv", ordering_summary)
    write_csv(f"{OUT}/exp_accuracy_raw.csv", accuracy_raw)
    write_csv(f"{OUT}/exp_accuracy_summary.csv", accuracy_summary)
    cutoff_summary = summarize_cutoff_depth(ordering_raw)
    write_csv(f"{OUT}/exp_cutoff_depth.csv", cutoff_summary)
    for row in cutoff_summary:
        print("浅/深剪枝统计", row)
    if cutoff_summary[-1]["浅剪枝次数"] != 0:
        print("预期不符：反向排序出现浅剪枝，按实际统计报告，未修改数据或排序。")
    best = [r for r in ordering_raw if r["顺序"] == "最优"]
    print("Part2最优理论不符次数：", sum(r["最优理论不符"] for r in best), "/", len(best))
    for row in accuracy_summary:
        print("Part3", row)
    print(f"实验完成，用时{time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
