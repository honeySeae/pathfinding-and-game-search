"""实验 A：平局规则对扩展节点数的影响。

现象：不同平局规则下扩展节点数差异很大。
假设：差异全部来自 f = C* 的节点；f < C* 的节点在任何平局规则下都必须被扩展。
验证：
  1. 理论计数：用 BFS 求出每个格子的真实 g*(n)，统计
       N_lt = #{n : g*(n) + h(n) <  C*}   （A* 理论上必须扩展的节点）
       N_eq = #{n : g*(n) + h(n) == C*}   （可能被扩展，取决于平局规则）
  2. 实际计数：把每次搜索扩展的节点按 f 值拆成 f<C* 与 f=C* 两部分。
  3. 检查：实际 f<C* 的数量是否在所有规则、所有地图上都恰好等于 N_lt。

运行：python -m experiments.q1_exp_a_tiebreak
输出：results/q1/exp_a_raw.csv, exp_a_summary.csv, exp_a_tiebreak.png
"""
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np

from src.astar import TIE_BREAKS, astar, manhattan
from src.grid_utils import all_distances, open_map, pick_pair, random_map

from .common import INK, INK_2, PALETTE, out_dir, save_fig, write_csv

OUT = out_dir("q1")
N = 40
N_MAPS = 30
RULE_NAMES = {"h_low": "优先h小", "h_high": "优先h大", "fifo": "FIFO", "lifo": "LIFO", "random": "随机"}
MAP_TYPES = [("空旷", 0.0), ("随机障碍20%", 0.2), ("随机障碍30%", 0.3)]


def theoretical_counts(grid, s, g):
    gstar = all_distances(grid, s)
    c_star = gstar[g]
    n_lt = n_eq = 0
    for node, d in gstar.items():
        f = d + manhattan(node, g)
        if f < c_star:
            n_lt += 1
        elif f == c_star:
            n_eq += 1
    return c_star, n_lt, n_eq


def main(seed=0):
    rng = random.Random(seed)
    raw = []
    mismatches = 0
    for map_name, density in MAP_TYPES:
        made = 0
        while made < N_MAPS:
            grid = open_map(N) if density == 0 else random_map(N, density, rng)
            pair = pick_pair(grid, rng, min_manhattan=N)
            if pair is None:
                continue
            s, g = pair
            c_star, n_lt, n_eq = theoretical_counts(grid, s, g)
            for rule in TIE_BREAKS:
                res = astar(s, g, grid, heuristic=manhattan, tie_break=rule, seed=rng.randrange(10**9))
                assert res.cost == c_star
                e_lt = sum(f < c_star for f in res.stats.expanded_f)
                e_eq = sum(f == c_star for f in res.stats.expanded_f)
                e_gt = sum(f > c_star for f in res.stats.expanded_f)
                if e_lt != n_lt or e_gt != 0:
                    mismatches += 1
                raw.append({"地图类型": map_name, "地图编号": made, "平局规则": RULE_NAMES[rule],
                            "C*": c_star, "理论N_lt": n_lt, "理论N_eq": n_eq,
                            "扩展总数": res.stats.expanded, "扩展f<C*": e_lt, "扩展f=C*": e_eq,
                            "扩展f>C*": e_gt, "f=C*扩展占比": e_eq / n_eq})
            made += 1
    write_csv(f"{OUT}/exp_a_raw.csv", raw)
    print(f"验证：f<C* 的实际扩展数 ≠ 理论 N_lt（或出现 f>C* 扩展）的次数 = {mismatches} / {len(raw)}")

    # 汇总表
    summary = []
    for map_name, _ in MAP_TYPES:
        for rule in TIE_BREAKS:
            rows = [r for r in raw if r["地图类型"] == map_name and r["平局规则"] == RULE_NAMES[rule]]
            tot = [r["扩展总数"] for r in rows]
            summary.append({
                "地图类型": map_name, "平局规则": RULE_NAMES[rule],
                "扩展总数均值": round(statistics.mean(tot), 1),
                "扩展总数标准差": round(statistics.stdev(tot), 1),
                "f<C*均值": round(statistics.mean(r["扩展f<C*"] for r in rows), 1),
                "f=C*均值": round(statistics.mean(r["扩展f=C*"] for r in rows), 1),
                "理论N_eq均值": round(statistics.mean(r["理论N_eq"] for r in rows), 1),
                "f=C*扩展占比均值": round(statistics.mean(r["f=C*扩展占比"] for r in rows), 3),
            })
    write_csv(f"{OUT}/exp_a_summary.csv", summary)
    for r in summary:
        print(r)
    plot(summary)


def plot(summary):
    """每种地图一个面板；每条柱子分两段：f<C*（灰，各规则应完全相同）+ f=C*（蓝，随规则变化）。"""
    fig, axes = plt.subplots(1, len(MAP_TYPES), figsize=(12, 3.8), sharey=False)
    for ax, (map_name, _) in zip(axes, MAP_TYPES):
        rows = [r for r in summary if r["地图类型"] == map_name]
        x = np.arange(len(rows))
        lt = [r["f<C*均值"] for r in rows]
        eq = [r["f=C*均值"] for r in rows]
        ax.bar(x, lt, width=0.6, color="#b8b7b0", label="f < C*（必须扩展）", edgecolor="white", linewidth=1)
        ax.bar(x, eq, width=0.6, bottom=lt, color=PALETTE[0], label="f = C*（取决于平局规则）",
               edgecolor="white", linewidth=1)
        for xi, a, b in zip(x, lt, eq):
            ax.text(xi, a + b, f"{a + b:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, [r["平局规则"] for r in rows], fontsize=9)
        ax.set_title(map_name, fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("平均扩展节点数")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=9)
    fig.suptitle("实验 A：平局规则只改变 f = C* 部分的扩展数", fontsize=11, color=INK, y=1.14)
    save_fig(fig, f"{OUT}/exp_a_tiebreak.png")


if __name__ == "__main__":
    main()