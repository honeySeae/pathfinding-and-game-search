"""题一基础对比：h=0 与曼哈顿距离的扩展节点数、热力图。

运行：python -m experiments.q1_baseline
输出：results/q1/baseline_given.csv, baseline_random.csv, baseline_heatmap_*.png
"""
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.astar import astar, manhattan, zero_heuristic
from src.grid_utils import pick_pair, random_map
from src.q1_maze import TEST_END, TEST_GRID, TEST_START

from .common import INK, out_dir, save_fig, write_csv

HEURISTICS = [("h=0", zero_heuristic), ("曼哈顿", manhattan)]
OUT = out_dir("q1")


def draw_expansion(ax, grid, res, start, goal, title):
    """热力图：灰色=障碍，蓝色深浅=扩展次序（越深越晚），橙线=最终路径。"""
    n_r, n_c = len(grid), len(grid[0])
    order = np.full((n_r, n_c), np.nan)
    for i, (r, c) in enumerate(res.stats.expanded_order):
        order[r, c] = i
    walls = np.array([[1.0 if ch == "X" else np.nan for ch in row] for row in grid])
    ax.imshow(np.ones((n_r, n_c)), cmap=ListedColormap(["#fcfcfb"]), vmin=0, vmax=1)
    ax.imshow(order, cmap="Blues", vmin=-0.15 * np.nanmax(order), vmax=np.nanmax(order))
    ax.imshow(walls, cmap=ListedColormap(["#5f5e5a"]), vmin=0, vmax=1)
    if res.path:
        ys, xs = zip(*res.path)
        ax.plot(xs, ys, color="#eb6834", linewidth=2)
    ax.scatter([start[1], goal[1]], [start[0], goal[0]], s=40, c=[INK, "#e34948"], zorder=3)
    ax.set_title(f"{title}\n扩展 {res.stats.expanded} 个节点", fontsize=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def run_given():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for ax, (name, h) in zip(axes, HEURISTICS):
        res = astar(TEST_START, TEST_END, TEST_GRID, heuristic=h)
        rows.append({"启发函数": name, "路径长度": len(res.path) - 1, "扩展节点数": res.stats.expanded,
                     "入堆次数": res.stats.pushed, "open表峰值": res.stats.max_open,
                     "路径": " ".join(map(str, res.path))})
        draw_expansion(ax, TEST_GRID, res, TEST_START, TEST_END, name)
    fig.suptitle("给定 5×5 地图上的扩展节点（颜色越深扩展越晚）", fontsize=11, color=INK, y=1.06)
    save_fig(fig, f"{OUT}/baseline_heatmap_given.png")
    write_csv(f"{OUT}/baseline_given.csv", rows)
    for r in rows:
        print(r)


def run_random(n=40, density=0.25, n_maps=50, seed=0):
    """小地图差异不明显，所以在 50 张 40×40 随机地图上重复比较，报均值与标准差。"""
    rng = random.Random(seed)
    rows = []
    example = None
    while len(rows) < n_maps:
        grid = random_map(n, density, rng)
        pair = pick_pair(grid, rng, min_manhattan=n)
        if pair is None:
            continue
        s, g = pair
        r0 = astar(s, g, grid, heuristic=zero_heuristic)
        r1 = astar(s, g, grid, heuristic=manhattan)
        assert r0.cost == r1.cost
        rows.append({"地图": len(rows), "最短路长度": r0.cost, "h=0扩展": r0.stats.expanded,
                     "曼哈顿扩展": r1.stats.expanded,
                     "节省率": 1 - r1.stats.expanded / r0.stats.expanded})
        if example is None:
            example = (grid, s, g, r0, r1)
    write_csv(f"{OUT}/baseline_random.csv", rows)
    for key in ["h=0扩展", "曼哈顿扩展", "节省率"]:
        vals = [r[key] for r in rows]
        print(f"{key}: 均值 {statistics.mean(vals):.3f}  标准差 {statistics.stdev(vals):.3f}")

    grid, s, g, r0, r1 = example
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    draw_expansion(axes[0], grid, r0, s, g, "h=0")
    draw_expansion(axes[1], grid, r1, s, g, "曼哈顿")
    fig.suptitle(f"{n}×{n} 随机地图（障碍密度 {density:.0%}）上的扩展节点", fontsize=11, color=INK, y=1.03)
    save_fig(fig, f"{OUT}/baseline_heatmap_random.png")


if __name__ == "__main__":
    run_given()
    run_random()