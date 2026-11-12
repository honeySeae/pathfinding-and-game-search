"""实验 B：地图类型与启发函数的信息量。

现象：曼哈顿相对 h=0 的节点节省，在迷宫、陷阱地图上明显变小。
假设：节省多少取决于"曼哈顿距离与真实距离的接近程度"（启发函数质量），
      而这由障碍的空间结构决定，不只是障碍密度。
验证：
  1. 对每个实例计算启发函数质量
       q_start = 曼哈顿(s, t) / 真实距离(s, t)
       q_mean  = 所有可达格 n 上 曼哈顿(n, t) / h*(n) 的平均值
     与节省率 = 1 − 曼哈顿扩展数 / h=0 扩展数 做相关分析（Pearson、Spearman）。
  2. 配对置乱对照：同一地图、同一起终点、障碍数量不变，只打乱障碍位置。
     置乱最多尝试 20 次以保证端点连通，失败则丢弃整对；结论限于此筛选条件。
     （走廊宽 1 的迷宫障碍密度约 52%，超过方格点渗流阈值约 41%，置乱后地图基本不连通，
       因此只对走廊宽 2、3 的迷宫做置乱对照。）
  3. 使用全部330个实例做含截距的最小二乘回归：分别以q_mean、密度及两者联合
     解释节省率；通过残差相关计算控制另一个变量后的偏相关，仅作线性描述。

运行：python -m experiments.q1_exp_b_maptypes
输出：results/q1/exp_b_raw.csv, exp_b_summary.csv, exp_b_correlation.csv, exp_b_paired.csv,
      exp_b_regression.csv,
      exp_b_saving.png, exp_b_quality_scatter.png, exp_b_heatmap_utrap.png
"""
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np

from src.astar import astar, manhattan, zero_heuristic
from src.grid_utils import (all_distances, maze_map, obstacle_density, open_map, pick_pair,
                            random_map, shuffle_obstacles, u_trap_map)

from .common import INK, INK_2, PALETTE, out_dir, save_fig, write_csv
from .q1_baseline import draw_expansion

OUT = out_dir("q1")
N = 61
N_MAPS = 30


def gen_instance(kind, rng):
    """返回 (地图, 起点, 终点)；失败返回 None。"""
    if kind == "空旷":
        grid = open_map(N)
    elif kind.startswith("随机"):
        grid = random_map(N, int(kind[2:4]) / 100, rng)
    elif kind == "迷宫(走廊1)":
        grid = maze_map(N, rng, corridor=1)
    elif kind == "迷宫(走廊2)":
        grid = maze_map(N, rng, corridor=2)
    elif kind == "迷宫(走廊3)":
        grid = maze_map(N, rng, corridor=3)
    elif kind == "U型陷阱":
        grid, s, g = u_trap_map(N, rng)
        return grid, s, g
    else:
        raise ValueError(kind)
    pair = pick_pair(grid, rng, min_manhattan=N // 2)
    return None if pair is None else (grid, *pair)


def gen_paired_instance(kind, rng):
    """生成一对同图同端点的原始/置乱实例；20 次均不连通则丢弃整对。"""
    inst = gen_instance(kind, rng)
    if inst is None:
        return None
    grid, s, g = inst
    # 每次都重新置乱原图并保留端点；不修改密度、不按节省率筛选结果。
    for _ in range(20):
        shuffled = shuffle_obstacles(grid, rng, keep_free=[s, g])
        if g in all_distances(shuffled, s):
            return grid, shuffled, s, g
    return None


KINDS = ["空旷", "随机10%", "随机20%", "随机30%",
         "迷宫(走廊3)", "迷宫(走廊3)-置乱", "迷宫(走廊2)", "迷宫(走廊2)-置乱", "迷宫(走廊1)",
         "U型陷阱", "U型陷阱-置乱"]


def heuristic_quality(grid, s, g):
    h_star = all_distances(grid, g)        # 题一代价对称，从终点出发的距离即 h*(n)
    ratios = [manhattan(n, g) / d for n, d in h_star.items() if d > 0]
    return manhattan(s, g) / h_star[s], statistics.mean(ratios)


def rank(a):
    # Spearman 的并列值必须取平均名次；零基与一基排名的相关系数相同。
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return r


def regression_analysis(raw):
    """在实例级数据上计算含截距OLS的R²及线性偏相关，不改变采样或已有统计。"""
    saving = np.asarray([r["节省率"] for r in raw], dtype=float)
    quality = np.asarray([r["q_mean"] for r in raw], dtype=float)
    density = np.asarray([r["障碍密度"] for r in raw], dtype=float)

    def residuals(values, predictors):
        # 显式加入截距，使用lstsq而非求逆，避免正规方程放大数值误差。
        design = np.column_stack((np.ones(len(values)), *predictors))
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        return values - design @ coefficients

    total_ss = float(np.sum((saving - saving.mean()) ** 2))
    rows = []
    for label, predictors in (("q_mean", (quality,)), ("障碍密度", (density,)),
                              ("q_mean + 障碍密度", (quality, density))):
        error = residuals(saving, predictors)
        r_squared = 1 - float(error @ error) / total_ss if total_ss > 0 else float("nan")
        rows.append({"分析": "最小二乘回归", "自变量": label, "控制变量": "",
                     "R²": r_squared, "偏相关系数": "", "样本数": len(raw)})

    for label, values, control_label, control in (("障碍密度", density, "q_mean", quality),
                                                 ("q_mean", quality, "障碍密度", density)):
        # 双方各自对控制变量（含截距）回归，再计算两组残差的Pearson相关。
        x_res = residuals(values, (control,))
        y_res = residuals(saving, (control,))
        tolerance = np.finfo(float).eps * max(1.0, np.linalg.norm(values), np.linalg.norm(saving)) * len(raw)
        partial = (float(np.corrcoef(x_res, y_res)[0, 1])
                   if min(np.linalg.norm(x_res), np.linalg.norm(y_res)) > tolerance else float("nan"))
        rows.append({"分析": "偏相关", "自变量": label, "控制变量": control_label,
                     "R²": "", "偏相关系数": partial, "样本数": len(raw)})
    return rows


def main(seed=0):
    rng = random.Random(seed)
    raw = []
    example_u = None
    for kind in KINDS:
        # 置乱组由对应原图产生，不能独立再生成一组地图和端点。
        if kind.endswith("-置乱"):
            continue
        paired = kind + "-置乱" in KINDS
        made = 0
        while made < N_MAPS:
            inst = gen_paired_instance(kind, rng) if paired else gen_instance(kind, rng)
            if inst is None:
                continue
            if paired:
                grid, shuffled, s, g = inst
                instances = [(kind, grid), (kind + "-置乱", shuffled)]
            else:
                grid, s, g = inst
                instances = [(kind, grid)]
            for name, current_grid in instances:
                r0 = astar(s, g, current_grid, heuristic=zero_heuristic)
                r1 = astar(s, g, current_grid, heuristic=manhattan)
                assert r0.cost == r1.cost
                q_start, q_mean = heuristic_quality(current_grid, s, g)
                # 配对编号在同一结构内对应同一原图；独立组留空，避免伪造配对。
                raw.append({"地图类型": name, "编号": made, "配对编号": made if paired else "",
                            "起点": s, "终点": g,
                            "障碍密度": round(obstacle_density(current_grid), 3),
                            "C*": r0.cost, "h=0扩展": r0.stats.expanded, "曼哈顿扩展": r1.stats.expanded,
                            "节省率": 1 - r1.stats.expanded / r0.stats.expanded,
                            "q_start": q_start, "q_mean": q_mean})
                if name == "U型陷阱" and example_u is None:
                    example_u = (current_grid, s, g, r0, r1)
            made += 1
    write_csv(f"{OUT}/exp_b_raw.csv", raw)

    # 对同一编号求“置乱−原始”，以配对差而非两组独立均值评估结构变化。
    paired_summary = []
    for kind in ["迷宫(走廊2)", "迷宫(走廊3)", "U型陷阱"]:
        original = {r["配对编号"]: r for r in raw if r["地图类型"] == kind}
        shuffled = {r["配对编号"]: r for r in raw if r["地图类型"] == kind + "-置乱"}
        assert original.keys() == shuffled.keys() and len(original) == N_MAPS
        diffs = [shuffled[i]["节省率"] - original[i]["节省率"] for i in original]
        paired_summary.append({"地图类型": kind, "节省率差值均值": statistics.mean(diffs),
                               "节省率差值标准差": statistics.stdev(diffs),
                               "置乱组更高对数": sum(d > 0 for d in diffs), "总对数": len(diffs)})
    write_csv(f"{OUT}/exp_b_paired.csv", paired_summary)
    for row in paired_summary:
        print(row)

    summary = []
    for kind in KINDS:
        rows = [r for r in raw if r["地图类型"] == kind]
        sv = [r["节省率"] for r in rows]
        summary.append({"地图类型": kind,
                        "障碍密度": round(statistics.mean(r["障碍密度"] for r in rows), 3),
                        "h=0扩展均值": round(statistics.mean(r["h=0扩展"] for r in rows), 1),
                        "曼哈顿扩展均值": round(statistics.mean(r["曼哈顿扩展"] for r in rows), 1),
                        "节省率均值": round(statistics.mean(sv), 3),
                        "节省率标准差": round(statistics.stdev(sv), 3),
                        "q_start均值": round(statistics.mean(r["q_start"] for r in rows), 3),
                        "q_mean均值": round(statistics.mean(r["q_mean"] for r in rows), 3)})
    write_csv(f"{OUT}/exp_b_summary.csv", summary)
    for r in summary:
        print(r)

    saving = np.array([r["节省率"] for r in raw])
    corr = []
    for key in ["q_start", "q_mean", "障碍密度"]:
        x = np.array([r[key] for r in raw])
        pearson = float(np.corrcoef(x, saving)[0, 1])
        spearman = float(np.corrcoef(rank(x), rank(saving))[0, 1])
        corr.append({"变量": key, "Pearson": round(pearson, 3), "Spearman": round(spearman, 3), "样本数": len(raw)})
    write_csv(f"{OUT}/exp_b_correlation.csv", corr)
    for c in corr:
        print(c)

    # 仅新增分析输出；沿用全部原始实例和密度精度，不改变旧CSV字段与绘图。
    regression = regression_analysis(raw)
    write_csv(f"{OUT}/exp_b_regression.csv", regression)
    for row in regression:
        # Windows 的 GBK 终端无法打印上标 ²；仅终端改用 ASCII，CSV 列名仍为 R²。
        print({("R^2" if key == "R²" else key): value for key, value in row.items()})

    plot_saving(summary)
    plot_scatter(raw, summary)
    grid, s, g, r0, r1 = example_u
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    draw_expansion(axes[0], grid, r0, s, g, "h=0")
    draw_expansion(axes[1], grid, r1, s, g, "曼哈顿")
    fig.suptitle("U 型陷阱：曼哈顿先把 U 形内部填满，再绕出去", fontsize=11, color=INK, y=1.03)
    save_fig(fig, f"{OUT}/exp_b_heatmap_utrap.png")


def plot_saving(summary):
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(len(summary))
    colors = [PALETTE[1] if "置乱" in r["地图类型"] else PALETTE[0] for r in summary]
    means = [r["节省率均值"] for r in summary]
    stds = [r["节省率标准差"] for r in summary]
    ax.bar(x, means, yerr=stds, width=0.62, color=colors, ecolor=INK_2, capsize=3,
           error_kw={"linewidth": 1})
    labels = [r["地图类型"].replace("-置乱", "\n置乱") + f"\n密度{r['障碍密度']:.0%}" for r in summary]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel("节省率（1 − 曼哈顿扩展 / h=0扩展）")
    ax.set_ylim(min(0, min(m - s for m, s in zip(means, stds)) - 0.05), 1.18)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], label="原始结构"),
                       Patch(color=PALETTE[1], label="置乱对照（密度相同、结构打乱）")],
              loc="upper center", ncol=2, fontsize=9)
    ax.set_title("实验 B：不同地图上曼哈顿相对 h=0 的节点节省率（均值 ± 标准差，每类 30 张）",
                 fontsize=11, color=INK)
    save_fig(fig, f"{OUT}/exp_b_saving.png")


def plot_scatter(raw, summary):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter([r["q_mean"] for r in raw], [r["节省率"] for r in raw], s=12, color=PALETTE[0],
               alpha=0.35, linewidths=0, label="单个实例")
    ax.scatter([r["q_mean均值"] for r in summary], [r["节省率均值"] for r in summary], s=36,
               color=INK, zorder=3, label="各地图类型均值")
    crowded = [r for r in summary if r["节省率均值"] > 0.95]   # 右上角挤在一起的几类合并标注
    for r in summary:
        if r in crowded:
            continue
        ax.annotate(r["地图类型"], (r["q_mean均值"], r["节省率均值"]), fontsize=7.5, color=INK_2,
                    xytext=(4, 3), textcoords="offset points")
    if crowded:
        ax.annotate(" / ".join(r["地图类型"] for r in crowded), (1.0, crowded[0]["节省率均值"]),
                    fontsize=7.5, color=INK_2, xytext=(-8, 10), textcoords="offset points", ha="right")
    ax.set_xlabel("启发函数质量 q_mean（曼哈顿 / 真实距离 的平均值）")
    ax.set_ylabel("节省率")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("启发函数质量与节点节省率", fontsize=11, color=INK)
    save_fig(fig, f"{OUT}/exp_b_quality_scatter.png")


if __name__ == "__main__":
    main()
