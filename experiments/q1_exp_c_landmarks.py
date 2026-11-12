"""实验 C：地标启发函数（ALT），针对实验 B 发现的"曼哈顿在迷宫/陷阱中很弱"。

设计：
  - 地图：迷宫(走廊1)、迷宫(走廊2)、U 型陷阱、随机30%、空旷（空旷作为负面对照：曼哈顿已完美）
  - 方法：h=0、曼哈顿、ALT-随机选点、ALT-最远点、完美启发函数 h*（从终点 BFS，作为下界参照）
  - 地标数 k ∈ {1, 2, 4, 8, 16, 32}：两侧都有取值，观察饱和而不是只报一个"最优"
  - 指标：扩展节点数、单次查询耗时、预处理耗时、启发函数准确度 mean(h/h*)
  - 摊销：预处理一次可服务多次查询，计算相对曼哈顿的"回本查询次数"
验证"最远点为什么比随机好"：比较两种选点方式的启发函数准确度。

计时方法（减少噪声）：
  - 每次查询重复运行 TIME_REPEATS 次，取中位数；每个地标启发函数重复构建 PRE_REPEATS 次，取中位数
  - 计时期间关闭 Python 垃圾回收，避免 GC 停顿混入
  - 回本查询次数 = 预处理耗时 / (曼哈顿单次耗时 − ALT 单次耗时)；
    若 ALT 提速不足曼哈顿耗时的 MIN_GAIN（5%），记为"提速低于5%"；
    5% 是人为设定的工程阈值，不是统计显著性检验；ALT 更慢则记为"不回本"。
    回本计算使用未舍入均值，仅在写入汇总 CSV 时舍入，避免小差值被舍入扭曲。

运行：python -m experiments.q1_exp_c_landmarks（约 3–8 分钟）
输出：results/q1/exp_c_raw.csv, exp_c_summary.csv, exp_c_nodes.png, exp_c_time.png, exp_c_accuracy.png
"""
import gc
import random
import statistics
import time

import matplotlib.pyplot as plt

from src.astar import astar, manhattan, zero_heuristic
from src.grid_utils import all_distances, maze_map, open_map, pick_pair, random_map, u_trap_map
from src.landmarks import LandmarkHeuristic

from .common import INK, INK_2, PALETTE, out_dir, save_fig, write_csv

OUT = out_dir("q1")
N = 61
KS = [1, 2, 4, 8, 16, 32]
MAPS_PER_TYPE = 10
QUERIES_PER_MAP = 10
ACC_SAMPLES = 300
MAP_TYPES = ["迷宫(走廊1)", "迷宫(走廊2)", "U型陷阱", "随机30%", "空旷"]
TIME_REPEATS = 5     # 每次查询重复计时次数
PRE_REPEATS = 3      # 每个地标启发函数重复构建次数
MIN_GAIN = 0.05      # 人为设定的 5% 提速阈值，不代表统计显著性


def gen_map(kind, rng):
    """返回 (地图, 查询列表)。U 型陷阱固定使用陷阱起终点作为查询。"""
    if kind == "U型陷阱":
        grid, s, g = u_trap_map(N, rng)
        return grid, [(s, g)]
    if kind == "迷宫(走廊1)":
        grid = maze_map(N, rng, 1)
    elif kind == "迷宫(走廊2)":
        grid = maze_map(N, rng, 2)
    elif kind == "随机30%":
        grid = random_map(N, 0.3, rng)
    else:
        grid = open_map(N)
    queries = []
    while len(queries) < QUERIES_PER_MAP:
        pair = pick_pair(grid, rng, N // 2)
        if pair:
            queries.append(pair)
    return grid, queries


def timed_search(s, g, grid, h):
    """重复 TIME_REPEATS 次，返回 (结果, 耗时中位数 ms)。扩展节点数每次都相同。"""
    times = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(TIME_REPEATS):
            res = astar(s, g, grid, heuristic=h)
            times.append(res.stats.time_ms)
    finally:
        gc.enable()
    return res, statistics.median(times)


def timed_build(grid, k, method, sub_seed):
    """用同一个种子重复构建 PRE_REPEATS 次（每次选出的地标相同），返回 (启发函数, 预处理耗时中位数 ms)。"""
    times = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(PRE_REPEATS):
            h = LandmarkHeuristic(grid, k, method, rng=random.Random(sub_seed))
            times.append(h.preprocess_ms)
    finally:
        gc.enable()
    return h, statistics.median(times)


def accuracy(h, h_star, goal, sample):
    vals = [h(n, goal) / h_star[n] for n in sample if h_star.get(n, 0) > 0]
    return statistics.mean(vals)


def main(seed=0):
    rng = random.Random(seed)
    raw = []
    t_start = time.time()
    for kind in MAP_TYPES:
        # U 型陷阱每张图只有一个陷阱查询，因此多生成一些地图（20 张 × 1 次查询）
        n_maps = MAPS_PER_TYPE * 2 if kind == "U型陷阱" else MAPS_PER_TYPE
        for m in range(n_maps):
            grid, queries = gen_map(kind, rng)
            alts = {}
            for method in ["random", "farthest"]:
                # 每张图每种选点方法共用一个种子，使所有 k 的地标前缀嵌套。
                sub_seed = rng.random()
                for k in KS:
                    alts[(method, k)] = timed_build(grid, k, method, sub_seed)
            for s, g in queries:
                h_star = all_distances(grid, g)
                sample = rng.sample(list(h_star), min(ACC_SAMPLES, len(h_star)))
                perfect = lambda n, goal, _h=h_star: _h.get(n, 0)  # noqa: E731
                methods = [("h=0", None, 0, zero_heuristic, 0.0),
                           ("曼哈顿", None, 0, manhattan, 0.0),
                           ("完美h*", None, 0, perfect, 0.0)]
                for (method, k), (h, pre_ms) in alts.items():
                    name = "ALT-随机" if method == "random" else "ALT-最远点"
                    methods.append((name, method, k, h, pre_ms))
                for name, method, k, h, pre_ms in methods:
                    res, t_ms = timed_search(s, g, grid, h)
                    assert res.cost == h_star[s], (name, k)       # 所有方法都必须最优
                    raw.append({"地图类型": kind, "地图": m, "方法": name, "k": k,
                                # 保留端点，结合地图编号和固定生成顺序可重放单个查询。
                                "起点": s, "终点": g,
                                "扩展节点数": res.stats.expanded, "查询耗时ms": t_ms,
                                "预处理耗时ms": pre_ms, "准确度": accuracy(h, h_star, g, sample)})
        print(f"{kind} 完成，累计 {time.time() - t_start:.0f}s")
    write_csv(f"{OUT}/exp_c_raw.csv", raw)

    summary = []
    for kind in MAP_TYPES:
        for name in ["h=0", "曼哈顿", "完美h*", "ALT-随机", "ALT-最远点"]:
            for k in (KS if name.startswith("ALT") else [0]):
                rows = [r for r in raw if r["地图类型"] == kind and r["方法"] == name and r["k"] == k]
                summary.append({"地图类型": kind, "方法": name, "k": k,
                                # 保留完整精度计算 gain 和回本，不在这里提前舍入。
                                "扩展节点均值": statistics.mean(r["扩展节点数"] for r in rows),
                                "查询耗时ms均值": statistics.mean(r["查询耗时ms"] for r in rows),
                                "预处理耗时ms均值": statistics.mean(r["预处理耗时ms"] for r in rows),
                                "准确度均值": statistics.mean(r["准确度"] for r in rows)})
    # 回本查询次数，见文件开头"计时方法"
    for kind in MAP_TYPES:
        manh = next(r for r in summary if r["地图类型"] == kind and r["方法"] == "曼哈顿")
        for r in summary:
            if r["地图类型"] == kind and r["方法"].startswith("ALT"):
                gain = manh["查询耗时ms均值"] - r["查询耗时ms均值"]
                r["提速比例"] = gain / manh["查询耗时ms均值"]
                if gain <= 0:
                    r["回本查询次数"] = "不回本"
                elif gain < MIN_GAIN * manh["查询耗时ms均值"]:
                    r["回本查询次数"] = "提速低于5%"
                else:
                    r["回本查询次数"] = r["预处理耗时ms均值"] / gain
    # 只为 CSV 构造舍入副本；后续绘图仍使用未舍入均值。
    digits = {"扩展节点均值": 1, "查询耗时ms均值": 3, "预处理耗时ms均值": 2,
              "准确度均值": 3, "提速比例": 3, "回本查询次数": 1}
    csv_summary = [{key: round(value, digits[key]) if key in digits and isinstance(value, (int, float))
                    else value for key, value in row.items()} for row in summary]
    write_csv(f"{OUT}/exp_c_summary.csv", csv_summary,
              fieldnames=["地图类型", "方法", "k", "扩展节点均值", "查询耗时ms均值", "预处理耗时ms均值",
                          "准确度均值", "提速比例", "回本查询次数"])
    for r in summary:
        print(r)
    plot(summary)


def _panel_lines(ax, summary, kind, key, refs=True):
    for name, color in [("ALT-随机", PALETTE[1]), ("ALT-最远点", PALETTE[0])]:
        ys = [next(r[key] for r in summary if r["地图类型"] == kind and r["方法"] == name and r["k"] == k)
              for k in KS]
        ax.plot(KS, ys, marker="o", markersize=4, color=color, label=name)
    if refs:
        for name, style in [("曼哈顿", "--"), ("完美h*", ":")]:
            y = next(r[key] for r in summary if r["地图类型"] == kind and r["方法"] == name)
            ax.axhline(y, color=INK_2, linestyle=style, linewidth=1.2, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_title(kind, fontsize=10, color=INK)
    ax.set_xlabel("地标数 k", fontsize=9)


def plot(summary):
    # 1. 扩展节点数
    fig, axes = plt.subplots(1, len(MAP_TYPES), figsize=(15, 3.6))
    for ax, kind in zip(axes, MAP_TYPES):
        _panel_lines(ax, summary, kind, "扩展节点均值")
        h0 = next(r["扩展节点均值"] for r in summary if r["地图类型"] == kind and r["方法"] == "h=0")
        ax.text(0.98, 0.04, f"h=0: {h0:.0f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=INK_2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("平均扩展节点数")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.suptitle("实验 C：地标数与扩展节点数", fontsize=11, color=INK, y=1.15)
    save_fig(fig, f"{OUT}/exp_c_nodes.png")

    # 2. 查询耗时与预处理耗时（两个量纲不同，分两行画，不用双坐标轴）
    fig, axes = plt.subplots(2, len(MAP_TYPES), figsize=(15, 6.4))
    for j, kind in enumerate(MAP_TYPES):
        _panel_lines(axes[0, j], summary, kind, "查询耗时ms均值")
        _panel_lines(axes[1, j], summary, kind, "预处理耗时ms均值", refs=False)
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_ylim(bottom=0)
        axes[0, j].set_xlabel("")
    axes[0, 0].set_ylabel("单次查询耗时 (ms)")
    axes[1, 0].set_ylabel("预处理耗时 (ms)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle(f"实验 C：查询耗时（上，{TIME_REPEATS} 次取中位数）与预处理耗时（下，{PRE_REPEATS} 次取中位数）", fontsize=11, color=INK, y=1.08)
    fig.tight_layout()
    save_fig(fig, f"{OUT}/exp_c_time.png")

    # 3. 准确度：解释"最远点为什么更好"
    fig, axes = plt.subplots(1, len(MAP_TYPES), figsize=(15, 3.4), sharey=True)
    for ax, kind in zip(axes, MAP_TYPES):
        _panel_lines(ax, summary, kind, "准确度均值")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("启发函数准确度 mean(h / h*)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.suptitle("实验 C：启发函数准确度（越接近 1 越好，完美 h* = 1）", fontsize=11, color=INK, y=1.15)
    save_fig(fig, f"{OUT}/exp_c_accuracy.png")


if __name__ == "__main__":
    main()
