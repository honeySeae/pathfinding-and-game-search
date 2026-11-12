"""实验 C 消融：固定地图和查询，只改变地标选择的一个因素。

现象 → 假设 → 验证：
1. U 型陷阱旧结果中 k=2 比 k=1 扩展更多。
   假设：不同 k 独立选点导致启发函数没有逐点占优关系。
   验证：同图同查询比较嵌套/非嵌套；拆分总扩展与 f<C*，检查相邻 k 逆转，
   在每个查询的同一批300个可达格上检查 h_{k+1}>=h_k。嵌套时理论保证
   f<C* 必扩展集合单调不增，不将该保证直接套到 f=C* 的平局部分。
2. 随机地图的封闭小区域可能浪费地标。
   假设：落入小区域的地标对最大区域查询无用，限制选点区域可降低扩展数。
   验证：同图同查询同种子，比较限制/不限制；两者均前缀嵌套，按是否存在
   区域外地标分组报告配对差值，允许结果不支持假设。

所有地图与查询先固定，再搜索；地图生成、查询、选点、占优抽样使用独立种子。
区域外地标比例分母为实际选出个数，最远点法不限制时可能在小区域提前耗尽。
配对差以查询为单位，同时报告地图数及地图平均差的标准差，避免将同图查询
误认为独立地图。只衡量节点数，不将本消融作为计时实验。

运行：py -m experiments.q1_exp_c_ablation
输出：results/q1/exp_c_ablation_{nesting,component}_{raw,summary}.csv 及两幅PNG。
"""
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# 复用现有风格与 Agg 后端，不改动任何已有实验脚本或已有输出。
from .common import INK, INK_2, PALETTE, out_dir, plt, save_fig, write_csv
from src.astar import astar, manhattan
from src.grid_utils import all_distances, free_cells, maze_map, random_map, u_trap_map
from src.landmarks import LandmarkHeuristic, largest_component

N = 61
SEED = 0
METHODS = ("random", "farthest")
METHOD_NAMES = {"random": "ALT-随机", "farthest": "ALT-最远点"}
NESTING_KS = (1, 2, 4, 8, 16, 32)
COMPONENT_KS = (1, 2, 4, 8)
SAMPLES = 300
OUT = out_dir("q1")


@dataclass
class FixedMap:
    kind: str
    index: int
    map_seed: int
    query_seed: int
    grid: list
    queries: list
    largest_fraction: float
    # 种子计划在运行任一条件前固定，不让搜索结果改变后续随机数序列。
    method_seeds: dict
    sample_seeds: list


def _queries_in_largest(grid, count, rng):
    """本消融的查询固定在最大区域，使小区域地标的无用性可被检验。"""
    cells = largest_component(grid)
    pairs = []
    for _ in range(20000):
        s, g = rng.choice(cells), rng.choice(cells)
        if manhattan(s, g) >= N // 2:
            pairs.append((s, g))
        if len(pairs) == count:
            return pairs
    raise RuntimeError("固定地图无法采到足够远的查询；不静默改变距离下限")


def fixed_maps(design, seed=SEED):
    """仅由种子生成全部地图、查询和子种子，可用于单实例重放。"""
    rng = random.Random(seed)
    specs = (("U型陷阱", 20, 1), ("迷宫(走廊2)", 10, 10)) if design == "nesting" else (
        ("随机30%", 20, 10), ("随机35%", 20, 10))
    cases = []
    for kind, count, query_count in specs:
        for index in range(count):
            map_seed, query_seed = rng.randrange(2**63), rng.randrange(2**63)
            map_rng = random.Random(map_seed)
            if kind == "U型陷阱":
                grid, s, g = u_trap_map(N, map_rng)
                queries = [(s, g)]
            else:
                grid = (maze_map(N, map_rng, 2) if kind == "迷宫(走廊2)" else
                        random_map(N, 0.30 if kind == "随机30%" else 0.35, map_rng))
                queries = _queries_in_largest(grid, query_count, random.Random(query_seed))
            largest = set(largest_component(grid))
            assert all(s in largest and g in largest for s, g in queries)
            seeds = {method: [rng.randrange(2**63) for _ in NESTING_KS] for method in METHODS}
            cases.append(FixedMap(kind, index, map_seed, query_seed, grid, queries,
                                  len(largest) / len(free_cells(grid)), seeds,
                                  [rng.randrange(2**63) for _ in queries]))
    return cases


def _metadata(case, query_id, method, k):
    s, g = case.queries[query_id]
    return {"地图类型": case.kind, "地图编号": case.index, "查询编号": query_id,
            "起点": s, "终点": g, "主种子": SEED, "地图种子": case.map_seed,
            "查询种子": case.query_seed, "方法": METHOD_NAMES[method], "k": k,
            "最大区域占比": case.largest_fraction}


def _search(case, query_id, h, distances):
    s, g = case.queries[query_id]
    result = astar(s, g, case.grid, heuristic=h)
    c_star = distances[s]
    assert result.cost == c_star, (case.kind, case.index, query_id, result.cost, c_star)
    # 单位权重和 ALT 距离均为整数，严格比较无浮点容差歧义。
    n_lt = sum(f < c_star for f in result.stats.expanded_f)
    n_eq = sum(f == c_star for f in result.stats.expanded_f)
    assert n_lt + n_eq == result.stats.expanded
    return result.stats.expanded, n_lt, n_eq


def run_nesting(cases):
    raw = []
    for case in cases:
        distances = [all_distances(case.grid, g) for _, g in case.queries]
        samples = [random.Random(seed).sample(list(d), min(SAMPLES, len(d)))
                   for seed, d in zip(case.sample_seeds, distances)]
        for method in METHODS:
            for condition in ("嵌套", "非嵌套"):
                hs = []
                seeds = case.method_seeds[method]
                for i, k in enumerate(NESTING_KS):
                    # 两条件 k=1 共用同一个种子；只改变后续 k 的种子耦合方式。
                    sub_seed = seeds[0] if condition == "嵌套" else seeds[i]
                    h = LandmarkHeuristic(case.grid, k, method, random.Random(sub_seed))
                    if hs and condition == "嵌套":
                        assert hs[-1].landmarks == h.landmarks[:len(hs[-1].landmarks)]
                    hs.append(h)
                for query_id, (_, g) in enumerate(case.queries):
                    previous = None
                    for i, (k, h) in enumerate(zip(NESTING_KS, hs)):
                        total, n_lt, n_eq = _search(case, query_id, h, distances[query_id])
                        eligible = int(i > 0)
                        dominance_ok = (sum(h(n, g) >= hs[i-1](n, g) for n in samples[query_id])
                                        if eligible else 0)
                        total_reversal = int(bool(previous) and total > previous[0])
                        lt_reversal = int(bool(previous) and n_lt > previous[1])
                        if condition == "嵌套" and lt_reversal:
                            print("理论预期不成立，需要排查：", case.kind, "地图", case.index,
                                  "查询", query_id, method, NESTING_KS[i-1], "→", k,
                                  "f<C*", previous[1], "→", n_lt, flush=True)
                        row = _metadata(case, query_id, method, k)
                        row.update({"条件": condition, "地标种子": case.method_seeds[method][0 if condition == "嵌套" else i],
                                    "抽样种子": case.sample_seeds[query_id], "C*": distances[query_id][case.queries[query_id][0]],
                                    "扩展总数": total, "扩展f<C*": n_lt, "扩展f=C*": n_eq,
                                    "前一k": NESTING_KS[i-1] if eligible else "",
                                    "相邻对数": eligible, "总扩展逆转": total_reversal,
                                    "f<C*逆转": lt_reversal, "占优通过格数": dominance_ok,
                                    "占优检查格数": len(samples[query_id]) if eligible else 0})
                        raw.append(row)
                        previous = (total, n_lt)
        print("消融1完成", case.kind, case.index, flush=True)
    return raw


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def summarize_nesting(raw):
    keys = ("地图类型", "方法", "条件", "k")
    summary = []
    for values, rows in _group(raw, keys).items():
        pairs = sum(r["相邻对数"] for r in rows)
        checks = sum(r["占优检查格数"] for r in rows)
        row = dict(zip(keys, values))
        row.update({"扩展总数均值": statistics.mean(r["扩展总数"] for r in rows),
                    "扩展f<C*均值": statistics.mean(r["扩展f<C*"] for r in rows),
                    "查询数": len(rows), "相邻对数": pairs,
                    "总扩展逆转次数": sum(r["总扩展逆转"] for r in rows),
                    "f<C*逆转次数": sum(r["f<C*逆转"] for r in rows),
                    "占优通过格数": sum(r["占优通过格数"] for r in rows),
                    "占优检查格数": checks,
                    "占优比例": sum(r["占优通过格数"] for r in rows) / checks if checks else ""})
        summary.append(row)
    return summary


def run_component(cases):
    raw = []
    for case in cases:
        distances = [all_distances(case.grid, g) for _, g in case.queries]
        for method in METHODS:
            sub_seed = case.method_seeds[method][0]
            previous = {}
            for k in COMPONENT_KS:
                hs = {restricted: LandmarkHeuristic(
                    case.grid, k, method, random.Random(sub_seed),
                    restrict_to_largest_component=restricted) for restricted in (True, False)}
                for restricted, h in hs.items():
                    if restricted in previous:
                        assert previous[restricted] == h.landmarks[:len(previous[restricted])]
                    previous[restricted] = h.landmarks
                outside = hs[False].outside_largest_component_count
                actual = len(hs[False].landmarks)
                for query_id in range(len(case.queries)):
                    limited, _, _ = _search(case, query_id, hs[True], distances[query_id])
                    unlimited, _, _ = _search(case, query_id, hs[False], distances[query_id])
                    row = _metadata(case, query_id, method, k)
                    row.update({"地标种子": sub_seed, "限制扩展节点数": limited,
                                "不限制扩展节点数": unlimited, "配对差值": unlimited - limited,
                                "限制地标数": len(hs[True].landmarks), "不限制地标数": actual,
                                "区域外地标数": outside, "区域外地标比例": outside / actual if actual else 0,
                                "有小区域地标": int(outside > 0),
                                # 保留实际坐标以区分“浪费地标”与候选次序变化造成的选点差异。
                                "限制地标": hs[True].landmarks, "不限制地标": hs[False].landmarks})
                    raw.append(row)
        print("消融2完成", case.kind, case.index, flush=True)
    return raw


def summarize_component(raw):
    summary = []
    keys = ("地图类型", "方法", "k")
    for values, rows in _group(raw, keys).items():
        for label, subset in (("全部", rows),
                              ("有小区域地标", [r for r in rows if r["有小区域地标"]]),
                              ("无小区域地标", [r for r in rows if not r["有小区域地标"]])):
            by_map = _group(subset, ("地图编号",))
            diffs = [r["配对差值"] for r in subset]
            map_diffs = [statistics.mean(r["配对差值"] for r in group) for group in by_map.values()]
            # 地标统计按地图去重，不能把同一地标集合随10条查询重复计数。
            maps = [group[0] for group in by_map.values()]
            total_landmarks = sum(r["不限制地标数"] for r in maps)
            row = dict(zip(keys, values))
            row.update({"分组": label, "地图数": len(maps), "查询配对数": len(subset),
                        "限制扩展均值": statistics.mean(r["限制扩展节点数"] for r in subset) if subset else "",
                        "不限制扩展均值": statistics.mean(r["不限制扩展节点数"] for r in subset) if subset else "",
                        "配对差值均值": statistics.mean(diffs) if diffs else "",
                        "配对差值标准差": statistics.stdev(diffs) if len(diffs) > 1 else (0 if diffs else ""),
                        # 单张地图无法估计地图间标准差，留空而不是伪装成零波动。
                        "地图平均差标准差": statistics.stdev(map_diffs) if len(map_diffs) > 1 else "",
                        "不限制更差对数": sum(d > 0 for d in diffs),
                        "有小区域地标地图数": sum(r["有小区域地标"] for r in maps),
                        "有小区域地标地图占比": sum(r["有小区域地标"] for r in maps) / len(maps) if maps else "",
                        "区域外地标比例": sum(r["区域外地标数"] for r in maps) / total_landmarks if total_landmarks else "",
                        "最大区域占比均值": statistics.mean(r["最大区域占比"] for r in maps) if maps else ""})
            summary.append(row)
    return summary


def plot_nesting(summary):
    kinds = ("U型陷阱", "迷宫(走廊2)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, kind in enumerate(kinds):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            notes = []
            for condition, color in zip(("嵌套", "非嵌套"), PALETTE):
                rows = sorted([r for r in summary if r["地图类型"] == kind and
                               r["方法"] == METHOD_NAMES[method] and r["条件"] == condition], key=lambda r: r["k"])
                ax.plot([r["k"] for r in rows], [r["扩展总数均值"] for r in rows],
                        marker="o", color=color, label=condition)
                denom = sum(r["相邻对数"] for r in rows)
                notes.append(f'{condition}：总逆转 {sum(r["总扩展逆转次数"] for r in rows)}/{denom}，'
                             f'f<C*逆转 {sum(r["f<C*逆转次数"] for r in rows)}/{denom}')
            # 逆转计数置于标题，避免覆盖k=1附近的折线。
            ax.set_title(f"{kind} · {METHOD_NAMES[method]}\n" + "\n".join(notes), color=INK, fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_xticks(NESTING_KS, [str(k) for k in NESTING_KS])
            ax.set_xlabel("地标数 k")
            ax.set_ylabel("平均扩展总数（含终点）")
            ax.legend(loc="lower left")
    fig.suptitle("消融1：固定地图与查询，仅改变跨 k 的选点种子是否共享", color=INK)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_fig(fig, f"{OUT}/exp_c_ablation_nesting.png")


def plot_component(summary):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, kind in enumerate(("随机30%", "随机35%")):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            x = np.arange(len(COMPONENT_KS))
            for offset, label, color in ((-.18, "有小区域地标", PALETTE[1]),
                                         (.18, "无小区域地标", PALETTE[0])):
                rows = [next(r for r in summary if r["地图类型"] == kind and
                             r["方法"] == METHOD_NAMES[method] and r["k"] == k and r["分组"] == label)
                        for k in COMPONENT_KS]
                # 空组用缺口和 n=0 标注，不能画成“差值恰为零”的假观测。
                means = [r["配对差值均值"] if r["地图数"] else np.nan for r in rows]
                errors = [r["地图平均差标准差"] if r["地图数"] > 1 else 0 for r in rows]
                ax.bar(x + offset, means, yerr=errors, width=.34, color=color,
                       label=label, capsize=3, ecolor=INK_2)
                for xi, row in zip(x + offset, rows):
                    ax.annotate(f'n={row["地图数"]}' + ("*" if row["地图数"] == 1 else ""), (xi, 0), xytext=(0, 4),
                                textcoords="offset points", ha="center", fontsize=7)
            ax.axhline(0, color=INK_2, linewidth=.8)
            ax.set_xticks(x, [str(k) for k in COMPONENT_KS])
            ax.set_xlabel("地标数 k")
            ax.set_ylabel("扩展数差值（不限制 − 限制）")
            ax.set_title(f"{kind} · {METHOD_NAMES[method]}", color=INK, fontsize=11)
            ax.legend(fontsize=8)
    fig.suptitle("消融2：误差线为地图平均差的样本标准差；n为地图数，*表示n=1无法估计", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_fig(fig, f"{OUT}/exp_c_ablation_component.png")


def main(seed=SEED):
    if seed != SEED:
        raise ValueError("本次固定消融主种子为0；改动实验配置需同步记录元数据")
    started = time.perf_counter()
    # 先冻结两套完整数据，再运行任何条件，排除条件循环改变地图或查询的混杂。
    nesting_cases = fixed_maps("nesting", seed)
    component_cases = fixed_maps("component", seed)
    nesting_raw = run_nesting(nesting_cases)
    nesting_summary = summarize_nesting(nesting_raw)
    write_csv(f"{OUT}/exp_c_ablation_nesting_raw.csv", nesting_raw)
    write_csv(f"{OUT}/exp_c_ablation_nesting_summary.csv", nesting_summary)
    plot_nesting(nesting_summary)
    component_raw = run_component(component_cases)
    component_summary = summarize_component(component_raw)
    write_csv(f"{OUT}/exp_c_ablation_component_raw.csv", component_raw)
    write_csv(f"{OUT}/exp_c_ablation_component_summary.csv", component_summary)
    plot_component(component_summary)
    for values, rows in _group(nesting_raw, ("地图类型", "方法", "条件")).items():
        denom = sum(r["相邻对数"] for r in rows)
        checks = sum(r["占优检查格数"] for r in rows)
        print("消融1汇总", values, "总逆转", sum(r["总扩展逆转"] for r in rows), "/", denom,
              "f<C*逆转", sum(r["f<C*逆转"] for r in rows), "/", denom,
              "占优比例", sum(r["占优通过格数"] for r in rows) / checks)
    for row in component_summary:
        print("消融2汇总", row)
    print(f"消融完成，用时 {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
