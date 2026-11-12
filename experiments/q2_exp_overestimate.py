"""题二拓展：不可采纳的启发函数何时仍最优，何时会出错。

现象：给定例中曼哈顿距离在起点高估(8>7)，却返回最优代价。
假设：本实验每张地图w_min=1，非终点上Manhattan=h_adm+1；一般数字最小值
大于1时，这个等式并不成立。h_adm+c只对非终点加同一常数，不改变非终点
之间的堆排序（五种平局规则均如此）。每个终点前驱的h_adm=0，第一次被取出
的终点前驱已具有最优代价，因最后一条边为0，此时生成的终点也已最优；
终点的h保持0只会使它更早出堆。因此常数平移在该结构下仍最优，不能推广为
任意图/启发函数加常数都安全。比例放大w*h_adm则改变各节点相对优先级，
可能次优；h_adm一致、边权非负时，即使当前A*不重开closed也有w倍代价界。
验证：两类30×30随机图各200张、端点距离>=15、主种子0；以h=0为最优值。
默认h_low比较10种启发式，另对Manhattan与h_adm+50补齐其余4种平局规则。
记录成本比、出错率与扩展数，并检查w倍界；不根据结果重新挑图或调种子。

运行：py -m experiments.q2_exp_overestimate
输出：results/q2/exp_overestimate_raw.csv、exp_overestimate_summary.csv、exp_overestimate_hratio.csv。
已有raw时默认只补算起点下界质量，不重跑搜索、不改写原始/汇总CSV；--rerun显式重跑。
"""
import ast
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

from src.astar import TIE_BREAKS, astar, manhattan, zero_heuristic
from src.q2_terrain import make_terrain_cost, minimum_terrain_weight, terrain_heuristic
from .common import out_dir, write_csv

N = 30
MAPS_PER_TYPE = 200
MIN_DISTANCE = 15
SEED = 0
MAP_TYPES = ("平原为主", "混合地形")
WEIGHTS = (1.2, 1.5, 2, 3, 5)
OUT = out_dir("q2")


def make_instance(kind, seed):
    """地图种子独立生成权重、端点及平局子种子，可直接重放单图。"""
    rng = random.Random(seed)
    if kind == "平原为主":
        values = rng.choices((1, 2, 3, 8), weights=(.70, .15, .10, .05), k=N * N)
    elif kind == "混合地形":
        values = [rng.randint(1, 9) for _ in range(N * N)]
    else:
        raise ValueError(kind)
    grid = [[str(v) for v in values[r*N:(r+1)*N]] for r in range(N)]
    for _ in range(10000):
        start = (rng.randrange(N), rng.randrange(N))
        end = (rng.randrange(N), rng.randrange(N))
        if manhattan(start, end) >= MIN_DISTANCE:
            break
    else:
        raise RuntimeError("无法选到距离至少15的端点")
    grid[start[0]][start[1]], grid[end[0]][end[1]] = "S", "E"
    return grid, start, end, {rule: rng.randrange(2**63) for rule in TIE_BREAKS}


def configurations(adm):
    """终点始终h=0；c只加在非终点上，w作用于整个一致下界。"""
    rows = [("h=0", 0, 0, zero_heuristic), ("h_adm", 1, 0, adm), ("曼哈顿距离", 1, 1, manhattan)]
    for c in (5, 50):
        rows.append((f"h_adm+{c}", 1, c, lambda n, g, c=c: 0 if n == g else adm(n, g) + c))
    for w in WEIGHTS:
        rows.append(("w×h_adm", w, 0, lambda n, g, w=w: w * adm(n, g)))
    return rows


def summarize(raw):
    groups = defaultdict(list)
    for row in raw:
        groups[(row["地图类型"], row["启发函数"], row["平局规则"], row["w"], row["c"])].append(row)
    summary = []
    for (kind, method, rule, w, c), rows in groups.items():
        compared = [r["曼哈顿扩展≤h_adm"] for r in rows if r["曼哈顿扩展≤h_adm"] != ""]
        row = {"地图类型": kind, "启发函数": method, "平局规则": rule, "w": w, "c": c,
               "实例数": len(rows), "出错数": sum(r["出错"] for r in rows),
               "出错率": statistics.mean(r["出错"] for r in rows),
               "代价比均值": statistics.mean(r["代价比"] for r in rows),
               "代价比最大值": max(r["代价比"] for r in rows),
               "平均扩展节点数": statistics.mean(r["扩展节点数"] for r in rows),
               "h=0平均扩展节点数": statistics.mean(r["h=0扩展节点数"] for r in rows),
               # 用均值之比画图；与逐图比例的平均不同，口径在此固定。
               "扩展相对h=0比例": statistics.mean(r["扩展节点数"] for r in rows) / statistics.mean(r["h=0扩展节点数"] for r in rows),
               "曼哈顿扩展≤h_adm次数": sum(compared) if compared else "",
               "曼哈顿扩展比较实例数": len(compared) if compared else "",
               "曼哈顿扩展≤h_adm比例": statistics.mean(compared) if compared else "",
               "w倍界违反次数": sum(r["w倍界违反"] for r in rows)}
        summary.append(row)
    return summary


def write_hratio(raw):
    """按实例去重计算h_adm(start)/C*，不把18种搜索配置重复计作独立样本。"""
    instances = {}
    for row in raw:
        key = (row["地图类型"], int(row["地图编号"]))
        start = ast.literal_eval(row["起点"]) if isinstance(row["起点"], str) else row["起点"]
        end = ast.literal_eval(row["终点"]) if isinstance(row["终点"], str) else row["终点"]
        identity = (int(row["地图种子"]), start, end, float(row["w_min"]), float(row["最优代价"]))
        if key in instances and instances[key] != identity:
            raise ValueError(f"同一实例的端点或最优代价不一致：{key}")
        instances[key] = identity
    summary = []
    for kind in MAP_TYPES:
        ratios = []
        for (map_kind, _), (_, start, end, w_min, optimal) in instances.items():
            if map_kind != kind:
                continue
            if optimal <= 0:
                raise ValueError("本实验端点距离至少15，应有正的最优代价")
            # 使用raw已保存的端点、w_min和C*，无需重新生成地图或执行A*。
            ratio = max(0, manhattan(start, end) - 1) * w_min / optimal
            assert 0 <= ratio <= 1 + 1e-12
            ratios.append(ratio)
        if len(ratios) != MAPS_PER_TYPE:
            raise ValueError(f"{kind}实例数应为{MAPS_PER_TYPE}，实际为{len(ratios)}")
        summary.append({"地图类型": kind, "实例数": len(ratios),
                        "h_adm起点/C*均值": statistics.mean(ratios),
                        "h_adm起点/C*标准差": statistics.stdev(ratios)})
    write_csv(f"{OUT}/exp_overestimate_hratio.csv", summary)
    for row in summary:
        print("起点下界质量：", row)
    return summary


def main(seed=SEED, reuse_existing=True):
    raw_path = Path(OUT) / "exp_overestimate_raw.csv"
    if reuse_existing and raw_path.exists():
        with raw_path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        if len(existing) != len(MAP_TYPES) * MAPS_PER_TYPE * 18 or any(int(r["主种子"]) != seed for r in existing):
            raise ValueError("已有raw的数量或主种子不符；如需重跑请显式使用--rerun")
        print(f"读取已有{len(existing)}条记录，仅补算下界质量；不重跑搜索或覆盖已有统计。")
        write_hratio(existing)
        return
    rng = random.Random(seed)
    raw = []
    # 全部地图及端点先固定；添加某种搜索配置不会移动之后地图的随机序列。
    cases = [(kind, index, rng.randrange(2**63)) for kind in MAP_TYPES for index in range(MAPS_PER_TYPE)]
    for kind, index, map_seed in cases:
        grid, start, end, search_seeds = make_instance(kind, map_seed)
        cost, adm = make_terrain_cost(grid), terrain_heuristic(grid)
        w_min = minimum_terrain_weight(grid)
        # 检验常数+1的前提，不静默删除不符合前提的实例。
        assert w_min == 1, (kind, index, "本图不能用Manhattan=h_adm+1解释")
        assert manhattan(start, end) == adm(start, end) + 1
        configs = configurations(adm)
        optimal_result = astar(start, end, grid, cost_fn=cost, heuristic=zero_heuristic,
                               tie_break="h_low", seed=search_seeds["h_low"], goal_cost_override=0)
        optimal = optimal_result.cost
        assert optimal > 0
        adm_result = astar(start, end, grid, cost_fn=cost, heuristic=adm,
                           tie_break="h_low", seed=search_seeds["h_low"], goal_cost_override=0)
        assert adm_result.cost == optimal
        jobs = [(config, "h_low") for config in configs]
        jobs += [(config, rule) for rule in TIE_BREAKS if rule != "h_low" for config in configs
                 if config[0] in ("曼哈顿距离", "h_adm+50")]
        for (method, w, c, h), rule in jobs:
            if method == "h=0":
                result = optimal_result
            elif method == "h_adm":
                result = adm_result
            else:
                result = astar(start, end, grid, cost_fn=cost, heuristic=h, tie_break=rule,
                               seed=search_seeds[rule], goal_cost_override=0)
            assert result.cost >= optimal
            bound_failed = int(method == "w×h_adm" and result.cost > w * optimal + 1e-9)
            assert not bound_failed, (kind, index, method, w, result.cost, optimal)
            raw.append({"地图类型": kind, "地图编号": index, "主种子": seed, "地图种子": map_seed,
                        "搜索种子": search_seeds[rule], "起点": start, "终点": end, "w_min": w_min,
                        "启发函数": method, "平局规则": rule, "w": w, "c": c,
                        "最优代价": int(optimal), "解代价": int(result.cost),
                        "出错": int(result.cost != optimal), "代价比": result.cost / optimal,
                        "扩展节点数": result.stats.expanded, "h=0扩展节点数": optimal_result.stats.expanded,
                        "h_adm扩展节点数(h_low)": adm_result.stats.expanded,
                        "曼哈顿扩展≤h_adm": int(result.stats.expanded <= adm_result.stats.expanded)
                        if method == "曼哈顿距离" and rule == "h_low" else "",
                        "w倍界违反": bound_failed})
        if (index + 1) % 50 == 0:
            print(f"{kind}：{index + 1}/{MAPS_PER_TYPE} 完成", flush=True)
    write_csv(f"{OUT}/exp_overestimate_raw.csv", raw)
    summary = summarize(raw)
    write_csv(f"{OUT}/exp_overestimate_summary.csv", summary)
    for row in summary:
        print(row)
    write_hratio(raw)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="显式重跑全部搜索并覆盖本实验raw/summary")
    args = parser.parse_args()
    main(reuse_existing=not args.rerun)
