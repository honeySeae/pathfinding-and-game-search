# 实验二：寻路与博弈搜索

实现迷宫寻路、加权地形、Alpha-Beta剪枝和竞争宝藏分配四题；用对照实验分析启发函数、访问顺序与搜索深度的影响。
以下先说明题目一的环境、实验与复现方式，题目二至四见后续各节。
`solve_maze(start, end, grid)` 返回包含起终点的坐标列表，无解返回 `[]`；坐标为 `(row, col)`，`X` 为障碍。

## 环境与安装

本次实测：Windows 64 位，Python **3.14.2**、NumPy **2.4.6**、Matplotlib **3.11.0**、pytest **9.0.2**。
`requirements.txt` 的依赖下限为 NumPy 1.24、Matplotlib 3.7、pytest 7.0；无 SciPy 依赖。
报告图脚本生成 Word 预览，另需 `python-docx>=1.2`（已列入依赖）。
建议使用上述实测版本进行严格核验。代码也适用于有相应 Python/依赖的 macOS、Linux，但本次未在这些平台复测。

Windows PowerShell 中进入 `lab2/` 后安装：

```powershell
py -m pip install -r requirements.txt
```

可选：按本次实测版本安装，并查看实际解释器：

```powershell
py -m pip install numpy==2.4.6 matplotlib==3.11.0 pytest==9.0.2
py -c "import sys; print(sys.executable); print(sys.version)"
```

macOS/Linux 通常使用 `python3` 替代 `py`。绘图使用 Agg 后端，不需要图形界面。
Windows 优先使用微软雅黑/黑体；其他系统若中文显示为方块，需要安装 Noto Sans CJK SC 等中文字体。
CSV 采用 UTF-8 BOM，方便 Windows Excel 正确显示中文。

## 目录结构

|文件或目录|说明|
|---|---|
|`src/grid_utils.py`|四邻居、单位代价、BFS 对拍、Dijkstra 距离表及各种地图生成器。|
|`src/astar.py`|A* 主体、五种平局规则、搜索结果与统计结构。|
|`src/landmarks.py`|最大连通区域内选地标及 ALT 距离差启发函数。|
|`src/q1_maze.py`|作业接口 `solve_maze` 和给定地图的运行示例。|
|`tests/test_q1.py`|给定用例、边界输入、随机 BFS 对拍、排名及地标回归测试。|
|`experiments/common.py`|结果目录、CSV 保存、中文字体与绘图风格。|
|`experiments/q1_baseline.py`|h=0 与曼哈顿的基础节点数和热力图对比。|
|`experiments/q1_exp_a_tiebreak.py`|五种平局规则与 f<C*、f=C* 理论计数对照。|
|`experiments/q1_exp_b_maptypes.py`|地图类型、启发质量、相关分析及同图同端点的配对置乱实验。|
|`experiments/q1_exp_c_landmarks.py`|地标数量、选点方式、准确度、查询时间与预处理摊销比较。|
|`experiments/q1_exp_c_ablation.py`|固定地图/查询，分别消融地标前缀嵌套和最大连通区域限制，记录占优及配对差值。|
|`experiments/fig_style.py`|报告图共用样式。|
|`experiments/q1_report_figures.py`|由题一结果生成报告图及表。|
|`experiments/q2_report_figures.py`|由题二结果生成报告图。|
|`experiments/q3_report_figures.py`|由题三结果生成报告图。|
|`experiments/q4_report_figures.py`|由题四结果生成报告图。|
|`requirements.txt`|运行、测试与报告图预览所需的第三方依赖及版本下限。|
|`results/q1/`|实验 CSV；运行脚本会在本地生成 PNG，图片不纳入仓库。|
|`README.md`|环境、配置、执行顺序、结果解释及复现说明。|

## 运行顺序

**必须在 `lab2/` 根目录下用 `py -m 模块名` 运行**，不要直接执行 `py src/q1_maze.py`。
|顺序|命令|
|---|---|
|1|`py -m pytest -q`|
|2|`py -m src.q1_maze`|
|3|`py -m experiments.q1_baseline`|
|4|`py -m experiments.q1_exp_a_tiebreak`|
|5|`py -m experiments.q1_exp_b_maptypes`|
|6|`py -m experiments.q1_exp_c_landmarks`|
|7|`py -m experiments.q1_exp_c_ablation`|

各题 CSV 生成后，可按需重画报告图；这些脚本会在 `results/q1/report/`～`results/q4/report/` 写入 PNG/SVG 和 Word 预览，生成文件不纳入仓库：

```powershell
py -m experiments.q1_report_figures
py -m experiments.q2_report_figures
py -m experiments.q3_report_figures
py -m experiments.q4_report_figures
```

## 实验配置

默认入口不接收命令行参数；配置常量或 `main(seed=0)` 位于相应脚本内。

|实验|地图类型、尺寸|数量与查询数|方法/参数|随机种子|
|---|---|---|---|---|
|给定示例/基础用例|题面 5×5|1 图、1 查询；s=(0,0)、g=(3,4)|h=0、曼哈顿；默认 h_low 平局规则|A* 默认 0；无随机地图|
|基础随机对比|随机障碍 25%，40×40|50 张有效图，每图 1 查询，曼哈顿端点距离≥40|h=0、曼哈顿|地图/查询 RNG=0|
|A|空旷、随机20%、随机30%，40×40|每类30张，每图1查询；共90图×5规则=450条|曼哈顿；h_low、h_high、fifo、lifo、random；端点距离≥40|主 RNG=0；random 平局子种子由其生成|
|B 独立组|空旷、随机10/20/30%、迷宫走廊1，61×61|每类30张，每图1查询|h=0、曼哈顿；端点距离≥30|共享主 RNG=0|
|B 配对组|迷宫走廊2、走廊3、U型陷阱，61×61|每种结构30对，每对原始/置乱各1查询；B合计330实例|同图、同端点、同障碍数；每原图最多置乱20次，仍失败重生成原图|共享主 RNG=0，保留固定类型/循环顺序|
|C 普通组|迷宫走廊1/2、随机30%、空旷，61×61|每类10图×10查询=100查询|k∈{1,2,4,8,16,32}；random、farthest；h=0、曼哈顿、h*参考|主 RNG=0；每图每种选点方法生成一个子种子，各k共用|
|C U型陷阱|U型陷阱，61×61|20图×1个指定陷阱查询=20查询|方法与k同上；不随机替换陷阱端点|同 C 主 RNG=0|
|消融1：嵌套|U型陷阱、迷宫走廊2，61×61|U型20图×1查询；迷宫10图×10查询；两条件共用固定地图和查询|random/farthest；k=1,2,4,8,16,32；跨k共享种子 vs 各k独立种子；每查询共用300个占优采样点|独立主 RNG=0；raw保存地图/查询/地标/抽样种子|
|消融2：区域|随机30%、随机35%，61×61|每密度20图×10查询，查询固定在最大区域|random/farthest；k=1,2,4,8；限制 vs 不限制；同图同方法同种子，均嵌套|独立主 RNG=0；raw保存地图/查询/地标种子|

C 共420查询×15种方法=6300条原始记录。每次查询重复5次取内部耗时中位数；ALT构建重复3次取中位数；
准确度每个目标最多抽300个可达节点，同一查询的方法共用采样节点。
U型以外查询通过可达性及距离下限筛选；地标只放在最大连通区域。地标数量超过该区域格数时截断。
同一种子下 random/farthest 的 k 个地标恰好是 k+1 个的前 k 个。
测试随机地图使用 seed=2026，原 ALT 对拍使用 seed=7，新增前缀/退化测试使用固定 seed=42/1。

消融实验先冻结全部数据，再运行任何条件。消融1在k=1时两条件共用种子，从后续k开始改变种子耦合，
共2880条搜索记录；消融2每行同时记录限制/不限制的一对结果，共3200行、6400次搜索。
消融2“不限制”保留历史最远点法只在种子连通块采样的行为；实际地标数可能小于k。
区域外地标比例以实际地标数为分母，地图占比按方法和k分别计算。

## 结果与提交

仓库提交算法源码、测试、对照实验及报告图脚本、README、依赖清单与 `results/q1/`～`results/q4/` 的 CSV。
报告图已放在实验报告中，仓库保留生成脚本；题一实验脚本另生成辅助 PNG，图片本身不随仓库提交。
本实验没有训练模型或预测文件，各题最终输出分别是路径、代价或博弈得分。
以下题一结果均位于 `results/q1/`。

|文件|内容|
|---|---|
|`baseline_given.csv`|给定图两种启发函数的路径、节点数、入堆次数与堆峰值。|
|`baseline_random.csv`|50张随机图的最短路、两种方法节点数与节省率。|
|`exp_a_raw.csv`|逐图逐平局规则的实际扩展及 N_lt/N_eq 理论计数。|
|`exp_a_summary.csv`|实验A按类型/规则的均值、标准差与平台扩展占比。|
|`exp_b_raw.csv`|逐实例节点数、密度、q_start/q_mean、节省率；含编号、配对编号、起终点。|
|`exp_b_summary.csv`|各地图类型的均值和节省率标准差。|
|`exp_b_correlation.csv`|q_start、q_mean、密度分别与节省率的 Pearson/Spearman 相关。|
|`exp_b_regression.csv`|全部330实例的含截距OLS：仅q_mean、仅障碍密度、联合模型的R²，以及分别控制另一变量后的两个偏相关；使用NumPy实现，运行实验B时同时打印。|
|`exp_b_paired.csv`|三种结构的“置乱−原始”节省率差值均值、样本标准差及置乱更高的对数/总对数。|
|`exp_c_raw.csv`|逐地图/起终点/方法/k的节点数、时间中位数、预处理时间与准确度。|
|`exp_c_summary.csv`|按类型/方法/k汇总均值、提速比例与回本估计。|
|`report/per_node_cost.csv`|由exp_c_summary.csv计算的走廊2迷宫/U型陷阱每节点耗时（微秒）与ALT线性拟合；每节点耗时=查询耗时均值/扩展数均值×1000。|
|`exp_c_ablation_nesting_raw.csv`|逐地图/查询/方法/条件/k的总扩展、f<C*/f=C*扩展、相邻k逆转、占优采样及完整种子。|
|`exp_c_ablation_nesting_summary.csv`|按地图类型/方法/条件/k汇总均值、逆转次数/分母和占优比例；k=1无相邻比较，比例留空。|
|`exp_c_ablation_component_raw.csv`|限制/不限制的逐查询配对节点数、地标坐标、区域外数量/比例、最大区域占比及重放种子。|
|`exp_c_ablation_component_summary.csv`|逐密度/方法/k按全部、有/无区域外地标分组的配对均值、标准差、更差对数、地图数及区域外比例。|

`expanded` 的现有口径是有效出堆节点数，**包含终点**；堆峰值含重复入堆条目。
节省率为 `1 − 曼哈顿扩展 / h=0扩展`。q_start为起点处曼哈顿/真实距离，q_mean为目标可达节点上该比值的平均，排除目标处0/0。
配对编号只在同一结构内对应；独立组该列为空。

## 复现核验

给定5×5用例应输出7步、8个坐标，例如：

```text
[(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
```

存在多条最短路时输出任意合法最短路径均符合题意。当前固定邻居/平局次序可稳定得到本实现的路径。
`py -m pytest tests/test_q1.py -q` 预期 **33 passed**；覆盖原有测试以及平均排名、前缀嵌套、非正k/空图/全障碍退化、最大区域选点、
不限制时可选到小区域、区域外计数只读及不限制时仍保持前缀。

保持**相同代码、配置、依赖和随机数调用顺序**，扩展节点数等非时间字段可逐位复现；耗时受运行环境影响，
不能要求逐位一致，只比较多次测量下相对大小，接近的数值不宜强下结论。
修改前后共享主 RNG 的消耗次数变化会改变后续地图，因此旧/新同编号地图不保证相同。

地图编号不是种子，**仅凭编号与端点不能直接重新随机生成该图**。下列重放代码按固定主种子从头恢复生成顺序，
跳过绘图和不必要的计时；需使用生成该 CSV 的同一版本代码。先选定 CSV 的一行，然后用下面方式验证。

### 重放 B 的单条记录

在 `lab2/` 下启动 `py` 后粘贴，或将片段保存到根目录下的临时 Python 文件并运行。
例子选择走廊2原图编号0，改 `目标类型/目标编号` 可选其他记录。

```python
import ast
import csv
import random
from experiments.q1_exp_b_maptypes import KINDS, N_MAPS, gen_instance, gen_paired_instance
from src.astar import astar, manhattan

目标类型, 目标编号 = "迷宫(走廊2)", 0
with open("results/q1/exp_b_raw.csv", encoding="utf-8-sig") as f:
    row = next(r for r in csv.DictReader(f)
               if r["地图类型"] == 目标类型 and int(r["编号"]) == 目标编号)

def replay_b():
    rng = random.Random(0)
    for kind in KINDS:
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
                variants = [(kind, grid), (kind + "-置乱", shuffled)]
            else:
                grid, s, g = inst
                variants = [(kind, grid)]
            for name, current in variants:
                if name == row["地图类型"] and made == int(row["编号"]):
                    assert (s, g) == (ast.literal_eval(row["起点"]), ast.literal_eval(row["终点"]))
                    result = astar(s, g, current, heuristic=manhattan)
                    assert result.stats.expanded == int(row["曼哈顿扩展"])
                    return result
            made += 1

result = replay_b()
print(result.cost, result.stats.expanded)
```

### 重放 C 的单条记录

例子选择U型图0、最远点k=2的记录。预处理/计时内部不消耗主 RNG；准确度抽样会消耗，必须保留这一过程。
同图同端点出现多条相同方法记录时，节点数相同；CSV端点可用于核验你选择的查询。

```python
import ast
import csv
import random
from experiments.q1_exp_c_landmarks import (
    MAP_TYPES, MAPS_PER_TYPE, ACC_SAMPLES, gen_map,
)
from src.grid_utils import all_distances
from src.landmarks import LandmarkHeuristic
from src.astar import astar, manhattan, zero_heuristic

with open("results/q1/exp_c_raw.csv", encoding="utf-8-sig") as f:
    row = next(r for r in csv.DictReader(f) if r["地图类型"] == "U型陷阱"
               and int(r["地图"]) == 0 and r["方法"] == "ALT-最远点" and int(r["k"]) == 2)

def replay_c():
    rng = random.Random(0)
    for kind in MAP_TYPES:
        count = MAPS_PER_TYPE * 2 if kind == "U型陷阱" else MAPS_PER_TYPE
        for m in range(count):
            grid, queries = gen_map(kind, rng)
            seeds = {method: rng.random() for method in ["random", "farthest"]}
            for s, g in queries:
                distances = all_distances(grid, g)
                rng.sample(list(distances), min(ACC_SAMPLES, len(distances)))
                if (kind == row["地图类型"] and m == int(row["地图"])
                        and (s, g) == (ast.literal_eval(row["起点"]), ast.literal_eval(row["终点"]))):
                    name = row["方法"]
                    if name.startswith("ALT"):
                        method = "random" if name == "ALT-随机" else "farthest"
                        h = LandmarkHeuristic(grid, int(row["k"]), method, random.Random(seeds[method]))
                    elif name == "曼哈顿":
                        h = manhattan
                    elif name == "h=0":
                        h = zero_heuristic
                    else:
                        h = lambda n, goal: distances.get(n, 0)
                    result = astar(s, g, grid, heuristic=h)
                    assert result.stats.expanded == int(row["扩展节点数"])
                    return result

result = replay_c()
print(result.cost, result.stats.expanded)
```

## 已知限制

- Python 函数调用、堆操作、轨迹统计和计时本身有开销；中位数与禁用GC不能消除所有噪声，运行顺序也可能影响微小时间差。
- ALT 的绝对距离差使用距离对称性，适用于本题静态单位无向图；题二“进入格子”的加权代价通常不对称，不能直接复用。
- A* 不重新打开 closed，要求启发函数对实际边权一致，且 h(goal)=0；仅可采纳并不充分。进入终点代价改为0后需重新设计/检查启发函数。
- C 中5%是人为工程阈值，“提速低于5%”不是统计不显著；回本数是未舍入均值比值的摊销估计，仍保留小数，不保证每张图都在该次数回本。
- 地标只放在最大连通区域，避免随机种子落入零星小区域。并列最大区域按行列扫描遇到的第一个选择；其他区域查询默认退化为曼哈顿。
  查询生成器保留原有可达性/距离条件，并没有额外强制最大区域；如极端地图在其他大区域也能采到查询，该查询仍正确但得不到ALT帮助。
  新增参数 `restrict_to_largest_component=False` 仅供显式关闭限制；默认True保持原实验行为，
  `outside_largest_component_count` 是实际区域外地标数的只读属性。
- C 的100次普通查询只来自10张地图，不能视为100张独立图；U型是20张×1次指定查询，跨类型直接混合平均会改变权重。
- B 配对组是在“最多20次置乱能得到连通端点”的条件下采样；丢弃整对会引入筛选条件，配对差值支持该条件下的比较，不能单凭总体相关断言因果。
- h*是已知真实距离的参考启发函数；当前查询耗时不包括计算其距离表的成本，不能将它当作免费预处理的可部署方案。
- 消融中同数值种子作用于不同候选列表，不保证选中相同的主区域地标；因此“无区域外地标”组的差值也可能非零。
  不根据是否符合假设重新挑选地图或种子；空组不是零效应证据。配对统计以查询为单位，另给地图平均差标准差，
  图中n为地图数；本实验不提供统计显著性检验。

### 重放新增消融的实例

原始CSV包含独立地图种子，随机图可直接用 `random_map(61, density, random.Random(map_seed))` 重建，
U型用 `u_trap_map`，走廊2用 `maze_map(61, random.Random(map_seed), 2)`；坐标列可用 `ast.literal_eval` 读取。
也可调用 `experiments.q1_exp_c_ablation.fixed_maps("nesting")` 或 `fixed_maps("component")` 按类型和编号取固定实例，
它们不会搜索或写实验结果。根据raw行的地标种子、方法、k构造LandmarkHeuristic；消融1默认限制True，
消融2分别传True/False。占优检查用raw抽样种子从 `list(all_distances(grid, goal))` 抽样，便可逐位重放。

## 题目二 加权地形

复用 `src/astar.py` 和 `src/grid_utils.py`，不修改两者及题一结果。数字是进入格子的代价，
**8也是可通行地形**；仅 `X` 作为兼容障碍，进入 `E` 为0，起点初始代价为0。
题面未规定重入S的代价，`make_terrain_cost(grid)` 将其设为数字格最小值w_min：正权最优路径不会再次进入起点，
该选择不影响从S出发的解，并保证所有非终点格进入代价至少为w_min。
无数字格时w_min约定为1，不支持负代价。

`terrain_heuristic(grid)` 返回 `max(0, Manhattan-1)*w_min`。
到终点至少走m步，前m−1次进入非终点格的代价至少为w_min，最后进入终点为0，所以可采纳；
普通移动时h最多减少w_min，进入终点时两端h均为0，所以一致。
`solve_terrain(start,end,grid)` 返回最小总代价int，无解为−1；
`path_cost_trace` 返回最优路径上 `(坐标,进入代价,累计代价)`，首项起点为 `(start,0,0)`，无解为空列表。

|新增文件|说明|
|---|---|
|`src/q2_terrain.py`|地形代价函数工厂、可采纳启发函数、求解接口、路径代价跟踪及题面示例。|
|`tests/test_q2.py`|题面和边界测试、方向性距离、逐格可采纳/逐边一致，以及300图五规则对拍。|
|`experiments/q2_exp_overestimate.py`|比较常数平移与加权A*，检验最优性、w倍界与扩展数。|

### 运行命令与核验

仍需在 `lab2/` 根目录执行。

```powershell
py -m pip install -r requirements.txt
py -m pytest -q
py -m src.q2_terrain
py -m experiments.q2_exp_overestimate
```

当前完整测试套件预期94项通过。
给定5×5用例最优代价7，路径为
`(0,0)→(1,0)→(2,0)→(3,0)→(3,1)→(3,2)→(4,2)→(4,3)→(4,4)`；
初始累计0，前7步每步加1，进入E加0，最终7。起点曼哈顿8高于真实代价7，但本例仍最优。

### 拓展实验配置

|项目|配置|
|---|---|
|地图与样本|30×30；平原为主200张（代价1/2/3/8概率70%/15%/10%/5%），混合地形200张（1–9均匀）；主种子0，起终点曼哈顿距离≥15。|
|启发函数|h=0、h_adm、曼哈顿、h_adm+5/+50（终点为0）、w×h_adm（w=1.2、1.5、2、3、5）。|
|平局与指标|默认优先h小；曼哈顿及h_adm+50另测其余4种平局规则。比较出错率、解代价比、扩展节点数，并逐例检查w倍代价界。|

实验地图均含代价1，故非终点上曼哈顿距离=h_adm+1；该关系和常数平移的最优性结论不推广到任意地图或启发函数。加权代价具有方向性，逐格可采纳测试从每个格子正向求真实剩余代价。

### 题二结果位置

|路径|内容|
|---|---|
|`results/q2/exp_overestimate_raw.csv`|逐图逐方法的最优/返回代价、代价比、扩展数、错误和w倍界标记，包含地图/搜索种子及端点。|
|`results/q2/exp_overestimate_summary.csv`|36组汇总：出错率、代价比均值/最大值、平均扩展数、相对h=0比例、曼哈顿不多于下界的比例。|
|`results/q2/exp_overestimate_hratio.csv`|每类200个独立实例的h_adm(起点)/C*均值和样本标准差；从raw按实例去重计算，不重复计入18种配置。|

扩展节点数包含终点；相对h=0的扩展比例为两组扩展数均值之比，节点数配对只在同一平局规则下比较。
通过 `make_instance(地图类型, 地图种子)` 可重建单张图及其端点，原始CSV的搜索种子重放random规则。
已有raw时运行 `py -m experiments.q2_exp_overestimate` 默认仅补算下界质量，不重跑搜索或改写旧统计；
如确需重新生成实验，显式运行 `py -m experiments.q2_exp_overestimate --rerun`。
新增实验不要求时间复现，也不使用计时作结论。

## 题目三 Alpha-Beta剪枝

`solve_game_tree(root)`以MAX为根、默认从左到右搜索，返回最佳得分。手工TreeNode按层交替，
JSON的显式 `is_maximizing_player` 优先，否则继承父角色的交替值。
`from_json`接受带/不带 `tree_structure` 的字典或JSON文本，叶子为 `{"value": 3}`。
`build_test_tree`严格保留同一个max2被min1和min2共同引用的结构。
搜索不做共享节点缓存，每次递归访问分别计数；只有预计算理想排序时缓存真实值，缓存键包含节点及有效角色。

|新增文件|说明|
|---|---|
|`src/game_tree.py`|TreeNode、JSON解析、题面共享树、Minimax、Alpha-Beta、统计/剪枝跟踪及真实值排序工具。|
|`tests/test_q3.py`|共享节点计数、JSON角色、等号剪枝、300随机树多种顺序对拍及24组精确理论计数。|
|`experiments/q3_exp_ordering.py`|三部分排序实验：题面树、分支/深度影响、带噪声的排序准确度。|

### 题三运行顺序

在lab2根目录执行；不修改题一、二代码及结果，不启动任何桌面应用。

```powershell
py -m pytest -q
py -m src.game_tree
py -m experiments.q3_exp_ordering
```

题面例返回5，Minimax评估8个叶子、访问15次节点；
从左到右Alpha-Beta评估7个叶子、访问14次节点，只在第一次访问max2时剪去叶子9。
再次访问共享max2时会重新搜索，不能用visited集合跳过。
`GameStats.nodes/leaves/prunes`分别为访问次数、评估叶子次数及实际跳过后续子节点的剪枝事件数。
trace记录节点、alpha/beta和被跳过的直接子节点名称。

### 三部分实验配置

|部分|配置与统计|
|---|---|
|Part1|题面共享树；Minimax与Alpha-Beta从左到右、从右到左、真实值最优排序、反向排序。|
|Part2|主种子0；b=2、3、5、8，d=2、4、6、8，跳过b^d>400000；其余15种组合各30棵、共450棵，叶子值从[0,1)抽取。比较随机、最优、反向排序的叶子数及有效分支因子（叶子数^(1/d)）。|
|Part3|复用b=5、d=6的30棵树；按真实值加高斯噪声排序，σ=0、0.01、0.02、0.05、0.1、0.2、0.5、1，另设随机排序；记录叶子比例及首选准确率。|

最优排序预先使用真实Minimax值，仅作理想参照，排序成本不计入搜索开销；反向排序为MAX按真实值升序、MIN降序。Part2逐树核对理论叶子数 `b^ceil(d/2)+b^floor(d/2)-1`；Part3首选准确率按实际访问节点计算，各树等权平均。实验主指标为节点数，不测时间；原始CSV保存树种子和排序种子。

### 题三结果位置

|路径|内容|
|---|---|
|`results/q3/exp_ordering_case.csv`|Part1的5行结果、叶子/节点/剪枝次数和JSON剪枝记录。|
|`results/q3/exp_ordering_raw.csv`|Part2逐树逐顺序的1350条记录、种子、叶子数、节点数、有效分支因子与理论核验。|
|`results/q3/exp_ordering_summary.csv`|Part2的45组均值/样本标准差、最优理论不符次数。|
|`results/q3/exp_accuracy_raw.csv`|Part3的270条记录、叶子比例、首选准确度、全树准确度与种子。|
|`results/q3/exp_accuracy_summary.csv`|Part3各σ的叶子均值/标准差、相对Minimax比例及排序准确度。|
|`results/q3/exp_cutoff_depth.csv`|b=5、d=6的30棵树，三种顺序的浅剪枝/深剪枝事件总数及每树均值。|

### 题三界来源与深剪枝分析

alpha/beta记录严格收紧时的来源层数；MAX剪枝看beta来源，MIN剪枝看alpha来源。来源于父节点为浅剪枝，更上层为深剪枝；统计单位是实际跳过后续子节点的剪枝事件，结果见 `results/q3/exp_cutoff_depth.csv`。

## 题目四 竞争环境下的宝藏分配

地图中数字为进入代价，X不可通行，A、B及宝藏格的进入代价统一为1。
每回合必须取一个尚未获取的宝藏，收益为价值减去最短移动代价；途经其他宝藏不自动收集，另一玩家不阻挡。
A最大化、B最小化当前累计收益差U，H按单个玩家回合计数。题面结果只使用`eval_score`。
本题直接调用已有`src/astar.py`，不修改题一至三的算法、脚本或结果。

### 题四文件与运行

|文件|用途|
|---|---|
|`src/q4_treasure.py`|内置原题、A*距离矩阵、不可变状态、Minimax/Alpha-Beta、三种评价函数、首步值与最优对局。|
|`tests/test_q4.py`|题面回归、Dijkstra距离对拍、100个随机实例的三评价×四排序对拍、实验真值缓存核验。|
|`experiments/q4_exp_order_and_eval.py`|原题访问顺序比较与700个固定随机实例的评价函数实验。|

在`lab2/`目录下按顺序运行，无新增依赖：

```powershell
py -m pytest -q
py -m src.q4_treasure
py -m experiments.q4_exp_order_and_eval
py -m experiments.q4_exp_order_and_eval --supplement
```

`--supplement`仅补测已保存700个实例的指定深度，不重新生成实例或覆盖历史计时；重复运行会跳过已有配置。
已经有原始实验CSV时，只需运行补测命令。
原题H=2输出`{"best_target": "T2", "best_value": -1}`，H=4输出`{"best_target": "T3", "best_value": 3}`。

### 题四实验配置与核验口径

|部分|配置|
|---|---|
|Part1|原题7×7、4个宝藏；H=2、4；Minimax及Alpha-Beta的fixed、reverse、gain、gain_asc顺序。另比较H=1–4的三种评价。|
|随机实例|主种子0；11×11，每格15%为X，其余代价1/2/3/5/8的概率为60%/15%/10%/10%/5%；标记格互异且连通、代价为1，宝藏价值为5–12。n=6取500例，n=8取200例。|
|方法与深度|eval_score、eval_closer、eval_rollout；Alpha-Beta按眼前收益降序访问。原测n=6的H=1–4、n=8的H=1–5；补测原评价至H=n、贪心推演至H=n−1。|
|指标|以H=n、eval_score求各首步真实值；并列最优均算选对。记录选对率及Wilson 95%区间、平均收益损失（最优真实值−所选首步真实值）、访问节点数；每配置搜索5次取耗时中位数。|

`GameState`记录双方位置、剩余宝藏、累计收益、当前玩家和剩余深度；`gain`按当前玩家眼前收益降序，`eval_closer`忽略轮次，`eval_rollout`按轮次贪心至结束。叶子推演步骤不计入搜索节点，节点数不能单独代表耗时。标记格进入代价均为1，点间距离双向计算并验证对称；不连通实例被拒绝。

### 题四结果位置

|路径（位于`results/q4/`）|内容|
|---|---|
|`exp_case.csv`|原题两深度的10组算法/排序结果、节点、叶子与剪枝次数。|
|`exp_first_steps.csv`|H=2、4时A每个首步的精确截断值。|
|`exp_pv.csv`|H=2、4最优对局中的玩家、目标、单步收益与U。|
|`exp_case_eval.csv`|原题H=1–4、三种评价函数的首步与估值。|
|`exp_instances.csv`|700个实例的种子、重生次数、完整地图、坐标、价值、距离矩阵和各首步真实值。|
|`exp_order_and_eval_raw.csv`|补测后11500条逐实例/深度/评价记录，原9000条保留；含真实收益损失、统计、5次耗时与中位数。|
|`exp_order_and_eval_summary.csv`|补测后35组结果；增加r=n−H（截断后剩余回合数），其余为选对率、Wilson区间、平均损失、平均节点/叶子/剪枝与平均中位耗时。|

单例重放：从原始CSV取`n`与`seed`，调用`random_instance(n, seed)`；再以`dataclasses.replace(state, depth_left=H)`设定深度，
调用`alphabeta(..., order="gain", evaluate=EVALUATORS[name])`。实例表另保存完整输入，可核验重建是否一致。
得分、首步和节点计数可复现；耗时受Python对象分配、系统调度与后台负载影响，仅作相对比较。
历史深度的计时原样保留，新增深度为另次补测；不能据小幅耗时差异断言提速。
