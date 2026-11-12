"""题目一报告图表：读取已有CSV，仅为图1重放地图并各运行一次A*。

运行：py -m experiments.q1_report_figures
输出：results/q1/report/ 内两图的300 dpi PNG和SVG、表1的CSV/Markdown、
      每节点耗时与线性拟合per_node_cost.csv、图注及preview.docx。
不调用任何已有实验的main，不改写已有实验输出，也不重新测量时间。
"""
import ast
import csv
import hashlib
import json
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from . import fig_style as style
from src.astar import astar, manhattan, zero_heuristic
from src.grid_utils import all_distances, maze_map, open_map, pick_pair, random_map, shuffle_obstacles, u_trap_map

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "q1"
OUT = SOURCE / "report"
CM = style.CM
plt = style.plt
TABLE_COLUMNS = ["地图", "方法", "扩展节点数", "查询耗时 (ms)", "相对曼哈顿提速",
                 "预处理耗时 (ms)", "回本查询次数"]
TABLE_TITLE = "表 1 地标启发函数的搜索开销与预处理摊销"
FIG1_CAPTION = ("图 1 U 型陷阱中 h=0 与曼哈顿启发函数的扩展范围。两图使用实验 B 的同一地图和起终点。"
                "两个子图分别将扩展次序归一化，蓝色由浅至深仅表示各自搜索中的先后，不能跨图比较绝对扩展编号；"
                "橙线为最短路径。该实例中曼哈顿减少约 {saving:.1%} 的扩展节点，节点数包含终点。")
FIG2_CAPTION = ("图2　(a) 各类地图上曼哈顿距离相对h=0的节点节省率（61×61，每类30张），"
                "蓝点为原始地图，橙点为打乱障碍位置后的结果，两点间连线越长表示障碍形状的影响越大；"
                "走廊宽1的迷宫障碍密度达51.7%，打乱后起终点基本不连通，未做置乱对照。"
                "(b) 走廊宽2的迷宫上，两种地标选取方式的扩展节点数随地标数k的变化。")


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def make_table(c_summary):
    rows = []
    for kind, label in (("迷宫(走廊2)", "迷宫（走廊 2）"), ("U型陷阱", "U 型陷阱")):
        for method, k in (("曼哈顿", 0), ("ALT-最远点", 2), ("ALT-最远点", 8), ("ALT-最远点", 32)):
            source = next(r for r in c_summary if r["地图类型"] == kind and r["方法"] == method and int(r["k"]) == k)
            rows.append({"地图": label, "方法": method if k == 0 else f"{method} k={k}",
                         "扩展节点数": f'{float(source["扩展节点均值"]):.1f}',
                         "查询耗时 (ms)": f'{float(source["查询耗时ms均值"]):.3f}',
                         "相对曼哈顿提速": "—" if k == 0 else f'{float(source["提速比例"]):.1%}',
                         "预处理耗时 (ms)": "—" if k == 0 else f'{float(source["预处理耗时ms均值"]):.3f}',
                         "回本查询次数": "—" if k == 0 else source["回本查询次数"]})
    return rows


def save_table(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "table1_alt_cost.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    markdown = [TABLE_TITLE, "", "|" + "|".join(TABLE_COLUMNS) + "|",
                "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"]
    markdown += ["|" + "|".join(row[key] for key in TABLE_COLUMNS) + "|" for row in rows]
    (OUT / "table1_alt_cost.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def per_node_cost_analysis(c_summary):
    """从已有汇总计算均值之比，不重测时间；仅对ALT六点拟合每节点耗时=a+b*k。"""
    rows = []
    for kind in ("迷宫(走廊2)", "U型陷阱"):
        alt_points = []
        for method, k in [("曼哈顿", 0)] + [("ALT-最远点", k) for k in (1, 2, 4, 8, 16, 32)]:
            source = next(r for r in c_summary if r["地图类型"] == kind and r["方法"] == method and int(r["k"]) == k)
            nodes = float(source["扩展节点均值"])
            time_ms = float(source["查询耗时ms均值"])
            # 毫秒转微秒：这里是查询耗时均值/扩展数均值，并非逐查询比值的平均。
            cost_us = time_ms / nodes * 1000
            rows.append({"记录类型": "逐点", "地图": kind, "方法": method, "k": k,
                         "扩展节点均值": nodes, "查询耗时ms均值": time_ms,
                         "每节点耗时(微秒)": cost_us, "a": "", "b": "", "相关系数": ""})
            print(f"每节点耗时：{kind}，{method}，k={k}，{cost_us:.6f} 微秒")
            if method == "ALT-最远点":
                alt_points.append((k, cost_us))
        points = np.asarray(alt_points, dtype=float)
        ks, costs = points[:, 0], points[:, 1]
        # 含截距OLS使用lstsq；r是k与每节点耗时的Pearson相关，不是R²。
        a, b = np.linalg.lstsq(np.column_stack((np.ones(len(ks)), ks)), costs, rcond=None)[0]
        correlation = float(np.corrcoef(ks, costs)[0, 1])
        rows.append({"记录类型": "线性拟合", "地图": kind, "方法": "ALT-最远点", "k": "",
                     "扩展节点均值": "", "查询耗时ms均值": "", "每节点耗时(微秒)": "",
                     "a": float(a), "b": float(b), "相关系数": correlation})
        print(f"线性拟合：{kind}，每节点耗时(微秒) = {a:.6f} + {b:.6f}·k，r={correlation:.6f}")
    with (OUT / "per_node_cost.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def reconstruct_first_utrap(b_raw, b_summary):
    """恢复B的RNG消耗顺序；只生成地图和筛选连通性，不重跑B的搜索/统计。"""
    target = next(r for r in b_raw if r["地图类型"] == "U型陷阱")
    counts = Counter(r["地图类型"] for r in b_raw)
    kinds = [r["地图类型"] for r in b_summary]
    rng = random.Random(0)
    for kind in kinds:
        if kind.endswith("-置乱"):
            continue
        made = 0
        while made < counts[kind]:
            if kind == "U型陷阱":
                grid, s, g = u_trap_map(61, rng)
            else:
                if kind == "空旷":
                    grid = open_map(61)
                elif kind.startswith("随机"):
                    grid = random_map(61, int(kind[2:4]) / 100, rng)
                else:
                    grid = maze_map(61, rng, int(kind.split("走廊")[1][0]))
                pair = pick_pair(grid, rng, 61 // 2)
                if pair is None:
                    continue
                s, g = pair
            if kind + "-置乱" in kinds:
                for _ in range(20):
                    shuffled = shuffle_obstacles(grid, rng, keep_free=[s, g])
                    if g in all_distances(shuffled, s):
                        break
                else:
                    continue
            if kind == "U型陷阱" and made == int(target["编号"]):
                assert (s, g) == (ast.literal_eval(target["起点"]), ast.literal_eval(target["终点"]))
                # 热力图只运行这两次A*；节点数须与B原始CSV逐位一致。
                r0 = astar(s, g, grid, heuristic=zero_heuristic)
                r1 = astar(s, g, grid, heuristic=manhattan)
                assert r0.stats.expanded == int(target["h=0扩展"])
                assert r1.stats.expanded == int(target["曼哈顿扩展"])
                assert r0.cost == r1.cost == float(target["C*"])
                return grid, s, g, r0, r1
            made += 1
    raise RuntimeError("无法恢复B的第一张U型陷阱；请核对CSV与生成器版本")


def save_fixed_size(fig, name):
    """样式文件原样保留；本脚本不tight裁切，保证SVG及Word的物理尺寸。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        if item.get_fontsize() < 7:
            raise ValueError(f"字号小于7 pt: {item.get_text()}")
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            raise ValueError(f"文字超出画布: {item.get_text()}")
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)


def figure1(instance):
    grid, start, goal, r0, r1 = instance
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 7.5 * CM), gridspec_kw={"wspace": .06})
    fig.subplots_adjust(left=.015, right=.91, bottom=.035, top=.80)
    walls = np.ma.masked_where(np.array(grid) != "X", np.ones((len(grid), len(grid[0]))))
    for ax, letter, title, result in zip(axes, ("a", "b"), ("h = 0", "曼哈顿距离"), (r0, r1)):
        order = np.full((len(grid), len(grid[0])), np.nan)
        for i, (r, c) in enumerate(result.stats.expanded_order):
            order[r, c] = i / max(1, result.stats.expanded - 1)
        ax.set_facecolor("white")
        ax.imshow(np.ma.masked_invalid(order), cmap=style.EXPAND_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(walls, cmap=ListedColormap([style.WALL]), vmin=0, vmax=1, interpolation="nearest")
        yy, xx = zip(*result.path)
        ax.plot(xx, yy, color=style.ORANGE, linewidth=1.4)
        ax.plot(start[1], start[0], "o", color="black", markersize=5, markeredgecolor="white", markeredgewidth=.8)
        ax.plot(goal[1], goal[0], "*", color=style.ORANGE, markersize=8.5, markeredgecolor="white", markeredgewidth=.6)
        ax.set_xlim(-.5, len(grid[0]) - .5)
        ax.set_ylim(len(grid) - .5 + .8, -.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#c9c8c2")
            spine.set_linewidth(.6)
        ax.set_title(f"({letter}) {title}，扩展 {result.stats.expanded} 个节点", loc="left", fontsize=8, pad=4)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=style.EXPAND_CMAP),
                       ax=list(axes), fraction=.025, pad=.02)
    bar.set_ticks([0, 1], labels=["先", "后"])
    bar.ax.tick_params(length=0, labelsize=7.5)
    bar.set_label("扩展次序（先 → 后）", fontsize=7.5, labelpad=3)
    bar.outline.set_visible(False)
    handles = [Patch(facecolor=style.WALL, label="障碍"),
               Line2D([], [], color=style.ORANGE, linewidth=1.4, label="最短路径"),
               Line2D([], [], marker="o", linestyle="none", color="black", markersize=5,
                      markeredgecolor="white", markeredgewidth=.8, label="起点"),
               Line2D([], [], marker="*", linestyle="none", color=style.ORANGE, markersize=8.5,
                      markeredgecolor="white", markeredgewidth=.6, label="终点")]
    style.top_legend(fig, handles, ncol=4, y=.91)
    save_fixed_size(fig, "fig1_utrap_heatmap")


def figure2(b_summary, c_summary):
    fig, (left, right) = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM),
                                    gridspec_kw={"width_ratios": [1.05, 1], "wspace": .5})
    fig.subplots_adjust(left=.16, right=.985, bottom=.25, top=.83)
    kinds = ["空旷", "随机10%", "随机20%", "随机30%", "迷宫(走廊3)", "迷宫(走廊2)", "迷宫(走廊1)", "U型陷阱"]
    labels = ["空旷", "随机 10%", "随机 20%", "随机 30%", "迷宫·走廊 3", "迷宫·走廊 2", "迷宫·走廊 1", "U 型陷阱"]
    by_kind = {row["地图类型"]: row for row in b_summary}
    for y, kind in enumerate(kinds):
        row = by_kind[kind]
        mean = float(row["节省率均值"])
        if kind + "-置乱" in by_kind:
            shuffled = float(by_kind[kind + "-置乱"]["节省率均值"])
            left.plot([mean, shuffled], [y, y], color=style.NEUTRAL, linewidth=1.6, zorder=1)
            left.plot(shuffled, y, "o", color=style.ORANGE, markersize=5,
                      markeredgecolor="white", markeredgewidth=.6, zorder=3)
        left.plot(mean, y, "o", color=style.BLUE, markersize=5,
                  markeredgecolor="white", markeredgewidth=.6, zorder=3)
        # 保留一位有效小数，整数百分比去掉无意义的“.0”。
        density_label = f'{float(row["障碍密度"]) * 100:.1f}'.removesuffix(".0") + "%"
        left.text(1.17, y, density_label, transform=left.get_yaxis_transform(),
                  fontsize=7.5, color=style.INK_2, va="center", ha="center", clip_on=False)
    left.text(1.17, -.95, "障碍密度", transform=left.get_yaxis_transform(),
              fontsize=7.5, color=style.INK_2, ha="center", clip_on=False)
    left.set_ylim(7.6, -.6)
    left.set_yticks(range(8), labels, fontsize=7.5)
    left.tick_params(axis="y", length=0)
    left.set_xlim(0, 1.03)
    left.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1.0"])
    left.set_xlabel("曼哈顿相对 h=0 的节点节省率", fontsize=8, labelpad=3)
    left.grid(False)
    left.grid(axis="x", visible=True)
    left.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=color,
                                markersize=5, markeredgecolor="white", markeredgewidth=.6, label=label)
                         for color, label in ((style.BLUE, "原始地图"), (style.ORANGE, "置乱对照"))],
                loc="upper left", bbox_to_anchor=(0, .9), fontsize=7.5, borderaxespad=.2)
    left.set_title("(a) 不同地图结构下的节省率", loc="left", fontsize=8.5, pad=4)
    rows = [r for r in c_summary if r["地图类型"] == "迷宫(走廊2)"]
    for method, color in (("ALT-最远点", style.BLUE), ("ALT-随机", style.ORANGE)):
        data = sorted([r for r in rows if r["方法"] == method], key=lambda r: int(r["k"]))
        right.plot([int(r["k"]) for r in data], [float(r["扩展节点均值"]) for r in data],
                   color=color, marker="o", label=method)
    for method, color, linestyle in (("h=0", style.NEUTRAL, "-"),
                                      ("曼哈顿", style.NEUTRAL_DARK, (0, (4, 2))),
                                      ("完美h*", style.NEUTRAL_DARK, (0, (1, 1.5)))):
        value = float(next(r["扩展节点均值"] for r in rows if r["方法"] == method))
        right.axhline(value, color=color, linestyle=linestyle, linewidth=1, label=method)
    right.set_xscale("log", base=2)
    right.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    right.minorticks_off()
    right.set_xlim(.8, 40)
    right.set_ylim(bottom=0)
    right.set_xlabel("地标数 k", fontsize=8, labelpad=3)
    # 纵轴标题靠近本轴，为左图的密度列多留空隙。
    right.set_ylabel("平均扩展节点数", fontsize=8, labelpad=1)
    right.set_title("(b) 迷宫（走廊宽2）上的扩展节点数", loc="left", fontsize=8, pad=4)
    right.legend(loc="center right", bbox_to_anchor=(1, .6), fontsize=7.5, borderaxespad=.2)
    save_fixed_size(fig, "fig2_structure_and_alt")


def _word_font(run, size=10.5, bold=False):
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")


def preview(rows, fig1_caption):
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.5)
    section.top_margin = section.bottom_margin = Cm(2)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1
    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(5)
    _word_font(title.add_run("题目一图表预览"), 14)
    intro = doc.add_paragraph("两张图按原始尺寸插入；表 1 汇总已有实验 C 的搜索与预处理开销。")
    intro.paragraph_format.space_after = Pt(5)
    for filename, height, caption in (("fig1_utrap_heatmap.png", 7.5, fig1_caption),
                                      ("fig2_structure_and_alt.png", 6.2, FIG2_CAPTION)):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.space_after = Pt(1)
        paragraph.add_run().add_picture(str(OUT / filename), width=Cm(16), height=Cm(height))
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(5)
        _word_font(paragraph.add_run(caption), 9)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    _word_font(paragraph.add_run(TABLE_TITLE), 10.5)
    table = doc.add_table(rows=1, cols=7)
    table.alignment, table.autofit = WD_TABLE_ALIGNMENT.CENTER, False
    widths = [2, 3.1, 1.75, 2.1, 2.35, 2.1, 2.6]
    for column, width in zip(table.columns, widths):
        column.width = Cm(width)
    headers = ["地图", "方法", "扩展\n节点数", "查询耗时\n(ms)", "相对曼哈顿\n提速", "预处理耗时\n(ms)", "回本\n查询次数"]
    for cell, text, width in zip(table.rows[0].cells, headers, widths):
        cell.text, cell.width = text, Cm(width)
    for row in rows:
        cells = table.add_row().cells
        for cell, key, width in zip(cells, TABLE_COLUMNS, widths):
            text = row[key]
            if key == "地图" and text.startswith("迷宫"):
                text = "迷宫\n（走廊 2）"
            if key == "方法" and " k=" in text:
                text = text.replace(" k=", "\nk=")
            cell.text, cell.width = text, Cm(width)
    # 三线表：明确清除内外其他边框，仅保留表顶、表底和表头下横线。
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{name}")
        element.set(qn("w:val"), "single" if name in ("top", "bottom") else "nil")
        element.set(qn("w:sz"), "8")
        element.set(qn("w:color"), "000000")
        borders.append(element)
    table._tbl.tblPr.append(borders)
    for i, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if i == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                edge = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                for key, value in (("val", "single"), ("sz", "4"), ("color", "000000")):
                    bottom.set(qn("w:" + key), value)
                edge.append(bottom)
                cell._tc.get_or_add_tcPr().append(edge)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = i < len(table.rows) - 1
                for run in p.runs:
                    _word_font(run, 10.5, bold=i == 0)
    doc.save(OUT / "preview.docx")


def main():
    import matplotlib
    if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) < (3, 6):
        raise RuntimeError("需要Matplotlib>=3.6以使用指定字体回退；不自行替换字体")
    # 不依赖样式文件的日志屏蔽判断字体是否存在，缺失时明确报错。
    for name in ("Times New Roman", "SimSun"):
        font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = list(SOURCE.glob("*.csv")) + list(SOURCE.glob("*.png"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    b_summary, c_summary, b_raw = read_csv("exp_b_summary.csv"), read_csv("exp_c_summary.csv"), read_csv("exp_b_raw.csv")
    table = make_table(c_summary)
    save_table(table)
    per_node_cost_analysis(c_summary)
    instance = reconstruct_first_utrap(b_raw, b_summary)
    style.apply_style()
    # 将缺字警告升级为错误，不能通过隐藏字体日志交付方框文字。
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        figure1(instance)
        figure2(b_summary, c_summary)
    # 图注保留到独立Markdown，便于直接粘贴报告，不必打开Word。
    fig1_caption = FIG1_CAPTION.format(saving=1 - instance[4].stats.expanded / instance[3].stats.expanded)
    (OUT / "captions.md").write_text(fig1_caption + "\n\n" + FIG2_CAPTION + "\n\n" + TABLE_TITLE + "\n", encoding="utf-8")
    preview(table, fig1_caption)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in before.items())
    meta = {"fig1_cm": [16, 7.5], "fig2_cm": [16, 6.2],
            "start": instance[1], "goal": instance[2],
            "expanded_h0": instance[3].stats.expanded, "expanded_manhattan": instance[4].stats.expanded,
            "fig1_caption": fig1_caption, "fig2_caption": FIG2_CAPTION, "table_title": TABLE_TITLE}
    # 核验信息写到临时目录，报告目录只保留用户要求的图、表和Word。
    audit_dir = ROOT / "tmp" / "report_qa"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("生成完成：", OUT)
    print("热力图节点数：", meta["expanded_h0"], meta["expanded_manhattan"])
    print((OUT / "table1_alt_cost.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
