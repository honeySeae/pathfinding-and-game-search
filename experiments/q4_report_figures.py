"""题四报告图：评价图只读汇总CSV；路线图重算原题最优对局与A*路径。

评价图16×6 cm、路线图16×7 cm，输出PNG/SVG及图注；不重跑随机实验，不启动桌面应用。
"""
import csv
import warnings
from pathlib import Path

import numpy as np
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from . import fig_style as style

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'q4'
OUT = SOURCE / 'report'
plt = style.plt


def figure(rows):
    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16*style.CM, 6*style.CM), gridspec_kw={'wspace': .44})
    fig.subplots_adjust(left=.085, right=.985, bottom=.25, top=.77)
    methods = [('eval_score', '原评价（当前分差）', style.BLUE),
               ('eval_rollout', '贪心推演', style.ORANGE),
               ('eval_closer', '按收益分配（不考虑轮次）', style.NEUTRAL_DARK)]
    annotations = []
    for name, label, color in methods:
        for n in (6, 8):
            if name == 'eval_closer' and n == 8:
                continue
            data = sorted([r for r in rows if int(r['n']) == n and r['evaluate'] == name],
                          key=lambda r: int(r['r']))
            expected = (6 if name == 'eval_score' else 5 if name == 'eval_rollout' else 4) if n == 6 else (
                8 if name == 'eval_score' else 7)
            assert len(data) == expected and all(int(r['instances']) == (500 if n == 6 else 200) for r in data)
            # 仅在左图排除r=0、1；右图仍需要保留原评价H=6完整搜索参照。
            left_data = [r for r in data if 2 <= int(r['r']) <= 7]
            remaining = [int(r['r']) for r in left_data]
            assert all(int(r['r']) == n-int(r['H']) for r in data)
            mean = np.array([float(r['accuracy'])*100 for r in left_data])
            low = np.array([float(r['ci95_low'])*100 for r in left_data])
            high = np.array([float(r['ci95_high'])*100 for r in left_data])
            line = ':' if name == 'eval_closer' else '-' if n == 6 else '--'
            marker = 'o' if n == 6 else 's'
            # r为真实剩余回合数，不对重合配置的横坐标抖动；Wilson区间保持非对称。
            axes[0].errorbar(remaining, mean, yerr=[mean-low, high-mean], color=color,
                             linestyle=line, marker=marker, capsize=1.8, elinewidth=.7,
                             markersize=3, linewidth=1.1)
            if n == 6 and name != 'eval_closer':
                # 贪心H=5仅剩强制最后一步，去掉该特例，不修改源CSV。
                data = sorted([r for r in data if not (name == 'eval_rollout' and int(r['H']) == 5)],
                              key=lambda r: int(r['H']))
                x = [float(r['mean_median_ms']) for r in data]
                y = [float(r['accuracy'])*100 for r in data]
                axes[1].plot(x, y, color=color, marker='o', linewidth=1.2, markersize=3)
                for r, xx, yy in zip(data, x, y):
                    # 先放置标签，后面按当前CSV的计时点检查并调整，避免不同机器的计时差异造成碰撞。
                    H = int(r['H'])
                    offset = (0, 4) if name == 'eval_score' else (0, -4)
                    annotations.append(axes[1].annotate(f'H={H}', (xx, yy), xytext=offset,
                                       textcoords='offset points', fontsize=7, color=color,
                                       ha='center', va='bottom' if name == 'eval_score' else 'top'))
    axes[0].set_title('(a) 选对率与剩余推演步数', loc='left', fontsize=8.5, pad=5)
    axes[1].set_title('(b) 选对率与搜索耗时', loc='left', fontsize=8.5, pad=5)
    for ax in axes:
        ax.set_ylabel('选对率（%）', fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylim(20, 100)
    axes[0].set_yticks([20, 40, 60, 80, 100])
    axes[0].set_xticks(range(2, 8))
    axes[0].set_xlim(1.75, 7.25)
    axes[1].set_ylim(0, 115)  # 为完整搜索点上方的紧邻标签保留空间。
    axes[0].set_xlabel('剩余推演步数 r（从小到大）', fontsize=8)
    axes[1].set_xscale('log')
    times = [float(r['mean_median_ms']) for r in rows if int(r['n']) == 6 and r['evaluate'] != 'eval_closer']
    axes[1].set_xlim(min(times)*.67, max(times)*1.65)
    ticks = [v for v in (.01, .03, .1, .3, 1, 3, 10) if min(times)*.67 <= v <= max(times)*1.65]
    axes[1].set_xticks(ticks, [f'{v:g}' for v in ticks])
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].set_xlabel('平均搜索耗时（毫秒，对数轴）', fontsize=8)
    # 颜色对应评价函数，线型/标记对应宝藏数，合并为一行中文图例。
    handles = [Line2D([], [], color=c, linestyle=':' if name == 'eval_closer' else '-', label=label)
               for name, label, c in methods]
    handles += [Line2D([], [], color=style.INK_2, linestyle=line, marker=mark, label=label, markersize=3)
                for line, mark, label in [('-', 'o', '六个宝藏'), ('--', 's', '八个宝藏')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=5,
               fontsize=7, columnspacing=.85, handlelength=1.5, handletextpad=.45)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Glyph .* missing from font')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # 各点的实际耗时因机器而异；在6 pt范围内寻找不相交的位置。
        placed = []
        for i, item in enumerate(annotations):
            preferred = 1 if i < 6 else -1
            candidates = [(dx, side * dy) for side in (preferred, -preferred)
                          for dy in (6, 5, 4, 3, 2, 1)
                          for dx in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6)
                          if np.hypot(dx, dy) <= 6]
            for dx, dy in candidates:
                item.set_position((dx, dy))
                item.set_ha('center' if dx == 0 else 'right' if dx < 0 else 'left')
                item.set_va('bottom' if dy > 0 else 'top')
                box = item.get_window_extent(renderer).padded(.5)
                if (not any(box.overlaps(other) for other in placed)
                        and not any(line.get_path().transformed(line.get_transform()).intersects_bbox(
                            box, filled=False) for line in axes[1].lines)):
                    placed.append(box)
                    break
            else:
                raise ValueError('无法放置深度标签：' + item.get_text())
        # 深度标签单独检查两两相交，防止密集点附近的H标注挤在一起。
        boxes = [item.get_window_extent(renderer).padded(.5) for item in annotations]
        for i, box in enumerate(boxes):
            if any(box.overlaps(other) for other in boxes[i+1:]):
                raise ValueError('深度标注重叠，请调整标签偏移')
            # 同时检查标签与两条耗时折线的相交，避免标签虽不重叠却压住曲线。
            if any(line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False)
                   for line in axes[1].lines):
                raise ValueError('深度标注与折线重叠：' + annotations[i].get_text())
        for item in fig.findobj(match=Text):
            if item.get_visible() and item.get_text():
                assert item.get_fontsize() >= 7
                box = item.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                    raise ValueError('文字超出画布：' + item.get_text())
        # 不使用tight裁边，保证物理尺寸固定为16×6 cm。
        for ext in ('png', 'svg'):
            fig.savefig(OUT / f'fig4_eval.{ext}', dpi=300, bbox_inches=None, facecolor='white')
    plt.close(fig)


def preview(caption):
    """独立的本地A4预览；只创建文件，不操作Word界面。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.5)
    section.top_margin = section.bottom_margin = Cm(2)
    for name in ('Normal', 'Title'):
        font = doc.styles[name].font
        font.name, font.size = 'Times New Roman', Pt(10.5 if name == 'Normal' else 14)
        font.color.rgb = RGBColor(0, 0, 0)
        doc.styles[name]._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'SimSun')
    doc.styles['Normal'].paragraph_format.line_spacing = 1.15
    doc.add_paragraph('题目四评价函数与首步决策', style='Title')
    doc.add_paragraph('比较不同剩余回合数下的首步选对率，以及相同搜索耗时下的决策质量。')
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(OUT / 'fig4_eval.png'), width=Cm(16), height=Cm(6))
    doc.add_paragraph(caption)
    doc.save(OUT / 'preview.docx')


def routes_figure():
    """按报告的四面板布局展示两个首步各自的前两步、后两步。"""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from src.astar import astar
    from src.q4_treasure import (GRID, POINTS, VALUES, GameState, eval_score,
                                 first_step_values, immediate_gain, move,
                                 principal_variation, terrain_cost)

    style.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(16*style.CM, 7*style.CM),
                             gridspec_kw={'wspace': .02})
    fig.subplots_adjust(left=.006, right=.994, bottom=.20, top=.73)
    values2 = first_step_values(GameState(depth_left=2))
    values4 = first_step_values(GameState(depth_left=4))
    cmap = LinearSegmentedColormap.from_list('terrain_light', ['white', style.NEUTRAL])
    costs = [terrain_cost(c) for row in GRID for c in row if c != 'X']
    max_cost = max(costs)
    expected = {'T2': ['T2', 'T3', 'T1', 'T4'], 'T3': ['T3', 'T1', 'T2', 'T4']}
    all_steps = {}
    for first in ('T2', 'T3'):
        state = GameState(depth_left=4)
        child = move(state, first)
        steps = [{'玩家': 'A', '目标': first, '收益': immediate_gain(state, first), 'U': eval_score(child)}]
        steps += principal_variation(child)
        all_steps[first] = steps
        actual = [step['目标'] for step in steps]
        if actual != expected[first]:
            print('路线与给定示意不一致，使用principal_variation输出：', first, actual)
        assert len(steps) == 4 and steps[1]['U'] == values2[first]
        assert steps[-1]['U'] == values4[first]

    # 报告图4按“首步T2/T3 × 前两步/后两步”拆开；每面板圆点表示该阶段开始时的位置。
    panels = [('T2', 0, 'a'), ('T2', 2, 'b'), ('T3', 0, 'c'), ('T3', 2, 'd')]
    for ax, (first, begin, letter) in zip(axes, panels):
        steps = all_steps[first]
        state = GameState(depth_left=4)
        for step in steps[:begin]:
            state = move(state, step['目标'])
        start_pos_a, start_pos_b = state.pos_a, state.pos_b
        current_positions = {start_pos_a, start_pos_b}
        ax.grid(False)
        ax.set_aspect('equal')
        ax.set_xlim(-.5, 6.5)
        ax.set_ylim(6.5, -.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(style.GRID)
            spine.set_linewidth(.5)
        for row in range(7):
            for col in range(7):
                cell = GRID[row][col]
                cost = terrain_cost(cell)
                color = style.WALL if cost is None else cmap((cost-1)/(max_cost-1))
                ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, facecolor=color,
                                       edgecolor=style.GRID, linewidth=.35, zorder=0))
        # 已取得的宝藏格仍可通行；与玩家当前位置重合的星形由圆点覆盖。
        for target in VALUES:
            row, col = POINTS[target]
            ax.plot(col, row, marker='*', markersize=9, markerfacecolor=style.NEUTRAL_DARK,
                    markeredgecolor='white', markeredgewidth=.4, zorder=5)
            if target not in current_positions and not (target == 'T1' and begin == 2):
                ax.annotate(target, (col, row), xytext=(3, -6), textcoords='offset points',
                            ha='left', va='top', fontsize=7, color=style.INK_2, zorder=7)
        for i in range(begin, begin+2):
            step = steps[i]
            player, target = step['玩家'], step['目标']
            source = state.pos_a if player == 'A' else state.pos_b
            result = astar(POINTS[source], POINTS[target], GRID, cost_fn=terrain_cost)
            assert result.cost == state.game.distance(source, target)
            assert VALUES[target]-result.cost == step['收益']
            x = [p[1] for p in result.path]
            y = [p[0] for p in result.path]
            color = style.BLUE if player == 'A' else style.ORANGE
            ax.plot(x, y, color=color, linewidth=1.4, zorder=3+i*.1)
            # 箭头与折线均来自A*的实际最小代价路径，不使用示意直线替代。
            ax.annotate('', xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.2,
                                    'mutation_scale': 7, 'shrinkA': 0, 'shrinkB': 2}, zorder=6)
            state = move(state, target)
        # 玩家圆点标在面板开始时的位置，报告中的“双方当前位置”即此含义。
        for player, position, color in (('A', start_pos_a, style.BLUE),
                                        ('B', start_pos_b, style.ORANGE)):
            row, col = POINTS[position]
            ax.plot(col, row, marker='o', markersize=10, markerfacecolor=color,
                    markeredgecolor='white', markeredgewidth=.5, zorder=8)
            ax.text(col, row, player, fontsize=7, color='white', ha='center', va='center', zorder=9)
        phase = '前两步' if begin == 0 else '后两步'
        ax.set_title(f'({letter}) 首步{first}·{phase}', loc='left', fontsize=9, pad=4)
        lines = []
        for i in range(begin, begin+2):
            step = steps[i]
            reward = (f"{step['收益']:+d}" if step['收益'] else '0').replace('-', '−')
            lines.append(f"{'①②③④'[i]}{step['玩家']}→{step['目标']} {reward}")
        ax.text(0, -.055, '\n'.join(lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=7.5, color=style.INK)
        ax.text(1, -.265, f"累计 U={steps[begin+1]['U']}".replace('=-', '=−'),
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=style.INK_2)
    handles = [Line2D([], [], color=style.BLUE, label='A 的移动'),
               Line2D([], [], color=style.ORANGE, label='B 的移动'),
               Line2D([], [], color=style.INK_2, marker='o', linestyle='None', markersize=7,
                      label='双方当前位置'),
               Line2D([], [], color=style.INK_2, marker='*', linestyle='None', markersize=8,
                      label='宝藏')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=4,
               fontsize=8.5, handlelength=1.7, columnspacing=1.4, handletextpad=.5)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Glyph .* missing from font')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for item in fig.findobj(match=Text):
            if not item.get_visible() or not item.get_text():
                continue
            assert item.get_fontsize() >= 7
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
                raise ValueError('路线图文字超出画布：'+item.get_text())
        for ext in ('png', 'svg'):
            fig.savefig(OUT / f'fig4_routes.{ext}', dpi=300, bbox_inches=None, facecolor='white')
    plt.close(fig)
    return all_steps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / 'exp_order_and_eval_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    # 新增路线图时保留已存在的评价图及Word预览，避免改动已有报告输出。
    create_eval = not all((OUT / name).exists() for name in ('fig4_eval.png', 'fig4_eval.svg'))
    if create_eval:
        figure(rows)
    caption = ('图5　剩余回合数与计算开销对宝藏分配决策的影响（11×11，种子0；六个宝藏500个实例，八个宝藏200个实例）。'
               '(a) r=n−H表示搜索截断后尚未行动的回合数，横轴从小到大；实线圆点为六个宝藏，虚线方块为八个宝藏，'
               '只画r=2–7，灰色点线仅展示六个宝藏的按收益分配对照。误差线为Wilson 95%置信区间。'
               'r=0的完整搜索及r=1的贪心推演因必然最优而未在(a)画出，同时排除原评价的r=1点以保持相同比较范围。'
               '所选首步的完整博弈真实值等于最优值即记为选对，包含并列最优。'
               '(b) 仅使用六个宝藏实例，比较选对率与平均搜索耗时，横轴为对数尺度，点旁标注搜索深度H；'
               '去掉贪心推演H=5，保留原评价H=6完整搜索作为参照。'
               '原评价使用当前分差；贪心推演按轮次取眼前收益最高的宝藏至结束；按收益分配忽略轮次。'
               '搜索统一使用Alpha-Beta与眼前收益降序排序，每配置5次计时取中位数后按实例求均值，包含并列首步复核。'
               '历史配置计时原样保留，新增深度另次补测，耗时比较受运行批次和系统负载影响。')
    if create_eval:
        preview(caption)
    steps = routes_figure()
    routes_caption = ('图4　首步为T2和T3时双方的最优对局。原题7×7地图上限定A首步后的H=4搜索。'
                      '(a)(b)为首步T2的前两步、后两步，(c)(d)为首步T3的前两步、后两步。'
                      '蓝色为A、橙色为B；圆点表示各面板开始时双方的位置，星形表示宝藏。'
                      '箭头沿A*求得的最小代价路径指向本回合选定的宝藏；途经其他宝藏不自动获取。'
                      '图下依次列出回合、玩家、目标、单步收益及该面板结束时的累计收益差U。')
    caption_path = OUT / 'captions.md'
    # 图号与五页报告一致：路线为图4，评价函数为图5；重跑不会重复追加图注。
    caption_path.write_text(routes_caption+'\n\n'+caption+'\n', encoding='utf-8')
    print('已生成', OUT)
    for first, sequence in steps.items():
        print('限定首步', first, sequence)
    # Windows的GBK终端不支持数学减号，仅终端回退为ASCII，图注文件保持原文。
    print(caption.replace('−', '-'))


if __name__ == '__main__':
    main()
