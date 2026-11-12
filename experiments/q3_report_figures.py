"""题三报告图：只读排序实验CSV，生成16×6 cm PNG/SVG、图注和Word预览。

运行：py -m experiments.q3_report_figures；不重跑博弈搜索，不启动桌面应用。
"""
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from matplotlib.text import Text

from . import fig_style as style

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "q3"
OUT = SOURCE / "report"
SIGMAS = (0, .01, .02, .05, .1, .2, .5, 1, math.inf)
plt = style.plt


def read(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def figure(ordering, accuracy):
    style.apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(16 * style.CM, 6 * style.CM),
                                    gridspec_kw={"wspace": .48})
    fig.subplots_adjust(left=.085, right=.985, bottom=.26, top=.84)
    for name, color in (("随机顺序", style.BLUE), ("最优", style.AQUA), ("反向排序", style.ORANGE)):
        rows = sorted([r for r in ordering if int(r["d"]) == 6 and r["顺序"] == name], key=lambda r: int(r["b"]))
        left.plot([int(r["b"]) for r in rows], [float(r["有效分支因子均值"]) for r in rows],
                  marker="o", color=color, label=name)
    xx = np.linspace(2, 8, 150)
    left.plot(xx, xx, color=style.NEUTRAL_DARK, linestyle="--", linewidth=1)
    left.plot(xx, np.sqrt(xx), color=style.NEUTRAL_DARK, linestyle=":", linewidth=1)
    # 两个参照标注均使用普通文本，避免数学字体与正文字体不一致。
    left.text(8.12, 8, "y = b", color=style.NEUTRAL_DARK, fontsize=7.5, va="center")
    left.text(8.12, math.sqrt(8), "y = √b", color=style.NEUTRAL_DARK, fontsize=7.5, va="center")
    left.set_xlim(1.8, 9.1)
    left.set_ylim(0, 8.6)
    left.set_xticks([2, 3, 5, 8])
    left.set_xlabel("分支数 b", fontsize=8)
    left.set_ylabel("有效分支因子", fontsize=8)
    left.set_title("(a) d=6 时的有效分支因子", loc="left", fontsize=8.5, pad=4)
    left.legend(loc="upper left", fontsize=7.5, borderaxespad=.2)
    # 使用实测准确率作数值横轴；只改变展示方式，保留原来的逐树均值和标准差。
    data = sorted(accuracy, key=lambda r: float(r["首选准确率均值"]))
    positions = [100 * float(r["首选准确率均值"]) for r in data]
    right.errorbar(positions, [100 * float(r["叶子比例均值"]) for r in data],
                   yerr=[100 * float(r["叶子比例标准差"]) for r in data], marker="o", color=style.BLUE,
                   linewidth=1.4, capsize=2, elinewidth=.8)
    # 理论参照从实验CSV读取，避免图中手写数值与实验配置脱节。
    reference = next(r for r in ordering if int(r["b"]) == 5 and int(r["d"]) == 6 and r["顺序"] == "最优")
    theory = 100 * float(reference["理论最优叶子数"]) / float(reference["Minimax叶子数"])
    right.axhline(theory, color=style.NEUTRAL_DARK, linestyle="--", linewidth=1)
    right.text(.02, theory, f"最优顺序理论值 {theory:.1f}%", transform=right.get_yaxis_transform(),
               fontsize=7, color=style.NEUTRAL_DARK, ha="left", va="bottom")
    right.set_xlim(18, 102)
    right.set_ylim(bottom=0)
    right.set_xticks([20, 40, 60, 80, 100])
    right.tick_params(axis="x", labelsize=7)
    right.set_xlabel("首选准确率（%）", fontsize=8)
    right.set_ylabel("评估叶子数占 Minimax 的比例（%）", fontsize=8)
    right.set_title("(b) b=5、d=6 时排序准确度与剪枝效果", loc="left", fontsize=7.5, pad=4)
    # 固定画布不tight裁切，以保留原始16×6 cm尺寸；检查缺字和文字越界。
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for item in fig.findobj(match=Text):
            if not item.get_visible() or not item.get_text():
                continue
            assert item.get_fontsize() >= 7
            bounds = item.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1:
                raise ValueError("文字超出画布：" + item.get_text())
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fig4_ordering.{ext}", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)


def preview(caption):
    """仅生成文件并写明字体与尺寸，不启动Word或操作桌面。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.5)
    section.top_margin = section.bottom_margin = Cm(2)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(10.5)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")
    normal.paragraph_format.line_spacing = 1.15
    paragraph = doc.add_paragraph(style="Title")
    run = paragraph.add_run("题目三访问顺序与剪枝效率")
    run.font.name, run.font.size = "Times New Roman", Pt(14)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")
    doc.add_paragraph("本图比较访问顺序及排序噪声对Alpha-Beta搜索叶子数的影响。")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(OUT / "fig4_ordering.png"), width=Cm(16), height=Cm(6))
    doc.add_paragraph(caption)
    doc.save(OUT / "preview.docx")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ordering, accuracy = read("exp_ordering_summary.csv"), read("exp_accuracy_summary.csv")
    figure(ordering, accuracy)
    mismatch = sum(int(r["最优理论不符次数"]) for r in ordering if r["顺序"] == "最优")
    caption = ("图4　访问顺序对Alpha-Beta剪枝效率的影响（每配置30棵均匀随机树，叶子值互不相同）。"
               "(a) 深度d=6时的有效分支因子L^(1/d)，L为实际评估叶子次数；虚线y=b为不剪枝参照，"
               "点线y=√b为深度趋于无穷时的渐近值，有限深度下最优顺序的理论有效分支因子略高于它。精确最优叶子数为"
               f"b^⌈d/2⌉+b^⌊d/2⌋−1，本实验逐树不符次数为{mismatch}。"
               "(b) b=5、d=6时，以实际访问节点的首选准确率均值为横轴，叶子数占Minimax比例为纵轴，"
               "均以百分数表示；误差线为30棵树的叶子比例样本标准差。各点由不同大小的高斯排序噪声生成，"
               "另含独立随机排序，按实测准确率递增连接；水平线为有限深度的精确最优比例。"
               "各噪声档复用同一批树和标准正态噪声；真实Minimax值的预计算与排序预处理不计入搜索节点开销。")
    (OUT / "captions.md").write_text(caption + "\n", encoding="utf-8")
    preview(caption)
    print("已生成", OUT)
    # Windows默认GBK终端可能无法编码公式符号，仅终端用转义回退；文件仍保留原文。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print(caption)


if __name__ == "__main__":
    main()
