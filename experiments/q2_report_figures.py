"""题二报告图：只读已有高估实验CSV，导出16×6 cm的图3和Word预览。

运行：py -m experiments.q2_report_figures
不重跑搜索或计时，不修改题一输出；只生成results/q2/report/中的新文件。
"""
import csv
import warnings
from pathlib import Path

from matplotlib.text import Text

from . import fig_style as style

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "q2"
OUT = SOURCE / "report"
WS = (1, 1.2, 1.5, 2, 3, 5)
plt = style.plt


def read_summary():
    with (SOURCE / "exp_overestimate_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def weighted_rows(summary, kind):
    rows = []
    for w in WS:
        method = "h_adm" if w == 1 else "w×h_adm"
        rows.append(next(r for r in summary if r["地图类型"] == kind and r["平局规则"] == "h_low"
                         and r["启发函数"] == method and float(r["w"]) == w))
    return rows


def figure(summary):
    style.apply_style()
    # 图例中的下界质量从统计CSV读取，两位小数仅用于显示，不写死数据。
    with (SOURCE / "exp_overestimate_hratio.csv").open(encoding="utf-8-sig", newline="") as stream:
        ratios = {row["地图类型"]: float(row["h_adm起点/C*均值"]) for row in csv.DictReader(stream)}
    legend_labels = [f"{kind}（起点处 h_adm/C* ≈ {ratios[kind]:.2f}）" for kind in ("平原为主", "混合地形")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16 * style.CM, 6 * style.CM),
                                    gridspec_kw={"wspace": .43})
    fig.subplots_adjust(left=.09, right=.985, bottom=.26, top=.80)
    # 等间距类别轴，避免1、1.2、1.5挤在一起；实际w仅用于取数和显示标签。
    positions = list(range(len(WS)))
    legend_handles = []
    # 不根据预期硬写0：从实际统计计算全部常数条件的出错次数。
    constants = [r for r in summary if float(r["c"]) > 0]
    errors = sum(int(r["出错数"]) for r in constants)
    checked = sum(int(r["实例数"]) for r in constants)
    for kind, color in (("平原为主", style.BLUE), ("混合地形", style.ORANGE)):
        rows = weighted_rows(summary, kind)
        handle, = left.plot(positions, [float(r["出错率"]) * 100 for r in rows], color=color,
                            marker="o", linewidth=1.4, label=kind, clip_on=False)
        legend_handles.append(handle)
        right.plot(positions, [float(r["扩展相对h=0比例"]) for r in rows], color=color,
                   marker="o", label=kind, clip_on=False)
    left.text(.02, .97, f"常数高估（曼哈顿距离、+5、+50）：\n{errors}/{checked} 次出错",
              transform=left.transAxes, fontsize=7, va="top")
    left.set_ylim(0, 100)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_ylabel("出错率（%）", fontsize=8)
    left.set_title("(a) 出错率", loc="left", fontsize=8.5, pad=4)
    right.set_ylim(bottom=0)
    right.set_ylabel("相对 h=0 的扩展比例", fontsize=8)
    right.set_title("(b) 扩展节点数", loc="left", fontsize=8.5, pad=4)
    legend = fig.legend(handles=legend_handles, labels=legend_labels, loc="upper center", bbox_to_anchor=(.5, .995),
                        ncol=2, fontsize=8.5, columnspacing=2, borderaxespad=.1)
    for ax in (left, right):
        ax.set_xlim(-.2, len(WS) - .8)
        ax.set_xticks(positions, ["1（h_adm）", "1.2", "1.5", "2", "3", "5"])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("放大系数 w", fontsize=8, labelpad=3)
    # 固定画布而非tight裁切，以保留16×6 cm的Word插入尺寸。
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # 两个图例项保持同一行；现有画布边界检查同时验证其未被裁切。
        legend_texts = legend.get_texts()
        assert all("\n" not in text.get_text() for text in legend_texts)
        centers = [text.get_window_extent(renderer).y0 + text.get_window_extent(renderer).height / 2 for text in legend_texts]
        assert max(centers) - min(centers) < 1e-6
        for item in fig.findobj(match=Text):
            if not item.get_visible() or not item.get_text():
                continue
            assert item.get_fontsize() >= 7
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                raise ValueError("文字超出画布：" + item.get_text())
        for ext in ("png", "svg"):
            fig.savefig(OUT / f"fig3_overestimate.{ext}", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return errors, checked


def preview(caption):
    """仅用python-docx生成本地预览，不启动Word或操作桌面。"""
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
    run = paragraph.add_run("题目二启发函数高估实验")
    run.font.name, run.font.size = "Times New Roman", Pt(14)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "SimSun")
    doc.add_paragraph("本图比较比例放大对返回次优解的比例和扩展节点开销的影响。")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(OUT / "fig3_overestimate.png"), width=Cm(16), height=Cm(6))
    doc.add_paragraph(caption)
    doc.save(OUT / "preview.docx")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_summary()
    errors, checked = figure(summary)
    caption = ("图3　比例放大启发函数（w × h_adm）对搜索的影响（30×30，每类200张）。"
               "(a) 返回次优解的比例；(b) 扩展节点数相对h=0的比例。"
               "w=1即可采纳下界h_adm，曼哈顿距离的结果与之几乎相同。")
    (OUT / "captions.md").write_text(caption + "\n", encoding="utf-8")
    preview(caption)
    print("已生成", OUT)
    print(f"常数高估：{errors}/{checked} 次出错")
    print(caption)


if __name__ == "__main__":
    main()
