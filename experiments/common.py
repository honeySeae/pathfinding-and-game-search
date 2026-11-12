"""实验脚本共用：结果目录、CSV 写出、绘图风格（中文字体与配色）。"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 分类配色（按固定顺序使用，不循环）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_COLOR = "#e4e3df"

plt.rcParams.update({
    # 中文字体：按顺序回退，Windows / macOS / Linux 都能找到其一
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                        "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def out_dir(sub: str) -> str:
    path = os.path.join(RESULTS, sub)
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(path: str, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig 便于 Excel 打开中文
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_fig(fig, path: str):
    fig.savefig(path)
    plt.close(fig)
    print("已保存", os.path.relpath(path, ROOT))