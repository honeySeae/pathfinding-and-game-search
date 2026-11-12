"""报告用统一绘图样式：按 Word 报告实际尺寸作图（A4 正文宽约 16 cm），插入时按 100% 使用。"""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 屏蔽"找不到字体"的刷屏警告

CM = 1 / 2.54                 # 厘米 → 英寸
FULL_W = 16 * CM              # 单栏图宽
HALF_W = 7.8 * CM             # 半宽图

# ---- 颜色 ----
INK = "#1b1b1b"               # 主要文字
INK_2 = "#5a5955"             # 次要文字、刻度
GRID = "#e6e5e0"              # 网格线
SPINE = "#9c9b95"             # 坐标轴
BLUE = "#2a78d6"              # 方法 / 系列 1
ORANGE = "#e0601f"            # 方法 / 系列 2，也用作路径
AQUA = "#1baf7a"              # 系列 3
NEUTRAL = "#b9b8b2"           # 中性灰：基线、"必须扩展"等背景量
NEUTRAL_DARK = "#6f6e69"      # 参照线
WALL = "#3b3a36"              # 障碍
SERIES = [BLUE, ORANGE, AQUA]
# 扩展次序：极浅蓝 → 中蓝，避免大面积深色
EXPAND_CMAP = LinearSegmentedColormap.from_list("expand", ["#eef4fc", "#9cc0ea", "#3f78c4"])


def apply_style():
    plt.rcParams.update({
        # 英文数字 Times New Roman，中文宋体；其后为 macOS / Linux 回退
        "font.family": ["Times New Roman", "SimSun", "Songti SC", "Noto Serif CJK SC"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def panel_label(ax, letter, text=""):
    """左上角面板标号，如 (a) 迷宫(走廊1)。"""
    ax.set_title(f"({letter}) {text}".strip(), loc="left", fontsize=9, pad=4, color=INK)


def top_legend(fig, handles, ncol=None, y=1.0):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(handles), columnspacing=1.4, borderaxespad=0.2)


def save(fig, path_no_ext):
    """同时导出 300 dpi PNG 和 SVG。"""
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
