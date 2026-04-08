import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font_size = 15
width = 4.5

crop = True
sam2 = False


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D

Times = 5
if True:
    # 数据示例
    data = {
        "Overlap ratio": [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4],
        "ScanNet200": [0.157,0.373,0.387,0.382,0.376,0.369,0.360,0.351,0.339],
        "ScanNet++": [0.117,0.323,0.382,0.389,0.391,0.388,0.387,0.389,0.388],
        "MatterPort3D": [0.060,0.162,0.184,0.189,0.191,0.189,0.188,0.187,0.188],
    }

    df = pd.DataFrame(data)

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": font_size,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(width,4))

    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
    colors = plt.cm.tab10.colors

    x = np.array(df["Overlap ratio"])
    legend_handles = []

    for i, col in enumerate(df.columns[1:]):
        y = np.array(df[col])
        
        # 原始散点
        ax.scatter(x, y,
                marker=markers[i % len(markers)],
                s=40,
                color=colors[i % len(colors)],
                edgecolor="black",
                zorder=3)
        
        # 光滑拟合曲线 (三次样条)
        x_smooth = np.linspace(x.min(), x.max(), 400)
        spline = make_interp_spline(x, y, k=Times)
        y_smooth = spline(x_smooth)
        
        ax.plot(x_smooth, y_smooth,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2)
        
        # 构造 legend 句柄：线+点
        handle = Line2D([0], [0],
                        color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)],
                        marker=markers[i % len(markers)],
                        markersize=6,
                        linewidth=2,
                        label=col)
        legend_handles.append(handle)

    # 标签和图例
    ax.set_xlabel("Overlap ratio", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    ax.legend(handles=legend_handles,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            edgecolor="black",
            fontsize=font_size,
            ncol=1,
            loc="lower right")

    plt.tight_layout()
    plt.savefig("Overlap_ratio.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

# -------------------------------------------------------
if True:
    # 数据示例
    data = {
        "Overlap ratio": [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4],
        "ScanNet200":   [0.059,0.264,0.329,0.343,0.345,0.343,0.336,0.325,0.309],
        "ScanNet++":    [0.077,0.246,0.302,0.332,0.340,0.349,0.356,0.356,0.360],
        "MatterPort3D": [0.033,0.162,0.197,0.201,0.204,0.207,0.208,0.208,0.207],
    }

    df = pd.DataFrame(data)

    plt.rcParams.update({
    "font.family": "serif",
    "font.size": font_size,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(width,4))

    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
    colors = plt.cm.tab10.colors

    x = np.array(df["Overlap ratio"])
    legend_handles = []

    for i, col in enumerate(df.columns[1:]):
        y = np.array(df[col])
        
        # 原始散点
        ax.scatter(x, y,
                marker=markers[i % len(markers)],
                s=40,
                color=colors[i % len(colors)],
                edgecolor="black",
                zorder=3)
        
        # 光滑拟合曲线 (三次样条)
        x_smooth = np.linspace(x.min(), x.max(), 400)
        spline = make_interp_spline(x, y, k=Times)
        y_smooth = spline(x_smooth)
        
        ax.plot(x_smooth, y_smooth,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2)
        
        # 构造 legend 句柄：线+点
        handle = Line2D([0], [0],
                        color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)],
                        marker=markers[i % len(markers)],
                        markersize=6,
                        linewidth=2,
                        label=col)
        legend_handles.append(handle)

    # 标签和图例
    ax.set_xlabel("Overlap ratio", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    ax.legend(handles=legend_handles,
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            edgecolor="black",
            fontsize=font_size,
            ncol=1,
            loc="lower right")

    plt.tight_layout()
    plt.savefig("Overlap_ratio_crop.png", dpi=300, bbox_inches="tight", pad_inches=0.05)  # 高分辨率保存

# -----------------------------------------------
if sam2:
    # 数据示例
    data = {
        "Manifold distance": [0.01, 0.03, 0.05, 0.07, 0.09],
        "ScanNet200": [0.362, 0.353, 0.376, 0.374, 0.368],
        "ScanNet++": [0.267, 0.381, 0.391, 0.368, 0.356],
        "MatterPort3D": [0.103, 0.169, 0.191, 0.193, 0.183],
    }

    df = pd.DataFrame(data)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.linewidth": 1.5,   # 坐标轴框线更粗
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


    fig, ax = plt.subplots(figsize=(width,4))

    # 使用不同线型和标记，避免全靠颜色区分
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']

    for i, col in enumerate(df.columns[1:]):
        ax.plot(df["Manifold distance"], df[col],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=6,
                linewidth=2,
                label=col)

    # 标签和图例
    ax.set_xlabel("Manifold distance", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    # 设置横坐标刻度
    ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    # 图例加不透明背景
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=1,
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig("Manifold_distance.png", dpi=300, bbox_inches="tight", pad_inches=0.05)  # 高分辨率保存

# --------------------------------------------
if crop:
    # 数据示例
    data = {
        "Manifold distance": [0.01, 0.03, 0.05, 0.07, 0.09],
        "ScanNet200":   [0.281, 0.328, 0.345, 0.344, 0.333],
        "ScanNet++":    [0.180, 0.342, 0.340, 0.332, 0.319],
        "MatterPort3D": [0.091, 0.183, 0.204, 0.205, 0.194],
    }

    df = pd.DataFrame(data)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.linewidth": 1.5,   # 坐标轴框线更粗
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


    fig, ax = plt.subplots(figsize=(width,4))

    # 使用不同线型和标记，避免全靠颜色区分
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']

    for i, col in enumerate(df.columns[1:]):
        ax.plot(df["Manifold distance"], df[col],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=6,
                linewidth=2,
                label=col)

    # 标签和图例
    ax.set_xlabel("Manifold distance", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    # 设置横坐标刻度
    ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    # 图例加不透明背景
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=1,
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig("Manifold_distance_crop.png", dpi=300, bbox_inches="tight", pad_inches=0.05)  # 高分辨率保存

# --------------------------------------------------
if sam2:
    # 数据
    data = {
        "Local frames": [5, 10, 15, 20, 25],
        "ScanNet200": [0.383, 0.382, 0.381, 0.376, 0.375],
        "ScanNet++": [0.385, 0.389, 0.386, 0.391, 0.393],
        "MatterPort3D": [0.190, 0.190, 0.187, 0.191, 0.188],
    }
    df = pd.DataFrame(data)

    # 科研风格
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(width,4))

    x = np.arange(len(df["Local frames"]))  # 横坐标位置
    width = 0.15                 # 每个柱子的宽度

    # 定义颜色和边框图案
    colors = plt.cm.tab10.colors   # 10 种常用颜色
    hatches = ['/', '\\', '-', '+', 'x', 'o', 'O', '.', '*']

    # 遍历每一列画柱子
    for i, col in enumerate(df.columns[1:]):
        ax.bar(x + i*width, df[col], width=width,
            label=col,
            color=colors[i % len(colors)],       # 填充颜色
            hatch=hatches[i % len(hatches)],     # 边框图案
            edgecolor="black", linewidth=1.5)    # 黑色边框

    # 设置横坐标为实际数值
    ax.set_xticks(x + width*len(df.columns[1:])/2)
    ax.set_xticklabels([f"{v}" for v in df["Local frames"]])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    # 标签加粗
    ax.set_xlabel("Local frames", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    # 图例加不透明背景
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=1,
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig("Local_frames.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

# -------------------------
if crop:
    # 数据
    data = {
        "Local frames": [5, 10, 15, 20, 25],
        "ScanNet200":   [0.349, 0.351, 0.349, 0.345, 0.345],
        "ScanNet++":    [0.334, 0.338, 0.339, 0.340, 0.338],
        "MatterPort3D": [0.199, 0.198, 0.203, 0.204, 0.202],
    }
    df = pd.DataFrame(data)

    # 科研风格
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=(width,4))

    x = np.arange(len(df["Local frames"]))  # 横坐标位置
    width = 0.15                 # 每个柱子的宽度

    # 定义颜色和边框图案
    colors = plt.cm.tab10.colors   # 10 种常用颜色
    hatches = ['/', '\\', '-', '+', 'x', 'o', 'O', '.', '*']

    # 遍历每一列画柱子
    for i, col in enumerate(df.columns[1:]):
        ax.bar(x + i*width, df[col], width=width,
            label=col,
            color=colors[i % len(colors)],       # 填充颜色
            hatch=hatches[i % len(hatches)],     # 边框图案
            edgecolor="black", linewidth=1.5)    # 黑色边框

    # 设置横坐标为实际数值
    ax.set_xticks(x + width*len(df.columns[1:])/2)
    ax.set_xticklabels([f"{v}" for v in df["Local frames"]])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    # 标签加粗
    ax.set_xlabel("Local frames", fontsize=font_size)
    ax.set_ylabel("AP$_{50}$", fontsize=font_size)

    # 图例加不透明背景
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=1,
        loc="lower right"
    )

    plt.tight_layout()
    plt.savefig("Local_frames_crop.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

# ---------------------------
if crop:
    font_size = 14

    fig, axes = plt.subplots(1, 3, figsize=(9,4))

    # -------------------------
    # 子图1：MAGE 系列
    # -------------------------
    ax = axes[0]
    # 数据
    x_vals = [0.122, 0.626, 0.11]
    y_vals = [30.1, 35, 34.5]
    labels = ["MaskClustering*", "OnlineAnySeg", "Stream3D (ours)"]
    colors = ["#377eb8", "#ffff33", "#e41a1c"]
    markers = ["h", "^", "*"]

    for x, y, lab, c, m in zip(x_vals, y_vals, labels, colors, markers):
        ax.scatter(x, y, c=c, marker=m, s=80, edgecolor="black", linewidth=0.8, label=lab)
        

    ax.text(x_vals[0]+0.02, y_vals[0], labels[0], fontsize=10)
    ax.text(x_vals[1]-0.28, y_vals[1], labels[1], fontsize=10)
    ax.text(x_vals[2]+0.02, y_vals[2], labels[2], fontsize=10)

    ax.set_ylabel("AP$_{50}$", fontsize=font_size)
    ax.set_xlabel("Sec. / frame", fontsize=font_size)
    ax.set_title("(a) ScanNet200", fontsize=font_size)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(loc="upper right", fontsize=8, frameon=True)

    # -------------------------
    # 子图2：DiT 系列
    # -------------------------
    ax = axes[1]
    x_vals = [0.359, 0.889, 0.326]
    y_vals = [29.7, 32.7, 34]
    labels = ["MaskClustering*", "OnlineAnySeg", "Stream3D (ours)"]
    colors = ["#377eb8", "#ffff33", "#e41a1c"]
    markers = ["h", "^", "*"]

    for x, y, lab, c, m in zip(x_vals, y_vals, labels, colors, markers):
        ax.scatter(x, y, c=c, marker=m, s=80, edgecolor="black", linewidth=0.8, label=lab)
        # ax.text(x+0.8, y+0.3, lab, fontsize=8)


    ax.text(x_vals[0]+0.02, y_vals[0], labels[0], fontsize=10)
    ax.text(x_vals[1]-0.31, y_vals[1], labels[1], fontsize=10)
    ax.text(x_vals[2]+0.02, y_vals[2], labels[2], fontsize=10)

    ax.set_ylabel("AP$_{50}$", fontsize=font_size)
    ax.set_xlabel("Sec. / frame", fontsize=font_size)
    ax.set_title("(b) ScanNet++", fontsize=font_size)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(loc="upper right", fontsize=8, frameon=True)

    # -------------------------
    # 子图3：基线模型
    # -------------------------
    ax = axes[2]
    x_vals = [0.945, 1.087, 0.505]
    y_vals = [13.2, 12.0, 20.4]
    labels = ["MaskClustering*", "OnlineAnySeg", "Stream3D (ours)"]
    colors = ["#377eb8", "#ffff33", "#e41a1c"]
    markers = ["h", "^", "*"]

    for x, y, lab, c, m in zip(x_vals, y_vals, labels, colors, markers):
        ax.scatter(x, y, c=c, marker=m, s=80, edgecolor="black", linewidth=0.8, label=lab)
        # ax.text(x+0.8, y+0.3, lab, fontsize=8)


    ax.text(x_vals[0]-0.38, y_vals[0], labels[0], fontsize=10)
    ax.text(x_vals[1]-0.32, y_vals[1], labels[1], fontsize=10)
    ax.text(x_vals[2]+0.02, y_vals[2], labels[2], fontsize=10)

    ax.set_ylabel("AP$_{50}$", fontsize=font_size)
    ax.set_xlabel("Sec. / frame", fontsize=font_size)
    ax.set_title("(c) MatterPort3D", fontsize=font_size)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig("Time_frames.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    # plt.show()