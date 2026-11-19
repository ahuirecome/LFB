import matplotlib.pyplot as plt
import numpy as np
import pickle
from curves import get_curve_function


plt.rcParams.update({
    # 字体：SCI强制无衬线字体 Arial
    "font.family": "Arial",
    "font.size": 10,        # 基础字号（期刊正文一般8~11pt）
    "axes.labelsize": 11,   # X/Y轴标签字号（比基础大1号）
    "axes.titlesize": 12,   # 子图标题字号
    "xtick.labelsize": 9,   # x轴刻度数字
    "ytick.labelsize": 9,   # y轴刻度数字
    "legend.fontsize": 9,   # 图例字号

    # 线条与边框（论文线条不能太细，印刷会糊）
    "axes.linewidth": 1.0,  # 坐标轴边框粗细
    "lines.linewidth": 1.0, # 曲线/折线粗细
    "lines.markersize": 2,   # 散点标记大小

    # 刻度设置
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,  # 刻度短线长度
    "ytick.major.size": 4,

    # 图片DPI与输出
    "figure.dpi": 300,       # 画布分辨率，导出矢量图时无效
    "figure.figsize": (6, 4),# 单张子图标准尺寸（英寸）
    "savefig.dpi": 600,      # 保存图片分辨率（位图png/tif）
    "savefig.bbox": "tight", # 自动裁剪空白边距，论文必备
    "savefig.pad_inches": 0.1# 裁剪后预留少量边距
})




def load_log(fname):
    with open(fname, 'rb') as f:
        log = pickle.load(f)
    return log


def plot_fit_curve(title, X_all, y_pred, dense_x, dense_y,  label1, label2):
    #plt.figure(figsize=(12, 8))
    plt.plot(dense_x, dense_y, c="blue", label=label1)
    plt.plot(X_all, y_pred, c="red", ls="--", label=label2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.ylim(-8, 8)
    plt.title(title)
    plt.show()


def plot_loss_history(loss_history):
    plt.plot(loss_history, c="#2E86AB")
    plt.title("CMA-ES Training Loss Variation")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(alpha=0.3)
    plt.show()


def compare_loss(title, log1, label1, log2, label2, log3, label3):

    #plt.figure(figsize=(12, 7))

    plt.plot(log1['loss_history'], c="r", label=label1, linewidth=2)
    plt.plot(log2['loss_history'], c="g", label=label2, linewidth=2)
    plt.plot(log3['loss_history'], c="b", label=label3, linewidth=2)

    # plt.tick_params(axis='both', labelsize=13)

    plt.title(title)
    plt.legend(pos="top right")
    plt.xlabel("Generation")
    plt.ylabel("MSE Loss")
    plt.grid(alpha=0.3)
    plt.show()



if __name__ == '__main__':
    curve_id = 5

    fname1 = f'./logs/openai/openai_es_popsize(40)_dim(32)_tanh_curve({curve_id}).np'
    log1 = load_log(fname1)

    fname2 = f'./logs/cma/cma_es_popsize(40)_dim(32)_tanh_curve({curve_id}).np'
    log2 = load_log(fname2)

    fname3= f'./logs/lfb/lfb_es_popsize(40)_generation(2000)_layer(8_8)_dim(32)_cos_lr(0.0001)_steps(20)_curve({curve_id}).np'
    log3 = load_log(fname3)

    dense_x = np.linspace(-1, 1, 10000)
    dense_y = get_curve_function(curve_id)(dense_x)

    title = 'Black-box Curve Fitting with OpenAI-ES'
    plot_fit_curve(title, log1['x'], log1['y_pred'], dense_x, dense_y, "Real Black-box Curve", "OpenAI-ES Fitted Curve")


    title = 'Black-box Curve Fitting with CMA-ES'
    plot_fit_curve(title, log2['x'], log2['y_pred'], dense_x, dense_y,"Real Black-box Curve", "CMA-ES Fitted Curve")


    title = 'Black-box Curve Fitting with LFB-ES'
    plot_fit_curve(title, log3['x'], log3['y_pred'], dense_x, dense_y,"Real Black-box Curve", "LFB-ES Fitted Curve")

    plt.show()
    print('-------finished--------')
