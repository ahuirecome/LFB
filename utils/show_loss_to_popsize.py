import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle


# 全局绘图参数（SCI英文图标准）
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
    "lines.linewidth": 1.5, # 曲线/折线粗细
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


def plot_fit_curve(X_all, y_true, y_pred):
    plt.figure(figsize=(12, 5))
    plt.plot(X_all, y_true, c="red", lw=2.2, label="Real Black-box Curve")
    plt.plot(X_all, y_pred, c="blue", ls="--", lw=2, label="CMA-ES Fitting Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.title("Black-box curve fitting with only terminal global rewards and no intermediate rewards")
    plt.show()


def plot_loss_history(loss_history):
    plt.plot(loss_history, c="#2E86AB")
    plt.title("CMA-ES Training Loss Variation")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(alpha=0.3)
    plt.show()


def compare_loss(title, log1, label1, log2, label2, log3, label3,log4, label4):

    plt.plot(log1['loss_history'], c="#4472C4", label=label1)
    plt.plot(log2['loss_history'], c="#ED7D31", label=label2)
    plt.plot(log3['loss_history'], c="#A5A5A5", label=label3)
    plt.plot(log4['loss_history'], c="#FFC000", label=label4)

    # plt.tick_params(axis='both', labelsize=13)

    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel("Generation")
    plt.ylabel("Fitting Error")
    plt.grid(alpha=0.3)
    plt.xlim([0,1000])
    plt.show(block=True)



if __name__ == '__main__':
    fname1 = './logs/lfb/lfb_es_popsize(10)_layer(8)_dim(32)_cos_iter(10)_curve(0).np'
    log1 = load_log(fname1)

    fname2 = './logs/lfb/lfb_es_popsize(20)_layer(8)_dim(32)_cos_iter(10)_curve(0).np'
    log2 = load_log(fname2)

    fname3 = './logs/lfb/lfb_es_popsize(30)_layer(8)_dim(32)_cos_iter(10)_curve(0).np'
    log3 = load_log(fname3)

    fname4 = './logs/lfb/lfb_es_popsize(40)_layer(8)_dim(32)_cos_iter(10)_curve(0).np'
    log4 = load_log(fname4)

    title = 'Fitting Error for Different Population Sizes'
    compare_loss(title,log1,"population size=10", log2,"population size=20", log3, "population size=30", log4, "population size=40")

    print('-------finished--------')
