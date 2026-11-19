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
    "lines.markersize": 5,   # 散点标记大小

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


def compare_loss(title, log1, label1, log2, label2, log3, label3):

    #plt.figure(figsize=(12, 7))

    plt.plot(log1['loss_history'], c="r", label=label1)
    plt.plot(log2['loss_history'], c="g", label=label2)
    plt.plot(log3['loss_history'], c="b", label=label3)

    # plt.tick_params(axis='both', labelsize=13)

    plt.title(title)
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitting Error")
    plt.grid(alpha=0.3)
    plt.ylim(0, 2.0)
    plt.xlim(0, 1000)
    plt.show()



if __name__ == '__main__':
    fname1 = '../logs/openai/layer4/curve(0)_seed(0)_openai_es_popsize(40)_evaluation(40000)_layer(4)_dim(32)_tanh_lr(0.09)_noise_std(0.1)_loss(0.13325).np'
    log1 = load_log(fname1)

    fname2 = '../logs/cma/layer2/curve(0)_seed(0)_cma_es_popsize(40)_evaluation(40000)_layer(2)_dim(32)_tanh_sigma(1.4)_loss(0.02235).np'
    log2 = load_log(fname2)

    fname3= '../logs/lfb/layer2_8/curve(0)_seed(0)_lfb_es_popsize(40)_evaluation(40000)_layer(2_8)_dim(32)_cos_lr(0.0001)_steps(10)_loss(0.00633).np'
    log3 = load_log(fname3)

    print(log1['loss_history'][999])
    print(log2['loss_history'][999])
    print(log3['loss_history'][999])

    title = 'Convergence Comparison for Different Methods'
    compare_loss(title,
                 log1,"OpenAI-ES",
                 log2,"CMA-ES",
                 log3, "LFB")



    print('-------finished--------')
