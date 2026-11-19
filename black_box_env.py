import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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


class BlackBoxEnv(object):
    def __init__(self, curve_id):
        self.curve_function = get_curve_function(curve_id)
        self.curve_id = curve_id

        self.states = np.linspace(-1, 1, 100).reshape(-1, 1)
        self.real_values = self.curve_function(self.states)

    def gen_state_values(self):
        states = np.random.uniform(-1, 1, size=(100, 1))
        values = self.curve_function(states)
        return states, values

    def get_loss(self, agent_outs):
        agent_outs = np.array(agent_outs)
        x = np.power(agent_outs - self.real_values, 2)
        loss = np.mean(x)
        return (float)(loss)

    def run(self, agent):
        [states, real_values] = [self.states, self.real_values] #self.gen_state_values()
        outs = agent.get_batch_out(states)
        loss = np.mean(np.power(outs - real_values, 2))
        agent.loss = loss
        agent.states = states
        agent.outs = outs
        #self.predict_values = outs
        return loss

    def plot_sample_curve(self):
        x_sample = self.states
        y_sample = self.real_values

        x_dense = np.linspace(-1, 1, 10000)
        y_dense = self.curve_function(x_dense)

        plt.plot(x_dense, y_dense, c="blue", label="Real Curve")
        plt.plot(x_sample, y_sample, marker="*", linestyle='', c="red", label="Sample Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.title("Test Curve and Sample Points")
        plt.ylim(-1.2, 1.6)
        plt.show()
        plt.pause(100000)


if __name__ == "__main__":
    env = BlackBoxEnv(curve_id=0)
    env.plot_sample_curve()
    print('----finished----')