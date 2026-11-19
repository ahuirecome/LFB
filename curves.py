import numpy as np
import matplotlib.pyplot as plt


def curve_mexican_hat_peak(x):
    base =  (1 - x**2) * np.exp(-0.5 * x**2)
    peak = 7 * np.exp(-120 * (x + 0.4) ** 2)
    valley = -5 * np.exp(-80 * (x - 0.3) ** 2)
    return base + peak + valley

def curve_poly_osc(x):
    y = np.zeros_like(x)
    m1 = x < -0.5
    m2 = (x >= -0.5) & (x < 0)
    m3 = (x >= 0) & (x < 0.5)
    m4 = x >= 0.5
    y[m1] = -12*(x[m1]+1)**2 + 2
    y[m2] = 10*(x[m2]+0.5)**2 - 1.5
    y[m3] = -10*(x[m3]-0)**2 +1.8
    y[m4] = 12*(x[m4]-0.5)**2 -2
    return y

def curve_multi_gauss(x):
    y = 3*np.exp(-50*(x+0.6)**2) \
        -4*np.exp(-60*(x+0.2)**2) \
        +2.5*np.exp(-45*(x-0.3)**2) \
        -3.2*np.exp(-70*(x-0.7)**2)
    return y

def curve_rational_osc(x):
    y = 1/(1 + 100*(x+0.5)**2) \
        -0.8/(1 + 80*(x-0.2)**2) \
        +0.6/(1 + 120*(x-0.6)**2)
    return y

def curve_dense_gauss_osc(x):
    centers = [-0.8, -0.5, -0.2, 0.1, 0.4, 0.7]
    amps = [2.2, -1.6, 1.9, -2.1, 1.7, -1.4]
    widths = [60, 55, 70, 65, 58, 62]
    y = np.zeros_like(x)
    for c,a,w in zip(centers,amps,widths):
        y += a * np.exp(-w*(x - c)**2)
    return y

def curve_beat_gauss(xs):
    wave_low = np.sin(9 * xs)
    wave_high = 1.8 * np.sin(10 * xs)
    peak = 7 * np.exp(-120 * (xs + 0.4) ** 2)
    valley = -5 * np.exp(-80 * (xs - 0.3) ** 2)
    return wave_low + wave_high + peak + valley

def curve_am_osc_gauss(x):
    return (x ** 2 + 0.5) * np.sin(8 * x) + 4 * np.exp(-90 * (x - 0.2) ** 2) - 2 * np.exp(-110 * (x + 0.3) ** 2)

def curve_piecewise_jump(x):
    y = np.zeros_like(x)
    mask1 = x < -0.5
    mask2 = (x >= -0.5) & (x < 0)
    mask3 = (x >= 0) & (x < 0.5)
    mask4 = x >= 0.5

    y[mask1] = 2.0 * np.sin(5 * x[mask1])
    y[mask2] = 1.5
    y[mask3] = -1.0 + np.cos(10 * x[mask3])
    y[mask4] = 2.5
    return y


def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2))

def curve_seg_with_gauss(x):
    x_nodes = np.linspace(-1, 1, 11)
    y_nodes = np.array([0.0,0.4,0.2,0.7,0.1,-0.3,0.5,0.3,-0.2,0.6,0.0])
    x = np.asarray(x)
    y_base = np.zeros_like(x)
    for seg_idx in range(10):
        x0, x1 = x_nodes[seg_idx], x_nodes[seg_idx+1]
        y0, y1 = y_nodes[seg_idx], y_nodes[seg_idx+1]
        mask = (x >= x0) & (x <= x1)
        t = (x[mask] - x0) / (x1 - x0)
        y_base[mask] = y0 * (1 - t) + y1 * t

    g1 = gaussian(x, mu=-0.75, sigma=0.06, amp=0.8)
    g2 = gaussian(x, mu=-0.40, sigma=0.05, amp=-0.6)
    g3 = gaussian(x, mu=0.15, sigma=0.07, amp=0.7)
    g4 = gaussian(x, mu=0.60, sigma=0.05, amp=-0.5)
    g5 = gaussian(x, mu=0.82, sigma=0.04, amp=0.6)
    return y_base + g1 + g2 + g3 + g4 + g5

def curve_seg_with_gauss_v2(x):
    x_nodes = np.linspace(-1, 1, 11)
    y_nodes = np.array([0.0,-0.3,0.5,-0.2,0.4,0.6,0.1,-0.4,0.3,-0.1,0.0])
    x = np.asarray(x)
    y_base = np.zeros_like(x)
    for seg_idx in range(10):
        x0, x1 = x_nodes[seg_idx], x_nodes[seg_idx+1]
        y0, y1 = y_nodes[seg_idx], y_nodes[seg_idx+1]
        mask = (x >= x0) & (x <= x1)
        t = (x[mask] - x0) / (x1 - x0)
        y_base[mask] = y0 * (1 - t) + y1 * t

    g1 = gaussian(x, mu=-0.82, sigma=0.05, amp=0.7)
    g2 = gaussian(x, mu=-0.50, sigma=0.06, amp=-0.55)
    g3 = gaussian(x, mu=-0.10, sigma=0.04, amp=0.65)
    g4 = gaussian(x, mu=0.42, sigma=0.07, amp=-0.7)
    g5 = gaussian(x, mu=0.74, sigma=0.05, amp=0.55)
    return y_base + g1 + g2 + g3 + g4 + g5


curve_functions = {
    0:curve_poly_osc,
    1:curve_multi_gauss,
    2:curve_rational_osc,
    3:curve_dense_gauss_osc,
    4:curve_beat_gauss,
    5:curve_am_osc_gauss,
    6:curve_piecewise_jump,
    7:curve_mexican_hat_peak,
    8:curve_seg_with_gauss,
    9:curve_seg_with_gauss_v2
}


def get_curve_function(id):
    func = curve_functions[id]
    return func

# ===================== draw curve =====================

def plot_all_curves():
    curve_list = list(curve_functions.keys())
    n_curves = len(curve_list)
    cols = 2
    rows = (n_curves + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(14, 2.8*rows))
    axs = axs.flatten()

    x = np.linspace(-1, 1, 8000)

    for idx, name in enumerate(curve_list):
        func = curve_functions[name]
        y = func(x)
        axs[idx].plot(x, y, linewidth=1.2)
        axs[idx].set_title(name, fontsize=10)
        axs[idx].grid(alpha=0.3)

    for idx in range(n_curves, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plot_all_curves()

