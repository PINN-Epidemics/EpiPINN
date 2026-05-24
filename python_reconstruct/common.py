"""Shared utilities for the refactored EpiPINN scripts.

The original files in ``EpiPINN/python`` are notebook exports.  This module
keeps the numerical choices unchanged while providing small reusable helpers
for data loading, training, plotting, and error reporting.

中文导读：
本文件不直接定义某一个流行病学模型，而是把论文所有案例都会重复用到的
“时间缩放、数据读取、SciANN 训练、相对误差、绘图、参数构造”集中起来。
这些工具服务于论文 Methods 中的共同设定：

- paper: Section 2.2, Eq. (5)-(6) 使用无量纲时间 ts=(t-t0)/(tf-t0)
  和人口数量缩放常数 C，把 S/I/R 等大数量级变量缩放到更适合 NN 训练的范围。
- paper: Section 2.2, Eq. (7)-(11) 把数据误差、ODE 残差、初值误差组合为
  PINN 损失；这里的 ``BaseRunner.train`` 是所有这些损失实际进入 SciANN 的入口。
- paper: Section 3.1 / Implementation details 使用相对 L2 误差评价重构结果，
  使用随机 collocation points 计算 ODE 残差。
"""

import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciann as sn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TimeGrid(object):
    """Container for physical and scaled time grids.

    paper: Eq. (5) 的时间缩放 ``ts`` 在代码中表现为 ``t_data_sc`` 和
    ``t_test_sc``。SciANN 的输入变量只接收缩放后的时间，绘图和误差统计再
    回到真实天数或日期。
    """

    def __init__(self, timespan, t0, tf, test_step=0.1):
        # What this block does:
        # 构造三套时间：真实日期 timespan、每天一个点的数据时间 t_data、
        # 以及更密的测试/绘图时间 t_test。
        # How it maps to the paper:
        # t_data 对应论文中的观测时刻 \tilde{t}_j；t_test 用来在连续函数
        # \hat{S}, \hat{I}, \hat{R}, \hat{Rt} 上采样，画出平滑曲线。
        self.timespan = timespan
        self.t0 = t0
        self.tf = tf
        self.t_data = np.arange(t0, tf)
        self.t_test = np.arange(t0, tf, test_step)
        self.t_data_sc = self.t_data / tf
        self.t_test_sc = self.t_test / tf

    @classmethod
    def from_dataframe_dates(cls, dataframe, test_step=0.1):
        # 数据文件中的 Date 列是每日观测；论文 Cases 2-7 都按 90 天左右的
        # 时间窗组织，因此这里把日期列统一变成 TimeGrid。
        timespan = dataframe["Date"].values.astype("datetime64[D]")
        return cls(timespan=timespan, t0=0.0, tf=len(timespan), test_step=test_step)


class TimedRun(object):
    """Training history with elapsed wall-clock time.

    paper: Tables 2-5 报告 joint/split 的训练耗时；这里保存的是同类指标。
    """

    def __init__(self, history, elapsed):
        self.history = history
        self.elapsed = elapsed

    @property
    def history_dict(self):
        return getattr(self.history, "history", self.history)


class BaseRunner(object):
    """Base class for script runners.

    每个案例 Runner 都继承它，以保持数据目录、绘图开关、SciANN 训练调用一致。
    """

    def __init__(self, data_dir=None, plot=True, verbose=1):
        self.data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT
        self.plot = plot
        self.verbose = verbose

    def data_path(self, filename):
        # Things to watch:
        # 默认 data_dir 是 EpiPINN 项目根目录，而不是 python_reconstruct 目录。
        # 因此 Case2.txt、RealData.txt 等数据文件会从 EpiPINN 根目录读取。
        path = self.data_dir / filename
        if path.exists():
            return path
        fallback = PROJECT_ROOT / filename
        if fallback.exists():
            return fallback
        raise FileNotFoundError("Cannot find data file: {0}".format(filename))

    def read_table(self, filename):
        return pd.read_table(str(self.data_path(filename)))

    def maybe_show(self):
        if self.plot:
            plt.show()
        else:
            plt.close()

    def train(self, model, x, y, epochs, batch_size, **kwargs):
        # What this block does:
        # SciANN/PyTorch interpretation:
        # ``model.train`` 相当于 Keras/TensorFlow 的 fit。x 是时间输入 ts，
        # y 是各个损失项的目标：PDE 残差通常用 "zeros"，数据项用观测值。
        # How it maps to the paper:
        # joint 方法把 L_D、L_ODE、L_IC 放进同一个 SciModel；
        # split 方法先训练数据网络，再训练 ODE 残差网络。
        start = time.time()
        history = model.train(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=self.verbose,
            **kwargs
        )
        elapsed = time.time() - start
        print("Training time: {0}".format(elapsed))
        return TimedRun(history=history, elapsed=elapsed)


def column(values):
    """Return an array as a two-dimensional column, matching the original code.

    SciANN 的 Data 目标通常要求 shape=(N, 1)，所以所有观测序列都转成列向量。
    """

    return np.asarray(values).reshape(-1, 1)


def relative_l2(reference, prediction):
    """Compute the relative L2 error used in the notebooks.

    paper: Eq. (36) e_r = ||prediction-reference||_2 / ||reference||_2。
    论文表格中的 S/I/R/Rt/beta/sigma 误差都对应这个形式。
    """

    return np.linalg.norm(reference - prediction, 2) / np.linalg.norm(reference, 2)


def print_errors(errors):
    for label, value in errors:
        print("{0}: {1:.3e}".format(label, value))


def uniform_collocation(t0, tf, count, include_initial=True):
    """Collocation points generated with the same uniform rule as the notebooks.

    paper: Section 3.1 implementation details 中 N_C=6000 个 collocation points。
    这些点没有观测标签，只用于让 ODE/PDE 残差在时间域内接近 0。
    """

    if include_initial:
        # include_initial=True 时把 ts=0 强行放进 collocation 集合，方便初值
        # 条件或疫情初期残差被训练看到。
        points = np.random.uniform(t0 / tf, 1.0, count - 1)
        return np.insert(points, 0, 0.0)
    return np.random.uniform(t0 / tf, 1.0, count)


def log_collocation(t0, tf, count):
    """Log-biased collocation points used by the constant-beta SIR case.

    Case 1 初期感染人数很小，beta 识别对早期增长很敏感；这里保留原 notebook
    的 log-biased 采样，使更多 collocation 点落在前期。
    """

    points = np.random.uniform(np.log1p(t0 / tf), np.log1p(1.0), count)
    return np.exp(points) - 1.0


def data_ode_training_grid(t_data_sc, t_train_ode):
    """Combine observed data times with ODE collocation times.

    How it maps to the paper:
    ``t_data_sc`` 对应数据损失 L_D/L_H/L_I 的观测点；
    ``t_train_ode`` 对应 L_ODE 的 collocation points。
    SciANN 允许用 ``ids_data`` 指定只有前 N_D 个训练点参与数据项。
    """

    t_train = np.concatenate([column(t_data_sc), column(t_train_ode)])
    ids_data = np.arange(np.asarray(t_data_sc).size, dtype=np.intp)
    return SimpleNamespace(t_train=t_train, t_train_ode=t_train_ode, ids_data=ids_data)


def save_history(path, history):
    # 保存训练曲线，便于复现实验图中关于收敛过程的比较。
    with open(path, "wb") as handle:
        pickle.dump(history, handle)


def save_history_and_weights(history_path, history, weights_path, model):
    save_history(history_path, history)
    model.save_weights(weights_path)


def plot_input_sir(timespan, s_data, i_data, r_data, i_obs):
    # What this block does:
    # 可视化合成数据的真实 S/I/R 轨迹和带噪声的感染观测。
    # How it maps to the paper:
    # 对应 Cases 1-3 中“由 SIR ODE 生成参考解，再从 I(t) 采样观测”的设置。
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(timespan, s_data, "b", label="Susceptible")
    ax[0].plot(timespan, r_data, "g", label="Recovered")
    ax[0].legend()
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("individuals")
    ax[0].set_title("Susceptible and Recovered")

    ax[1].plot(timespan, i_data, "r", label="Infectious")
    ax[1].plot(timespan, i_obs, "xm", label="samples")
    ax[1].legend()
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("individuals")
    ax[1].set_title("Infectious and Observed Data")
    fig.tight_layout()


def plot_sir_prediction(t_data, t_test, s_data, i_data, r_data, i_obs, s_pred, i_pred, r_pred):
    # 绘图比较 reference solution 与 PINN approximation。
    # 论文图 3-7 的核心视觉结构就是：实线为参考解，虚线为 PINN，叉号为数据。
    plt.plot(t_data, s_data, c="b", linewidth=4)
    plt.plot(t_test, s_pred, "--", c="k", linewidth=4)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend([r"$S$", r"$\hat{S}$"])
    plt.show()

    plt.plot(t_data, i_data, c="r", linewidth=4)
    plt.plot(t_test, i_pred, "--", c="k", linewidth=4)
    plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend([r"$I$", r"$\hat{I}$", "samples"])
    plt.show()

    plt.plot(t_data, r_data, c="g", linewidth=4)
    plt.plot(t_test, r_pred, "--", c="k", linewidth=4)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend([r"$R$", r"$\hat{R}$"])
    plt.show()


def build_sigma(case, ts, add_floor=False):
    """Build the hospitalization parameter for cases 6 and 7.

    paper: Section 2.4 and Cases 6-7.
    Case 6 把住院比例 sigma 视作常数参数；Case 7 把 sigma(t) 也作为 NN。
    ``add_floor`` 只在 split real-data 中使用，防止 Is = DeltaH/(sigma*C)
    因 sigma 过小而出现数值爆炸。
    """

    if case == 6:
        return sn.Parameter(name="sigma", inputs=ts, non_neg=True)
    if case == 7:
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        return sigma + 1e-3 if add_floor else sigma
    raise ValueError("case must be 6 or 7")
