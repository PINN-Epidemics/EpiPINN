"""Refactored SIR-Rt SciANN workflow for cases 2 and 3.

中文导读：
本文件复现论文 Case 2 和 Case 3：仍使用完整 SIR 模型，但把传播率
beta(t) 作为时间依赖函数由神经网络学习。训练数据仍只有带噪声的感染人数
I_obs，因此这是一个典型 PINN 反问题：用观测数据约束 I_s_hat，用 ODE
残差同时推断 S_s_hat、R_s_hat 和 beta_hat(t)。

- paper: Section 2.1, Eq. (1)-(2) 基础 SIR 模型。
- paper: Section 2.2, Eq. (5)-(11) 缩放 SIR、PINN 损失、joint/split。
- paper: Section 3.1, Cases 2-3：Case 2 的 beta(t) 是合成时变函数，
  Case 3 的 beta(t)=delta*Rt(t) 来自意大利 COVID-19 早期 Rt 估计。
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import sciann as sn

try:
    from .common import (
        BaseRunner,
        TimeGrid,
        column,
        data_ode_training_grid,
        plot_sir_prediction,
        print_errors,
        relative_l2,
        uniform_collocation,
    )
except ImportError:  # pragma: no cover - script execution fallback.
    from common import (
        BaseRunner,
        TimeGrid,
        column,
        data_ode_training_grid,
        plot_sir_prediction,
        print_errors,
        relative_l2,
        uniform_collocation,
    )


class SIRRtData(object):
    """Data and scaling for the SIR-Rt cases.

    What this block does:
    从 Case2.txt 或 Case3.txt 读入参考 S/I/R、带噪声感染观测、Rt 参考值，
    并构造 beta_data=delta*Rt_data 作为误差评估目标。

    How it maps to the paper:
    paper: Section 3.1 Cases 2-3 使用每日感染数据作为训练数据，
    参考 beta(t) 只用于验证，并不会作为训练标签输入 PINN。
    """

    def __init__(self, dataframe):
        self.frame = dataframe
        # 与论文 simulation setup 一致：意大利规模人口、5 天平均感染期。
        self.N = 56e6
        self.delta = 1 / 5
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        # Eq. (5)-(6) 的缩放常数；C1/C2 出现在缩放后的 SIR 残差中。
        self.C = 1e5
        self.C1 = self.grid.tf * self.C / self.N
        self.C2 = self.grid.tf * self.delta
        # 初值条件对应 paper: Eq. (2)。
        self.S0 = self.N - 1
        self.I0 = 1
        self.R0 = 0

        # 这些列来自合成数据文件：真值用于画图和误差，I data 才是训练数据。
        self.S_data = dataframe["Susceptible"].values
        self.I_data = dataframe["Infectious"].values
        self.R_data = dataframe["Recovered"].values
        self.I_obs = dataframe["I data"].values
        self.Rt_data = dataframe["Rt"].values
        # paper: Rt = beta/delta * S/N。这里的合成数据文件提供 Rt，
        # 代码用 beta=delta*Rt 作为传播率参考；S/N 的有效再生产数因子在数据生成侧已体现。
        self.beta_data = self.Rt_data * self.delta

        # 观测感染人数缩放后进入 L_D；I_data_sc 仅用于评估第一阶段数据拟合误差。
        self.I_obs_sc = column(self.I_obs / self.C)
        self.I_data_sc = self.I_data / self.C


class SIRRtRunner(BaseRunner):
    """Run the full SIR-Rt PINN workflow.

    本 Runner 是 Case 1 的自然推广：beta 不再是 SciANN Parameter，
    而是一个 Functional NN，即 beta_hat(ts)。
    """

    def __init__(self, case=2, data_dir=None, plot=True, verbose=1):
        super(SIRRtRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        if case not in (2, 3):
            raise ValueError("case must be 2 or 3")
        self.case = case
        # paper: MSE + Adam + NTK adaptive weights + 6000 collocation points。
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.adaptive_ntk = {"method": "NTK", "freq": 100}
        self.epochs_joint = 5000
        self.epochs_data = 3000
        self.epochs_ode = 1000
        self.batch_size = 100
        self.batch_data = 10
        self.collocation_points = 6000

    def load_data(self):
        # Case2: 合成时变 beta；Case3: 由 COVID-19 Rt 轨迹构造的时变 beta。
        filename = "Case2.txt" if self.case == 2 else "Case3.txt"
        return SIRRtData(self.read_table(filename))

    def run(self):
        # 固定随机种子用于复现 SciANN 初始化和 collocation 采样。
        sn.set_random_seed(234)
        data = self.load_data()
        self.plot_inputs(data)

        training = self.build_training_grid(data)
        joint = self.run_joint(data, training)
        split = self.run_split(data, training)
        return SimpleNamespace(data=data, joint=joint, split=split)

    def plot_inputs(self, data):
        # What this block does:
        # 展示训练前的数据结构：S/R 真值、I 真值和 noisy samples、beta 参考轨迹。
        # How it maps to the paper:
        # 对应论文 Case 2-3 的输入/参考曲线；实际训练时只使用 I_obs。
        if not self.plot:
            return

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        plot_s = ax[0].plot(data.grid.timespan, data.S_data, "b", label="Susceptible")
        ax[0].tick_params(axis="y", labelcolor="b")
        ax[0].set_xlabel("date")
        ax[0].set_ylabel("individuals")
        ax[0].set_title("Susceptible and Recovered")

        ax2 = ax[0].twinx()
        plot_r = ax2.plot(data.grid.timespan, data.R_data, "g", label="Recovered")
        ax2.set_ylabel("individuals")
        ax2.tick_params(axis="y", labelcolor="g")

        lines = plot_s + plot_r
        ax[0].legend(lines, [line.get_label() for line in lines], loc=6)

        ax[1].plot(data.grid.timespan, data.I_data, "r", label="Infectious")
        ax[1].plot(data.grid.timespan, data.I_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[1].set_xlabel("date")
        ax[1].set_ylabel("individuals")
        ax[1].set_title("Infectious and Observed Data")

        ax[2].plot(data.grid.timespan, data.beta_data, label=r"$\beta$")
        ax[2].legend(loc=6)
        ax[2].set_xlabel("date")
        ax[2].set_title("Transimission rate")
        fig.tight_layout()
        self.maybe_show()

    def build_training_grid(self, data):
        # paper: N_C=6000 collocation points，均匀随机采样在缩放时间 [0,1]。
        # 与 Case 1 不同，这里不使用 log-biased 采样。
        t_train_ode = uniform_collocation(
            data.grid.t0,
            data.grid.tf,
            self.collocation_points,
            include_initial=True,
        )
        return data_ode_training_grid(data.grid.t_data_sc, t_train_ode)

    def build_sir_rt_model(self, data, beta_trainable=True):
        # What this block does:
        # 构造完整 SIR-Rt PINN 的四个数学对象：
        # \hat{S}_s(ts)、\hat{I}_s(ts)、\hat{R}_s(ts)、\hat{beta}(ts)。
        # How it maps to the paper:
        # paper: Eq. (6) 的 \beta_s(ts) 是待估时间函数；
        # paper: Section 2.2 说明 S/I/beta 由 NN 近似，R 由守恒关系推出。
        ts = sn.Variable("ts")
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        # beta_trainable 这个参数名沿用原重构代码；实际控制的是 Is 是否可训练。
        # split 第二阶段会冻结 Is，使其等于第一阶段拟合出的数据函数。
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=beta_trainable)
        # paper: Implementation details 中 beta/Rt 网络使用 4 层、每层 100 个神经元。
        beta = sn.Functional("Beta", ts, 4 * [100], output_activation="square")
        # N=S+I+R，避免额外训练 R 网络，也把总人口守恒硬编码进模型。
        rs = data.N / data.C - is_ - ss
        return SimpleNamespace(ts=ts, Ss=ss, Is=is_, Rs=rs, Beta=beta)

    def sir_rt_residuals(self, data, model, include_i_initial=True):
        # What this block does:
        # 完整 SIR 的缩放残差和初值残差。
        # How it maps to the paper:
        # - l_dsdt/l_didt/l_drdt 对应 Eq. (9) 中三条 ODE 残差。
        # - l_s0/l_i0/l_r0 对应 Eq. (8) 初始条件损失。
        # Things to watch:
        # 数据损失 L_D 仍然在 run_joint/run_split 中单独添加，不在这里。
        l_s0 = sn.rename((model.Ss - data.S0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_S0")
        l_r0 = sn.rename((model.Rs - data.R0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_R0")
        # dSs/dts + C1*Beta*Is*Ss = 0
        l_dsdt = sn.rename((sn.diff(model.Ss, model.ts) + data.C1 * model.Beta * model.Is * model.Ss), "L_dSdt")
        # dIs/dts - C1*Beta*Is*Ss + C2*Is = 0
        l_didt = sn.rename(
            (sn.diff(model.Is, model.ts) - data.C1 * model.Beta * model.Is * model.Ss + data.C2 * model.Is),
            "L_dIdt",
        )
        # dRs/dts - C2*Is = 0
        l_drdt = sn.rename((sn.diff(model.Rs, model.ts) - data.C2 * model.Is), "L_dRdt")
        residuals = [sn.PDE(l_dsdt), sn.PDE(l_didt), sn.PDE(l_drdt), sn.PDE(l_s0)]
        if include_i_initial:
            # joint 训练时 I 也是未知网络，所以要加 I0；split 训练时 I 已由数据网络确定。
            l_i0 = sn.rename((model.Is - data.I0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_I0")
            residuals.append(sn.PDE(l_i0))
        residuals.append(sn.PDE(l_r0))
        return residuals

    def run_joint(self, data, training):
        # What this block does:
        # joint approach 同时训练 Ss/Is/Beta，损失为 ODE 残差 + 初值 + 感染数据。
        # How it maps to the paper:
        # paper: Eq. (10) L_joint。
        sn.reset_session()
        model = self.build_sir_rt_model(data)
        loss_joint = self.sir_rt_residuals(data, model, include_i_initial=True) + [
            # 下面三个 Data(...*0) 保留 notebook 中的零目标项/弱正则项，
            # 用于稳定多目标训练并配合 NTK adaptive_weights。
            sn.Data(model.Ss * 0.0),
            sn.Data(model.Rs * 0.0),
            sn.Data(model.Beta * 0.0),
            # L_D: 只对 ids_data 指定的每日观测点拟合 I_obs。
            sn.Data(model.Is),
        ]
        sci_model = sn.SciModel(model.ts, loss_joint, self.loss_err, self.optimizer)
        loss_train = ["zeros"] * 9 + [(training.ids_data, data.I_obs_sc)]
        history = self.train(
            sci_model,
            training.t_train,
            loss_train,
            epochs=self.epochs_joint,
            batch_size=self.batch_size,
            adaptive_weights=self.adaptive_ntk,
        )
        predictions = self.evaluate_sir_rt(data, sci_model, model)
        self.plot_sir_rt(data, predictions)
        errors = self.report_sir_rt_errors(data, predictions)
        return SimpleNamespace(model=sci_model, functions=model, history=history, predictions=predictions, errors=errors)

    def run_split(self, data, training):
        # What this block does:
        # split approach:
        # 1) Isc 只拟合感染数据 L_D；
        # 2) 固定 Isc 权重为 Is，训练 Ss 和 Beta 以满足 SIR 方程。
        # How it maps to the paper:
        # paper: Eq. (11) L_split；论文强调 split 避免一开始就解复杂多目标优化。
        sn.reset_session()
        ts = sn.Variable("ts")
        # 第一阶段是普通深度回归，得到平滑、可微的 \hat{I}_s(ts)。
        isc = sn.Functional("Isc", ts, 4 * [50], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(isc), self.loss_err, self.optimizer)
        data_history = self.train(
            data_model,
            data.grid.t_data_sc,
            data.I_obs_sc,
            epochs=self.epochs_data,
            batch_size=self.batch_data,
        )
        isc_pred = isc.eval(data_model, data.grid.t_data_sc)
        print("Isc error: {0:.3e}".format(relative_l2(data.I_data_sc, isc_pred)))

        weights = isc.get_weights()
        # 第二阶段复制并冻结 Isc；SciANN 仍可对固定网络输出求导，用于 L_dIdt。
        is_fixed = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=False)
        is_fixed.set_weights(weights)
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        beta = sn.Functional("Beta", ts, 4 * [100], output_activation="square")
        rs = data.N / data.C - is_fixed - ss
        model = SimpleNamespace(ts=ts, Ss=ss, Is=is_fixed, Rs=rs, Beta=beta)

        loss_ode = self.sir_rt_residuals(data, model, include_i_initial=False) + [
            # ODE 阶段没有直接数据标签；这些零目标项与 PDE 残差一起训练。
            sn.Data(ss * 0.0),
            sn.Data(rs * 0.0),
            sn.Data(beta * 0.0),
            sn.Data(is_fixed * 0.0),
        ]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
            training.t_train_ode,
            ["zeros"] * 9,
            epochs=self.epochs_ode,
            batch_size=self.batch_size,
            adaptive_weights=self.adaptive_ntk,
        )
        predictions = self.evaluate_sir_rt(data, ode_model, model)
        self.plot_sir_rt(data, predictions)
        errors = self.report_sir_rt_errors(data, predictions)
        return SimpleNamespace(
            data_model=data_model,
            ode_model=ode_model,
            data_function=isc,
            functions=model,
            data_history=data_history,
            ode_history=ode_history,
            predictions=predictions,
            errors=errors,
        )

    def evaluate_sir_rt(self, data, sci_model, model):
        # 缩放变量乘以 C 回到人数；beta 不乘 C，因为它本身是 day^-1 量纲的参数函数。
        return SimpleNamespace(
            S_test=model.Ss.eval(sci_model, data.grid.t_test_sc) * data.C,
            I_test=model.Is.eval(sci_model, data.grid.t_test_sc) * data.C,
            R_test=model.Rs.eval(sci_model, data.grid.t_test_sc) * data.C,
            beta_test=model.Beta.eval(sci_model, data.grid.t_test_sc),
            S=model.Ss.eval(sci_model, data.grid.t_data_sc) * data.C,
            I=model.Is.eval(sci_model, data.grid.t_data_sc) * data.C,
            R=model.Rs.eval(sci_model, data.grid.t_data_sc) * data.C,
            beta=model.Beta.eval(sci_model, data.grid.t_data_sc),
        )

    def plot_sir_rt(self, data, predictions):
        # 对应论文 Fig 4-7 中状态变量和 beta(t) 的参考/预测对比。
        if not self.plot:
            return
        plot_sir_prediction(
            data.grid.t_data,
            data.grid.t_test,
            data.S_data,
            data.I_data,
            data.R_data,
            data.I_obs,
            predictions.S_test,
            predictions.I_test,
            predictions.R_test,
        )
        plt.plot(data.grid.t_data, data.beta_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.beta_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\beta$", r"$\hat{\beta}$"])
        plt.show()

    def report_sir_rt_errors(self, data, predictions):
        # paper: Eq. (36) 的相对 L2 误差。
        # “last 70 days” 排除前 20 天，是因为论文讨论中指出疫情初期 I 很小，
        # beta/Rt 的识别更不稳定，单独报告后段误差更能反映主体动态。
        errors = [
            ("S error", relative_l2(data.S_data, predictions.S)),
            ("I error", relative_l2(data.I_data, predictions.I)),
            ("R error", relative_l2(data.R_data, predictions.R)),
            ("Beta error", relative_l2(data.beta_data, predictions.beta)),
            ("Beta error last 70 days", relative_l2(data.beta_data[20:], predictions.beta[20:])),
        ]
        print_errors(errors)
        return errors


def main(case=2, plot=True, verbose=1):
    return SIRRtRunner(case=case, plot=plot, verbose=verbose).run()


if __name__ == "__main__":
    main()
