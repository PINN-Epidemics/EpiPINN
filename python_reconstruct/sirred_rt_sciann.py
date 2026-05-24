"""Refactored reduced SIR-Rt SciANN workflow for cases 4 and 5.

中文导读：
本文件复现论文的 reduced SIR 设定。与完整 SIR 不同，这里不再显式学习
S(t)、R(t) 和 beta(t)，而是直接使用有效再生产数 Rt(t) 控制感染人数 I(t)：

    dI/dt = delta * (Rt - 1) * I

论文提出这个 reduced 模型是为了减少 NN 数量和损失项冗余，让 PINN 更稳定。
Case 4 只使用高噪声感染数据估计 Rt；Case 5 进一步加入住院数据 Delta_H，
同时估计 Rt(t) 和住院比例 sigma(t)。

- paper: Section 2.3, Eq. (12)-(16) reduced SIR、reduced ODE 残差、
  joint/split 损失。
- paper: Section 2.4, Eq. (17)-(27) 引入住院数据 Delta_H 和 sigma(t)。
- paper: Section 3.1, Cases 4-5 使用强噪声感染数据，并在 Case 5 加入更可靠的住院数据。
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import sciann as sn

# 本文件既可以作为包内模块导入，也可以被入口脚本直接执行。
# 包内导入使用 ".common"，直接执行时退回到同目录的 "common"。
try:
    from .common import (
        BaseRunner,
        TimeGrid,
        column,
        data_ode_training_grid,
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
        print_errors,
        relative_l2,
        uniform_collocation,
    )


class ReducedSIRRtData(object):
    """Data and scaling shared by cases 4 and 5.

    What this block does:
    读取 Case4.txt，并构造 Case 4/5 都需要的感染、Rt、sigma、住院数据。

    How it maps to the paper:
    - Case 4: 训练数据是带较大误差的 I_obs，用来估计 Rt。
    - Case 5: 额外使用 Delta_H 数据，假设 Delta_H = delta * sigma * I。
    """

    def __init__(self, dataframe):
        # dataframe 通常来自 EpiPINN/Case4.txt。
        # 这个文件虽然叫 Case4，但其中同时保存了 Case 5 需要的 sigma 和 H data。
        self.frame = dataframe
        # N 在 reduced 方程里不直接进入 dI/dt，但保留与论文合成设置一致。
        self.N = 56e6
        # delta 是感染者移出感染舱室的速率，D=1/delta=5 天。
        # 在 reduced 方程 dI/dt = delta*(Rt-1)*I 中它决定感染存量变化的时间尺度。
        self.delta = 1 / 5
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        # paper: Case 5 设置 I 的缩放 SI=1e5，住院日增量的缩放 SH=1e3。
        self.SI = 1e5
        self.SH = 1e3
        # C = SI*delta/SH，使 delta*sigma*I 的缩放形式可以写成 sigma*Is*C。
        # 见 paper: Eq. (22) CH*DeltaHs = delta*C*sigma*Is。
        self.C = self.SI * self.delta / self.SH

        # 真值序列仅用于评估/绘图；I data 和 H data 是 noisy training data。
        self.I_data = dataframe["Infectious"].values
        self.I_obs = dataframe["I data"].values
        self.Rt_data = dataframe["Rt"].values
        self.sigma_data = dataframe["sigma"].values
        # 合成参考住院量：H(t)=delta*sigma(t)*I(t)，对应 paper: Eq. (17)。
        # 这里 H_data 是无噪声参考，用于误差评估；真正参与训练的是 H_obs。
        self.H_data = (self.delta * self.sigma_data * self.I_data).reshape(-1)
        self.H_obs = dataframe["H data"].values

        # 缩放后的观测进入 SciANN Data loss。
        # I_obs_sc/H_obs_sc 是训练标签；I_data_sc/H_data_sc 只用于评估拟合误差。
        self.I_obs_sc = column(self.I_obs / self.SI)
        self.I_data_sc = self.I_data / self.SI
        self.H_obs_sc = column(self.H_obs / self.SH)
        self.H_data_sc = self.H_data / self.SH


class ReducedSIRRtRunner(BaseRunner):
    """Run the reduced SciANN examples from the original script.

    本 Runner 一次跑 Case 4 的 joint/split 和 Case 5 的 joint/split。
    """

    def __init__(self, data_dir=None, plot=True, verbose=1):
        super(ReducedSIRRtRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        # paper: reduced 模型不使用 NTK adaptive weights，因为论文说明各损失量级较一致。
        # 与 sir_r0.py/sir_rt.py 的完整 SIR 不同，这里只有 dI/dt 一条主残差，
        # 且 I、Delta_H 都被缩放到接近 1 的量级，所以没有启用 adaptive_NTK。
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.collocation_points = 6000
        # joint 仍使用论文实现细节中的 5000 epochs。
        self.epochs_joint = 5000
        self.batch_size = 100
        # split 的第一阶段是纯数据拟合，batch_data=10 对应较小的每日观测集。
        self.batch_data = 10

    def run(self):
        # What this block does:
        # 一次性运行 reduced 模型的四个实验分支：
        # - Case 4 joint
        # - Case 4 split
        # - Case 5 joint
        # - Case 5 split
        #
        # Things to watch:
        # 运行 main() 会触发所有四个分支的长训练；如果只是调试某个分支，
        # 建议在交互环境里实例化 Runner 后单独调用 run_case*_ 方法。
        # Case4.txt 同时包含 Case 4 的高噪声感染数据和 Case 5 所需的 sigma/H 数据。
        data = ReducedSIRRtData(self.read_table("Case4.txt"))
        self.plot_inputs(data)
        training = self.build_training_grid(data)
        case4_joint = self.run_case4_joint(data, training)
        case4_split = self.run_case4_split(data, training)
        case5_joint = self.run_case5_joint(data, training)
        case5_split = self.run_case5_split(data, training)
        return SimpleNamespace(
            data=data,
            case4=SimpleNamespace(joint=case4_joint, split=case4_split),
            case5=SimpleNamespace(joint=case5_joint, split=case5_split),
        )

    def build_training_grid(self, data):
        # reduced 残差仍在随机 collocation points 上计算。
        # data_ode_training_grid 会把“有观测标签的每日时间点”和“只用于 ODE 的随机点”
        # 拼接到同一个 t_train；training.ids_data 指向前半段观测点。
        t_train_ode = uniform_collocation(
            data.grid.t0,
            data.grid.tf,
            self.collocation_points,
            include_initial=True,
        )
        return data_ode_training_grid(data.grid.t_data_sc, t_train_ode)

    def plot_inputs(self, data):
        # What this block does:
        # 显示 Case 4/5 的观测环境：感染、住院、Rt 真值、sigma 真值。
        # How it maps to the paper:
        # 对应 Fig 8-9 的输入和参考轨迹；训练不会直接看到 Rt/sigma 真值。
        if not self.plot:
            return
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))
        # 第 1 幅：感染存量 I 的参考曲线与带强噪声的感染观测。
        # Case 4 只使用这些 magenta x 观测点训练。
        ax[0].plot(data.grid.timespan, data.I_data, "r", label="Infectious")
        ax[0].plot(data.grid.timespan, data.I_obs, "xm", label="samples")
        ax[0].legend(loc=6)
        ax[0].set_xlabel("date")
        ax[0].set_ylabel("individuals")
        ax[0].set_title("Infectious and Observed Data")

        # 第 2 幅：住院日增量 Delta_H 的参考曲线与 noisy observations。
        # Case 5 split 第一阶段优先拟合这个序列。
        ax[1].plot(data.grid.timespan, data.H_data, "b", label="Hospitalizations")
        ax[1].plot(data.grid.timespan, data.H_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[1].set_xlabel("date")
        ax[1].set_ylabel("individuals")
        ax[1].set_title("Hospitalizations and Observed Data")

        # 第 3 幅：Rt 真值。它只用于结果对照，不作为训练标签。
        ax[2].plot(data.grid.timespan, data.Rt_data, label=r"$\mathcal{R}_t$")
        ax[2].legend(loc=6)
        ax[2].set_xlabel("date")
        ax[2].set_title("Reproduction number")

        # 第 4 幅：sigma 真值。Case 5 要从 I 与 Delta_H 的关系中反推出它。
        ax[3].plot(data.grid.timespan, data.sigma_data, "orange", label=r"$\sigma$")
        ax[3].legend(loc=6)
        ax[3].set_xlabel("date")
        ax[3].set_title("Hospitalization rate")
        fig.tight_layout()
        self.maybe_show()

    def reduced_residual(self, data, ts, is_, rt):
        # What this block does:
        # 定义 reduced SIR 的唯一物理残差。
        # How it maps to the paper:
        # paper: Eq. (13)-(14)
        # dIs/dts - delta*(tf-t0)*(Rt-1)*Is = 0。
        # SciANN/PyTorch interpretation:
        # sn.diff(is_, ts) 使用自动微分计算神经网络输出对输入 ts 的导数。
        # 因为 ts 是缩放时间，所以右端要乘以 tf-t0，把物理时间导数换成缩放时间导数。
        #
        # Things to watch:
        # 这里 grid.tf 就是 (tf-t0)，因为 t0=0。若以后换成非零 t0，需要检查缩放定义。
        return sn.rename((sn.diff(is_, ts) - data.grid.tf * data.delta * (rt - 1) * is_), "L_dIdt")

    def run_case4_joint(self, data, training):
        # What this block does:
        # Case 4 joint: 同时训练 \hat{I}_s 和 \hat{Rt}。
        # How it maps to the paper:
        # paper: Eq. (15) L_r_joint = L_D(Is) + L_r,ODE(Is,Rt)。
        sn.reset_session()
        sn.set_random_seed(34)
        # ts 是唯一输入变量，即缩放后的时间 t_s。
        ts = sn.Variable("ts")
        # I 使用 4x50 网络，Rt 使用 4x100 网络，对应 implementation details。
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        # Data(rt*0) 是零目标占位/弱正则；Data(is_) 是感染数据损失 L_D。
        # loss_joint 的顺序决定 train() 中 y 列表的顺序：
        #   1. PDE(l_didt)  -> "zeros"
        #   2. Data(rt*0)   -> "zeros"
        #   3. Data(is_)    -> (ids_data, I_obs_sc)
        loss_joint = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(is_)]
        model = sn.SciModel(ts, loss_joint, self.loss_err, self.optimizer)
        history = self.train(
            model,
            # t_train 同时包含每日观测点和 collocation points；
            # Data(is_) 只在 ids_data 指向的观测点上计算。
            training.t_train,
            ["zeros"] * 2 + [(training.ids_data, data.I_obs_sc)],
            epochs=self.epochs_joint,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate_case4(data, model, is_, rt)
        self.plot_case4(data, predictions)
        errors = self.report_case4_errors(data, predictions)
        return SimpleNamespace(model=model, Is=is_, Rt=rt, history=history, predictions=predictions, errors=errors)

    def run_case4_split(self, data, training):
        # What this block does:
        # Case 4 split:
        # 1) Isc 只拟合高噪声感染观测；
        # 2) 固定 Isc，再仅用 reduced ODE 残差估计 Rt。
        # How it maps to the paper:
        # paper: Eq. (16) L_r_split(Rt)=L_r,ODE(Rt)。
        # paper: Eq. (14) 的残差在第二阶段成为 Rt 的主要识别信息。
        sn.reset_session()
        sn.set_random_seed(34)
        ts = sn.Variable("ts")
        # 第一阶段网络名叫 Isc，c 可理解为 calibrated/data-constrained。
        # 它只是平滑拟合 I_obs，不知道任何流行病学方程。
        isc = sn.Functional("Isc", ts, 4 * [50], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(isc), self.loss_err, self.optimizer)
        data_history = self.train(
            data_model,
            # 纯数据阶段只使用真实观测时间，不使用 collocation points。
            data.grid.t_data_sc,
            data.I_obs_sc,
            epochs=1000,
            batch_size=self.batch_data,
        )
        isc_pred = isc.eval(data_model, data.grid.t_data_sc)
        print("Isc error: {0:.3e}".format(relative_l2(data.I_data_sc, isc_pred)))

        # 冻结由数据得到的 I 函数；这体现了 split 方法先“拟合数据”，再“满足物理”。
        is_fixed = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=False)
        is_fixed.set_weights(isc.get_weights())
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_fixed, rt)
        # 第二阶段没有新的感染观测标签；它只要求 fixed I 与待估 Rt 满足 reduced ODE。
        # Data(is_fixed*0) 是零目标占位项，因为 is_fixed 已冻结，不会改变 I 网络权重。
        loss_ode = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(is_fixed * 0.0)]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
            # 只用 ODE collocation points；这正是论文 Eq. (16) 的 fully physics-informed 阶段。
            training.t_train_ode,
            ["zeros"] * 3,
            epochs=3000,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate_case4(data, ode_model, is_fixed, rt)
        self.plot_case4(data, predictions)
        errors = self.report_case4_errors(data, predictions)
        return SimpleNamespace(
            data_model=data_model,
            ode_model=ode_model,
            Is=is_fixed,
            Rt=rt,
            data_history=data_history,
            ode_history=ode_history,
            predictions=predictions,
            errors=errors,
        )

    def run_case5_joint(self, data, training):
        # What this block does:
        # Case 5 joint: 同时训练 Is、Rt、sigma，并拟合感染数据和住院数据。
        # How it maps to the paper:
        # - paper: Eq. (17) H = delta*sigma*I。
        # - paper: Eq. (22) 缩放后 Delta_Hs 与 sigma、Is 的代数关系。
        # - paper: Eq. (25) L_H_joint = L_D + L_H + L_H,ODE。
        sn.set_random_seed(34)
        sn.reset_session()
        ts = sn.Variable("ts")
        # Case 5 joint 的未知函数：
        # - Is: 缩放感染存量
        # - Rt: 有效再生产数
        # - sigma: 感染者中进入住院数据通道的比例
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        # paper: Implementation details 中 sigma 网络为 10 层、每层 5 个神经元。
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        # deltaHs 是 scaled daily hospitalizations 的模型值。
        # 因 data.C = SI*delta/SH，sigma*Is*C 等价于 (delta*sigma*I)/SH。
        delta_hs = sn.rename(sigma * is_ * data.C, "deltaHs")
        # loss_joint 的四个目标依次是：
        #   1. reduced ODE 残差 -> 0
        #   2. Rt 零目标占位/弱正则 -> 0
        #   3. Is 在观测点拟合 I_obs_sc
        #   4. DeltaHs 在观测点拟合 H_obs_sc
        loss_joint = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(is_), sn.Data(delta_hs)]
        model = sn.SciModel(ts, loss_joint, self.loss_err, self.optimizer)
        history = self.train(
            model,
            training.t_train,
            ["zeros"] * 2 + [(training.ids_data, data.I_obs_sc), (training.ids_data, data.H_obs_sc)],
            epochs=self.epochs_joint,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate_case5(data, model, is_, rt, sigma, delta_hs)
        self.plot_case5(data, predictions)
        errors = self.report_case5_errors(data, predictions)
        return SimpleNamespace(
            model=model,
            Is=is_,
            Rt=rt,
            sigma=sigma,
            deltaHs=delta_hs,
            history=history,
            predictions=predictions,
            errors=errors,
        )

    def run_case5_split(self, data, training):
        # What this block does:
        # Case 5 split:
        # 1) 先用住院数据训练 \hat{Delta}_{H,s}，因为论文认为住院数据比感染数据更可靠。
        # 2) 冻结 Delta_H，再通过 I = Delta_H/(delta*sigma) 和 reduced ODE
        #    同时估计 sigma(t) 与 Rt(t)，并用 I_obs 作为数据约束。
        # How it maps to the paper:
        # paper: Eq. (26) \hat{I}_s = CH/(delta*C) * DeltaHs/sigma_s。
        # paper: Eq. (27) L_H_split 同时含感染数据失配和 ODE 残差。
        sn.reset_session()
        sn.set_random_seed(34)
        ts = sn.Variable("ts")
        # Hsc 是 scaled daily hospitalizations 的 data-only 网络。
        # 论文选择先拟合 Delta_H，是因为住院数据通常比感染报告更稳定、更可靠。
        hsc = sn.Functional("Hsc", ts, 4 * [100], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(hsc), self.loss_err, self.optimizer)
        data_history = self.train(
            data_model,
            data.grid.t_data_sc,
            data.H_obs_sc,
            epochs=3000,
            batch_size=self.batch_data,
        )
        hsc_pred = hsc.eval(data_model, data.grid.t_data_sc)
        print("Hsc error: {0:.3e}".format(relative_l2(data.H_data_sc, hsc_pred)))

        # 第二阶段冻结住院曲线 Delta_Hs，把更可靠的数据作为已知函数输入物理方程。
        delta_hs = sn.Functional("deltaHs", ts, 4 * [100], output_activation="square", trainable=False)
        delta_hs.set_weights(hsc.get_weights())
        # 待估函数仍是 Rt 和 sigma；但 sigma 采用倒数参数化，见下方 Things to watch。
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        # Things to watch:
        # 这里让 sigma 网络实际学习的是 1/sigma 的形式：
        # is_ = (1/C) * DeltaHs * sigma，evaluate_case5(..., inverse_sigma=True)
        # 再把输出取倒数还原为论文中的 sigma。这样可避免直接除以接近 0 的 sigma。
        is_ = sn.rename((1 / data.C) * delta_hs * sigma, "Is")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        # loss_ode 的最后一个 Data(is_) 在 ids_data 上拟合 I_obs，对应 Eq. (27) 的感染数据项。
        # 训练目标顺序：
        #   1. PDE(l_didt)      -> "zeros"
        #   2. Data(rt*0)       -> "zeros"
        #   3. Data(sigma*0)    -> "zeros"
        #   4. Data(is_)        -> (ids_data, I_obs_sc)
        loss_ode = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(sigma * 0.0), sn.Data(is_)]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
            # 这里用 t_train 而不是 t_train_ode，因为第二阶段同时需要：
            # - collocation points 上的 ODE 残差
            # - 观测点上的感染数据拟合
            training.t_train,
            ["zeros"] * 3 + [(training.ids_data, data.I_obs_sc)],
            epochs=1000,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate_case5(data, ode_model, is_, rt, sigma, delta_hs, inverse_sigma=True)
        self.plot_case5(data, predictions)
        errors = self.report_case5_errors(data, predictions)
        return SimpleNamespace(
            data_model=data_model,
            ode_model=ode_model,
            Is=is_,
            Rt=rt,
            sigma=sigma,
            deltaHs=delta_hs,
            data_history=data_history,
            ode_history=ode_history,
            predictions=predictions,
            errors=errors,
        )

    def evaluate_case4(self, data, model, is_, rt):
        # Is 是缩放量，乘以 SI 回到人数；Rt 是无量纲量，不做缩放。
        # *_test 用更密的 t_test_sc 画平滑曲线；无后缀的 I/Rt 用每日点计算误差。
        return SimpleNamespace(
            I_test=is_.eval(model, data.grid.t_test_sc) * data.SI,
            Rt_test=rt.eval(model, data.grid.t_test_sc),
            I=is_.eval(model, data.grid.t_data_sc) * data.SI,
            Rt=rt.eval(model, data.grid.t_data_sc),
        )

    def evaluate_case5(self, data, model, is_, rt, sigma, delta_hs, inverse_sigma=False):
        # Case 5 split 中 sigma 网络可能表示 1/sigma，因此用 inverse_sigma 统一还原。
        sigma_test = sigma.eval(model, data.grid.t_test_sc)
        sigma_data = sigma.eval(model, data.grid.t_data_sc)
        if inverse_sigma:
            sigma_test = 1 / sigma_test
            sigma_data = 1 / sigma_data
        return SimpleNamespace(
            # delta_hs 是 scaled Delta_H，乘以 SH 回到每日住院人数。
            # I/H/Rt/sigma 同时返回测试点曲线和每日点误差序列。
            I_test=is_.eval(model, data.grid.t_test_sc) * data.SI,
            H_test=delta_hs.eval(model, data.grid.t_test_sc) * data.SH,
            Rt_test=rt.eval(model, data.grid.t_test_sc),
            sigma_test=sigma_test,
            I=is_.eval(model, data.grid.t_data_sc) * data.SI,
            H=delta_hs.eval(model, data.grid.t_data_sc) * data.SH,
            Rt=rt.eval(model, data.grid.t_data_sc),
            sigma=sigma_data,
        )

    def plot_case4(self, data, predictions):
        # 对应论文 Fig 8：高噪声感染数据下 I 和 Rt 的参考/预测对比。
        if not self.plot:
            return
        # 感染图：红线是真值，黑虚线是 PINN，紫色叉号是训练观测。
        plt.plot(data.grid.t_data, data.I_data, c="r", linewidth=4)
        plt.plot(data.grid.t_test, predictions.I_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.I_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()

        # Rt 图：真值只用于验证；训练过程中没有直接 Rt 标签。
        plt.plot(data.grid.t_data, data.Rt_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.Rt_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()

    def plot_case5(self, data, predictions):
        # 对应论文 Fig 9：加入住院数据后，同时比较 I、Delta_H、Rt、sigma。
        if not self.plot:
            return
        # 1) 感染存量 I。
        plt.plot(data.grid.t_data, data.I_data, c="r", linewidth=4)
        plt.plot(data.grid.t_test, predictions.I_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.I_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()

        # 2) 日住院 Delta_H。Case 5 split 中这是第一阶段直接拟合的对象。
        plt.plot(data.grid.t_data, data.H_data, c="b", linewidth=4)
        plt.plot(data.grid.t_test, predictions.H_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.H_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$\Delta_H$", r"$\hat{\Delta}_H$", "samples"])
        plt.show()

        # 3) 有效再生产数 Rt。
        plt.plot(data.grid.t_data, data.Rt_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.Rt_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()

        # 4) 住院比例 sigma。joint 直接输出 sigma；split 会在 evaluate 中还原倒数。
        plt.plot(data.grid.t_data, data.sigma_data, c="orange", linewidth=4)
        plt.plot(data.grid.t_test, predictions.sigma_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\sigma$", r"$\hat{\sigma}$"])
        plt.show()

    def report_case4_errors(self, data, predictions):
        # paper: Eq. (36) relative L2 error；后 100 天误差排除最早 20 天不稳定段。
        errors = [
            ("I error", relative_l2(data.I_data, predictions.I)),
            ("Rt error", relative_l2(data.Rt_data, predictions.Rt)),
            ("Rt error last 100 days", relative_l2(data.Rt_data[20:], predictions.Rt[20:])),
        ]
        print_errors(errors)
        return errors

    def report_case5_errors(self, data, predictions):
        # Case 5 除 I/Rt 外，还评估住院量 Delta_H 和时间变化住院比例 sigma。
        errors = [
            ("I error", relative_l2(data.I_data, predictions.I)),
            ("DeltaH error", relative_l2(data.H_data, predictions.H)),
            ("Rt error", relative_l2(data.Rt_data, predictions.Rt)),
            ("Rt error last 100 days", relative_l2(data.Rt_data[20:], predictions.Rt[20:])),
            ("sigma error", relative_l2(data.sigma_data, predictions.sigma)),
            ("sigma error last 100 days", relative_l2(data.sigma_data[20:], predictions.sigma[20:])),
        ]
        print_errors(errors)
        return errors


def main(plot=True, verbose=1):
    return ReducedSIRRtRunner(plot=plot, verbose=verbose).run()


if __name__ == "__main__":
    main()
