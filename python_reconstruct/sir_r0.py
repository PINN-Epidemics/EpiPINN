"""Refactored constant-transmission SIR SciANN workflow for case 1.

中文导读：
本文件复现论文 Case 1：在基础 SIR 模型中假设传播率 beta 为常数，
用带 Poisson 计数噪声的感染人数 I(t) 作为观测数据，比较 joint PINN 与
split PINN 对 S/I/R 状态和 beta_0 的识别效果。

- paper: Section 2.1, Eq. (1)-(2) 基础 SIR ODE 和初值条件。
- paper: Section 2.2, Eq. (5)-(11) 变量缩放、数据损失、ODE 残差、
  初值损失、joint/split 两种训练方式。
- paper: Section 3.1.1 Case 1 使用 beta_0=0.6 day^-1、delta=1/5、
  R0=3、N=56e6、I0=1 的合成疫情。
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import sciann as sn
from scipy.integrate import odeint

try:
    from .common import (
        BaseRunner,
        TimeGrid,
        column,
        log_collocation,
        plot_input_sir,
        print_errors,
        relative_l2,
        save_history_and_weights,
    )
except ImportError:  # pragma: no cover - script execution fallback.
    from common import (
        BaseRunner,
        TimeGrid,
        column,
        log_collocation,
        plot_input_sir,
        print_errors,
        relative_l2,
        save_history_and_weights,
    )


def sir_ode(x, t, delta, beta, population, t0):
    """Reference SIR ODE used to generate case 1 synthetic data.

    paper: Eq. (1) 的未缩放强形式：
    dS/dt = -beta * I * S / N
    dI/dt =  beta * I * S / N - delta * I
    dR/dt =  delta * I

    这里用于生成“真值”轨迹，不是 PINN 本身；PINN 后面只看到 I_obs。
    """

    del t0
    susceptible, infectious, recovered = x
    lambda_val = beta * infectious / population
    return [
        -lambda_val * susceptible,
        lambda_val * susceptible - delta * infectious,
        delta * infectious,
    ]


class ConstantSIRData(object):
    """Synthetic case 1 data and scaling.

    What this block does:
    生成 Case 1 的参考 S/I/R 轨迹、带噪声观测 I_obs、以及缩放常数。

    How it maps to the paper:
    - N=56e6、delta=1/5、I0=1 来自 Section 3.1 simulation setup。
    - r0=3 使 beta=delta*r0=0.6，对应 Case 1 的常数 beta_0。
    - C=1e5 对应 Eq. (5) 的人口缩放，避免 NN 直接拟合百万级数值。
    """

    def __init__(self, weekly=False):
        # 人口与流行病学参数。delta=1/D，D=5 天表示平均感染期。
        self.N = 56e6
        self.delta = 1 / 5
        self.r0 = 3.0
        self.beta = self.delta * self.r0

        # 时间窗为 90 天，对应论文合成实验中模拟疫情早期 90 天。
        self.t0 = 0.0
        self.tf = 90.0

        # C 是 S/I/R 的缩放因子；C1/C2 是 Eq. (6) 缩放后 ODE 中出现的系数。
        # 注意论文中 C2 写作 (tf-t0)*delta；代码中的 C2 与此一致。
        self.C = 1e5
        self.C1 = self.tf * self.C / self.N
        self.C2 = self.tf * self.delta

        # 初值条件对应 paper: Eq. (2)：几乎全体易感，只有 1 个初始感染者。
        self.S0 = self.N - 1
        self.I0 = 1
        self.R0 = 0
        self.weekly = weekly

        timespan = np.arange("2020-02-01", "2020-05-01", dtype="datetime64[D]")
        tspan = timespan.astype(int)
        # Reference solver:
        # 使用 scipy.odeint 生成合成真值；论文用这些真值生成 noisy reported infections。
        solution = odeint(sir_ode, [self.S0, self.I0, self.R0], tspan, args=(self.delta, self.beta, self.N, tspan[0]))

        self.grid = TimeGrid(timespan=timespan, t0=self.t0, tf=self.tf)
        self.S_data = solution[:, 0]
        self.I_data = solution[:, 1]
        self.R_data = solution[:, 2]
        # paper: Section 3.1 提到感染数据来自以 I(t) 为均值的 Poisson 采样，
        # 模拟“病例数是计数过程”的报告误差。
        self.I_obs = np.random.poisson(self.I_data)

        # 缩放后的感染数据进入 SciANN 的数据损失 L_D。
        self.I_obs_sc = self.I_obs / self.C
        self.I_data_sc = self.I_data / self.C
        if weekly:
            # 可选 weekly 训练不是论文主表格的每日 N_D=90 设置；
            # 它保留了 notebook 中用于缺测/稀疏观测实验的入口。
            self.I_obs_train = column(self.I_obs_sc[::7])
            self.t_data_train = self.grid.t_data_sc[::7]
        else:
            self.I_obs_train = column(self.I_obs_sc)
            self.t_data_train = self.grid.t_data_sc


class ConstantSIRRunner(BaseRunner):
    """Run the case 1 PINN workflow.

    SciANN/PyTorch interpretation:
    Runner 把“定义网络 -> 定义损失 -> train -> eval/plot/error”组织成可调用流程。
    """

    def __init__(self, weekly=False, data_dir=None, plot=True, verbose=1, save_outputs=True):
        super(ConstantSIRRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        self.weekly = weekly
        self.save_outputs = save_outputs
        # paper: Section 2.2 的损失都用 MSE；Section 3.1 使用 Adam 优化。
        self.loss_err = "mse"
        self.optimizer = "adam"
        # paper: Implementation details 提到使用 NTK 自适应权重平衡多目标损失。
        self.adaptive_ntk = {"method": "NTK", "freq": 100}
        # paper: N_C=6000 collocation points；joint 5000 epochs；
        # split 中数据拟合和 ODE 拟合分开训练。
        self.collocation_points = 6000
        self.epochs_joint = 5000
        self.epochs_ode = 1000
        self.batch_size = 100

    def run(self):
        # What this block does:
        # 先生成合成数据，再依次跑 joint 和 split，最后比较 beta 的训练轨迹。
        data = ConstantSIRData(weekly=self.weekly)
        if self.plot:
            plot_input_sir(data.grid.timespan, data.S_data, data.I_data, data.R_data, data.I_obs)
            self.maybe_show()

        training = self.build_training_grid(data)
        joint = self.run_joint(data, training)
        split = self.run_split(data, training, joint)
        if self.plot:
            plt.plot(joint.history.history_dict["Beta"], linewidth=2.5)
            plt.plot(split.ode_history.history_dict["Beta"], linewidth=2.5)
            plt.legend(["Joint", "Split"])
            plt.ylabel(r"$\hat{\beta}_0$")
            plt.xlabel("epochs")
            self.maybe_show()
        return SimpleNamespace(data=data, joint=joint, split=split)

    def build_training_grid(self, data):
        # Case 1 使用 log-biased collocation，保留原 notebook 设置。
        # 这些点服务于 L_ODE/L_IC，不带病例观测标签。
        t_train_ode = log_collocation(data.t0, data.tf, self.collocation_points)
        t_train = np.concatenate([column(data.t_data_train), column(t_train_ode)])
        # ids_data 告诉 SciANN：拼接后的训练点里，只有前 N_D 个点有 I_obs 数据。
        ids_data = np.arange(data.t_data_train.size, dtype=np.intp)
        return SimpleNamespace(t_train=t_train, t_train_ode=t_train_ode, ids_data=ids_data)

    def build_state(self, data, ts, beta, is_trainable=True):
        # What this block does:
        # 构造 PINN 近似函数 \hat{S}_s(ts)、\hat{I}_s(ts)、\hat{R}_s(ts) 和 beta。
        # How it maps to the paper:
        # paper: Eq. (5)-(6) 中 Ss/Is/Rs 是缩放变量；Case 1 的 beta 是常数参数。
        # SciANN interpretation:
        # Functional 是一个以 ts 为输入的前馈神经网络；output_activation="square"
        # 是论文 Section 2.2 采用的硬非负约束，避免人数或 beta 变成负数。
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=is_trainable)
        # paper: Eq. (6) 后的质量守恒约束 N=S+I+R；
        # 代码不用单独训练 R 网络，而是用 Rs=N/C-Is-Ss 推出 R。
        rs = data.N / data.C - is_ - ss
        return SimpleNamespace(ts=ts, Ss=ss, Is=is_, Rs=rs, Beta=beta)

    def sir_losses(self, data, state, include_i_initial=True, add_beta_regularizer=False):
        # What this block does:
        # 组装 Case 1 的物理残差和初值损失。SciANN 中 PDE(expr) 表示让 expr -> 0。
        # How it maps to the paper:
        # - L_S0/L_I0/L_R0 对应 Eq. (8) 的初值损失 L_IC。
        # - L_dSdt/L_dIdt/L_dRdt 对应 Eq. (9) 的 ODE 残差 L_ODE。
        # - 数据损失 L_D 不在本函数中添加，而是在 run_joint/run_split 中用 Data(Is) 加入。
        # Things to watch:
        # (1 - sign(ts - 0)) 只在 ts=0 为正，因此把初值约束限制在初始时刻。
        l_s0 = sn.rename((state.Ss - data.S0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_S0")
        l_r0 = sn.rename((state.Rs - data.R0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_R0")
        # 缩放后方程 paper: Eq. (6)：
        # dSs/dts + C1 * beta * Is * Ss = 0
        l_dsdt = sn.rename((sn.diff(state.Ss, state.ts) + data.C1 * state.Beta * state.Is * state.Ss), "L_dSdt")
        # dIs/dts - C1 * beta * Is * Ss + C2 * Is = 0
        l_didt = sn.rename(
            (sn.diff(state.Is, state.ts) - data.C1 * state.Beta * state.Is * state.Ss + data.C2 * state.Is),
            "L_dIdt",
        )
        # dRs/dts - C2 * Is = 0
        l_drdt = sn.rename((sn.diff(state.Rs, state.ts) - data.C2 * state.Is), "L_dRdt")
        losses = [sn.PDE(l_dsdt), sn.PDE(l_didt), sn.PDE(l_drdt), sn.PDE(l_s0)]
        if include_i_initial:
            # split 第二阶段把 I 网络冻结为数据回归结果，通常不再单独施加 I0。
            l_i0 = sn.rename((state.Is - data.I0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_I0")
            losses.append(sn.PDE(l_i0))
        losses.append(sn.PDE(l_r0))
        # Data(0) 项用于给 Ss/Rs 的输出加入零目标占位/弱正则，保持与 notebook 的损失列表一致。
        losses += [sn.Data(state.Ss * 0.0), sn.Data(state.Rs * 0.0)]
        if add_beta_regularizer:
            losses.append(sn.Data(state.Beta * 0.0))
        return losses

    def run_joint(self, data, training):
        # What this block does:
        # joint approach: S、I、beta 同时训练，单个 SciModel 同时最小化
        # L_ODE + L_IC + L_D。
        # How it maps to the paper:
        # paper: Eq. (10) L_joint = L_D + L_ODE + L_IC。
        sn.reset_session()
        ts = sn.Variable("ts")
        # Case 1 的 beta_0 是常数，所以用 Parameter 而不是 Functional 网络。
        beta = sn.Parameter(name="Beta", inputs=ts, non_neg=True)
        state = self.build_state(data, ts, beta)
        losses = self.sir_losses(data, state, include_i_initial=True, add_beta_regularizer=False) + [sn.Data(state.Is)]
        model = sn.SciModel(ts, losses, self.loss_err, self.optimizer)
        # log_parameters 记录 beta 在训练中的变化，对应论文 Fig 3b 的收敛曲线。
        log_params = {"parameters": beta, "freq": 1}
        history = self.train(
            model,
            training.t_train,
            ["zeros"] * 8 + [(training.ids_data, data.I_obs_train)],
            epochs=self.epochs_joint,
            batch_size=self.batch_size,
            log_parameters=log_params,
            adaptive_weights=self.adaptive_ntk,
        )
        # 训练后在每日数据点和更密测试点上评估 \hat{S}, \hat{I}, \hat{R}, beta。
        predictions = self.evaluate(data, model, state)
        self.plot_predictions(data, predictions)
        errors = self.report_errors(data, predictions)
        if self.save_outputs:
            save_history_and_weights("hJoint.txt", history.history_dict, "mJoint.hdf5", model)
        return SimpleNamespace(model=model, functions=state, history=history, predictions=predictions, errors=errors)

    def run_split(self, data, training, joint):
        # What this block does:
        # split approach 的两阶段：
        # 1) 只用感染观测 I_obs 训练 \hat{I}_s，即普通数据回归。
        # 2) 冻结 \hat{I}_s，仅训练 \hat{S}_s 和 beta，使 SIR 残差接近 0。
        # How it maps to the paper:
        # paper: Eq. (11) L_split = L_ODE + L_IC，其中 \hat{I}_s 来自第一阶段 L_D。
        sn.reset_session()
        ts = sn.Variable("ts")
        # 第一阶段的 Isc 是“data-only”网络，不带任何 SIR 方程约束。
        isc = sn.Functional("Isc", ts, 4 * [50], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(isc), self.loss_err, self.optimizer)
        if data.weekly:
            epochs_data = 1000
            batch_data = 13
        else:
            epochs_data = 3000
            batch_data = 10
        data_history = self.train(
            data_model,
            data.t_data_train,
            data.I_obs_train,
            epochs=epochs_data,
            batch_size=batch_data,
        )
        isc_pred = isc.eval(data_model, data.grid.t_data_sc)
        print("Isc error: {0:.3e}".format(relative_l2(data.I_data_sc, isc_pred)))

        # 第二阶段：新建同结构 Is，把第一阶段权重复制过来并设为 trainable=False。
        # 这样 Is 在 ODE 训练中是已知函数，beta 和 Ss 才是待求未知量。
        is_fixed = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=False)
        is_fixed.set_weights(isc.get_weights())
        beta = sn.Parameter(name="Beta", inputs=ts, non_neg=True)
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        rs = data.N / data.C - is_fixed - ss
        state = SimpleNamespace(ts=ts, Ss=ss, Is=is_fixed, Rs=rs, Beta=beta)
        losses = self.sir_losses(data, state, include_i_initial=False, add_beta_regularizer=False) + [
            sn.Data(is_fixed * 0.0)
        ]
        ode_model = sn.SciModel(ts, losses, self.loss_err, self.optimizer)
        log_params = {"parameters": beta, "freq": 1}
        ode_history = self.train(
            ode_model,
            training.t_train_ode,
            ["zeros"] * 8,
            epochs=self.epochs_ode,
            batch_size=self.batch_size,
            log_parameters=log_params,
            adaptive_weights=self.adaptive_ntk,
        )
        predictions = self.evaluate(data, ode_model, state)
        self.plot_predictions(data, predictions)
        errors = self.report_errors(data, predictions)
        if self.save_outputs:
            save_history_and_weights("hSplit_data.txt", data_history.history_dict, "mSplit_data.hdf5", data_model)
            save_history_and_weights("hSplit_ode.txt", ode_history.history_dict, "mSplit_ode.hdf5", ode_model)
        return SimpleNamespace(
            data_model=data_model,
            ode_model=ode_model,
            data_function=isc,
            functions=state,
            data_history=data_history,
            ode_history=ode_history,
            predictions=predictions,
            errors=errors,
            joint=joint,
        )

    def evaluate(self, data, model, state):
        # SciANN 的 eval 输出仍是缩放量；乘以 C 回到人数尺度，方便与参考解比较。
        return SimpleNamespace(
            S_test=state.Ss.eval(model, data.grid.t_test_sc) * data.C,
            I_test=state.Is.eval(model, data.grid.t_test_sc) * data.C,
            R_test=state.Rs.eval(model, data.grid.t_test_sc) * data.C,
            S=state.Ss.eval(model, data.grid.t_data_sc) * data.C,
            I=state.Is.eval(model, data.grid.t_data_sc) * data.C,
            R=state.Rs.eval(model, data.grid.t_data_sc) * data.C,
            beta=state.Beta.eval(model, data.grid.t_data_sc),
        )

    def plot_predictions(self, data, predictions):
        # 这些图对应论文中 Case 1 对 S/I/R 参考解、PINN 解、观测样本的比较。
        if not self.plot:
            return
        scatter_t = data.grid.t_data[::7] if data.weekly else data.grid.t_data
        scatter_i = data.I_obs[::7] if data.weekly else data.I_obs

        plt.plot(data.grid.t_data, data.S_data, c="b", linewidth=4)
        plt.plot(data.grid.t_test, predictions.S_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend(["$S$", r"$\hat{S}$"])
        plt.show()

        plt.plot(data.grid.t_data, data.I_data, c="r", linewidth=4)
        plt.plot(data.grid.t_test, predictions.I_test, "--", c="k", linewidth=4)
        plt.scatter(scatter_t, scatter_i, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend(["$I$", r"$\hat{I}$", "samples"])
        plt.show()

        plt.plot(data.grid.t_data, data.R_data, c="g", linewidth=4)
        plt.plot(data.grid.t_test, predictions.R_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend(["$R$", r"$\hat{R}$"])
        plt.show()

    def report_errors(self, data, predictions):
        # paper: Eq. (36) 相对 L2 误差；beta 是常数，只取预测序列第一个值即可。
        beta_pred = np.ravel(predictions.beta)[0]
        errors = [
            ("S error", relative_l2(data.S_data, predictions.S)),
            ("I error", relative_l2(data.I_data, predictions.I)),
            ("R error", relative_l2(data.R_data, predictions.R)),
            ("Beta error", abs(beta_pred - data.beta) / data.beta),
        ]
        print_errors(errors)
        return errors


def main(weekly=False, plot=True, verbose=1):
    return ConstantSIRRunner(weekly=weekly, plot=plot, verbose=verbose).run()


if __name__ == "__main__":
    main()
