"""Refactored real-data SciANN workflow for cases 6 and 7.

中文导读：
本文件复现论文真实数据应用：使用 2020-02-21 至 2020-05-20 意大利 COVID-19
早期的日新增感染 Delta_I 和日新增住院 Delta_H，估计有效再生产数 Rt(t)。
Case 6 把住院比例 sigma 视为常数；Case 7 把 sigma(t) 也作为神经网络函数学习。

与 synthetic Case 5 不同，真实数据中可直接观测的是“日新增感染”而不是当前感染
存量 I(t)。因此代码使用论文 Section 2.4 后半部分的扩展：

    Delta_I ~= dSigma_I/dt = delta * Rt * I
    Delta_H ~= dSigma_H/dt = delta * sigma * I

- paper: Section 2.4, Eq. (28)-(35) 真实数据可用 Delta_I/Delta_H 时的
  hospitalization-augmented reduced SIR 损失。
- paper: Section 3.2 Cases 6-7：真实意大利数据，Case 6 常数 sigma，
  Case 7 时间变化 sigma。
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import sciann as sn

try:
    from .common import (
        BaseRunner,
        TimeGrid,
        build_sigma,
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
        build_sigma,
        column,
        data_ode_training_grid,
        print_errors,
        relative_l2,
        uniform_collocation,
    )


class RealEpidemicData(object):
    """Real-data scaling for cases 6 and 7.

    What this block does:
    读取 RealData.txt，把日新增住院 H_obs、日新增感染 I_obs、ISS Rt 参考值缩放。

    How it maps to the paper:
    paper: Section 3.1 simulation setup / Cases 6-7 说明真实数据使用最大观测值
    作为 C 和 CH 的缩放基准；这里 SI=max(Delta_I)，SH=max(Delta_H)。
    """

    def __init__(self, dataframe):
        self.frame = dataframe
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        # 列名沿用数据文件：H data 是 daily hospitalizations Delta_H，
        # I data 是 daily reported infections Delta_I，不是当前 infectious stock I(t)。
        self.H_obs = dataframe["H data"].values
        self.I_obs = dataframe["I data"].values
        # Rt_data 是用论文引用的统计方法得到的参考轨迹，只用于对比，不参与训练。
        self.Rt_data = dataframe["Rt data"].values
        self.delta = 1 / 5
        # 缩放：真实数据中论文用样本最大值作为缩放常数。
        self.SI = self.I_obs.max()
        self.SH = self.H_obs.max()
        # data.C = SI*delta/SH，使 Delta_Hs = sigma*Is*C 的量级与住院观测匹配。
        self.C = self.SI * self.delta / self.SH
        self.I_obs_sc = column(self.I_obs / self.SI)
        self.H_obs_sc = column(self.H_obs / self.SH)


class RealDataRunner(BaseRunner):
    """Run cases 6 and 7 from the real-data script.

    Case 6: Rt(t) 是 NN，sigma 是 SciANN Parameter 常数。
    Case 7: Rt(t) 和 sigma(t) 都是 NN。
    """

    def __init__(self, case=6, data_dir=None, plot=True, verbose=1):
        super(RealDataRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        if case not in (6, 7):
            raise ValueError("case must be 6 or 7")
        self.case = case
        # paper: Section 3.2 真实数据案例仍使用 MSE + Adam + 6000 collocation points。
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.collocation_points = 6000
        self.epochs_joint = 5000
        self.epochs_data = 1000
        self.epochs_ode = 3000
        self.batch_size = 100
        self.batch_data = 10

    def run(self):
        # 固定随机种子，方便复现网络初始化和 collocation 采样。
        sn.set_random_seed(234)
        data = RealEpidemicData(self.read_table("RealData.txt"))
        self.plot_inputs(data)
        training = self.build_training_grid(data)
        joint = self.run_joint(data, training)
        split = self.run_split(data, training)
        return SimpleNamespace(data=data, joint=joint, split=split)

    def build_training_grid(self, data):
        # 真实数据案例同样把观测点和 ODE collocation points 拼接。
        t_train_ode = uniform_collocation(
            data.grid.t0,
            data.grid.tf,
            self.collocation_points,
            include_initial=True,
        )
        return data_ode_training_grid(data.grid.t_data_sc, t_train_ode)

    def plot_inputs(self, data):
        # What this block does:
        # 展示真实训练数据：日新增感染、日新增住院、参考 Rt。
        # How it maps to the paper:
        # 对应论文 Fig 10-11 中用于拟合/比较的观测序列。
        if not self.plot:
            return
        fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        ax[0].plot(data.grid.timespan, data.I_obs, "xm", label="samples")
        ax[0].legend(loc=6)
        ax[0].set_xlabel("date")
        ax[0].set_ylabel("individuals")
        ax[0].set_title("Infectious Data")

        ax[1].plot(data.grid.timespan, data.H_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[1].set_xlabel("date")
        ax[1].set_ylabel("individuals")
        ax[1].set_title("Hospitalizations Data")

        ax[2].plot(data.grid.timespan, data.Rt_data, label=r"$\mathcal{R}_t$")
        ax[2].legend(loc=6)
        ax[2].set_xlabel("date")
        ax[2].set_title("Reproduction number")
        fig.tight_layout()
        self.maybe_show()

    def infection_residual(self, data, ts, is_, rt):
        # What this block does:
        # reduced SIR 中感染存量 I 的动力学残差。
        # How it maps to the paper:
        # paper: Eq. (13) / Eq. (31) 第一项：
        # dIs/dts - delta*(tf-t0)*(Rt-1)*Is = 0。
        # 即使真实训练数据是 Delta_I/Delta_H，内部仍需要隐含的 I(t) 满足这个 ODE。
        return sn.rename((sn.diff(is_, ts) - data.grid.tf * data.delta * (rt - 1) * is_), "L_dIdt")

    def run_joint(self, data, training):
        # What this block does:
        # joint approach: 同时训练 I(t)、Rt(t)、sigma，并拟合 Delta_I 和 Delta_H。
        # How it maps to the paper:
        # paper: Eq. (32) L_HI_joint = L_I + L_H + L_HI,ODE。
        sn.reset_session()
        ts = sn.Variable("ts")
        # Is 是隐藏的 infectious stock，不直接等于观测的日新增感染。
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        # build_sigma 按 Case 6/7 返回常数 sigma 或时间函数 sigma(t)。
        sigma = build_sigma(self.case, ts)
        l_didt = self.infection_residual(data, ts, is_, rt)
        # paper: Eq. (22) 缩放后 Delta_Hs = delta*C/CH * sigma*Is。
        # 这里 data.C 已包含 delta、SI、SH 的比例，所以写作 sigma*Is*C。
        delta_hs = sn.rename(sigma * is_ * data.C, "deltaHs")
        # paper: Eq. (30) Delta_I ~= dSigma_I/dt = delta*Rt*I；
        # 缩放后用 delta*Rt*Is 拟合 I_obs_sc。
        delta_is = sn.rename(data.delta * rt * is_, "deltaIs")
        loss_joint = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(delta_is), sn.Data(delta_hs)]
        model = sn.SciModel(ts, loss_joint, self.loss_err, self.optimizer)
        history = self.train(
            model,
            training.t_train,
            ["zeros"] * 2 + [(training.ids_data, data.I_obs_sc), (training.ids_data, data.H_obs_sc)],
            epochs=self.epochs_joint,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate(data, model, rt, sigma, delta_is, delta_hs)
        self.plot_predictions(data, predictions)
        errors = self.report_errors(data, predictions)
        return SimpleNamespace(
            model=model,
            Rt=rt,
            sigma=sigma,
            deltaIs=delta_is,
            deltaHs=delta_hs,
            history=history,
            predictions=predictions,
            errors=errors,
        )

    def run_split(self, data, training):
        # What this block does:
        # split approach:
        # 1) 先只用更可靠的住院数据 Delta_H 训练 \hat{Delta}_{H,s}。
        # 2) 冻结 Delta_H，再通过 Is=Delta_H/(sigma*C)、Delta_I=delta*Rt*Is
        #    与感染动力学残差来训练 Rt 和 sigma。
        # How it maps to the paper:
        # paper: Eq. (33)-(35)，论文明确说明住院数据通常更可靠，所以 split
        # 第一阶段优先拟合 Delta_H。
        sn.reset_session()
        ts = sn.Variable("ts")
        # data-only 住院网络，输出 scaled Delta_H。
        delta_hsc = sn.Functional("deltaHsc", ts, 4 * [50], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(delta_hsc), self.loss_err, self.optimizer)
        data_history = self.train(
            data_model,
            data.grid.t_data_sc,
            data.H_obs_sc,
            epochs=self.epochs_data,
            batch_size=self.batch_data,
        )
        if self.plot:
            # 第一阶段拟合效果：只看住院数据曲线是否被平滑重构。
            delta_hsc_pred = delta_hsc.eval(data_model, data.grid.t_test_sc)
            plt.plot(data.grid.t_test, delta_hsc_pred * data.SH, "--", c="k", linewidth=4)
            plt.scatter(data.grid.t_data, data.H_obs, marker="x", c="m", s=100)
            plt.xlabel("days")
            plt.ylabel("individuals")
            plt.legend([r"$\hat{\Delta}_H$", "samples"])
            plt.show()

        # 第二阶段冻结 Delta_H，作为更可靠的已知函数。
        delta_hs = sn.Functional("deltaHs", ts, 4 * [50], output_activation="square", trainable=False)
        delta_hs.set_weights(delta_hsc.get_weights())
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        # add_floor=True 防止 sigma 接近 0 导致 Is=DeltaH/(sigma*C) 数值爆炸。
        sigma = build_sigma(self.case, ts, add_floor=True)
        # paper: Eq. (33) 的真实数据版本：Is = DeltaHs / (sigma*C)。
        is_ = sn.rename(delta_hs / sigma / data.C, "Is")
        # paper: Eq. (34) Delta_Is = (CH/C) * Rt*DeltaHs/sigma；
        # 这里代入 Is 后写成 delta*Rt*Is，与 joint 形式一致。
        delta_is = sn.rename(data.delta * rt * is_, "dSdt")
        l_didt = self.infection_residual(data, ts, is_, rt)
        # loss_ode 包含：I(t) 的 reduced ODE、Rt 零目标占位、Delta_I 数据拟合。
        loss_ode = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(delta_is)]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
            training.t_train,
            ["zeros"] * 2 + [(training.ids_data, data.I_obs_sc)],
            epochs=self.epochs_ode,
            batch_size=self.batch_size,
        )
        predictions = self.evaluate(data, ode_model, rt, sigma, delta_is, delta_hs)
        self.plot_predictions(data, predictions)
        errors = self.report_errors(data, predictions)
        return SimpleNamespace(
            data_model=data_model,
            ode_model=ode_model,
            deltaHsc=delta_hsc,
            Rt=rt,
            sigma=sigma,
            deltaIs=delta_is,
            deltaHs=delta_hs,
            data_history=data_history,
            ode_history=ode_history,
            predictions=predictions,
            errors=errors,
        )

    def evaluate(self, data, model, rt, sigma, delta_is, delta_hs):
        # deltaI/deltaH 乘以各自缩放常数回到每日人数；Rt/sigma 保持无量纲。
        return SimpleNamespace(
            deltaI_test=delta_is.eval(model, data.grid.t_test_sc) * data.SI,
            deltaH_test=delta_hs.eval(model, data.grid.t_test_sc) * data.SH,
            Rt_test=rt.eval(model, data.grid.t_test_sc),
            sigma_test=sigma.eval(model, data.grid.t_test_sc),
            Rt=rt.eval(model, data.grid.t_data_sc),
        )

    def plot_predictions(self, data, predictions):
        # 对应论文 Fig 10-11：Delta_I、Delta_H、Rt，以及 Case 7 的 sigma(t)。
        if not self.plot:
            return
        plt.plot(data.grid.t_test, predictions.deltaI_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.I_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$\hat{\Delta}_I$", "samples"])
        plt.show()

        plt.plot(data.grid.t_test, predictions.deltaH_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.H_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$\hat{\Delta}_H$", "samples"])
        plt.show()

        plt.plot(data.grid.t_data, data.Rt_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.Rt_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()

        if self.case == 7:
            plt.plot(data.grid.t_test, predictions.sigma_test, "--", c="k", linewidth=4)
            plt.xlabel("days")
            plt.legend([r"$\hat{\sigma}$"])
            plt.show()

    def report_errors(self, data, predictions):
        # 真实数据没有 S/I/R 真值，只有统计方法得到的 Rt 参考，因此这里只报告 Rt error。
        errors = [("Rt error", relative_l2(data.Rt_data, predictions.Rt))]
        print_errors(errors)
        if self.case == 6:
            # Case 6 中 sigma 是常数参数，打印估计值便于和论文表格/图对照。
            print("Estimated sigma: {0:.4f}".format(np.ravel(predictions.sigma_test)[0]))
        return errors


def main(case=6, plot=True, verbose=1):
    return RealDataRunner(case=case, plot=plot, verbose=verbose).run()


if __name__ == "__main__":
    main()
