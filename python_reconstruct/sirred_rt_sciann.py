"""Refactored reduced SIR-Rt SciANN workflow for cases 4 and 5."""

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
    """Data and scaling shared by cases 4 and 5."""

    def __init__(self, dataframe):
        self.frame = dataframe
        self.N = 56e6
        self.delta = 1 / 5
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        self.SI = 1e5
        self.SH = 1e3
        self.C = self.SI * self.delta / self.SH

        self.I_data = dataframe["Infectious"].values
        self.I_obs = dataframe["I data"].values
        self.Rt_data = dataframe["Rt"].values
        self.sigma_data = dataframe["sigma"].values
        self.H_data = (self.delta * self.sigma_data * self.I_data).reshape(-1)
        self.H_obs = dataframe["H data"].values

        self.I_obs_sc = column(self.I_obs / self.SI)
        self.I_data_sc = self.I_data / self.SI
        self.H_obs_sc = column(self.H_obs / self.SH)
        self.H_data_sc = self.H_data / self.SH


class ReducedSIRRtRunner(BaseRunner):
    """Run the reduced SciANN examples from the original script."""

    def __init__(self, data_dir=None, plot=True, verbose=1):
        super(ReducedSIRRtRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.collocation_points = 6000
        self.epochs_joint = 5000
        self.batch_size = 100
        self.batch_data = 10

    def run(self):
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
        t_train_ode = uniform_collocation(
            data.grid.t0,
            data.grid.tf,
            self.collocation_points,
            include_initial=True,
        )
        return data_ode_training_grid(data.grid.t_data_sc, t_train_ode)

    def plot_inputs(self, data):
        if not self.plot:
            return
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))
        ax[0].plot(data.grid.timespan, data.I_data, "r", label="Infectious")
        ax[0].plot(data.grid.timespan, data.I_obs, "xm", label="samples")
        ax[0].legend(loc=6)
        ax[0].set_xlabel("date")
        ax[0].set_ylabel("individuals")
        ax[0].set_title("Infectious and Observed Data")

        ax[1].plot(data.grid.timespan, data.H_data, "b", label="Hospitalizations")
        ax[1].plot(data.grid.timespan, data.H_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[1].set_xlabel("date")
        ax[1].set_ylabel("individuals")
        ax[1].set_title("Hospitalizations and Observed Data")

        ax[2].plot(data.grid.timespan, data.Rt_data, label=r"$\mathcal{R}_t$")
        ax[2].legend(loc=6)
        ax[2].set_xlabel("date")
        ax[2].set_title("Reproduction number")

        ax[3].plot(data.grid.timespan, data.sigma_data, "orange", label=r"$\sigma$")
        ax[3].legend(loc=6)
        ax[3].set_xlabel("date")
        ax[3].set_title("Hospitalization rate")
        fig.tight_layout()
        self.maybe_show()

    def reduced_residual(self, data, ts, is_, rt):
        return sn.rename((sn.diff(is_, ts) - data.grid.tf * data.delta * (rt - 1) * is_), "L_dIdt")

    def run_case4_joint(self, data, training):
        sn.reset_session()
        sn.set_random_seed(34)
        ts = sn.Variable("ts")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        loss_joint = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(is_)]
        model = sn.SciModel(ts, loss_joint, self.loss_err, self.optimizer)
        history = self.train(
            model,
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
        sn.reset_session()
        sn.set_random_seed(34)
        ts = sn.Variable("ts")
        isc = sn.Functional("Isc", ts, 4 * [50], output_activation="square")
        data_model = sn.SciModel(ts, sn.Data(isc), self.loss_err, self.optimizer)
        data_history = self.train(
            data_model,
            data.grid.t_data_sc,
            data.I_obs_sc,
            epochs=1000,
            batch_size=self.batch_data,
        )
        isc_pred = isc.eval(data_model, data.grid.t_data_sc)
        print("Isc error: {0:.3e}".format(relative_l2(data.I_data_sc, isc_pred)))

        is_fixed = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=False)
        is_fixed.set_weights(isc.get_weights())
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_fixed, rt)
        loss_ode = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(is_fixed * 0.0)]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
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
        sn.set_random_seed(34)
        sn.reset_session()
        ts = sn.Variable("ts")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        delta_hs = sn.rename(sigma * is_ * data.C, "deltaHs")
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
        sn.reset_session()
        sn.set_random_seed(34)
        ts = sn.Variable("ts")
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

        delta_hs = sn.Functional("deltaHs", ts, 4 * [100], output_activation="square", trainable=False)
        delta_hs.set_weights(hsc.get_weights())
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        is_ = sn.rename((1 / data.C) * delta_hs * sigma, "Is")
        l_didt = self.reduced_residual(data, ts, is_, rt)
        loss_ode = [sn.PDE(l_didt), sn.Data(rt * 0.0), sn.Data(sigma * 0.0), sn.Data(is_)]
        ode_model = sn.SciModel(ts, loss_ode, self.loss_err, self.optimizer)
        ode_history = self.train(
            ode_model,
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
        return SimpleNamespace(
            I_test=is_.eval(model, data.grid.t_test_sc) * data.SI,
            Rt_test=rt.eval(model, data.grid.t_test_sc),
            I=is_.eval(model, data.grid.t_data_sc) * data.SI,
            Rt=rt.eval(model, data.grid.t_data_sc),
        )

    def evaluate_case5(self, data, model, is_, rt, sigma, delta_hs, inverse_sigma=False):
        sigma_test = sigma.eval(model, data.grid.t_test_sc)
        sigma_data = sigma.eval(model, data.grid.t_data_sc)
        if inverse_sigma:
            sigma_test = 1 / sigma_test
            sigma_data = 1 / sigma_data
        return SimpleNamespace(
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
        if not self.plot:
            return
        plt.plot(data.grid.t_data, data.I_data, c="r", linewidth=4)
        plt.plot(data.grid.t_test, predictions.I_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.I_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()

        plt.plot(data.grid.t_data, data.Rt_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.Rt_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()

    def plot_case5(self, data, predictions):
        if not self.plot:
            return
        plt.plot(data.grid.t_data, data.I_data, c="r", linewidth=4)
        plt.plot(data.grid.t_test, predictions.I_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.I_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()

        plt.plot(data.grid.t_data, data.H_data, c="b", linewidth=4)
        plt.plot(data.grid.t_test, predictions.H_test, "--", c="k", linewidth=4)
        plt.scatter(data.grid.t_data, data.H_obs, marker="x", c="m", s=100)
        plt.xlabel("days")
        plt.ylabel("individuals")
        plt.legend([r"$\Delta_H$", r"$\hat{\Delta}_H$", "samples"])
        plt.show()

        plt.plot(data.grid.t_data, data.Rt_data, linewidth=4)
        plt.plot(data.grid.t_test, predictions.Rt_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()

        plt.plot(data.grid.t_data, data.sigma_data, c="orange", linewidth=4)
        plt.plot(data.grid.t_test, predictions.sigma_test, "--", c="k", linewidth=4)
        plt.xlabel("days")
        plt.legend([r"$\sigma$", r"$\hat{\sigma}$"])
        plt.show()

    def report_case4_errors(self, data, predictions):
        errors = [
            ("I error", relative_l2(data.I_data, predictions.I)),
            ("Rt error", relative_l2(data.Rt_data, predictions.Rt)),
            ("Rt error last 100 days", relative_l2(data.Rt_data[20:], predictions.Rt[20:])),
        ]
        print_errors(errors)
        return errors

    def report_case5_errors(self, data, predictions):
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
