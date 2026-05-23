"""Refactored SIR-Rt SciANN workflow for cases 2 and 3."""

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
    """Data and scaling for the SIR-Rt cases."""

    def __init__(self, dataframe):
        self.frame = dataframe
        self.N = 56e6
        self.delta = 1 / 5
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        self.C = 1e5
        self.C1 = self.grid.tf * self.C / self.N
        self.C2 = self.grid.tf * self.delta
        self.S0 = self.N - 1
        self.I0 = 1
        self.R0 = 0

        self.S_data = dataframe["Susceptible"].values
        self.I_data = dataframe["Infectious"].values
        self.R_data = dataframe["Recovered"].values
        self.I_obs = dataframe["I data"].values
        self.Rt_data = dataframe["Rt"].values
        self.beta_data = self.Rt_data * self.delta

        self.I_obs_sc = column(self.I_obs / self.C)
        self.I_data_sc = self.I_data / self.C


class SIRRtRunner(BaseRunner):
    """Run the full SIR-Rt PINN workflow."""

    def __init__(self, case=2, data_dir=None, plot=True, verbose=1):
        super(SIRRtRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        if case not in (2, 3):
            raise ValueError("case must be 2 or 3")
        self.case = case
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
        filename = "Case2.txt" if self.case == 2 else "Case3.txt"
        return SIRRtData(self.read_table(filename))

    def run(self):
        sn.set_random_seed(234)
        data = self.load_data()
        self.plot_inputs(data)

        training = self.build_training_grid(data)
        joint = self.run_joint(data, training)
        split = self.run_split(data, training)
        return SimpleNamespace(data=data, joint=joint, split=split)

    def plot_inputs(self, data):
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
        t_train_ode = uniform_collocation(
            data.grid.t0,
            data.grid.tf,
            self.collocation_points,
            include_initial=True,
        )
        return data_ode_training_grid(data.grid.t_data_sc, t_train_ode)

    def build_sir_rt_model(self, data, beta_trainable=True):
        ts = sn.Variable("ts")
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=beta_trainable)
        beta = sn.Functional("Beta", ts, 4 * [100], output_activation="square")
        rs = data.N / data.C - is_ - ss
        return SimpleNamespace(ts=ts, Ss=ss, Is=is_, Rs=rs, Beta=beta)

    def sir_rt_residuals(self, data, model, include_i_initial=True):
        l_s0 = sn.rename((model.Ss - data.S0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_S0")
        l_r0 = sn.rename((model.Rs - data.R0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_R0")
        l_dsdt = sn.rename((sn.diff(model.Ss, model.ts) + data.C1 * model.Beta * model.Is * model.Ss), "L_dSdt")
        l_didt = sn.rename(
            (sn.diff(model.Is, model.ts) - data.C1 * model.Beta * model.Is * model.Ss + data.C2 * model.Is),
            "L_dIdt",
        )
        l_drdt = sn.rename((sn.diff(model.Rs, model.ts) - data.C2 * model.Is), "L_dRdt")
        residuals = [sn.PDE(l_dsdt), sn.PDE(l_didt), sn.PDE(l_drdt), sn.PDE(l_s0)]
        if include_i_initial:
            l_i0 = sn.rename((model.Is - data.I0 / data.C) * (1 - sn.sign(model.ts - data.grid.t0 / data.grid.tf)), "L_I0")
            residuals.append(sn.PDE(l_i0))
        residuals.append(sn.PDE(l_r0))
        return residuals

    def run_joint(self, data, training):
        sn.reset_session()
        model = self.build_sir_rt_model(data)
        loss_joint = self.sir_rt_residuals(data, model, include_i_initial=True) + [
            sn.Data(model.Ss * 0.0),
            sn.Data(model.Rs * 0.0),
            sn.Data(model.Beta * 0.0),
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
        sn.reset_session()
        ts = sn.Variable("ts")
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
        is_fixed = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=False)
        is_fixed.set_weights(weights)
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        beta = sn.Functional("Beta", ts, 4 * [100], output_activation="square")
        rs = data.N / data.C - is_fixed - ss
        model = SimpleNamespace(ts=ts, Ss=ss, Is=is_fixed, Rs=rs, Beta=beta)

        loss_ode = self.sir_rt_residuals(data, model, include_i_initial=False) + [
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
