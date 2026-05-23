"""Refactored constant-transmission SIR SciANN workflow for case 1."""

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
    """Reference SIR ODE used to generate case 1 synthetic data."""

    del t0
    susceptible, infectious, recovered = x
    lambda_val = beta * infectious / population
    return [
        -lambda_val * susceptible,
        lambda_val * susceptible - delta * infectious,
        delta * infectious,
    ]


class ConstantSIRData(object):
    """Synthetic case 1 data and scaling."""

    def __init__(self, weekly=False):
        self.N = 56e6
        self.delta = 1 / 5
        self.r0 = 3.0
        self.beta = self.delta * self.r0
        self.t0 = 0.0
        self.tf = 90.0
        self.C = 1e5
        self.C1 = self.tf * self.C / self.N
        self.C2 = self.tf * self.delta
        self.S0 = self.N - 1
        self.I0 = 1
        self.R0 = 0
        self.weekly = weekly

        timespan = np.arange("2020-02-01", "2020-05-01", dtype="datetime64[D]")
        tspan = timespan.astype(int)
        solution = odeint(sir_ode, [self.S0, self.I0, self.R0], tspan, args=(self.delta, self.beta, self.N, tspan[0]))

        self.grid = TimeGrid(timespan=timespan, t0=self.t0, tf=self.tf)
        self.S_data = solution[:, 0]
        self.I_data = solution[:, 1]
        self.R_data = solution[:, 2]
        self.I_obs = np.random.poisson(self.I_data)

        self.I_obs_sc = self.I_obs / self.C
        self.I_data_sc = self.I_data / self.C
        if weekly:
            self.I_obs_train = column(self.I_obs_sc[::7])
            self.t_data_train = self.grid.t_data_sc[::7]
        else:
            self.I_obs_train = column(self.I_obs_sc)
            self.t_data_train = self.grid.t_data_sc


class ConstantSIRRunner(BaseRunner):
    """Run the case 1 PINN workflow."""

    def __init__(self, weekly=False, data_dir=None, plot=True, verbose=1, save_outputs=True):
        super(ConstantSIRRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        self.weekly = weekly
        self.save_outputs = save_outputs
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.adaptive_ntk = {"method": "NTK", "freq": 100}
        self.collocation_points = 6000
        self.epochs_joint = 5000
        self.epochs_ode = 1000
        self.batch_size = 100

    def run(self):
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
        t_train_ode = log_collocation(data.t0, data.tf, self.collocation_points)
        t_train = np.concatenate([column(data.t_data_train), column(t_train_ode)])
        ids_data = np.arange(data.t_data_train.size, dtype=np.intp)
        return SimpleNamespace(t_train=t_train, t_train_ode=t_train_ode, ids_data=ids_data)

    def build_state(self, data, ts, beta, is_trainable=True):
        ss = sn.Functional("Ss", ts, 4 * [50], output_activation="square")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square", trainable=is_trainable)
        rs = data.N / data.C - is_ - ss
        return SimpleNamespace(ts=ts, Ss=ss, Is=is_, Rs=rs, Beta=beta)

    def sir_losses(self, data, state, include_i_initial=True, add_beta_regularizer=False):
        l_s0 = sn.rename((state.Ss - data.S0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_S0")
        l_r0 = sn.rename((state.Rs - data.R0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_R0")
        l_dsdt = sn.rename((sn.diff(state.Ss, state.ts) + data.C1 * state.Beta * state.Is * state.Ss), "L_dSdt")
        l_didt = sn.rename(
            (sn.diff(state.Is, state.ts) - data.C1 * state.Beta * state.Is * state.Ss + data.C2 * state.Is),
            "L_dIdt",
        )
        l_drdt = sn.rename((sn.diff(state.Rs, state.ts) - data.C2 * state.Is), "L_dRdt")
        losses = [sn.PDE(l_dsdt), sn.PDE(l_didt), sn.PDE(l_drdt), sn.PDE(l_s0)]
        if include_i_initial:
            l_i0 = sn.rename((state.Is - data.I0 / data.C) * (1 - sn.sign(state.ts - data.t0 / data.tf)), "L_I0")
            losses.append(sn.PDE(l_i0))
        losses.append(sn.PDE(l_r0))
        losses += [sn.Data(state.Ss * 0.0), sn.Data(state.Rs * 0.0)]
        if add_beta_regularizer:
            losses.append(sn.Data(state.Beta * 0.0))
        return losses

    def run_joint(self, data, training):
        sn.reset_session()
        ts = sn.Variable("ts")
        beta = sn.Parameter(name="Beta", inputs=ts, non_neg=True)
        state = self.build_state(data, ts, beta)
        losses = self.sir_losses(data, state, include_i_initial=True, add_beta_regularizer=False) + [sn.Data(state.Is)]
        model = sn.SciModel(ts, losses, self.loss_err, self.optimizer)
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
        predictions = self.evaluate(data, model, state)
        self.plot_predictions(data, predictions)
        errors = self.report_errors(data, predictions)
        if self.save_outputs:
            save_history_and_weights("hJoint.txt", history.history_dict, "mJoint.hdf5", model)
        return SimpleNamespace(model=model, functions=state, history=history, predictions=predictions, errors=errors)

    def run_split(self, data, training, joint):
        sn.reset_session()
        ts = sn.Variable("ts")
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
