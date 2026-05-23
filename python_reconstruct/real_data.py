"""Refactored real-data SciANN workflow for cases 6 and 7."""

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
    """Real-data scaling for cases 6 and 7."""

    def __init__(self, dataframe):
        self.frame = dataframe
        self.grid = TimeGrid.from_dataframe_dates(dataframe)
        self.H_obs = dataframe["H data"].values
        self.I_obs = dataframe["I data"].values
        self.Rt_data = dataframe["Rt data"].values
        self.delta = 1 / 5
        self.SI = self.I_obs.max()
        self.SH = self.H_obs.max()
        self.C = self.SI * self.delta / self.SH
        self.I_obs_sc = column(self.I_obs / self.SI)
        self.H_obs_sc = column(self.H_obs / self.SH)


class RealDataRunner(BaseRunner):
    """Run cases 6 and 7 from the real-data script."""

    def __init__(self, case=6, data_dir=None, plot=True, verbose=1):
        super(RealDataRunner, self).__init__(data_dir=data_dir, plot=plot, verbose=verbose)
        if case not in (6, 7):
            raise ValueError("case must be 6 or 7")
        self.case = case
        self.loss_err = "mse"
        self.optimizer = "adam"
        self.collocation_points = 6000
        self.epochs_joint = 5000
        self.epochs_data = 1000
        self.epochs_ode = 3000
        self.batch_size = 100
        self.batch_data = 10

    def run(self):
        sn.set_random_seed(234)
        data = RealEpidemicData(self.read_table("RealData.txt"))
        self.plot_inputs(data)
        training = self.build_training_grid(data)
        joint = self.run_joint(data, training)
        split = self.run_split(data, training)
        return SimpleNamespace(data=data, joint=joint, split=split)

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
        return sn.rename((sn.diff(is_, ts) - data.grid.tf * data.delta * (rt - 1) * is_), "L_dIdt")

    def run_joint(self, data, training):
        sn.reset_session()
        ts = sn.Variable("ts")
        is_ = sn.Functional("Is", ts, 4 * [50], output_activation="square")
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        sigma = build_sigma(self.case, ts)
        l_didt = self.infection_residual(data, ts, is_, rt)
        delta_hs = sn.rename(sigma * is_ * data.C, "deltaHs")
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
        sn.reset_session()
        ts = sn.Variable("ts")
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
            delta_hsc_pred = delta_hsc.eval(data_model, data.grid.t_test_sc)
            plt.plot(data.grid.t_test, delta_hsc_pred * data.SH, "--", c="k", linewidth=4)
            plt.scatter(data.grid.t_data, data.H_obs, marker="x", c="m", s=100)
            plt.xlabel("days")
            plt.ylabel("individuals")
            plt.legend([r"$\hat{\Delta}_H$", "samples"])
            plt.show()

        delta_hs = sn.Functional("deltaHs", ts, 4 * [50], output_activation="square", trainable=False)
        delta_hs.set_weights(delta_hsc.get_weights())
        rt = sn.Functional("Rt", ts, 4 * [100], output_activation="square")
        sigma = build_sigma(self.case, ts, add_floor=True)
        is_ = sn.rename(delta_hs / sigma / data.C, "Is")
        delta_is = sn.rename(data.delta * rt * is_, "dSdt")
        l_didt = self.infection_residual(data, ts, is_, rt)
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
        return SimpleNamespace(
            deltaI_test=delta_is.eval(model, data.grid.t_test_sc) * data.SI,
            deltaH_test=delta_hs.eval(model, data.grid.t_test_sc) * data.SH,
            Rt_test=rt.eval(model, data.grid.t_test_sc),
            sigma_test=sigma.eval(model, data.grid.t_test_sc),
            Rt=rt.eval(model, data.grid.t_data_sc),
        )

    def plot_predictions(self, data, predictions):
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
        errors = [("Rt error", relative_l2(data.Rt_data, predictions.Rt))]
        print_errors(errors)
        if self.case == 6:
            print("Estimated sigma: {0:.4f}".format(np.ravel(predictions.sigma_test)[0]))
        return errors


def main(case=6, plot=True, verbose=1):
    return RealDataRunner(case=case, plot=plot, verbose=verbose).run()


if __name__ == "__main__":
    main()
