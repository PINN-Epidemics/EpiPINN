"""Shared utilities for the refactored EpiPINN scripts.

The original files in ``EpiPINN/python`` are notebook exports.  This module
keeps the numerical choices unchanged while providing small reusable helpers
for data loading, training, plotting, and error reporting.
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
    """Container for physical and scaled time grids."""

    def __init__(self, timespan, t0, tf, test_step=0.1):
        self.timespan = timespan
        self.t0 = t0
        self.tf = tf
        self.t_data = np.arange(t0, tf)
        self.t_test = np.arange(t0, tf, test_step)
        self.t_data_sc = self.t_data / tf
        self.t_test_sc = self.t_test / tf

    @classmethod
    def from_dataframe_dates(cls, dataframe, test_step=0.1):
        timespan = dataframe["Date"].values.astype("datetime64[D]")
        return cls(timespan=timespan, t0=0.0, tf=len(timespan), test_step=test_step)


class TimedRun(object):
    """Training history with elapsed wall-clock time."""

    def __init__(self, history, elapsed):
        self.history = history
        self.elapsed = elapsed

    @property
    def history_dict(self):
        return getattr(self.history, "history", self.history)


class BaseRunner(object):
    """Base class for script runners."""

    def __init__(self, data_dir=None, plot=True, verbose=1):
        self.data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT
        self.plot = plot
        self.verbose = verbose

    def data_path(self, filename):
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
    """Return an array as a two-dimensional column, matching the original code."""

    return np.asarray(values).reshape(-1, 1)


def relative_l2(reference, prediction):
    """Compute the relative L2 error used in the notebooks."""

    return np.linalg.norm(reference - prediction, 2) / np.linalg.norm(reference, 2)


def print_errors(errors):
    for label, value in errors:
        print("{0}: {1:.3e}".format(label, value))


def uniform_collocation(t0, tf, count, include_initial=True):
    """Collocation points generated with the same uniform rule as the notebooks."""

    if include_initial:
        points = np.random.uniform(t0 / tf, 1.0, count - 1)
        return np.insert(points, 0, 0.0)
    return np.random.uniform(t0 / tf, 1.0, count)


def log_collocation(t0, tf, count):
    """Log-biased collocation points used by the constant-beta SIR case."""

    points = np.random.uniform(np.log1p(t0 / tf), np.log1p(1.0), count)
    return np.exp(points) - 1.0


def data_ode_training_grid(t_data_sc, t_train_ode):
    """Combine observed data times with ODE collocation times."""

    t_train = np.concatenate([column(t_data_sc), column(t_train_ode)])
    ids_data = np.arange(np.asarray(t_data_sc).size, dtype=np.intp)
    return SimpleNamespace(t_train=t_train, t_train_ode=t_train_ode, ids_data=ids_data)


def save_history(path, history):
    with open(path, "wb") as handle:
        pickle.dump(history, handle)


def save_history_and_weights(history_path, history, weights_path, model):
    save_history(history_path, history)
    model.save_weights(weights_path)


def plot_input_sir(timespan, s_data, i_data, r_data, i_obs):
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
    """Build the hospitalization parameter for cases 6 and 7."""

    if case == 6:
        return sn.Parameter(name="sigma", inputs=ts, non_neg=True)
    if case == 7:
        sigma = sn.Functional("sigma", ts, 10 * [5], output_activation="square")
        return sigma + 1e-3 if add_floor else sigma
    raise ValueError("case must be 6 or 7")
