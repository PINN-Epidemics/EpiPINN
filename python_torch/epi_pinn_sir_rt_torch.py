#!/usr/bin/env python
"""PyTorch implementation of the full SIR model with time-dependent beta(t).

This translates ``python/EpiPINN-SIR-Rt.py`` from SciANN to PyTorch.  It covers
Case 2 and Case 3.
"""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from pinn_torch_common import (
    MLP,
    TrainConfig,
    as_column_tensor,
    derivative,
    freeze,
    make_collocation,
    mse,
    quick_config,
    random_batch,
    rel_l2,
    resolve_data_path,
    set_seed,
    sign_gated_initial_loss,
    train_loop,
    zero_expression_loss,
)


class JointSIRRt(nn.Module):
    """Joint S/I/beta(t) PINN; R is defined by conservation."""

    def __init__(self, cfg: TrainConfig, n_over_c: float) -> None:
        super().__init__()
        self.s_net = MLP(cfg.hidden_state, positive=True)
        self.i_net = MLP(cfg.hidden_state, positive=True)
        self.beta_net = MLP(cfg.hidden_beta, positive=True)
        self.n_over_c = float(n_over_c)

    def states(self, t: torch.Tensor):
        s_value = self.s_net(t)
        i_value = self.i_net(t)
        r_value = self.n_over_c - i_value - s_value
        beta_value = self.beta_net(t)
        return s_value, i_value, r_value, beta_value


class DataOnlyI(nn.Module):
    """First split stage: fit I(t) from infectious observations only."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.i_net = MLP(cfg.hidden_state, positive=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.i_net(t)


class SplitSIRRt(nn.Module):
    """Second split stage with fixed I(t), learned S(t) and beta(t)."""

    def __init__(self, fixed_i: nn.Module, cfg: TrainConfig, n_over_c: float) -> None:
        super().__init__()
        self.i_net = freeze(fixed_i)
        self.s_net = MLP(cfg.hidden_state, positive=True)
        self.beta_net = MLP(cfg.hidden_beta, positive=True)
        self.n_over_c = float(n_over_c)

    def states(self, t: torch.Tensor):
        s_value = self.s_net(t)
        i_value = self.i_net(t)
        r_value = self.n_over_c - i_value - s_value
        beta_value = self.beta_net(t)
        return s_value, i_value, r_value, beta_value


def plot_axis(df: pd.DataFrame) -> np.ndarray:
    """Return a plotting x-axis compatible with Case2 and Case3 data files."""

    if "Date" in df.columns:
        return df["Date"].values.astype("datetime64[D]")
    return np.arange(len(df))


def sir_rt_loss(model, t_batch, t_data, i_obs, constants, include_i0: bool, include_data: bool):
    s_value, i_value, r_value, beta_value = model.states(t_batch)
    ds_dt = derivative(s_value, t_batch)
    di_dt = derivative(i_value, t_batch)
    dr_dt = derivative(r_value, t_batch)
    loss_ode = (
        mse(ds_dt + constants["C1"] * beta_value * i_value * s_value, torch.zeros_like(s_value))
        + mse(di_dt - constants["C1"] * beta_value * i_value * s_value + constants["C2"] * i_value, torch.zeros_like(i_value))
        + mse(dr_dt - constants["C2"] * i_value, torch.zeros_like(r_value))
    )

    loss_s0 = sign_gated_initial_loss(t_batch, s_value, constants["S0"] / constants["C"], constants["t0"] / constants["tf"])
    loss_r0 = sign_gated_initial_loss(t_batch, r_value, constants["R0"] / constants["C"], constants["t0"] / constants["tf"])
    loss_ic = (
        loss_s0
        + loss_r0
    )
    if include_i0:
        loss_ic = loss_ic + sign_gated_initial_loss(
            t_batch,
            i_value,
            constants["I0"] / constants["C"],
            constants["t0"] / constants["tf"],
        )

    loss_data = torch.zeros((), dtype=torch.float32, device=t_batch.device)
    if include_data and t_data is not None and i_obs is not None:
        _, i_pred_data, _, _ = model.states(t_data)
        loss_data = mse(i_pred_data, i_obs)
    return loss_ode, loss_ic, loss_data, zero_expression_loss(s_value, r_value, beta_value)


def train_joint(cfg, device, t_train, t_data_sc, i_obs_sc, constants):
    model = JointSIRRt(cfg, constants["N"] / constants["C"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        loss_ode, loss_ic, loss_data, loss_zero = sir_rt_loss(
            model,
            tb,
            t_data,
            i_obs,
            constants,
            include_i0=True,
            include_data=True,
        )
        return loss_ode + loss_ic + loss_zero + loss_data, {
            "ode": loss_ode,
            "ic": loss_ic,
            "zero": loss_zero,
            "data": loss_data,
        }

    history = train_loop(model, optimizer, cfg.epochs_joint, loss_fn, cfg.print_every, "joint")
    return model, history


def train_data_i(cfg, device, t_data_sc, i_obs_sc):
    model = DataOnlyI(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_data), cfg.batch_size_data, device)
        loss_data = mse(model(t_data[idx]), i_obs[idx])
        return loss_data, {"data": loss_data}

    history = train_loop(model, optimizer, cfg.epochs_data, loss_fn, cfg.print_every, "split-data")
    return model, history


def train_split_ode(cfg, device, fixed_i, t_train_ode, constants):
    model = SplitSIRRt(fixed_i, cfg, constants["N"] / constants["C"]).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    t_all = as_column_tensor(t_train_ode, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        loss_ode, loss_ic, _, loss_zero = sir_rt_loss(
            model,
            tb,
            None,
            None,
            constants,
            include_i0=False,
            include_data=False,
        )
        loss_zero = loss_zero + zero_expression_loss(model.states(tb)[1])
        return loss_ode + loss_ic + loss_zero, {"ode": loss_ode, "ic": loss_ic, "zero": loss_zero}

    history = train_loop(model, optimizer, cfg.epochs_ode, loss_fn, cfg.print_every, "split-ode")
    return model, history


def eval_model(model, t_values, device, c_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        s_value, i_value, r_value, beta_value = model.states(t_tensor)
    return (
        s_value.cpu().numpy().reshape(-1) * c_scale,
        i_value.cpu().numpy().reshape(-1) * c_scale,
        r_value.cpu().numpy().reshape(-1) * c_scale,
        beta_value.cpu().numpy().reshape(-1),
    )


def report_errors(label, model, t_data_sc, s_data, i_data, r_data, beta_data, device, c_scale):
    s_pred, i_pred, r_pred, beta_pred = eval_model(model, t_data_sc, device, c_scale)
    print("{} S error: {:.3e}".format(label, rel_l2(s_data, s_pred)))
    print("{} I error: {:.3e}".format(label, rel_l2(i_data, i_pred)))
    print("{} R error: {:.3e}".format(label, rel_l2(r_data, r_pred)))
    print("{} Beta error: {:.3e}".format(label, rel_l2(beta_data, beta_pred)))
    print("{} Beta error last 70 days: {:.3e}".format(label, rel_l2(beta_data[20:], beta_pred[20:])))


def run(args: argparse.Namespace) -> None:
    cfg = TrainConfig(seed=args.seed, device=args.device)
    cfg.epochs_joint = args.epochs_joint
    cfg.epochs_data = args.epochs_data
    cfg.epochs_ode = args.epochs_ode
    if args.quick:
        cfg = quick_config(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    file_name = "Case2.txt" if args.case == 2 else "Case3.txt"
    df = pd.read_table(resolve_data_path(args.data_dir, file_name))
    timespan = plot_axis(df)

    population = 56e6
    delta = 1 / 5
    t0 = 0.0
    tf = float(len(df))
    c_scale = 1e5
    constants = {
        "N": population,
        "delta": delta,
        "t0": t0,
        "tf": tf,
        "C": c_scale,
        "C1": tf * c_scale / population,
        "C2": tf * delta,
        "S0": population - 1,
        "I0": 1,
        "R0": 0,
    }

    t_data = np.arange(t0, tf)
    t_test = np.arange(t0, tf, 0.1)
    t_data_sc = t_data / tf
    t_test_sc = t_test / tf
    s_data = df["Susceptible"].values
    i_data = df["Infectious"].values
    r_data = df["Recovered"].values
    i_obs = df["I data"].values
    rt_data = df["Rt"].values
    beta_data = rt_data * delta
    i_obs_sc = i_obs / c_scale

    if not args.no_plots:
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        plot_s = ax[0].plot(timespan, s_data, "b", label="Susceptible")
        ax2 = ax[0].twinx()
        plot_r = ax2.plot(timespan, r_data, "g", label="Recovered")
        lines = plot_s + plot_r
        ax[0].legend(lines, [line.get_label() for line in lines], loc=6)
        ax[1].plot(timespan, i_data, "r", label="Infectious")
        ax[1].plot(timespan, i_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[2].plot(timespan, beta_data, label=r"$\beta$")
        ax[2].legend(loc=6)
        fig.tight_layout()
        plt.show()

    t_train_ode = make_collocation(t0 / tf, cfg.n_collocation, include_zero=True)
    t_train = np.concatenate([t_data_sc.reshape(-1, 1), t_train_ode])

    print("Device:", device)
    start = time.time()
    joint, _ = train_joint(cfg, device, t_train, t_data_sc, i_obs_sc, constants)
    print("Joint training time: {:.1f}s".format(time.time() - start))
    report_errors("Joint", joint, t_data_sc, s_data, i_data, r_data, beta_data, device, c_scale)

    start = time.time()
    data_i, _ = train_data_i(cfg, device, t_data_sc, i_obs_sc)
    print("Split data training time: {:.1f}s".format(time.time() - start))
    split, _ = train_split_ode(cfg, device, data_i.i_net, t_train_ode, constants)
    report_errors("Split", split, t_data_sc, s_data, i_data, r_data, beta_data, device, c_scale)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(joint.state_dict(), out / "sir_rt_joint_torch.pt")
        torch.save(data_i.state_dict(), out / "sir_rt_split_data_torch.pt")
        torch.save(split.state_dict(), out / "sir_rt_split_ode_torch.pt")

    if not args.no_plots:
        s_pred, i_pred, r_pred, beta_pred = eval_model(joint, t_test_sc, device, c_scale)
        plt.plot(t_data, s_data, "b", linewidth=4)
        plt.plot(t_test, s_pred, "--k", linewidth=4)
        plt.legend([r"$S$", r"$\hat{S}$"])
        plt.show()
        plt.plot(t_data, i_data, "r", linewidth=4)
        plt.plot(t_test, i_pred, "--k", linewidth=4)
        plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()
        plt.plot(t_data, r_data, "g", linewidth=4)
        plt.plot(t_test, r_pred, "--k", linewidth=4)
        plt.legend([r"$R$", r"$\hat{R}$"])
        plt.show()
        plt.plot(t_data, beta_data, linewidth=4)
        plt.plot(t_test, beta_pred, "--k", linewidth=4)
        plt.legend([r"$\beta$", r"$\hat{\beta}$"])
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(2, 3), default=2)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-data", type=int, default=3000)
    parser.add_argument("--epochs-ode", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
