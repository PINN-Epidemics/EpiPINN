#!/usr/bin/env python
"""PyTorch implementation of the constant-transmission-rate SIR PINN example.

This translates ``python/EpiPINN-SIR-R0-sciann.py`` from SciANN to explicit
PyTorch modules and autograd residuals.
"""

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from torch import nn

from pinn_torch_common import (
    MLP,
    PositiveParameter,
    TrainConfig,
    as_column_tensor,
    derivative,
    freeze,
    make_log_collocation,
    mse,
    quick_config,
    random_batch,
    rel_l2,
    set_seed,
    sign_gated_initial_loss,
    train_loop,
    zero_expression_loss,
)


class JointSIRConstantBeta(nn.Module):
    """Joint S/I/beta PINN; R is recovered from population conservation."""

    def __init__(self, cfg: TrainConfig, n_over_c: float) -> None:
        super().__init__()
        self.s_net = MLP(cfg.hidden_state, positive=True)
        self.i_net = MLP(cfg.hidden_state, positive=True)
        self.beta = PositiveParameter(initial_value=0.5)
        self.n_over_c = float(n_over_c)

    def states(self, t: torch.Tensor):
        s_value = self.s_net(t)
        i_value = self.i_net(t)
        r_value = self.n_over_c - i_value - s_value
        beta_value = self.beta(t)
        return s_value, i_value, r_value, beta_value


class DataOnlyI(nn.Module):
    """First split stage: fit I(t) from observations only."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.i_net = MLP(cfg.hidden_state, positive=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.i_net(t)


class SplitSIRConstantBeta(nn.Module):
    """Second split stage: fixed I(t), train S(t) and constant beta."""

    def __init__(self, fixed_i: nn.Module, cfg: TrainConfig, n_over_c: float) -> None:
        super().__init__()
        self.i_net = freeze(fixed_i)
        self.s_net = MLP(cfg.hidden_state, positive=True)
        self.beta = PositiveParameter(initial_value=0.5)
        self.n_over_c = float(n_over_c)

    def states(self, t: torch.Tensor):
        s_value = self.s_net(t)
        i_value = self.i_net(t)
        r_value = self.n_over_c - i_value - s_value
        beta_value = self.beta(t)
        return s_value, i_value, r_value, beta_value


def sir_rhs(x, _t, delta, beta, population):
    """Reference SIR ODE used to synthesize Case 1 data."""

    susceptible, infectious, recovered = x
    lambda_value = beta * infectious / population
    return [
        -lambda_value * susceptible,
        lambda_value * susceptible - delta * infectious,
        delta * infectious,
    ]


def train_joint(
    cfg: TrainConfig,
    device: torch.device,
    t_train: np.ndarray,
    t_data_sc: np.ndarray,
    i_obs_sc: np.ndarray,
    constants: dict,
):
    model = JointSIRConstantBeta(cfg, constants["N"] / constants["C"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        s_value, i_value, r_value, beta_value = model.states(tb)
        ds_dt = derivative(s_value, tb)
        di_dt = derivative(i_value, tb)
        dr_dt = derivative(r_value, tb)
        loss_ode = (
            mse(ds_dt + constants["C1"] * beta_value * i_value * s_value, torch.zeros_like(s_value))
            + mse(di_dt - constants["C1"] * beta_value * i_value * s_value + constants["C2"] * i_value, torch.zeros_like(i_value))
            + mse(dr_dt - constants["C2"] * i_value, torch.zeros_like(r_value))
        )

        loss_s0 = sign_gated_initial_loss(tb, s_value, constants["S0"] / constants["C"], constants["t0"] / constants["tf"])
        loss_i0 = sign_gated_initial_loss(tb, i_value, constants["I0"] / constants["C"], constants["t0"] / constants["tf"])
        loss_r0 = sign_gated_initial_loss(tb, r_value, constants["R0"] / constants["C"], constants["t0"] / constants["tf"])
        # Mirrors sn.Data(Ss * 0.0) and sn.Data(Rs * 0.0) from the SciANN source.
        loss_zero_data = zero_expression_loss(s_value, r_value)
        loss_ic = (
            loss_s0
            + loss_i0
            + loss_r0
        )
        _, i_pred_data, _, _ = model.states(t_data)
        loss_data = mse(i_pred_data, i_obs)
        return loss_ode + loss_ic + loss_zero_data + loss_data, {
            "ode": loss_ode,
            "ic": loss_ic,
            "zero": loss_zero_data,
            "data": loss_data,
            "beta": beta_value.mean(),
        }

    history = train_loop(model, optimizer, cfg.epochs_joint, loss_fn, cfg.print_every, "joint")
    return model, history


def train_data_i(cfg: TrainConfig, device: torch.device, t_data_sc: np.ndarray, i_obs_sc: np.ndarray):
    model = DataOnlyI(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_data), cfg.batch_size_data, device)
        pred = model(t_data[idx])
        loss_data = mse(pred, i_obs[idx])
        return loss_data, {"data": loss_data}

    history = train_loop(model, optimizer, cfg.epochs_data, loss_fn, cfg.print_every, "split-data")
    return model, history


def train_split_ode(
    cfg: TrainConfig,
    device: torch.device,
    fixed_i: nn.Module,
    t_train_ode: np.ndarray,
    constants: dict,
):
    model = SplitSIRConstantBeta(fixed_i, cfg, constants["N"] / constants["C"]).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    t_all = as_column_tensor(t_train_ode, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        s_value, i_value, r_value, beta_value = model.states(tb)
        ds_dt = derivative(s_value, tb)
        di_dt = derivative(i_value, tb)
        dr_dt = derivative(r_value, tb)
        loss_ode = (
            mse(ds_dt + constants["C1"] * beta_value * i_value * s_value, torch.zeros_like(s_value))
            + mse(di_dt - constants["C1"] * beta_value * i_value * s_value + constants["C2"] * i_value, torch.zeros_like(i_value))
            + mse(dr_dt - constants["C2"] * i_value, torch.zeros_like(r_value))
        )
        loss_s0 = sign_gated_initial_loss(tb, s_value, constants["S0"] / constants["C"], constants["t0"] / constants["tf"])
        loss_r0 = sign_gated_initial_loss(tb, r_value, constants["R0"] / constants["C"], constants["t0"] / constants["tf"])
        # Mirrors sn.Data(Ss * 0.0), sn.Data(Rs * 0.0), sn.Data(Is * 0.0).
        loss_zero_data = zero_expression_loss(s_value, r_value, i_value)
        loss_ic = (
            loss_s0
            + loss_r0
        )
        return loss_ode + loss_ic + loss_zero_data, {
            "ode": loss_ode,
            "ic": loss_ic,
            "zero": loss_zero_data,
            "beta": beta_value.mean(),
        }

    history = train_loop(model, optimizer, cfg.epochs_ode, loss_fn, cfg.print_every, "split-ode")
    return model, history


def eval_sir(model, t_values: np.ndarray, device: torch.device, c_scale: float):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        s_value, i_value, r_value, beta_value = model.states(t_tensor)
    return (
        s_value.cpu().numpy().reshape(-1) * c_scale,
        i_value.cpu().numpy().reshape(-1) * c_scale,
        r_value.cpu().numpy().reshape(-1) * c_scale,
        beta_value.cpu().numpy().reshape(-1),
    )


def run(args: argparse.Namespace) -> None:
    cfg = TrainConfig(seed=args.seed, device=args.device)
    cfg.epochs_joint = args.epochs_joint
    cfg.epochs_data = args.epochs_data
    cfg.epochs_ode = args.epochs_ode
    if args.quick:
        cfg = quick_config(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    population = 56e6
    delta = 1 / 5
    r0_value = 3.0
    beta = delta * r0_value
    t0 = 0.0
    tf = 90.0
    c_scale = 1e5
    constants = {
        "N": population,
        "delta": delta,
        "beta": beta,
        "t0": t0,
        "tf": tf,
        "C": c_scale,
        "C1": tf * c_scale / population,
        "C2": tf * delta,
        "S0": population - 1,
        "I0": 1,
        "R0": 0,
    }

    timespan = np.arange("2020-02-01", "2020-05-01", dtype="datetime64[D]")
    tspan = timespan.astype(int)
    x0 = [constants["S0"], constants["I0"], constants["R0"]]
    solution = odeint(sir_rhs, x0, tspan, args=(delta, beta, population))
    s_data, i_data, r_data = solution[:, 0], solution[:, 1], solution[:, 2]
    i_obs = np.random.poisson(i_data)

    t_data = np.arange(t0, tf)
    t_test = np.arange(t0, tf, 0.1)
    i_obs_sc = i_obs / c_scale
    i_data_sc = i_data / c_scale
    t_data_sc = t_data / tf
    t_test_sc = t_test / tf

    if args.weekly:
        t_data_train = t_data_sc[::7]
        i_obs_train = i_obs_sc[::7]
        cfg.epochs_data = min(cfg.epochs_data, 1000)
        cfg.batch_size_data = 13
    else:
        t_data_train = t_data_sc
        i_obs_train = i_obs_sc

    if not args.no_plots:
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(timespan, s_data, "b", label="Susceptible")
        ax[0].plot(timespan, r_data, "g", label="Recovered")
        ax[0].legend()
        ax[1].plot(timespan, i_data, "r", label="Infectious")
        ax[1].plot(timespan, i_obs, "xm", label="samples")
        ax[1].legend()
        fig.tight_layout()
        plt.show()

    t_train_ode = make_log_collocation(t0 / tf, cfg.n_collocation)
    t_train = np.concatenate([t_data_train.reshape(-1, 1), t_train_ode])

    print("Device:", device)
    start = time.time()
    joint, hist_joint = train_joint(cfg, device, t_train, t_data_train, i_obs_train, constants)
    print("Joint training time: {:.1f}s".format(time.time() - start))
    s_pred, i_pred, r_pred, beta_pred = eval_sir(joint, t_data_sc, device, c_scale)
    print("Joint S error: {:.3e}".format(rel_l2(s_data, s_pred)))
    print("Joint I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Joint R error: {:.3e}".format(rel_l2(r_data, r_pred)))
    print("Joint Beta error: {:.3e}".format(abs(beta_pred[0] - beta) / beta))

    start = time.time()
    data_i, hist_data = train_data_i(cfg, device, t_data_train, i_obs_train)
    print("Split data training time: {:.1f}s".format(time.time() - start))
    split, hist_ode = train_split_ode(cfg, device, data_i.i_net, t_train_ode, constants)
    s_pred, i_pred, r_pred, beta_pred = eval_sir(split, t_data_sc, device, c_scale)
    print("Split S error: {:.3e}".format(rel_l2(s_data, s_pred)))
    print("Split I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Split R error: {:.3e}".format(rel_l2(r_data, r_pred)))
    print("Split Beta error: {:.3e}".format(abs(beta_pred[0] - beta) / beta))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(joint.state_dict(), output_dir / "sir_r0_joint_torch.pt")
        torch.save(data_i.state_dict(), output_dir / "sir_r0_split_data_torch.pt")
        torch.save(split.state_dict(), output_dir / "sir_r0_split_ode_torch.pt")
        with (output_dir / "sir_r0_history_torch.pkl").open("wb") as file:
            pickle.dump({"joint": hist_joint, "split_data": hist_data, "split_ode": hist_ode}, file)

    if not args.no_plots:
        s_test, i_test, r_test, beta_test = eval_sir(joint, t_test_sc, device, c_scale)
        plt.plot(t_data, s_data, "b", linewidth=4)
        plt.plot(t_test, s_test, "--k", linewidth=4)
        plt.legend([r"$S$", r"$\hat{S}$"])
        plt.show()
        plt.plot(t_data, i_data, "r", linewidth=4)
        plt.plot(t_test, i_test, "--k", linewidth=4)
        if args.weekly:
            plt.scatter(t_data[::7], i_obs[::7], marker="x", c="m", s=100)
        else:
            plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()
        plt.plot(t_data, r_data, "g", linewidth=4)
        plt.plot(t_test, r_test, "--k", linewidth=4)
        plt.legend([r"$R$", r"$\hat{R}$"])
        plt.show()
        plt.plot(t_test, beta_test, "--k", linewidth=4)
        plt.axhline(beta, color="tab:blue", linewidth=4)
        plt.legend([r"$\hat{\beta}$", r"$\beta$"])
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-data", type=int, default=3000)
    parser.add_argument("--epochs-ode", type=int, default=1000)
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
