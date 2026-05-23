#!/usr/bin/env python
"""PyTorch implementation of the reduced SIR-Rt PINN examples.

This translates ``python/EpiPINN-SIRred-Rt-sciann.py`` from SciANN.  Case 4
uses infectious observations only.  Case 5 additionally uses hospitalization
observations and estimates sigma(t).
"""

import argparse
from dataclasses import replace
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
    train_loop,
    zero_expression_loss,
)


class JointReducedI(nn.Module):
    """Joint PINN for I(t) and Rt(t) using the reduced equation."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.i_net = MLP(cfg.hidden_state, positive=True)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)

    def forward(self, t: torch.Tensor):
        return self.i_net(t), self.rt_net(t)


class DataOnlyI(nn.Module):
    """First split stage for infectious observations."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.i_net = MLP(cfg.hidden_state, positive=True)

    def forward(self, t: torch.Tensor):
        return self.i_net(t)


class SplitReducedI(nn.Module):
    """Second split stage: fixed I(t), learned Rt(t)."""

    def __init__(self, fixed_i: nn.Module, cfg: TrainConfig) -> None:
        super().__init__()
        self.i_net = freeze(fixed_i)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)

    def forward(self, t: torch.Tensor):
        return self.i_net(t), self.rt_net(t)


class JointReducedH(nn.Module):
    """Joint PINN for I(t), Rt(t), sigma(t), and Delta_H(t)."""

    def __init__(self, cfg: TrainConfig, c_factor: float) -> None:
        super().__init__()
        self.i_net = MLP(cfg.hidden_state, positive=True)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)
        self.sigma_net = MLP(cfg.hidden_sigma, positive=True)
        self.c_factor = float(c_factor)

    def values(self, t: torch.Tensor):
        i_value = self.i_net(t)
        rt_value = self.rt_net(t)
        sigma_value = self.sigma_net(t)
        delta_h = sigma_value * i_value * self.c_factor
        return i_value, rt_value, sigma_value, delta_h


class DataOnlyH(nn.Module):
    """First split stage for hospitalization observations."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.h_net = MLP(cfg.hidden_beta, positive=True)

    def forward(self, t: torch.Tensor):
        return self.h_net(t)


class SplitReducedH(nn.Module):
    """Second split stage: fixed Delta_H(t), learned Rt(t) and inverse sigma relation."""

    def __init__(self, fixed_h: nn.Module, cfg: TrainConfig, c_factor: float) -> None:
        super().__init__()
        self.h_net = freeze(fixed_h)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)
        self.sigma_net = MLP(cfg.hidden_sigma, positive=True)
        self.c_factor = float(c_factor)

    def values(self, t: torch.Tensor):
        delta_h = self.h_net(t)
        rt_value = self.rt_net(t)
        sigma_inverse = self.sigma_net(t)
        i_value = delta_h * sigma_inverse / self.c_factor
        return delta_h, rt_value, sigma_inverse, i_value


def reduced_ode_loss(i_value, rt_value, t_tensor, tf, delta):
    d_i_dt = derivative(i_value, t_tensor)
    residual = d_i_dt - tf * delta * (rt_value - 1.0) * i_value
    return mse(residual, torch.zeros_like(residual))


def train_joint_i(cfg, device, t_train, t_data_sc, i_obs_sc, tf, delta):
    model = JointReducedI(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        i_value, rt_value = model(tb)
        loss_ode = reduced_ode_loss(i_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0), which is a zero-valued SciANN expression.
        loss_zero = zero_expression_loss(rt_value)
        i_pred, _ = model(t_data)
        loss_data = mse(i_pred, i_obs)
        return loss_ode + loss_zero + loss_data, {"ode": loss_ode, "zero": loss_zero, "data": loss_data}

    return model, train_loop(model, optimizer, cfg.epochs_joint, loss_fn, cfg.print_every, "joint-I")


def train_data_i(cfg, device, t_data_sc, i_obs_sc):
    model = DataOnlyI(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_data), cfg.batch_size_data, device)
        loss_data = mse(model(t_data[idx]), i_obs[idx])
        return loss_data, {"data": loss_data}

    return model, train_loop(model, optimizer, cfg.epochs_data, loss_fn, cfg.print_every, "split-data-I")


def train_split_i(cfg, device, fixed_i, t_train_ode, tf, delta):
    model = SplitReducedI(fixed_i, cfg).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    t_all = as_column_tensor(t_train_ode, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        i_value, rt_value = model(tb)
        loss_ode = reduced_ode_loss(i_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0) and sn.Data(Is * 0.0).
        loss_zero = zero_expression_loss(rt_value, i_value)
        return loss_ode + loss_zero, {"ode": loss_ode, "zero": loss_zero}

    return model, train_loop(model, optimizer, cfg.epochs_ode, loss_fn, cfg.print_every, "split-ode-I")


def train_joint_h(cfg, device, t_train, t_data_sc, i_obs_sc, h_obs_sc, tf, delta, c_factor):
    model = JointReducedH(cfg, c_factor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)
    h_obs = as_column_tensor(h_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        i_value, rt_value, _, _ = model.values(tb)
        loss_ode = reduced_ode_loss(i_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0).
        loss_zero = zero_expression_loss(rt_value)
        i_pred, _, _, h_pred = model.values(t_data)
        loss_data_i = mse(i_pred, i_obs)
        loss_data_h = mse(h_pred, h_obs)
        return loss_ode + loss_zero + loss_data_i + loss_data_h, {
            "ode": loss_ode,
            "zero": loss_zero,
            "I": loss_data_i,
            "H": loss_data_h,
        }

    return model, train_loop(model, optimizer, cfg.epochs_joint, loss_fn, cfg.print_every, "joint-H")


def train_data_h(cfg, device, t_data_sc, h_obs_sc):
    model = DataOnlyH(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_data = as_column_tensor(t_data_sc, device)
    h_obs = as_column_tensor(h_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_data), cfg.batch_size_data, device)
        loss_data = mse(model(t_data[idx]), h_obs[idx])
        return loss_data, {"data": loss_data}

    return model, train_loop(model, optimizer, cfg.epochs_data, loss_fn, cfg.print_every, "split-data-H")


def train_split_h(cfg, device, fixed_h, t_train, t_data_sc, i_obs_sc, tf, delta, c_factor):
    model = SplitReducedH(fixed_h, cfg, c_factor).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        _, rt_value, sigma_inverse, i_value = model.values(tb)
        loss_ode = reduced_ode_loss(i_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0) and sn.Data(sigma * 0.0).
        loss_zero = zero_expression_loss(rt_value, sigma_inverse)
        _, _, _, i_pred = model.values(t_data)
        loss_data = mse(i_pred, i_obs)
        return loss_ode + loss_zero + loss_data, {"ode": loss_ode, "zero": loss_zero, "I": loss_data}

    return model, train_loop(model, optimizer, cfg.epochs_ode, loss_fn, cfg.print_every, "split-ode-H")


def eval_joint_i(model, t_values, device, si_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        i_value, rt_value = model(t_tensor)
    return i_value.cpu().numpy().reshape(-1) * si_scale, rt_value.cpu().numpy().reshape(-1)


def eval_split_i(model, t_values, device, si_scale):
    return eval_joint_i(model, t_values, device, si_scale)


def eval_joint_h(model, t_values, device, si_scale, sh_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        i_value, rt_value, sigma_value, h_value = model.values(t_tensor)
    return (
        i_value.cpu().numpy().reshape(-1) * si_scale,
        h_value.cpu().numpy().reshape(-1) * sh_scale,
        rt_value.cpu().numpy().reshape(-1),
        sigma_value.cpu().numpy().reshape(-1),
    )


def eval_split_h(model, t_values, device, si_scale, sh_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        h_value, rt_value, sigma_inverse, i_value = model.values(t_tensor)
    sigma = 1.0 / np.maximum(sigma_inverse.cpu().numpy().reshape(-1), 1e-12)
    return (
        i_value.cpu().numpy().reshape(-1) * si_scale,
        h_value.cpu().numpy().reshape(-1) * sh_scale,
        rt_value.cpu().numpy().reshape(-1),
        sigma,
    )


def run_case4(cfg, args, df, device, t_data, t_test, t_data_sc, t_test_sc, t_train, t_train_ode):
    cfg = replace(cfg, epochs_data=args.epochs_data or 1000, epochs_ode=args.epochs_ode or 3000)
    if args.quick:
        cfg = quick_config(cfg)
    delta = 1 / 5
    si_scale = 1e5
    i_data = df["Infectious"].values
    i_obs = df["I data"].values
    rt_data = df["Rt"].values
    i_obs_sc = i_obs / si_scale
    i_data_sc = i_data / si_scale
    tf = float(len(df))

    start = time.time()
    joint_i, _ = train_joint_i(cfg, device, t_train, t_data_sc, i_obs_sc, tf, delta)
    print("Joint I/Rt training time: {:.1f}s".format(time.time() - start))
    i_pred, rt_pred = eval_joint_i(joint_i, t_data_sc, device, si_scale)
    print("Joint I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Joint Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    print("Joint Rt error last 100 days: {:.3e}".format(rel_l2(rt_data[20:], rt_pred[20:])))

    start = time.time()
    data_i, _ = train_data_i(cfg, device, t_data_sc, i_obs_sc)
    print("Split I data training time: {:.1f}s".format(time.time() - start))
    i_smooth = data_i(as_column_tensor(t_data_sc, device)).detach().cpu().numpy().reshape(-1)
    print("Isc error: {:.3e}".format(rel_l2(i_data_sc, i_smooth)))
    split_i, _ = train_split_i(cfg, device, data_i.i_net, t_train_ode, tf, delta)
    i_pred, rt_pred = eval_split_i(split_i, t_data_sc, device, si_scale)
    print("Split I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Split Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    print("Split Rt error last 100 days: {:.3e}".format(rel_l2(rt_data[20:], rt_pred[20:])))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(joint_i.state_dict(), out / "sirred_case4_joint_torch.pt")
        torch.save(data_i.state_dict(), out / "sirred_case4_split_data_torch.pt")
        torch.save(split_i.state_dict(), out / "sirred_case4_split_ode_torch.pt")

    if not args.no_plots:
        i_test, rt_test = eval_joint_i(joint_i, t_test_sc, device, si_scale)
        plt.plot(t_data, i_data, "r", linewidth=4)
        plt.plot(t_test, i_test, "--k", linewidth=4)
        plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()
        plt.plot(t_data, rt_data, linewidth=4)
        plt.plot(t_test, rt_test, "--k", linewidth=4)
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()


def run_case5(cfg, args, df, device, t_data, t_test, t_data_sc, t_test_sc, t_train):
    cfg = replace(cfg, epochs_data=args.epochs_data or 3000, epochs_ode=args.epochs_ode or 1000)
    if args.quick:
        cfg = quick_config(cfg)
    delta = 1 / 5
    si_scale = 1e5
    sh_scale = 1e3
    c_factor = si_scale * delta / sh_scale
    i_data = df["Infectious"].values
    i_obs = df["I data"].values
    rt_data = df["Rt"].values
    sigma_data = df["sigma"].values
    h_data = (delta * sigma_data * i_data).reshape(-1)
    h_obs = df["H data"].values
    i_obs_sc = i_obs / si_scale
    h_obs_sc = h_obs / sh_scale
    h_data_sc = h_data / sh_scale
    tf = float(len(df))

    start = time.time()
    joint_h, _ = train_joint_h(cfg, device, t_train, t_data_sc, i_obs_sc, h_obs_sc, tf, delta, c_factor)
    print("Joint I/Rt/sigma/H training time: {:.1f}s".format(time.time() - start))
    i_pred, h_pred, rt_pred, sigma_pred = eval_joint_h(joint_h, t_data_sc, device, si_scale, sh_scale)
    print("Joint I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Joint DeltaH error: {:.3e}".format(rel_l2(h_data, h_pred)))
    print("Joint Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    print("Joint Rt error last 100 days: {:.3e}".format(rel_l2(rt_data[20:], rt_pred[20:])))
    print("Joint sigma error: {:.3e}".format(rel_l2(sigma_data, sigma_pred)))
    print("Joint sigma error last 100 days: {:.3e}".format(rel_l2(sigma_data[20:], sigma_pred[20:])))

    start = time.time()
    data_h, _ = train_data_h(cfg, device, t_data_sc, h_obs_sc)
    print("Split H data training time: {:.1f}s".format(time.time() - start))
    h_smooth = data_h(as_column_tensor(t_data_sc, device)).detach().cpu().numpy().reshape(-1)
    print("Hsc error: {:.3e}".format(rel_l2(h_data_sc, h_smooth)))
    split_h, _ = train_split_h(cfg, device, data_h.h_net, t_train, t_data_sc, i_obs_sc, tf, delta, c_factor)
    i_pred, h_pred, rt_pred, sigma_pred = eval_split_h(split_h, t_data_sc, device, si_scale, sh_scale)
    print("Split I error: {:.3e}".format(rel_l2(i_data, i_pred)))
    print("Split DeltaH error: {:.3e}".format(rel_l2(h_data, h_pred)))
    print("Split Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    print("Split Rt error last 100 days: {:.3e}".format(rel_l2(rt_data[20:], rt_pred[20:])))
    print("Split sigma error: {:.3e}".format(rel_l2(sigma_data, sigma_pred)))
    print("Split sigma error last 100 days: {:.3e}".format(rel_l2(sigma_data[20:], sigma_pred[20:])))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(joint_h.state_dict(), out / "sirred_case5_joint_torch.pt")
        torch.save(data_h.state_dict(), out / "sirred_case5_split_data_torch.pt")
        torch.save(split_h.state_dict(), out / "sirred_case5_split_ode_torch.pt")

    if not args.no_plots:
        i_test, h_test, rt_test, sigma_test = eval_joint_h(joint_h, t_test_sc, device, si_scale, sh_scale)
        plt.plot(t_data, i_data, "r", linewidth=4)
        plt.plot(t_test, i_test, "--k", linewidth=4)
        plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
        plt.legend([r"$I$", r"$\hat{I}$", "samples"])
        plt.show()
        plt.plot(t_data, h_data, "b", linewidth=4)
        plt.plot(t_test, h_test, "--k", linewidth=4)
        plt.scatter(t_data, h_obs, marker="x", c="m", s=100)
        plt.legend([r"$\Delta_H$", r"$\hat{\Delta}_H$", "samples"])
        plt.show()
        plt.plot(t_data, rt_data, linewidth=4)
        plt.plot(t_test, rt_test, "--k", linewidth=4)
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()
        plt.plot(t_data, sigma_data, "orange", linewidth=4)
        plt.plot(t_test, sigma_test, "--k", linewidth=4)
        plt.legend([r"$\sigma$", r"$\hat{\sigma}$"])
        plt.show()


def run(args: argparse.Namespace) -> None:
    cfg = TrainConfig(seed=args.seed, device=args.device)
    cfg.epochs_joint = args.epochs_joint
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    df = pd.read_table(resolve_data_path(args.data_dir, "Case4.txt"))
    timespan = df["Date"].values.astype("datetime64[D]")
    t0 = 0.0
    tf = float(len(timespan))
    t_data = np.arange(t0, tf)
    t_test = np.arange(t0, tf, 0.1)
    t_data_sc = t_data / tf
    t_test_sc = t_test / tf
    t_train_ode = make_collocation(t0 / tf, cfg.n_collocation, include_zero=True)
    t_train = np.concatenate([t_data_sc.reshape(-1, 1), t_train_ode])

    if not args.no_plots:
        delta = 1 / 5
        i_data = df["Infectious"].values
        i_obs = df["I data"].values
        rt_data = df["Rt"].values
        sigma_data = df["sigma"].values
        h_data = delta * sigma_data * i_data
        h_obs = df["H data"].values
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))
        ax[0].plot(timespan, i_data, "r", label="Infectious")
        ax[0].plot(timespan, i_obs, "xm", label="samples")
        ax[0].legend(loc=6)
        ax[1].plot(timespan, h_data, "b", label="Hospitalizations")
        ax[1].plot(timespan, h_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[2].plot(timespan, rt_data, label=r"$\mathcal{R}_t$")
        ax[2].legend(loc=6)
        ax[3].plot(timespan, sigma_data, "orange", label=r"$\sigma$")
        ax[3].legend(loc=6)
        fig.tight_layout()
        plt.show()

    print("Device:", device)
    if args.case in (4, "4", "both"):
        print("Running Case 4")
        run_case4(cfg, args, df, device, t_data, t_test, t_data_sc, t_test_sc, t_train, t_train_ode)
    if args.case in (5, "5", "both"):
        print("Running Case 5")
        run_case5(cfg, args, df, device, t_data, t_test, t_data_sc, t_test_sc, t_train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="both", choices=("4", "5", "both"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-data", type=int, default=None)
    parser.add_argument("--epochs-ode", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
