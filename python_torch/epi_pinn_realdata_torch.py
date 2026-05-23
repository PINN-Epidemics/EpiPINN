#!/usr/bin/env python
"""PyTorch implementation of the real-data EpiPINN examples.

This translates ``python/EpiPINN-RealData.py`` from SciANN.  Case 6 uses a
constant non-negative sigma parameter.  Case 7 uses a time-dependent sigma(t)
network.
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
    PositiveParameter,
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


class SigmaModule(nn.Module):
    """Case-dependent sigma definition matching the source script."""

    def __init__(self, cfg: TrainConfig, case: int) -> None:
        super().__init__()
        self.case = int(case)
        if self.case == 6:
            self.sigma = PositiveParameter(initial_value=0.1)
        elif self.case == 7:
            self.sigma = MLP(cfg.hidden_sigma, positive=True)
        else:
            raise ValueError("case must be 6 or 7")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma(t)


class JointRealData(nn.Module):
    """Joint real-data model for Is, Rt and sigma."""

    def __init__(self, cfg: TrainConfig, case: int, c_factor: float, delta: float) -> None:
        super().__init__()
        self.is_net = MLP(cfg.hidden_state, positive=True)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)
        self.sigma_net = SigmaModule(cfg, case)
        self.c_factor = float(c_factor)
        self.delta = float(delta)

    def values(self, t: torch.Tensor):
        is_value = self.is_net(t)
        rt_value = self.rt_net(t)
        sigma_value = self.sigma_net(t)
        delta_h = sigma_value * is_value * self.c_factor
        delta_i = self.delta * rt_value * is_value
        return is_value, rt_value, sigma_value, delta_i, delta_h


class DataOnlyDeltaH(nn.Module):
    """First split stage: fit Delta_H(t) from hospitalization observations."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.delta_h_net = MLP(cfg.hidden_state, positive=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.delta_h_net(t)


class SplitRealData(nn.Module):
    """Second split stage with fixed Delta_H(t), learned Rt and sigma."""

    def __init__(self, fixed_delta_h: nn.Module, cfg: TrainConfig, case: int, c_factor: float, delta: float) -> None:
        super().__init__()
        self.delta_h_net = freeze(fixed_delta_h)
        self.rt_net = MLP(cfg.hidden_beta, positive=True)
        self.case = int(case)
        if self.case == 6:
            self.sigma_net = PositiveParameter(initial_value=0.1)
        elif self.case == 7:
            # Source code adds 1e-3 to avoid division by zero.
            self.sigma_net = MLP(cfg.hidden_sigma, positive=True)
        else:
            raise ValueError("case must be 6 or 7")
        self.c_factor = float(c_factor)
        self.delta = float(delta)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        value = self.sigma_net(t)
        if self.case == 7:
            value = value + 1e-3
        return value

    def values(self, t: torch.Tensor):
        delta_h = self.delta_h_net(t)
        rt_value = self.rt_net(t)
        sigma_value = self.sigma(t)
        is_value = delta_h / sigma_value / self.c_factor
        delta_i = self.delta * rt_value * is_value
        return delta_h, rt_value, sigma_value, is_value, delta_i


def reduced_ode_loss(is_value, rt_value, t_tensor, tf, delta):
    d_i_dt = derivative(is_value, t_tensor)
    residual = d_i_dt - tf * delta * (rt_value - 1.0) * is_value
    return mse(residual, torch.zeros_like(residual))


def train_joint(cfg, device, model, t_train, t_data_sc, i_obs_sc, h_obs_sc, tf, delta):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)
    h_obs = as_column_tensor(h_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        is_value, rt_value, _, _, _ = model.values(tb)
        loss_ode = reduced_ode_loss(is_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0).
        loss_zero = zero_expression_loss(rt_value)
        _, _, _, delta_i, delta_h = model.values(t_data)
        loss_i = mse(delta_i, i_obs)
        loss_h = mse(delta_h, h_obs)
        return loss_ode + loss_zero + loss_i + loss_h, {
            "ode": loss_ode,
            "zero": loss_zero,
            "I": loss_i,
            "H": loss_h,
        }

    return train_loop(model, optimizer, cfg.epochs_joint, loss_fn, cfg.print_every, "joint")


def train_data_delta_h(cfg, device, t_data_sc, h_obs_sc):
    model = DataOnlyDeltaH(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t_data = as_column_tensor(t_data_sc, device)
    h_obs = as_column_tensor(h_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_data), cfg.batch_size_data, device)
        loss_h = mse(model(t_data[idx]), h_obs[idx])
        return loss_h, {"H": loss_h}

    history = train_loop(model, optimizer, cfg.epochs_data, loss_fn, cfg.print_every, "split-data")
    return model, history


def train_split(cfg, device, model, t_train, t_data_sc, i_obs_sc, tf, delta):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    t_all = as_column_tensor(t_train, device)
    t_data = as_column_tensor(t_data_sc, device)
    i_obs = as_column_tensor(i_obs_sc, device)

    def loss_fn():
        idx = random_batch(len(t_all), cfg.batch_size, device)
        tb = t_all[idx].clone().detach().requires_grad_(True)
        _, rt_value, _, is_value, _ = model.values(tb)
        loss_ode = reduced_ode_loss(is_value, rt_value, tb, tf, delta)
        # Mirrors sn.Data(Rt * 0.0).
        loss_zero = zero_expression_loss(rt_value)
        _, _, _, _, delta_i = model.values(t_data)
        loss_i = mse(delta_i, i_obs)
        return loss_ode + loss_zero + loss_i, {"ode": loss_ode, "zero": loss_zero, "I": loss_i}

    return train_loop(model, optimizer, cfg.epochs_ode, loss_fn, cfg.print_every, "split-ode")


def eval_joint(model, t_values, device, si_scale, sh_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        _, rt_value, sigma_value, delta_i, delta_h = model.values(t_tensor)
    return (
        delta_i.cpu().numpy().reshape(-1) * si_scale,
        delta_h.cpu().numpy().reshape(-1) * sh_scale,
        rt_value.cpu().numpy().reshape(-1),
        sigma_value.cpu().numpy().reshape(-1),
    )


def eval_split(model, t_values, device, si_scale, sh_scale):
    t_tensor = as_column_tensor(t_values, device)
    with torch.no_grad():
        delta_h, rt_value, sigma_value, _, delta_i = model.values(t_tensor)
    return (
        delta_i.cpu().numpy().reshape(-1) * si_scale,
        delta_h.cpu().numpy().reshape(-1) * sh_scale,
        rt_value.cpu().numpy().reshape(-1),
        sigma_value.cpu().numpy().reshape(-1),
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

    df = pd.read_table(resolve_data_path(args.data_dir, "RealData.txt"))
    timespan = df["Date"].values.astype("datetime64[D]")
    t0 = 0.0
    tf = float(len(timespan))
    t_data = np.arange(t0, tf)
    t_test = np.arange(t0, tf, 0.1)
    t_data_sc = t_data / tf
    t_test_sc = t_test / tf

    h_obs = df["H data"].values
    i_obs = df["I data"].values
    rt_data = df["Rt data"].values
    delta = 1 / 5
    si_scale = float(i_obs.max())
    sh_scale = float(h_obs.max())
    c_factor = si_scale * delta / sh_scale
    i_obs_sc = i_obs / si_scale
    h_obs_sc = h_obs / sh_scale

    if not args.no_plots:
        fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        ax[0].plot(timespan, i_obs, "xm", label="samples")
        ax[0].legend(loc=6)
        ax[0].set_title("Infectious Data")
        ax[1].plot(timespan, h_obs, "xm", label="samples")
        ax[1].legend(loc=6)
        ax[1].set_title("Hospitalizations Data")
        ax[2].plot(timespan, rt_data, label=r"$\mathcal{R}_t$")
        ax[2].legend(loc=6)
        fig.tight_layout()
        plt.show()

    t_train_ode = make_collocation(t0 / tf, cfg.n_collocation, include_zero=True)
    t_train = np.concatenate([t_data_sc.reshape(-1, 1), t_train_ode])

    print("Device:", device)
    print("Case:", args.case)
    joint_model = JointRealData(cfg, args.case, c_factor, delta).to(device)
    start = time.time()
    train_joint(cfg, device, joint_model, t_train, t_data_sc, i_obs_sc, h_obs_sc, tf, delta)
    print("Joint training time: {:.1f}s".format(time.time() - start))
    delta_i_pred, delta_h_pred, rt_pred, sigma_pred = eval_joint(joint_model, t_data_sc, device, si_scale, sh_scale)
    print("Joint Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    if args.case == 6:
        print("Joint estimated sigma: {:.4f}".format(float(sigma_pred[0])))

    start = time.time()
    data_h, _ = train_data_delta_h(cfg, device, t_data_sc, h_obs_sc)
    print("Split data training time: {:.1f}s".format(time.time() - start))
    split_model = SplitRealData(data_h.delta_h_net, cfg, args.case, c_factor, delta).to(device)
    train_split(cfg, device, split_model, t_train, t_data_sc, i_obs_sc, tf, delta)
    delta_i_pred, delta_h_pred, rt_pred, sigma_pred = eval_split(split_model, t_data_sc, device, si_scale, sh_scale)
    print("Split Rt error: {:.3e}".format(rel_l2(rt_data, rt_pred)))
    if args.case == 6:
        print("Split estimated sigma: {:.4f}".format(float(sigma_pred[0])))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(joint_model.state_dict(), out / "realdata_joint_torch.pt")
        torch.save(data_h.state_dict(), out / "realdata_split_data_torch.pt")
        torch.save(split_model.state_dict(), out / "realdata_split_ode_torch.pt")

    if not args.no_plots:
        delta_i_test, delta_h_test, rt_test, sigma_test = eval_joint(joint_model, t_test_sc, device, si_scale, sh_scale)
        plt.plot(t_test, delta_i_test, "--k", linewidth=4)
        plt.scatter(t_data, i_obs, marker="x", c="m", s=100)
        plt.legend([r"$\hat{\Delta}_I$", "samples"])
        plt.show()
        plt.plot(t_test, delta_h_test, "--k", linewidth=4)
        plt.scatter(t_data, h_obs, marker="x", c="m", s=100)
        plt.legend([r"$\hat{\Delta}_H$", "samples"])
        plt.show()
        plt.plot(t_data, rt_data, linewidth=4)
        plt.plot(t_test, rt_test, "--k", linewidth=4)
        plt.legend([r"$\mathcal{R}_t$", r"$\hat{\mathcal{R}}_t$"])
        plt.show()
        if args.case == 7:
            plt.plot(t_test, sigma_test, "--k", linewidth=4)
            plt.legend([r"$\hat{\sigma}$"])
            plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(6, 7), default=6)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-data", type=int, default=1000)
    parser.add_argument("--epochs-ode", type=int, default=3000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

