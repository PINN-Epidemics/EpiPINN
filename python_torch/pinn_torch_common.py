"""Shared PyTorch utilities for the EpiPINN examples.

The original examples use SciANN objects such as ``Functional``, ``Parameter``,
``PDE`` and ``Data``.  This module exposes the same ideas with explicit
``torch.nn.Module`` networks, trainable positive parameters, autograd
derivatives and MSE loss terms.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn


Tensor = torch.Tensor


@dataclass
class TrainConfig:
    """Runtime and optimizer settings shared by the translated scripts."""

    seed: int = 34
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_collocation: int = 6000
    lr: float = 1e-3
    batch_size: int = 100
    batch_size_data: int = 10
    epochs_joint: int = 5000
    epochs_data: int = 3000
    epochs_ode: int = 1000
    hidden_state: Tuple[int, ...] = (50, 50, 50, 50)
    hidden_beta: Tuple[int, ...] = (100, 100, 100, 100)
    hidden_sigma: Tuple[int, ...] = (5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    print_every: int = 500


def quick_config(cfg: TrainConfig) -> TrainConfig:
    """Shrink training for smoke tests."""

    cfg.n_collocation = min(cfg.n_collocation, 128)
    cfg.epochs_joint = min(cfg.epochs_joint, 5)
    cfg.epochs_data = min(cfg.epochs_data, 5)
    cfg.epochs_ode = min(cfg.epochs_ode, 5)
    cfg.batch_size = min(cfg.batch_size, 32)
    cfg.batch_size_data = min(cfg.batch_size_data, 16)
    cfg.print_every = 1
    return cfg


def set_seed(seed: int) -> None:
    """Set all random seeds used by these examples."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    """Return the EpiPINN directory from inside ``python_torch``."""

    return Path(__file__).resolve().parents[1]


def resolve_data_path(data_dir: Optional[str], file_name: str) -> Path:
    """Resolve a data file either from a user-supplied directory or EpiPINN root."""

    base = Path(data_dir).expanduser() if data_dir else project_root()
    return (base / file_name).resolve()


def as_column_tensor(values: np.ndarray, device: torch.device, requires_grad: bool = False) -> Tensor:
    """Convert a 1-D array-like value to a float32 column tensor."""

    tensor = torch.as_tensor(np.asarray(values).reshape(-1, 1), dtype=torch.float32, device=device)
    return tensor.requires_grad_(requires_grad)


def scalar_tensor(value: float, device: torch.device, requires_grad: bool = False) -> Tensor:
    """Create a scalar value as a one-row column tensor."""

    return torch.tensor([[value]], dtype=torch.float32, device=device, requires_grad=requires_grad)


class MLP(nn.Module):
    """Fully connected tanh network with optional square output activation."""

    def __init__(self, hidden_layers: Iterable[int], positive: bool = True) -> None:
        super().__init__()
        layers = []
        in_features = 1
        for width in hidden_layers:
            layers.append(nn.Linear(in_features, int(width)))
            layers.append(nn.Tanh())
            in_features = int(width)
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
        self.positive = positive

    def forward(self, t: Tensor) -> Tensor:
        value = self.net(t)
        return value.square() if self.positive else value


class PositiveParameter(nn.Module):
    """A non-negative scalar parameter, equivalent to SciANN ``Parameter(non_neg=True)``."""

    def __init__(self, initial_value: float = 0.5) -> None:
        super().__init__()
        raw_value = max(float(initial_value), 1e-6) ** 0.5
        self.raw = nn.Parameter(torch.tensor(raw_value, dtype=torch.float32))

    def forward(self, t: Tensor) -> Tensor:
        return self.raw.square().expand_as(t)

    def value(self) -> float:
        return float(self.raw.detach().square().cpu())


def freeze(module: nn.Module) -> nn.Module:
    """Disable parameter gradients while keeping input gradients available."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def derivative(y: Tensor, x: Tensor) -> Tensor:
    """Compute dy/dx for column tensors with PyTorch autograd."""

    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def mse(value: Tensor, target: Tensor) -> Tensor:
    """Mean squared error."""

    return torch.mean((value - target) ** 2)


def zero_expression_loss(*values: Tensor) -> Tensor:
    """Loss for SciANN expressions such as ``sn.Data(Rt * 0.0)``.

    These terms are mathematically zero but are kept in the PyTorch scripts so
    the loss layout mirrors the source code.
    """

    if not values:
        raise ValueError("At least one tensor is required.")
    total = torch.zeros((), dtype=values[0].dtype, device=values[0].device)
    for value in values:
        total = total + mse(value * 0.0, torch.zeros_like(value))
    return total


def sign_gated_initial_loss(t: Tensor, value: Tensor, target: float, t0_scaled: float) -> Tensor:
    """Match SciANN ``(u - u0) * (1 - sign(ts - t0/tf))`` initial terms."""

    gate = 1.0 - torch.sign(t - float(t0_scaled))
    residual = (value - float(target)) * gate
    return mse(residual, torch.zeros_like(residual))


def rel_l2(true: np.ndarray, pred: np.ndarray) -> float:
    """Relative L2 error used in the original notebooks."""

    true_flat = np.asarray(true).reshape(-1)
    pred_flat = np.asarray(pred).reshape(-1)
    denom = np.linalg.norm(true_flat, 2)
    return float(np.linalg.norm(true_flat - pred_flat, 2) / denom) if denom > 0 else float("nan")


def random_batch(n_items: int, batch_size: int, device: torch.device) -> Tensor:
    """Random mini-batch indices."""

    return torch.randint(0, n_items, (min(batch_size, n_items),), device=device)


def make_collocation(t0_scaled: float, n_points: int, include_zero: bool = True) -> np.ndarray:
    """Uniform collocation points on the scaled time interval."""

    count = max(int(n_points) - (1 if include_zero else 0), 1)
    points = np.random.uniform(t0_scaled, 1.0, count)
    if include_zero:
        points = np.insert(points, 0, t0_scaled)
    return points.reshape(-1, 1)


def make_log_collocation(t0_scaled: float, n_points: int) -> np.ndarray:
    """Log-spaced random collocation distribution used by the constant-beta case."""

    samples = np.random.uniform(np.log1p(t0_scaled), np.log1p(1.0), int(n_points))
    return (np.exp(samples) - 1.0).reshape(-1, 1)


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    loss_fn,
    print_every: int = 500,
    label: str = "train",
) -> Dict[str, list]:
    """Run a compact training loop and return scalar history."""

    history = {"loss": []}
    for epoch in range(1, int(n_epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = loss_fn()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history["loss"].append(loss_value)
        for key, value in metrics.items():
            history.setdefault(key, []).append(float(value.detach().cpu()))

        if print_every and (epoch == 1 or epoch == n_epochs or epoch % print_every == 0):
            metric_text = ", ".join(
                "{}={:.3e}".format(key, float(value.detach().cpu())) for key, value in metrics.items()
            )
            if metric_text:
                print("{} epoch {}/{}: loss={:.3e}, {}".format(label, epoch, n_epochs, loss_value, metric_text))
            else:
                print("{} epoch {}/{}: loss={:.3e}".format(label, epoch, n_epochs, loss_value))
    return history


def predict_numpy(model: nn.Module, values: np.ndarray, device: torch.device, fn):
    """Evaluate a model helper function without retaining gradients."""

    t = as_column_tensor(values, device)
    model.eval()
    with torch.no_grad():
        output = fn(t)
    if isinstance(output, tuple):
        return tuple(item.detach().cpu().numpy() for item in output)
    return output.detach().cpu().numpy()
