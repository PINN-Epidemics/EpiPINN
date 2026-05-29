"""Minimal CUDA smoke test for NVIDIA PhysicsNeMo.

Run with:
    conda run -n physicsnemo python test.py

This test requires CUDA by default. For a CPU-only import/logic check, run:
    conda run -n physicsnemo python test.py --allow-cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


DEFAULT_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "physicsnemo-warp-cache"

# PhysicsNeMo imports Warp in some model modules. Make its kernel cache writable
# even when the home cache directory is not writable.
os.environ.setdefault("WARP_CACHE_PATH", str(DEFAULT_CACHE_DIR))
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "physicsnemo-cache"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysicsNeMo CUDA smoke test")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run on CPU if CUDA is unavailable. By default CUDA is required.",
    )
    parser.add_argument("--steps", type=int, default=5, help="Training steps to run.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nFAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()

    import torch

    print("=" * 72)
    print("PhysicsNeMo CUDA smoke test")
    print("=" * 72)
    print(f"Python executable : {sys.executable}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"Warp cache        : {os.environ['WARP_CACHE_PATH']}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"CUDA devices      : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"CUDA device       : {index}")
        print(f"GPU name          : {torch.cuda.get_device_name(index)}")
        print(f"GPU memory        : {props.total_memory / 1024**3:.2f} GiB")
    elif args.allow_cpu:
        device = torch.device("cpu")
        print("CUDA is unavailable; running CPU fallback because --allow-cpu was set.")
    else:
        fail(
            "CUDA is not available to PyTorch. Activate the correct environment, "
            "check the NVIDIA driver/WSL GPU setup, then rerun this script."
        )

    import physicsnemo
    from physicsnemo.models.mlp.fully_connected import FullyConnected

    print(f"PhysicsNeMo       : {getattr(physicsnemo, '__version__', 'unknown')}")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = True

    model = FullyConnected(
        in_features=2,
        layer_size=32,
        out_features=1,
        num_layers=3,
        activation_fn="silu",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.rand(256, 2, device=device)
    target = torch.sin(x[:, :1]) + torch.cos(x[:, 1:])

    print(f"Model device      : {next(model.parameters()).device}")
    print(f"Input device      : {x.device}")
    print(f"Input shape       : {tuple(x.shape)}")

    model.train()
    loss = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        print(f"step {step:02d} loss    : {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        output = model(x[:4])

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Output shape      : {tuple(output.shape)}")
    print(f"Output device     : {output.device}")
    print("=" * 72)
    if device.type == "cuda":
        print("SUCCESS: PhysicsNeMo model ran with CUDA.")
    else:
        print("SUCCESS: PhysicsNeMo model ran on CPU.")


if __name__ == "__main__":
    main()
