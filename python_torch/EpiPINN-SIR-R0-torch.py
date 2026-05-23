#!/usr/bin/env python
"""Compatibility entry point for the PyTorch Case 1 script."""

from epi_pinn_sir_r0_torch import parse_args, run


if __name__ == "__main__":
    run(parse_args())

