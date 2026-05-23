#!/usr/bin/env python
"""Compatibility entry point for the PyTorch Case 4/5 script."""

from epi_pinn_sirred_rt_torch import parse_args, run


if __name__ == "__main__":
    run(parse_args())

