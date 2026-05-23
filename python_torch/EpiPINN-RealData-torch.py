#!/usr/bin/env python
"""Compatibility entry point for the PyTorch Case 6/7 script."""

from epi_pinn_realdata_torch import parse_args, run


if __name__ == "__main__":
    run(parse_args())

