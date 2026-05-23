#!/usr/bin/env python
# coding: utf-8

"""Compatibility entry point.

The original file imported ``epi_pinn_sirred_rt_torch``, which is not present
in this repository.  This entry point runs the refactored SciANN reduced
SIR-Rt workflow instead.
"""

from sirred_rt_sciann import main


main()
