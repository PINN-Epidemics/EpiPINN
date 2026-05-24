#!/usr/bin/env python
# coding: utf-8

"""Compatibility entry point.

The original file imported ``epi_pinn_sirred_rt_torch``, which is not present
in this repository.  This entry point runs the refactored SciANN reduced
SIR-Rt workflow instead.

中文说明：
这是兼容入口，不含模型实现。真正的 reduced SIR、Case 4/5、住院数据和
论文公式映射注释都在 ``sirred_rt_sciann.py`` 中。
"""

from sirred_rt_sciann import main


main()
