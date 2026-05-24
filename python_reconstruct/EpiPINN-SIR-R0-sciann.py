#!/usr/bin/env python
# coding: utf-8

"""Thin entry point for the refactored case 1 SciANN workflow.

运行入口：
调用 ``sir_r0.main()``，对应论文 Case 1：完整 SIR + 常数 beta_0 的
joint/split PINN 对比。详细论文映射注释在 ``sir_r0.py`` 中。
"""

from sir_r0 import main


main()
