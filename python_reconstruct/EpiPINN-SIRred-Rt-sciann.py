#!/usr/bin/env python
# coding: utf-8

"""Thin entry point for the refactored cases 4 and 5 SciANN workflow.

运行入口：
调用 ``sirred_rt_sciann.main()``，对应论文 Case 4-5：reduced SIR，
以及加入住院数据 Delta_H 后对 Rt(t)、sigma(t) 的估计。
详细中文论文映射注释在 ``sirred_rt_sciann.py`` 中。
"""

from sirred_rt_sciann import main


main()
