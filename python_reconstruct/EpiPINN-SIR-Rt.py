#!/usr/bin/env python
# coding: utf-8

"""Thin entry point for the refactored cases 2 and 3 SciANN workflow.

运行入口：
默认调用 ``sir_rt.main(case=2)``，对应论文 Case 2：完整 SIR + 时变 beta(t)。
如果要跑 Case 3，可在交互环境中调用 ``sir_rt.main(case=3)``。
详细中文论文映射注释在 ``sir_rt.py`` 中。
"""

from sir_rt import main


main(case=2)
