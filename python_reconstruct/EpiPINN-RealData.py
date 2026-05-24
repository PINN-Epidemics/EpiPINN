#!/usr/bin/env python
# coding: utf-8

"""Thin entry point for the refactored cases 6 and 7 SciANN workflow.

运行入口：
默认调用 ``real_data.main(case=6)``，对应论文 Case 6：真实意大利数据 +
常数住院比例 sigma。若要跑 Case 7，可调用 ``real_data.main(case=7)``。
详细中文论文映射注释在 ``real_data.py`` 中。
"""

from real_data import main


main(case=6)
