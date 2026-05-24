"""Refactored SciANN implementations for the EpiPINN examples.

中文导读：
``python_reconstruct`` 将论文的 notebook/exported scripts 重构成模块化代码。
主要文件对应关系：

- ``sir_r0.py``: Case 1，完整 SIR，常数 beta_0。
- ``sir_rt.py``: Case 2-3，完整 SIR，时间变化 beta(t)。
- ``sirred_rt_sciann.py``: Case 4-5，reduced SIR，Rt(t) 与住院数据。
- ``real_data.py``: Case 6-7，真实意大利 COVID-19 数据，Rt(t)/sigma(t) 估计。
"""
