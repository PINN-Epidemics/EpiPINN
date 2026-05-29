# EpiPINN Cases Summary

本文档总结仓库中各个 epidemic PINN case 的设置、SciANN 用法、共同点与差异，并提炼出构建类似模型时的实践指导。

## 1. 分案例总结

### Case 1: SIR + 常数传播率 beta/R0

对应文件：`python/EpiPINN-SIR-R0-sciann.py`，DeepXDE 复刻在 `DeepXDE/EpiPINN-SIR-R0-deepxde.py`。

数据来源：

- 代码内部用 `scipy.integrate.odeint` 从完整 SIR 方程生成合成真值 `S_data, I_data, R_data`。
- 观测数据 `I_obs` 是对 `I_data` 做 Poisson 采样得到的噪声感染数据。
- `S/R` 有真值用于评估和画图，但训练时只直接使用 `I_obs`。

模型设置：

- 总人口 `N=56e6`，恢复率 `delta=1/5`，真实 `r0=3`，所以真实 `beta=delta*r0`。
- 网络学习缩放后的 `Ss(t), Is(t)`，并用守恒关系构造 `Rs=N/C-Is-Ss`。
- `Beta = sn.Parameter(..., non_neg=True)`，表示一个非负常数参数，不是时间函数。
- 时间输入归一化为 `ts=t/tf`，人数用 `C=1e5` 缩放。

方程残差：

```text
dS/dt = - beta I S / N
dI/dt =   beta I S / N - delta I
dR/dt =                 delta I
```

在缩放变量中写成 SciANN residual：

```python
L_dSdt = sn.diff(Ss, ts) + C1*Beta*Is*Ss
L_dIdt = sn.diff(Is, ts) - C1*Beta*Is*Ss + C2*Is
L_dRdt = sn.diff(Rs, ts) - C2*Is
```

SciANN joint 训练：

- `sn.Variable('ts')` 定义输入。
- `sn.Functional('Ss', ts, 4*[50], output_activation='square')` 和 `Is` 定义非负状态网络。
- `sn.Parameter('Beta', inputs=ts, non_neg=True)` 定义常数待识别参数。
- `sn.PDE(...)` 用于 ODE residual 和初始条件 residual。
- `sn.Data(Is)` 用于感染观测监督。
- `adaptive_weights={'method':'NTK','freq':100}` 用于多 loss 项自适应加权。

loss 结构：

```python
loss_joint = [
    sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
    sn.PDE(L_S0),   sn.PDE(L_I0),   sn.PDE(L_R0),
    sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Is)
]
loss_train = ['zeros']*8 + [(ids_data, I_obs_sc)]
```

其中 `sn.Data(Ss*0.0)`、`sn.Data(Rs*0.0)` 是零输出占位项；配合 `'zeros'` 时数学上恒为 0，不约束模型。

Split 训练：

1. 先只用 `sn.Data(Isc)` 拟合感染曲线 `I(t)`。
2. 复制权重到 `Is = sn.Functional(..., trainable=False)` 冻结感染网络。
3. 再训练 `Ss` 和 `Beta`，通过 SIR residual 反演 `S/R/beta`。

适用场景：

- 已知完整 SIR 模型结构。
- 传播率可近似为常数。
- 想验证 PINN 是否能从少量感染观测恢复状态和常数参数。

### Case 2: SIR + 时变 Rt/beta，合成数据

对应文件：`python/EpiPINN-SIR-Rt.py`，`case=2`，读取 `Case2.txt`。

数据来源：

- `Case2.txt` 包含 `Susceptible, Infectious, Recovered, I data, Rt`。
- 从文件本身看是合成 benchmark 数据，因为它同时给出不可直接观测的完整 `S/I/R` 真值和 `Rt` 真值。
- 生成脚本不在当前仓库中。

模型设置：

- 仍使用完整 SIR。
- 不再学习常数 `Beta`，而是学习时间函数 `Beta(t)`。
- `beta_data = Rt_data * delta` 只用于画图和误差评估，训练中没有直接监督 `Beta`。

SciANN 差异：

```python
Beta = sn.Functional("Beta", ts, 4*[100], output_activation='square')
```

`sn.Functional` 表示一个神经网络函数，所以 `Beta` 可以随时间变化。`output_activation='square'` 保证非负。

loss 结构：

```python
loss_joint = [
    sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
    sn.PDE(L_S0), sn.PDE(L_I0), sn.PDE(L_R0),
    sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Beta*0.0),
    sn.Data(Is)
]
loss_train = ['zeros']*9 + [(ids_data, I_obs_sc)]
```

`sn.Data(Beta*0.0)` 也是零占位项，不会把 `Beta` 压向 0。

Split 训练：

- 第一阶段拟合 `I(t)`。
- 第二阶段冻结 `I(t)`，学习 `S(t)` 和 `Beta(t)`，并通过 SIR 方程约束。

适用场景：

- 传播率明显随时间变化。
- 有感染观测，但没有直接的 `S/R/beta` 观测。
- 仍相信完整 SIR 守恒关系 `S+I+R=N`。

### Case 3: SIR + 时变 Rt/beta，COVID 风格数据

对应文件：`python/EpiPINN-SIR-Rt.py`，`case=3`，读取 `Case3.txt`。

与 Case 2 的代码完全同一套，只是输入数据不同。

数据特征：

- `Case3.txt` 包含真实日期 `Date`，以及 `Susceptible, Infectious, Recovered, I data, Rt`。
- 由于文件中仍有完整 `S/I/R` 真值和 `Rt` 真值，它在代码使用方式上仍是合成或半合成 benchmark，不是纯真实观测实验。
- 真值列主要用于评估 PINN 识别质量。

模型、loss、SciANN 用法：

- 与 Case 2 相同。
- 仍是完整 SIR + `Beta(t)` 函数网络。
- 仍使用 NTK 自适应权重。

适用场景：

- 用更接近 COVID 时间轴和 Rt 曲线的数据测试完整 SIR-Rt 反演。
- 比 Case 2 更像实际疫情曲线，但训练逻辑仍依赖 benchmark 真值进行验证。

### Case 4: reduced SIR-Rt，只用 I 反演 Rt

对应文件：`python/EpiPINN-SIRred-Rt-sciann.py` 前半部分，读取 `Case4.txt`。

数据来源：

- `Case4.txt` 包含 `Infectious, I data, Rt, sigma, H data`。
- 文件中有 `Infectious/Rt/sigma` 真值，所以这是合成数据。
- 本 case 的训练主要使用感染观测 `I data`，`Rt` 真值用于评估。

模型设置：

- 不再显式建模 `S/R`。
- 使用 reduced SIR 关系：

```text
dI/dt = delta * (Rt - 1) * I
```

这个方程可由完整 SIR 中 `Rt = beta S / delta N` 的有效再生数形式得到。

SciANN 设置：

```python
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')
L_dIdt = sn.diff(Is, ts) - tf*delta*(Rt-1)*Is
loss_joint = [sn.PDE(L_dIdt), sn.Data(Rt*0.0), sn.Data(Is)]
loss_train = ['zeros']*2 + [(ids_data, I_obs_sc)]
```

重要特点：

- 没有 `S/R` 网络。
- 没有初始条件 loss。
- 没有 NTK 自适应权重。
- `sn.Data(Rt*0.0)` 仍是零占位项。

为什么可以不用初始条件：

- reduced 模型更偏真实数据反演场景，起始感染数不一定可信。
- `I(t)` 直接由数据项约束，初始值不再像完整 SIR 那样必须显式锁定。

Split 训练：

1. 先用 `I_obs` 拟合平滑 `I(t)`。
2. 冻结 `I(t)`。
3. 通过 `dI/dt = delta(Rt-1)I` 反推出 `Rt(t)`。

适用场景：

- 不想估计 `S/R`。
- 只关心从感染曲线反演有效再生数 `Rt`。
- 完整 SIR 的人口守恒、易感者数据或初值不可用。

### Case 5: reduced SIR-Rt + 住院数据 + sigma

对应文件：`python/EpiPINN-SIRred-Rt-sciann.py` 后半部分，仍读取 `Case4.txt`。

数据来源：

- 与 Case 4 同一个 `Case4.txt`。
- 训练使用感染观测 `I data` 和住院增量观测 `H data`。
- `Rt` 和 `sigma` 真值用于评估。

模型设置：

在 Case 4 的 reduced infection equation 基础上增加住院观测方程：

```text
dI/dt    = delta * (Rt - 1) * I
Delta_H = delta * sigma * I
```

代码缩放后写成：

```python
deltaHs = sn.rename(sigma*Is*C, "deltaHs")
```

其中 `C = SI*delta/SH`，所以 `deltaHs*SH = delta*sigma*I`。

Joint 训练：

```python
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')
sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')

loss_joint = [
    sn.PDE(L_dIdt),
    sn.Data(Rt*0.0),
    sn.Data(Is),
    sn.Data(deltaHs)
]
loss_train = ['zeros']*2 + [(ids_data, I_obs_sc), (ids_data, H_obs_sc)]
```

Split 训练：

1. 先拟合住院增量 `Hsc(t)`，不是先拟合 `I(t)`。
2. 冻结 `deltaHs(t)`。
3. 学习 `Rt(t)` 和一个辅助 `sigma` 网络，并由住院关系重构 `I(t)`。

代码里的 split 重构：

```python
c = 1/C
Is = sn.rename(c*deltaHs*sigma, "Is")
sigma_pred = 1/sigma_pred
```

这说明 split 阶段的 `sigma` 网络实际更像论文定义下 `sigma` 的倒数参数化，最后要取倒数才与真实 `sigma_data` 比较。

为什么 split 先拟合住院数据：

- 该实验想展示用住院数据辅助反演。
- 住院数据在现实中通常比感染数更稳定、漏报更少。
- 第二阶段仍通过 `sn.Data(Is)` 使用感染观测，所以不是完全不用 `I`。

适用场景：

- 感染数噪声较大，但有住院/重症/死亡等相对可靠的滞后或关联指标。
- 想同时识别 `Rt(t)` 和观测比例/住院率 `sigma(t)`。

### Case 6: RealData + 常数 sigma

对应文件：`python/EpiPINN-RealData.py`，`case=6`，读取 `RealData.txt`。

数据来源：

- `RealData.txt` 包含 `Date, I data, H data, Rt data`。
- 这里没有 `S/I/R` 真值，也没有 `sigma` 真值。
- `Rt data` 在代码中主要用于画图和误差评估，不作为训练监督项。

模型设置：

- 使用 reduced SIR-Rt。
- `sigma` 是非负常数参数：

```python
sigma = sn.Parameter(name="sigma", inputs=ts, non_neg=True)
```

观测建模：

代码定义：

```python
L_dIdt  = sn.diff(Is,ts) - tf*delta*(Rt-1)*Is
deltaHs = sigma*Is*C
deltaIs = delta*Rt*Is
```

训练时拟合的是：

- `deltaIs` 对应 `I data`
- `deltaHs` 对应 `H data`

注意这里的 `I data` 在真实数据脚本中被当成类似新增感染流量 `Delta_I` 的观测来拟合，而不是直接拟合状态变量 `Is`。

Joint loss：

```python
loss_joint = [
    sn.PDE(L_dIdt),
    sn.Data(Rt*0.0),
    sn.Data(deltaIs),
    sn.Data(deltaHs)
]
loss_train = ['zeros']*2 + [(ids_data, I_obs_sc), (ids_data, H_obs_sc)]
```

Split 训练：

1. 先拟合住院观测 `deltaHsc(t)`。
2. 冻结 `deltaHs(t)`。
3. 由 `Is = deltaHs/sigma/C` 反推出感染状态。
4. 同时通过 `deltaIs = delta*Rt*Is` 拟合感染观测，并用 ODE residual 约束 `Rt`。

适用场景：

- 真实疫情数据。
- 住院率或报告比例可以先假设为常数。
- 没有完整 compartment 真值。

### Case 7: RealData + 时变 sigma

对应文件：`python/EpiPINN-RealData.py`，`case=7`。

与 Case 6 的区别：

- `sigma` 从常数参数改为时间函数网络：

```python
sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')
```

- split 阶段额外加 `+1e-3`：

```python
sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square') + 1e-3
```

这是为了避免后续 `Is = deltaHs/sigma/C` 中除以过小的 sigma。

其余设置：

- reduced SIR-Rt 方程不变。
- 仍拟合 `deltaIs` 和 `deltaHs`。
- `Rt data` 仍不是训练监督，而是用于对比。

适用场景：

- 真实数据中住院率、检测率、报告率可能随时间变化。
- 常数 sigma 假设太强，需要给模型更多自由度。

风险：

- `Rt(t)` 和 `sigma(t)` 都是未知函数时，反问题更不适定。
- 必须依赖数据项、ODE 约束、网络容量和缩放来共同稳定训练。

## 2. 按场景汇总：不同 case 的异同

### 按数据类型

| 场景 | Case | 数据文件 | 数据性质 | 训练观测 | 真值用途 |
|---|---:|---|---|---|---|
| 内部 ODE 合成 | 1 | 无外部文件 | 完全合成 | `I_obs` | `S/I/R/beta` 评估 |
| 完整 SIR-Rt benchmark | 2 | `Case2.txt` | 合成 | `I data` | `S/I/R/Rt` 评估 |
| COVID 风格 benchmark | 3 | `Case3.txt` | 合成或半合成 | `I data` | `S/I/R/Rt` 评估 |
| Reduced benchmark | 4 | `Case4.txt` | 合成 | `I data` | `I/Rt` 评估 |
| Reduced + hospitalization benchmark | 5 | `Case4.txt` | 合成 | `I data`, `H data` | `I/Rt/sigma/H` 评估 |
| 真实数据，常数 sigma | 6 | `RealData.txt` | 真实观测 | `I data`, `H data` | `Rt data` 对比 |
| 真实数据，时变 sigma | 7 | `RealData.txt` | 真实观测 | `I data`, `H data` | `Rt data` 对比 |

### 按流行病模型

| 模型 | Case | 状态网络 | 参数/函数 | 核心方程 |
|---|---:|---|---|---|
| 完整 SIR，常数 beta | 1 | `Ss, Is`; `Rs` 守恒构造 | `Beta` 常数参数 | 三个 SIR ODE |
| 完整 SIR，时变 beta | 2, 3 | `Ss, Is`; `Rs` 守恒构造 | `Beta(t)` 函数网络 | 三个 SIR ODE |
| Reduced SIR-Rt | 4 | `Is` | `Rt(t)` | `dI/dt=delta(Rt-1)I` |
| Reduced + hospitalization | 5 | `Is` 或由 `deltaH/sigma` 重构 | `Rt(t), sigma(t)` | 感染 ODE + `Delta_H=delta*sigma*I` |
| RealData reduced | 6, 7 | `Is` 或由 `deltaH/sigma` 重构 | `Rt(t)`, `sigma` 常数或函数 | 感染 ODE + `Delta_I`, `Delta_H` 观测方程 |

### 按待识别量

| 待识别量 | SciANN 写法 | 对应 case |
|---|---|---|
| 非负状态变量 `S/I` | `sn.Functional(..., output_activation='square')` | 1-7 |
| 守恒构造的 `R` | `Rs = N/C - Is - Ss` | 1-3 |
| 常数 beta | `sn.Parameter(..., non_neg=True)` | 1 |
| 时变 beta | `sn.Functional("Beta", ..., output_activation='square')` | 2, 3 |
| 时变 Rt | `sn.Functional("Rt", ..., output_activation='square')` | 4-7 |
| 常数 sigma | `sn.Parameter("sigma", ..., non_neg=True)` | 6 |
| 时变 sigma | `sn.Functional("sigma", ..., output_activation='square')` | 5, 7 |

### 按训练策略

| 策略 | Case | 做法 | 优点 | 风险 |
|---|---:|---|---|---|
| Joint | 全部 | 所有状态、参数、方程、数据项一起训练 | 端到端，物理和数据同时作用 | 多 loss 难平衡，反问题可能不稳定 |
| Split: 先拟合 I | 1-4 | 先拟合 `I(t)`，再冻结并反演参数 | 感染曲线更平滑，二阶段更稳定 | 第一阶段误差会传递 |
| Split: 先拟合 H | 5-7 | 先拟合住院增量，再重构 `I` 和反演 `Rt/sigma` | 更适合真实数据和住院数据可靠场景 | `sigma` 与 `Rt` 可辨识性更弱 |

### 按初始条件

| Case | 是否显式使用初值 loss | 原因 |
|---:|---|---|
| 1-3 | 是，`L_S0/L_I0/L_R0` | 完整 SIR benchmark，初值明确 |
| 4-7 | 否 | reduced/真实数据设置中，真实起点和初始感染状态不一定可信；数据项直接约束观测曲线 |

### 按 NTK 自适应权重

| Case | 是否使用 `adaptive_weights=adaptive_NTK` |
|---:|---|
| 1-3 | 使用 |
| 4-7 | 未使用 |

合理解释：

- 完整 SIR 的 loss 项更多，包括 3 个 ODE residual、初值 residual、数据项，尺度更容易不平衡，所以使用 NTK adaptive weights。
- reduced 模型 loss 项少，变量也做了更直接缩放，作者没有继续使用 NTK。
- 这不是数学上“不能用 NTK”，而是代码中的实现选择。若训练不稳定，可以尝试重新加入 adaptive weights。

### 关于 `sn.PDE()` 和 `sn.Data()`

`sn.PDE(expr)`：

- 用于物理 residual、ODE residual、初始条件 residual。
- 通常训练目标是 `'zeros'`。
- 含义是让 `expr(t)` 接近 0。

`sn.Data(expr)`：

- 用于监督数据拟合。
- 训练目标通常是数组，或 `(ids, values)`。
- 含义是让 `expr(t_i)` 接近观测值。

是否能写反：

- 如果目标都是 `'zeros'`，有些情况下数值上可能还能跑。
- 但语义会变差，SciANN 内部日志、权重、数据索引行为可能不符合预期。
- 推荐规则：方程残差和初值残差用 `sn.PDE`；有观测值的量用 `sn.Data`。

### 关于零占位项

这些写法在仓库中反复出现：

```python
sn.Data(Rt*0.0)
sn.Data(Beta*0.0)
sn.Data(Ss*0.0)
sn.Data(Is*0.0)
```

如果对应 target 是 `'zeros'`，则 loss 为：

```text
MSE(0, 0 * variable) = 0
```

所以它们不约束变量，也不是有效正则化。更像是为了保持 loss 列表结构、日志结构或与论文/Notebook 模板对齐的占位项。

如果真想正则化 `Rt` 或 `Beta`，应该写成例如：

```python
sn.PDE(sn.diff(Rt, ts))
sn.PDE(sn.diff(Beta, ts))
sn.Data(Rt)
```

并提供合理 target 或权重，而不是乘以 `0.0`。

## 3. 构建 epidemic PINN 的指导

### 第一步：判断你有什么观测

只有感染观测 `I`：

- 合成 benchmark 且完整 SIR 可信：用 Case 1-3 的完整 SIR。
- 真实数据或不想估计 `S/R`：用 Case 4 的 reduced SIR-Rt。

有感染和住院观测 `I, H`：

- 合成数据、想验证 `sigma` 识别：用 Case 5。
- 真实数据：用 Case 6/7。

有完整 `S/I/R` 真值：

- 可以用于评估，但真实疫情中通常没有。
- 训练时不一定要把全部真值作为 data loss，否则问题会变成强监督拟合而不是典型 inverse PINN。

### 第二步：选择完整模型还是 reduced 模型

选择完整 SIR，当：

- 总人口 `N` 明确。
- 初始条件可信。
- `S+I+R=N` 守恒假设合理。
- 你关心 `S/R` 状态恢复。

选择 reduced SIR-Rt，当：

- 只关心 `Rt(t)`。
- `S/R` 不可观测或不可辨识。
- 真实数据初值不可信。
- 需要更少变量、更少 loss 项。

### 第三步：决定参数是常数还是时间函数

常数参数：

```python
beta = sn.Parameter(name="Beta", inputs=ts, non_neg=True)
sigma = sn.Parameter(name="sigma", inputs=ts, non_neg=True)
```

适合：

- 机制基本稳定。
- 数据量少。
- 想减少反问题自由度。

时间函数：

```python
Rt = sn.Functional("Rt", ts, 4*[100], output_activation="square")
sigma = sn.Functional("sigma", ts, 10*[5], output_activation="square")
```

适合：

- 干预、检测政策、住院率明显变化。
- 数据长度足够。
- 需要恢复时间变化趋势。

注意：

- 未知函数越多，可辨识性越弱。
- `Rt(t)` 和 `sigma(t)` 同时未知时，建议加入额外观测、平滑正则或更强先验。

### 第四步：写 SciANN 变量和网络

基本模板：

```python
ts = sn.Variable("ts")

Is = sn.Functional("Is", ts, 4*[50], output_activation="square")
Rt = sn.Functional("Rt", ts, 4*[100], output_activation="square")
```

完整 SIR：

```python
Ss = sn.Functional("Ss", ts, 4*[50], output_activation="square")
Rs = N/C - Is - Ss
```

住院模型：

```python
sigma = sn.Functional("sigma", ts, 10*[5], output_activation="square")
deltaHs = sn.rename(sigma*Is*C, "deltaHs")
```

真实数据 split 中如果要除以 `sigma`：

```python
sigma = sn.Functional("sigma", ts, 10*[5], output_activation="square") + 1e-3
Is = sn.rename(deltaHs/sigma/C, "Is")
```

### 第五步：写 residual

完整 SIR：

```python
L_dSdt = sn.rename(sn.diff(Ss,ts) + C1*Beta*Is*Ss, "L_dSdt")
L_dIdt = sn.rename(sn.diff(Is,ts) - C1*Beta*Is*Ss + C2*Is, "L_dIdt")
L_dRdt = sn.rename(sn.diff(Rs,ts) - C2*Is, "L_dRdt")
```

Reduced SIR-Rt：

```python
L_dIdt = sn.rename(sn.diff(Is,ts) - tf*delta*(Rt-1)*Is, "L_dIdt")
```

观测方程：

```python
deltaIs = sn.rename(delta*Rt*Is, "deltaIs")
deltaHs = sn.rename(sigma*Is*C, "deltaHs")
```

### 第六步：组织 loss

方程项：

```python
sn.PDE(L_dIdt)
```

观测项：

```python
sn.Data(Is)
sn.Data(deltaIs)
sn.Data(deltaHs)
```

训练 target：

```python
loss_train = ['zeros']*num_residuals + [(ids_data, observed_values)]
```

多个观测：

```python
loss_train = ['zeros']*2 + [
    (ids_data, I_obs_sc),
    (ids_data, H_obs_sc)
]
```

原则：

- residual 的 target 用 `'zeros'`。
- 观测数据只在真实观测点上监督，用 `(ids_data, values)`。
- 不要把不可观测变量硬塞进 `sn.Data`，除非确实有观测或真值监督。

### 第七步：选择 joint 还是 split

优先 joint，当：

- 数据干净。
- loss 项数量不多。
- 你希望端到端联合识别。

优先 split，当：

- 观测噪声大。
- 反演参数不稳定。
- 需要先得到平滑的 `I(t)` 或 `H(t)`。
- 想把复杂反问题拆成“数据平滑 + 物理反演”。

Split 模板：

```python
# stage 1: data fit
Isc = sn.Functional("Isc", ts, 4*[50], output_activation="square")
m_data = sn.SciModel(ts, sn.Data(Isc), "mse", "adam")
m_data.train(t_data_sc, I_obs_sc, ...)

# freeze
weights = Isc.get_weights()
Is = sn.Functional("Is", ts, 4*[50], output_activation="square", trainable=False)
Is.set_weights(weights)

# stage 2: physics inverse problem
Rt = sn.Functional("Rt", ts, 4*[100], output_activation="square")
L_dIdt = sn.diff(Is,ts) - tf*delta*(Rt-1)*Is
m_ode = sn.SciModel(ts, [sn.PDE(L_dIdt)], "mse", "adam")
```

### 第八步：缩放和采样

缩放：

- 时间统一缩放到 `[0,1]`：`t_data_sc = t_data/tf`。
- 感染人数用 `SI` 或 `C` 缩放。
- 住院人数用 `SH` 缩放。
- 观测方程中的 `C = SI*delta/SH` 用来保证 `deltaHs` 与 `H_obs_sc` 同尺度。

采样：

```python
t_train_ode = np.random.uniform(t0/tf, 1., Nc-1)
t_train_ode = np.insert(t_train_ode, 0, 0.0)
t_train = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
ids_data = np.arange(t_data_sc.size, dtype=np.intp)
```

含义：

- `t_data_sc` 是有监督观测点。
- `t_train_ode` 是 collocation points，用于物理 residual。
- `ids_data` 告诉 SciANN 哪些输入点有观测值。

### 第九步：什么时候加初始条件

加初始条件：

- 合成数据。
- 初始状态已知且可信。
- 完整 SIR 模型。

不加或谨慎加：

- 真实疫情数据。
- 起始日期不是疫情真正起点。
- 初始感染数高度不确定。
- reduced 模型中 `I(t)` 已由数据项约束。

SciANN 初值 residual 写法：

```python
L_I0 = sn.rename((Is - I0/C) * (1 - sn.sign(ts - t0/tf)), "L_I0")
loss.append(sn.PDE(L_I0))
```

### 第十步：什么时候使用 NTK adaptive weights

建议使用：

- loss 项很多。
- ODE residual、初值 residual、数据 loss 数量级差异明显。
- 完整 SIR 的 joint 训练。

代码写法：

```python
adaptive_NTK = {"method": "NTK", "freq": 100}
m.train(..., adaptive_weights=adaptive_NTK)
```

可以先不使用：

- reduced 模型。
- loss 项少。
- 缩放后训练已经稳定。

### 最后建议

构建 epidemic PINN 时，优先按下面顺序做设计：

1. 明确观测量：感染、住院、死亡、Rt 参考值分别是什么。
2. 明确未知量：状态变量、常数参数、时间函数参数。
3. 选择完整 SIR 或 reduced SIR-Rt。
4. 对所有输入输出做缩放。
5. 用 `sn.PDE` 写物理 residual，用 `sn.Data` 写观测监督。
6. 用 `(ids_data, values)` 区分观测点和 collocation points。
7. 先做 joint baseline；不稳定时改成 split。
8. 合成数据可加初值和真值误差评估；真实数据不要过度依赖不可观测真值。
9. 对同时未知的多个函数参数加入先验、平滑或额外观测，否则反问题容易不唯一。
10. 不要把 `sn.Data(x*0.0)` 当成正则项；它只是零损失占位。

## 4. 学完本代码后可以掌握的技巧

学完本代码后，可以掌握用 PINN 处理仓室传染病模型的一套完整流程：把 SIR 或 reduced SIR-Rt 等仓室模型写成可训练的物理残差，用神经网络同时拟合观测数据并满足微分方程约束，从而实现状态变量恢复和参数反演。

具体来说，可以掌握以下能力：

1. 使用 SciANN 构建 epidemic PINN，将时间变量、状态变量、常数参数和时变参数分别表示为 `sn.Variable`、`sn.Functional` 和 `sn.Parameter`。
2. 将 SIR 方程、reduced SIR-Rt 方程、初始条件和观测方程转化为 `sn.PDE` 与 `sn.Data` 组成的损失函数。
3. 利用感染数据拟合仓室模型中的隐藏状态，例如 `S(t)`、`I(t)`、`R(t)`，并校验未知参数，如常数传播率 `beta`、基本再生数 `R0` 或有效再生数 `Rt`。
4. 支持时变参数识别，用神经网络函数表示 `Beta(t)`、`Rt(t)`、`sigma(t)`，从而处理干预政策、传播环境、检测率或住院率随时间变化的情况。
5. 支持通过住院数据辅助拟合，在感染观测之外引入 `Delta_H = delta*sigma*I` 等观测方程，提高真实数据场景下的参数识别能力。
6. 掌握 joint 和 split 两种训练策略：joint 用于端到端联合训练，split 用于先平滑观测曲线、再通过物理方程反演参数。
7. 掌握真实数据与合成数据的不同处理方式：合成数据可用真值评估误差，真实数据则更关注可观测量拟合、参数趋势解释和模型假设合理性。
8. 理解缩放、collocation points、观测点索引、初值约束、NTK 自适应权重等 PINN 训练细节对稳定性和可辨识性的影响。

一句话概括：这套代码展示了如何把仓室流行病模型从“已知方程求解”扩展为“带噪声观测下的状态估计和参数反演”，并进一步支持时变传播参数和住院数据驱动的真实疫情建模。
