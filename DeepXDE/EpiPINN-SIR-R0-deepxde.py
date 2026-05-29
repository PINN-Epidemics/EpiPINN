#!/usr/bin/env python
# coding: utf-8

# # A Physics-Informed Neural Network approach for compartmental epidemiological models
#
#
# Notebooks for PINN solution of the SIR compartmental model presented [in the paper](https://arxiv.org/abs/2311.09944):
#
# ```
# @misc{millevoi2023physicsinformed,
#       title={A Physics-Informed Neural Network approach for compartmental epidemiological models},
#       author={Caterina Millevoi and Damiano Pasetto and Massimiliano Ferronato},
#       year={2023},
#       eprint={2311.09944},
#       archivePrefix={arXiv},
#       primaryClass={math.NA}
# }
# ```
#
# Note: The uploaded code is related to Case 1. For further information please contact the corresponding author.

# In[1]:


import os
import pickle
import time
from pathlib import Path

# DeepXDE 的 backend 必须在 import deepxde 之前设置。
# 这里使用 PyTorch backend，和本项目中的 DeepXDE 环境检查脚本保持一致。
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint

SEED = 1234


def set_random_seed(seed=SEED):
    """固定 NumPy、PyTorch 和 DeepXDE 的随机种子，减少两套实现之间的随机差异。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)


def print_device_info():
    """打印 DeepXDE/PyTorch 实际使用的计算设备。"""
    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    print("DeepXDE backend:", dde.backend.backend_name)
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Default torch device:", torch.get_default_device())
    else:
        print("CUDA is not available. The model will run on CPU.")


def print_model_device(name, model):
    """打印 DeepXDE 模型网络参数所在设备。"""
    if dde.backend.backend_name == "pytorch":
        print(f"{name} network device:", next(model.net.parameters()).device)


print_device_info()
set_random_seed()


# ## Case 1: constant trasmission rate

# In[ ]:


# SIR 模型，Case 1：常数传播率
# 参数 N 表示总人口，delta 表示平均感染期的倒数，r0 表示基本再生数
# beta = delta * r0，因此 r0 = beta / delta
# C 用于把 S/I/R 做尺度归一化，这里按 10^6 人口量级处理
N = 56e6  # (-) population (Italy)
delta = 1 / 5  # (1/T) 5 = mean reproduction period
r0 = 3.0  # (-) basic reproduction number (estimate for Italy)

beta = delta * r0  # (1/T) transmission rate
t0 = 0.0  # (days) initial time
tf = 90.0  # (days) final time

# 缩放因子：对应Eq. (6)
C = 1e5  # 缩放因子，按人口量级选取，便于 PINN 训练；C 的选择会影响训练稳定性和收敛速度
C1 = tf * C / N  # 对应论文 Eq. (6) 中的 C1 = tf*C/N，表示时间和人数的联合缩放因子
C2 = tf * delta  # 对应论文 Eq. (6) 中的 C2 = tf*delta，表示时间和感染期的联合缩放因子


# %%

# In[ ]:


# 下面定义 SIR 常微分方程，对应论文 Eq. (1)-(2)
# dS/dt = -lambda*S, dI/dt = lambda*S - delta*I, dR/dt = delta*I
# lambda_val = beta * I / N，表示感染力项 beta*I/N
def SIR(x, t, delta, beta, N, t0):
    S, I, R = x
    lambda_val = beta * I / N  # 感染力项，表示每个感染者每天感染的人数
    dSdt = -lambda_val * S
    dIdt = lambda_val * S - delta * I
    dRdt = delta * I
    return [dSdt, dIdt, dRdt]


# 初始条件设为 S(t0)=N-I0, I(t0)=I0, R(t0)=0，对应论文 Eq. (2)
# 这里取 I0=1，表示一次初始暴发
S0 = N - 1
I0 = 1
R0 = 0
x0 = [S0, I0, R0]

# 先用 ODE 生成参考解
# PINN 的时间变量按 t_s = (t - t0)/(tf - t0) 归一化
timespan = np.arange("2020-02-01", "2020-05-01", dtype="datetime64[D]")
tspan = timespan.astype(int)

# 用 ODE 解生成时间序列和参考数据
# 根据 SIR 这个变化率函数，从初始值 x0 出发，数值积分得到的 S(t), I(t), R(t) 在 tspan 各个时间点上的值。
x = odeint(SIR, x0, tspan, args=(delta, beta, N, tspan[0]))

S_data = x[:, 0]
I_data = x[:, 1]
R_data = x[:, 2]

# 观测数据 I_obs 是对 I_data 的泊松采样: 在固定时间段内，某类事件发生了多少次。
# 也就是说，假设 ODE 解出来的 I_data(t) 是某一天感染人数的“期望值”，真实观测到的 I_obs(t) 会围绕这个值随机波动。
I_obs = np.random.poisson(I_data)

# 画出 S/R 与 I/观测数据
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(timespan, S_data, "b", label="Susceptible")
plt.plot(timespan, R_data, "g", label="Recovered")
plt.legend()
plt.xlabel("date")
plt.ylabel("individuals")
plt.title("Susceptible and Recovered")
plt.subplot(2, 1, 2)
plt.plot(timespan, I_data, "r", label="Infectious")
plt.plot(timespan, I_obs, "xm", label="samples")
plt.legend()
plt.xlabel("date")
plt.ylabel("individuals")
plt.title("Infectious and Observed Data")
plt.show()


# In[ ]:


# 构造训练时间 t_data 和测试时间 t_test
t_data = np.arange(t0, tf)
t_test = np.arange(t0, tf, 0.1)

# 对变量进行缩放：I_s=I/C，t_s=(t-t0)/tf，对应论文 Eq. (6)
# 这里是对 PINN 输入输出的尺度变换，便于 DeepXDE 训练
I_obs_sc = I_obs / C
I_data_sc = I_data / C
t_data_sc = t_data / tf
t_test_sc = t_test / tf

# 如果按周采样，则每 7 天取一个点
# weekly=True 时只保留每周观测，模拟稀疏数据
weekly = False
if weekly:
    I_obs_sc = I_obs_sc[::7]


# In[ ]:


# 损失函数使用 MSE，优化器使用 Adam。
# DeepXDE 通过 model.compile(loss="MSE", optimizer="adam") 设置损失和优化器；
# 注意：DeepXDE 1.15.0 不提供 SciANN 的 adaptive_weights={'method': 'NTK'} 内置接口。
# 这里用 PyTorch 梯度按 SciANN 的 NTKLossWeight 思路实现自适应权重：
# 每隔 adaptive_NTK["freq"] 步逐样本计算各约束输出的梯度平方和，近似 diag-NTK trace，并动态更新 loss_weights。
loss_err = "MSE"
optimizer = "adam"
learning_rate = 1e-3
adaptive_NTK = {"method": "NTK", "freq": 100}
# SciANN 的 NTKLossWeight 使用 data_generator[0]，也就是当前 batch，而不是全量 collocation 点。
# 这里默认保持同样语义：用一个 batch 的点估计 NTK 权重。设为 None 才会使用全量点。
ntk_sample_size = 100


# In[ ]:


def column(values):
    """把一维数组转成 DeepXDE 约定的二维列向量。"""
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def square_output(_, outputs):
    """DeepXDE 输出变换：对应原脚本的 square 输出层，保证预测的 S_s/I_s 非负。"""
    return torch.square(outputs)


def positive_beta(beta_value):
    """DeepXDE 中的 Beta 直接对应 SciANN Parameter，并在每步后投影到非负。"""
    return torch.clamp(beta_value, min=0.0)


def tensor_to_float(value):
    """兼容 PyTorch tensor 和 NumPy 标量的取值函数，用于记录 beta。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.ravel(value)[0])


def reset_deepxde_session():
    """重置 DeepXDE backend 会话；PyTorch backend 下通常没有需要清理的全局图。"""
    if hasattr(dde.backend, "clear_session"):
        dde.backend.clear_session()


def initialize_sciann_like(net):
    """近似 SciANN 默认初始化：首层 fan_in，后续 fan_avg，权重截断正态，bias 随机均匀。"""
    linear_index = 0
    for module in net.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
        denominator = fan_in if linear_index == 0 else (fan_in + fan_out) / 2.0
        stddev = np.sqrt(1.0 / denominator) / 0.87962566103423978
        torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2.0 * stddev, b=2.0 * stddev)
        torch.nn.init.uniform_(module.bias, a=-0.05, b=0.05)
        linear_index += 1


def sciann_training_steps(num_samples, epochs, batch_size):
    """把 SciANN 的 epochs/batch_size 语义换算成 DeepXDE 的 optimizer update 次数。"""
    steps_per_epoch = int(np.ceil(num_samples / batch_size))
    return epochs * steps_per_epoch


def zero_like_loss(error):
    """返回一个保持梯度图连接的零损失，用于复刻 SciANN 中的零占位项。"""
    return torch.mean(torch.square(error * 0.0))


class SciANNStyleBatchData(dde.data.Data):
    """用 SciANN 的 y_true/sample_weight 语义组织 DeepXDE mini-batch 训练数据。"""

    def __init__(self, train_x, loss_targets, output_fn, batch_size=100, shuffle=True):
        self.train_x_all = column(train_x)
        self.loss_targets = self.prepare_targets(loss_targets)
        self.output_fn = output_fn
        self.batch_size = batch_size
        self.train_sampler = dde.data.BatchSampler(len(self.train_x_all), shuffle=shuffle)
        self._train_indices = np.arange(len(self.train_x_all))
        self._test_indices = np.arange(len(self.train_x_all))
        self.train_x = self.train_x_all
        self.test_x = self.train_x_all
        self.train_aux_vars = None
        self.test_aux_vars = None

    def prepare_targets(self, loss_targets):
        prepared = []
        for target_spec in loss_targets:
            if target_spec in ("zeros", "zero_placeholder"):
                prepared.append(target_spec)
                continue
            ids, values = target_spec
            ids = np.asarray(ids, dtype=np.intp)
            full_values = np.zeros((len(self.train_x_all), 1), dtype=np.float32)
            full_weights = np.zeros((len(self.train_x_all), 1), dtype=np.float32)
            full_values[ids] = column(values)
            full_weights[ids] = 1.0
            prepared.append({"values": full_values, "weights": full_weights})
        return prepared

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self._train_indices = self.train_sampler.get_next(batch_size)
        self.train_x = self.train_x_all[self._train_indices]
        return self.train_x, None, None

    def test(self):
        self._test_indices = np.arange(len(self.train_x_all))
        self.test_x = self.train_x_all
        return self.test_x, None, None

    def losses_for_indices(self, indices, outputs, loss_fn, inputs):
        loss_outputs = self.output_fn(inputs, outputs)
        losses = []
        for output, target_spec in zip(loss_outputs, self.loss_targets):
            if target_spec == "zeros":
                losses.append(loss_fn(torch.zeros_like(output), output))
            elif target_spec == "zero_placeholder":
                losses.append(zero_like_loss(output))
            else:
                target = torch.as_tensor(target_spec["values"][indices], dtype=output.dtype, device=output.device)
                weights = torch.as_tensor(target_spec["weights"][indices], dtype=output.dtype, device=output.device)
                losses.append(loss_fn(target * weights, output * weights))
        return losses

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self.losses_for_indices(self._train_indices, outputs, loss_fn, inputs)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self.losses_for_indices(self._test_indices, outputs, loss_fn, inputs)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return self.losses_train(targets, outputs, loss_fn, inputs, model, aux=aux)


class MiniBatchDataSet(dde.data.DataSet):
    """DeepXDE DataSet 默认忽略 batch_size，这里补上 SciANN 风格 mini-batch。"""

    def __init__(self, *args, shuffle=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler = dde.data.BatchSampler(len(self.train_x), shuffle=shuffle)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return self.train_x[indices], self.train_y[indices]


class BetaLogger(dde.callbacks.Callback):
    """记录 DeepXDE 外部可训练变量 Beta 的训练轨迹。"""

    def __init__(self, beta_raw, period=1):
        super().__init__()
        self.beta_raw = beta_raw
        self.period = period
        self.values = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.period == 0:
            self.values.append(tensor_to_float(positive_beta(self.beta_raw)))


class NonNegativeParameter(dde.callbacks.Callback):
    """模拟 SciANN Parameter(non_neg=True)：每次优化后把参数投影到非负半轴。"""

    def __init__(self, *parameters):
        super().__init__()
        self.parameters = parameters

    def on_batch_end(self):
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.clamp_(min=0.0)


class NTKAdaptiveWeights(dde.callbacks.Callback):
    """按 SciANN NTKLossWeight 公式动态平衡 DeepXDE 的多项损失。"""

    def __init__(self, freq=100, beta=0.9, alpha=0.0, min_max=None, sample_size=None, eps=1e-12):
        super().__init__()
        self.freq = freq
        self.beta = beta
        self.alpha = alpha
        self.min_max = [-np.inf, np.inf] if min_max is None else min_max
        self.sample_size = sample_size
        self.eps = eps
        self.values = []
        self.gradients = []
        self.base_losses = None

    def on_train_begin(self):
        if self.alpha > 0:
            self.base_losses = self.current_unweighted_losses()
        self.update_weights()

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.freq == 0:
            self.update_weights()

    def trainable_parameters(self):
        parameters = [p for p in self.model.net.parameters() if p.requires_grad]
        parameters.extend(self.model.external_trainable_variables)
        return parameters

    def model_device(self):
        return next(self.model.net.parameters()).device

    def model_dtype(self):
        return next(self.model.net.parameters()).dtype

    def as_model_tensor(self, values):
        return torch.as_tensor(values, dtype=self.model_dtype(), device=self.model_device())

    def input_tensor(self, x_value):
        return self.as_model_tensor(x_value).reshape(1, -1).requires_grad_()

    def loss_outputs(self, x_tensor):
        y = self.model.net(x_tensor)
        if hasattr(self.model.data, "output_fn"):
            return self.model.data.output_fn(x_tensor, y)
        pde = getattr(self.model.data, "pde", None)
        if pde is not None:
            outputs = pde(x_tensor, y)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            return outputs
        return [y]

    def squared_grad_norm(self, output):
        parameters = self.trainable_parameters()
        grads = torch.autograd.grad(torch.sum(output), parameters, retain_graph=True, allow_unused=True)
        trace = torch.zeros((), dtype=output.dtype, device=output.device)
        for grad_item in grads:
            if grad_item is not None:
                trace = trace + torch.sum(torch.square(grad_item.detach()))
        return trace.detach()

    def clear_gradients(self):
        if hasattr(dde.grad, "clear"):
            dde.grad.clear()

    def ntk_points(self, points):
        points = self.as_model_tensor(points)
        if self.sample_size is None or points.shape[0] <= self.sample_size:
            return points
        ids = torch.linspace(0, points.shape[0] - 1, self.sample_size, device=points.device).long()
        return points[ids]

    def eval_diag_ntk(self):
        points = self.as_model_tensor(self.model.train_state.X_train)
        global_indices = getattr(self.model.data, "_train_indices", np.arange(points.shape[0]))
        if self.sample_size is not None and points.shape[0] > self.sample_size:
            local_indices = torch.linspace(0, points.shape[0] - 1, self.sample_size, device=points.device).long()
        else:
            local_indices = torch.arange(points.shape[0], device=points.device)
        num_terms = len(self.loss_outputs(self.input_tensor(points[local_indices[0]])))
        traces = [torch.zeros((), dtype=self.model_dtype(), device=self.model_device()) for _ in range(num_terms)]
        for local_index in local_indices:
            x_value = points[local_index]
            global_index = int(global_indices[int(local_index.detach().cpu().item())])
            x_tensor = self.input_tensor(x_value)
            outputs = self.loss_outputs(x_tensor)
            for index, output in enumerate(outputs):
                target_spec = getattr(self.model.data, "loss_targets", [None] * num_terms)[index]
                if isinstance(target_spec, dict) and target_spec["weights"][global_index, 0] == 0.0:
                    continue
                traces[index] = traces[index] + self.squared_grad_norm(output)
            self.clear_gradients()
        return torch.stack(traces)

    def current_unweighted_losses(self):
        previous_weights = self.model.loss_weights
        self.model.loss_weights = None
        losses = self.model.outputs_losses_train(
            self.model.train_state.X_train,
            self.model.train_state.y_train,
            self.model.train_state.train_aux_vars,
        )[1]
        self.model.loss_weights = previous_weights
        return losses.detach()

    def update_weights(self):
        traces = self.eval_diag_ntk()
        normalization_grad = torch.sum(traces)
        mean_trace = torch.mean(traces)
        ntk_weights = normalization_grad / torch.where(traces > self.eps, traces, mean_trace + self.eps)

        if self.base_losses is not None:
            losses = self.current_unweighted_losses()
            scores = losses / self.base_losses
            scores = torch.pow(scores / torch.mean(scores), self.alpha)
            ntk_weights = ntk_weights * scores

        ntk_weights = torch.clamp(ntk_weights, min=self.min_max[0], max=self.min_max[1])
        ntk_weights = ntk_weights * (ntk_weights.numel() / torch.sum(ntk_weights))

        old_weights = self.as_model_tensor(self.model.loss_weights)
        new_weights = (1.0 - self.beta) * old_weights + self.beta * ntk_weights

        self.model.loss_weights = new_weights
        self.values.append(new_weights.detach().cpu().numpy())
        self.gradients.append(traces.detach().cpu().numpy())
        self.model.opt.zero_grad()


def make_deepxde_history(losshistory, beta_logger=None, ntk_logger=None):
    """把 DeepXDE 的 LossHistory 整理成包含 loss 和 Beta 轨迹的字典，便于后续分析。"""
    history = {
        "steps": np.asarray(losshistory.steps),
        "loss_train": np.asarray(losshistory.loss_train),
        "loss_test": np.asarray(losshistory.loss_test),
    }
    if beta_logger is not None:
        history["Beta"] = np.asarray(beta_logger.values)
    if ntk_logger is not None:
        history["loss_weights"] = np.asarray(ntk_logger.values)
        history["loss_gradients"] = np.asarray(ntk_logger.gradients)
    return history


def save_history_and_model(history, history_name, model, model_name):
    """保存训练历史和 DeepXDE 模型权重。"""
    output_dir = Path(__file__).resolve().parent
    with open(output_dir / history_name, "wb") as myFile:
        pickle.dump(history, myFile)
    model.save(str(output_dir / model_name), verbose=0)


def build_joint_pde(beta_raw):
    """构建 joint 方法中的 SIR ODE 残差。"""

    def sir_outputs(ts, y):
        Ss = y[:, 0:1]
        Is = y[:, 1:2]
        Rs = N / C - Is - Ss

        dSs_dts = dde.grad.jacobian(y, ts, i=0, j=0)
        dIs_dts = dde.grad.jacobian(y, ts, i=1, j=0)
        dRs_dts = -dSs_dts - dIs_dts

        initial_mask = 1.0 - torch.sign(ts - t0 / tf)
        Beta = positive_beta(beta_raw)
        L_dSdt = dSs_dts + C1 * Beta * Is * Ss
        L_dIdt = dIs_dts - C1 * Beta * Is * Ss + C2 * Is
        L_dRdt = dRs_dts - C2 * Is
        L_S0 = (Ss - S0 / C) * initial_mask
        L_I0 = (Is - I0 / C) * initial_mask
        L_R0 = (Rs - R0 / C) * initial_mask
        return [L_dSdt, L_dIdt, L_dRdt, L_S0, L_I0, L_R0, Ss * 0.0, Rs * 0.0, Is]

    return sir_outputs


def build_split_pde(beta_raw, fixed_i_net):
    """构建 split 第二阶段中的 SIR ODE 残差。"""

    def sir_outputs(ts, y):
        Ss = y[:, 0:1]
        Is = fixed_i_net(ts)
        Rs = N / C - Is - Ss

        dSs_dts = dde.grad.jacobian(y, ts, i=0, j=0)
        dIs_dts = dde.grad.jacobian(Is, ts, i=0, j=0)
        dRs_dts = dde.grad.jacobian(Rs, ts, i=0, j=0)

        initial_mask = 1.0 - torch.sign(ts - t0 / tf)
        Beta = positive_beta(beta_raw)
        L_dSdt = dSs_dts + C1 * Beta * Is * Ss
        L_dIdt = dIs_dts - C1 * Beta * Is * Ss + C2 * Is
        L_dRdt = dRs_dts - C2 * Is
        L_S0 = (Ss - S0 / C) * initial_mask
        L_R0 = (Rs - R0 / C) * initial_mask
        return [L_dSdt, L_dIdt, L_dRdt, L_S0, L_R0, Ss * 0.0, Rs * 0.0, Is * 0.0]

    return sir_outputs


def predict_joint(model, t_points):
    """获取 joint 模型的 S/I/R 与 beta 预测值。"""
    pred = model.predict(column(t_points))
    S_pred = pred[:, 0:1]
    I_pred = pred[:, 1:2]
    R_pred = N / C - I_pred - S_pred
    return S_pred, I_pred, R_pred


def predict_split(s_model, i_model, t_points):
    """获取 split 模型的 S/I/R 预测值。"""
    S_pred = s_model.predict(column(t_points))
    I_pred = i_model.predict(column(t_points))
    R_pred = N / C - I_pred - S_pred
    return S_pred, I_pred, R_pred


def relative_l2(reference, prediction):
    """计算相对 L2 误差：||u-u_hat||_2 / ||u||_2。"""
    reference = np.ravel(reference)
    prediction = np.ravel(prediction)
    return np.linalg.norm(reference - prediction, 2) / np.linalg.norm(reference, 2)


def plot_sir_predictions(S_pred_test, I_pred_test, R_pred_test):
    """绘制 S/I/R 参考解、PINN 预测和观测样本。"""
    plt.plot(t_data, S_data, c="b", linewidth=4)
    plt.plot(t_test, S_pred_test * C, "--", c="k", linewidth=4)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend(["$S$", "$\\hat{S}$"])
    plt.show()

    plt.plot(t_data, I_data, c="r", linewidth=4)
    plt.plot(t_test, I_pred_test * C, "--", c="k", linewidth=4)
    if weekly:
        plt.scatter(t_data[::7], I_obs[::7], marker="x", c="m", s=100)
    else:
        plt.scatter(t_data, I_obs, marker="x", c="m", s=100)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend(["$I$", "$\\hat{I}$", "samples"])
    plt.show()

    plt.plot(t_data, R_data, c="g", linewidth=4)
    plt.plot(t_test, R_pred_test * C, "--", c="k", linewidth=4)
    plt.xlabel("days")
    plt.ylabel("individuals")
    plt.legend(["$R$", "$\\hat{R}$"])
    plt.show()


def print_sir_errors(S_pred, I_pred, R_pred, beta_pred):
    """打印 S/I/R 相对 L2 误差和 Beta 识别误差。"""
    S_err = relative_l2(S_data, S_pred * C)
    I_err = relative_l2(I_data, I_pred * C)
    R_err = relative_l2(R_data, R_pred * C)
    beta_err = abs(beta_pred - beta) / beta

    print(f"S error: {S_err:.3e}")
    print(f"I error: {I_err:.3e}")
    print(f"R error: {R_err:.3e}")
    print(f"Beta error: {beta_err:.3e}")


# ### Joint

# In[ ]:


reset_deepxde_session()  # 重置 DeepXDE backend 会话，清除之前的模型和变量定义，确保后续代码从干净状态开始执行


# In[ ]:


# 构建神经网络 - joint 方法，同时学习 S、I 和 beta_s（对应论文 Section 2.2）

# ts 是归一化时间变量；Ss/Is 分别表示 S_s_hat(ts) 和 I_s_hat(ts)
# DeepXDE 这里通过自定义 Data 对象直接喂入归一化后的时间点，与 SciANN 的 t_train 语义一致。

# DeepXDE 的 PFNN 可以构建两个并行子网络，分别拟合 S_s 和 I_s；
# 输出层通过 apply_output_transform 使用 square 变换保证非负；
# 网络结构与原脚本保持一致：4 层、每层 50 个神经元；初始化随后覆盖成 SciANN-like 初始化。
net_joint = dde.nn.PFNN([1, [50, 50], [50, 50], [50, 50], [50, 50], [1, 1]], "tanh", "Glorot uniform")
initialize_sciann_like(net_joint)
net_joint.apply_output_transform(square_output)

# Beta 是 DeepXDE 的外部可训练变量；初值和 SciANN Parameter 默认值一致为 1.0，
# 训练时通过 NonNegativeParameter 回调模拟 SciANN 的 non_neg=True 约束。
Beta_raw = dde.Variable(np.float32(1.0))


# In[ ]:


# 初始条件按原 SciANN 脚本写成 sign(ts) 加权残差，包含在 build_joint_pde 返回的损失项中。


# In[ ]:


# 构建 joint 模型的损失函数，包含 ODE 残差、初始条件和数据项（对应论文 Eq. (10)）
# 整体形式为 L_joint = L_ODE + L_IC + L_D
# DeepXDE 中用自定义 Data 对象复刻 SciANN 的 y_true/sample_weight 语义；
# 观测感染人数只在 ids_data 对应的样本上约束 I_s 的预测值。
if weekly:
    t_data_train = t_data_sc[::7]
else:
    t_data_train = t_data_sc

I_obs_sc = I_obs_sc.reshape(-1, 1)  # 观测数据 I_s 的训练目标


# In[ ]:


# 组织训练数据和随机 collocation 点
# 前半部分使用 I_obs 观测数据，后半部分用于 ODE 物理残差约束

Nc = 6000  # collocation points

# 时间采样密度偏向早期时间，增加 t=0 附近的采样点密度，有助于 PINN 更好地拟合初始阶段的数据和满足物理约束
#   np.log1p / np.exp 用于在 [0, tf] 上生成更偏向早期时间的采样点
#   具体来说所以 t=0 附近的密度比 t=1 附近大约高一倍，时间范围没有改变
t_train_ode = np.random.uniform(np.log1p(t0 / tf), np.log1p(1.0), Nc)  # 在 [log1p(t0/tf), log1p(1)] 上生成 Nc 个随机数，偏向早期时间
t_train_ode = np.exp(t_train_ode) - 1.0  # 反变换回 [t0/tf, 1]，得到 ODE 约束的训练时间点

# 将观测数据点和 ODE 约束点合并成训练时间 t_train
if weekly:
    t_train = np.concatenate([t_data_sc[::7].reshape(-1, 1), t_train_ode.reshape(-1, 1)])
    ids_data = np.arange(t_data_sc[::7].size, dtype=np.intp)
else:
    t_train = np.concatenate([t_data_sc.reshape(-1, 1), t_train_ode.reshape(-1, 1)])
    ids_data = np.arange(t_data_sc.size, dtype=np.intp)

epochs_joint = 5000  # joint 模型的训练轮数
batch_size = 100  # 原脚本中的 batch size；DeepXDE 这里通过自定义 Data 对象执行同样的 mini-batch 训练
iterations_joint = sciann_training_steps(t_train.shape[0], epochs_joint, batch_size)

# DeepXDE 的 loss 列表由 3 个 ODE residual + 3 个初值 residual + 2 个零占位项 + 1 个 I 数据项组成：
#   0:3 对应 ODE 残差项
#   3:6 对应初始条件约束
#   6:8 对应原脚本中的 Data(Ss*0.0)、Data(Rs*0.0) 零损失占位项
#   8 对应 I_s 的数据约束
loss_train = ["zeros"] * 8 + [(ids_data, I_obs_sc)]
data_joint = SciANNStyleBatchData(t_train, loss_train, build_joint_pde(Beta_raw), batch_size=batch_size)

beta_logger_joint = BetaLogger(Beta_raw, period=1)
ntk_logger_joint = NTKAdaptiveWeights(freq=adaptive_NTK["freq"], sample_size=ntk_sample_size)
nonnegative_joint = NonNegativeParameter(Beta_raw)

# 构建 joint 模型
m = dde.Model(data_joint, net_joint)  # DeepXDE 的 Model 用于封装 PDE 数据、神经网络、损失函数和优化器
m.compile(
    optimizer,
    lr=learning_rate,
    loss=loss_err,
    loss_weights=[1.0] * 9,
    external_trainable_variables=[Beta_raw],
)
print_model_device("Joint", m)


# In[ ]:


# 训练 joint 模型，并记录训练耗时
# 通过 BetaLogger 记录 Beta 的训练过程，便于后续和 split 方法对比
time1 = time.time()
h, train_state = m.train(
    iterations=iterations_joint,
    batch_size=batch_size,
    callbacks=[nonnegative_joint, beta_logger_joint, ntk_logger_joint],
    display_every=100,
)
time2 = time.time()


# In[ ]:


print(f"Training time: {time2 - time1}")


# In[ ]:


# 获取 joint 模型在测试时间上的 S/I/R 预测结果
# 预测值仍是缩放变量，绘图时乘以 C 还原到人数尺度
S_pred_test, I_pred_test, R_pred_test = predict_joint(m, t_test_sc)

# 绘图
plot_sir_predictions(S_pred_test, I_pred_test, R_pred_test)


# In[ ]:


# 计算相对 L2 误差，并评估 Beta 的识别误差
# 相对误差形式为 ||u-u_hat||_2 / ||u||_2；Beta 在本 case 中是常数，因此取当前标量值
S_pred, I_pred, R_pred = predict_joint(m, t_data_sc)
beta_pred = tensor_to_float(positive_beta(Beta_raw))

print_sir_errors(S_pred, I_pred, R_pred, beta_pred)


# In[ ]:


# 保存 joint 阶段的训练历史和模型权重
# h 中包含 loss，BetaLogger 中包含 Beta 随 epoch 的变化，可用于后续分析
h_joint = make_deepxde_history(h, beta_logger_joint, ntk_logger_joint)
save_history_and_model(h_joint, "hJoint.txt", m, "mJoint")


# ### Split

# In[ ]:


reset_deepxde_session()


# In[ ]:


# 构建神经网络 - split 方法第一步，仅回归 I(t)（对应论文 split approach first step）
# 这一阶段只根据观测数据拟合缩放后的 I_s(t_s)
# 初始化随后覆盖成 SciANN-like 初始化。
net_data = dde.nn.FNN([1] + 4 * [50] + [1], "tanh", "Glorot uniform")
initialize_sciann_like(net_data)
net_data.apply_output_transform(square_output)


# In[ ]:


# 构建 split 第一阶段的数据回归模型
# DeepXDE 的 DataSet 直接接收 (t_s, I_obs_sc)，等价于只用观测数据监督 Isc
if weekly:
    t_data_train = t_data_sc[::7]
    epochs_data = 1000
    batch_data = 13
else:
    t_data_train = t_data_sc
    epochs_data = 3000
    batch_data = 10

data_fit = MiniBatchDataSet(
    X_train=column(t_data_train),
    y_train=I_obs_sc.astype(np.float32),
    X_test=column(t_data_sc),
    y_test=column(I_data_sc),
)

m_data = dde.Model(data_fit, net_data)
m_data.compile(optimizer, lr=learning_rate, loss=loss_err)
print_model_device("Split data", m_data)
iterations_data = sciann_training_steps(t_data_train.size, epochs_data, batch_data)


# In[ ]:


# 训练 split 第一阶段的数据回归模型
time1_data = time.time()
h_data, train_state_data = m_data.train(
    iterations=iterations_data,
    batch_size=batch_data,
    display_every=100,
)
time2_data = time.time()


# In[ ]:


print(f"Training time: {time2_data - time1_data}")


# In[ ]:


# 获取 split 第一阶段的 I(t) 预测结果
Isc_pred = m_data.predict(column(t_test_sc))

# 绘图
plt.plot(t_data, I_data, c="r", linewidth=4)
plt.plot(t_test, Isc_pred * C, "--", c="k", linewidth=4)
if weekly:
    plt.scatter(t_data[::7], I_obs[::7], marker="x", c="m", s=100)
else:
    plt.scatter(t_data, I_obs, marker="x", c="m", s=100)
plt.xlabel("days")
plt.ylabel("individuals")
plt.legend(["$I$", "$\\hat{I}$", "samples"])
plt.show()


# In[ ]:


# 计算回归得到的 I_s 与参考 I_data_sc 之间的相对误差
Isc_pred = m_data.predict(column(t_data_sc))
Isc_err = relative_l2(I_data_sc, Isc_pred)
print(f"Isc error: {Isc_err:.3e}")


# In[ ]:


# 冻结第一阶段学到的 I 网络权重，作为第二阶段的已知 I(t)
fixed_i_net = m_data.net
fixed_i_net.eval()
for parameter in fixed_i_net.parameters():
    parameter.requires_grad = False


# In[ ]:


# 构建神经网络 - split 方法第二步，在固定 I 后学习 S 和 beta（对应论文 split approach second step）
# 这里使用第一阶段拟合得到的 I(t)，继续识别 S(t) 和 beta(t)
# 初始化随后覆盖成 SciANN-like 初始化。
net_ode = dde.nn.FNN([1] + 4 * [50] + [1], "tanh", "Glorot uniform")
initialize_sciann_like(net_ode)
net_ode.apply_output_transform(square_output)

Beta_raw_split = dde.Variable(np.float32(1.0))


# In[ ]:


# S 和 R 的初始条件按原 SciANN 脚本写成 sign(ts) 加权残差；I 已由第一阶段网络固定。

# 构造 ODE 残差，并把固定的 I 网络纳入物理约束


# In[ ]:


epochs_ode = 1000
iterations_ode = sciann_training_steps(t_train_ode.shape[0], epochs_ode, batch_size)
loss_train_ode = ["zeros"] * 8
data_ode = SciANNStyleBatchData(
    t_train_ode.reshape(-1, 1),
    loss_train_ode,
    build_split_pde(Beta_raw_split, fixed_i_net),
    batch_size=batch_size,
)
beta_logger_split = BetaLogger(Beta_raw_split, period=1)
ntk_logger_split = NTKAdaptiveWeights(freq=adaptive_NTK["freq"], sample_size=ntk_sample_size)
nonnegative_split = NonNegativeParameter(Beta_raw_split)

m_ode = dde.Model(data_ode, net_ode)
m_ode.compile(
    optimizer,
    lr=learning_rate,
    loss=loss_err,
    loss_weights=[1.0] * 8,
    external_trainable_variables=[Beta_raw_split],
)
print_model_device("Split ODE", m_ode)


# In[ ]:


# 训练 split 第二阶段模型，只使用 collocation 点施加物理约束
time1_ode = time.time()
h_ode, train_state_ode = m_ode.train(
    iterations=iterations_ode,
    batch_size=batch_size,
    callbacks=[nonnegative_split, beta_logger_split, ntk_logger_split],
    display_every=100,
)
time2_ode = time.time()


# In[ ]:


print(f"Training time: {time2_ode - time1_ode}")


# In[ ]:


# 获取 split 模型在测试时间上的 S/I/R 预测结果
S_pred_test, I_pred_test, R_pred_test = predict_split(m_ode, m_data, t_test_sc)

# 绘图
plot_sir_predictions(S_pred_test, I_pred_test, R_pred_test)


# In[ ]:


# 计算 split 模型的相对 L2 误差和 Beta 识别误差
S_pred, I_pred, R_pred = predict_split(m_ode, m_data, t_data_sc)
beta_pred = tensor_to_float(positive_beta(Beta_raw_split))

print_sir_errors(S_pred, I_pred, R_pred, beta_pred)


# In[ ]:


# 保存 split 两个阶段的训练历史和模型权重
h_split_data = make_deepxde_history(h_data)
h_split_ode = make_deepxde_history(h_ode, beta_logger_split, ntk_logger_split)
save_history_and_model(h_split_data, "hSplit_data.txt", m_data, "mSplit_data")
save_history_and_model(h_split_ode, "hSplit_ode.txt", m_ode, "mSplit_ode")


# In[ ]:


# 绘制 beta 估计值随 epoch 的变化，对比 joint 和 split 方法
plt.plot(h_joint["Beta"], linewidth=2.5)
plt.plot(h_split_ode["Beta"], linewidth=2.5)
plt.legend(["Joint", "Split"])
plt.ylabel(r"$\hat{\beta}_0$")
plt.xlabel("epochs")
plt.show()


# In[ ]:
