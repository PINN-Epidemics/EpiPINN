# NTK 与代码中动态权重更新的区别

## 1. 传统 NTK（神经切线核）概念

NTK 关注的是神经网络输出相对于参数的雅可比矩阵及其 Gram 矩阵。

设模型输出为

$$
f(\mathbf{x};\boldsymbol{\theta})
$$

平方损失为

$$
L(\boldsymbol{\theta}) = \frac{1}{2}\|f(\mathbf{x};\boldsymbol{\theta}) - \mathbf{y}\|^2
$$

参数梯度为

$$
\nabla_{\boldsymbol{\theta}} L = J(\boldsymbol{\theta})^\top \bigl(f(\mathbf{x};\boldsymbol{\theta}) - \mathbf{y}\bigr)
$$

其中

$$
J = \frac{\partial f}{\partial \boldsymbol{\theta}}
$$

是模型输出对参数的雅可比矩阵。

NTK 定义为：

$$
K(\mathbf{x}, \mathbf{x}') = J(\boldsymbol{\theta}) J(\boldsymbol{\theta})^\top
$$

在宽网络的线性化近似下，训练输出可以近似为：

$$
f_{t+1} \approx f_t - \eta K (f_t - y)
$$

因此，NTK 的核心是：

- 关注网络本身的训练动力学；
- 关注输出变化与参数变化之间的关系；
- 关键对象是雅可比矩阵和核矩阵。

---

## 2. 代码中动态权重更新的实际做法

当前代码里 `update_loss_weights` 的流程为：

1. 对每个损失项 $L_i$ 计算参数梯度：

$$
\mathbf{g}_i = \nabla_{\boldsymbol{\theta}} L_i
$$

2. 计算梯度范数：

$$
n_i = \|\mathbf{g}_i\|_2
$$

3. 计算目标尺度：

$$
\bar{n} = \frac{1}{3}(n_{\text{data}} + n_{\text{ode}} + n_{\text{ic}})
$$

4. 生成新的权重：

$$
w_i = \text{clamp}\left(\frac{\bar{n}}{\max(n_i, \epsilon)},\, 10^{-3},\, 10^{3}\right)
$$

5. 使用动量平滑：

$$
w_i \leftarrow 0.9\,w_i + 0.1\,w_i^{\text{new}}
$$

最终总损失为：

$$
L_{\text{total}} = w_{\text{data}} L_{\text{data}} + w_{\text{ode}} L_{\text{ode}} + w_{\text{ic}} L_{\text{ic}}
$$

这一步骤的本质是“多目标损失之间的梯度规模平衡”。

---

## 3. 核心区别

| 方面 | 传统 NTK | 代码动态权重更新 |
|---|---|---|
| 关注点 | 网络训练动力学、雅可比和核矩阵 | 多损失项梯度大小的平衡 |
| 计算对象 | $J, K = JJ^\top$ | $\|\nabla L_i\|$ 以及权重 $w_i$ |
| 是否计算 NTK | 是 | 否 |
| 方法类型 | 训练动力学分析 | 自适应损失加权 |


## 4. 这段代码属于什么方法？

这类方法通常被称为：

- `GradNorm` 风格的损失平衡；
- gradient-norm based adaptive loss weighting；
- adaptive loss weighting / dynamic loss weighting；
- gradient norm balancing。

需要注意的是，这段代码不是严格的原始 `GradNorm` 论文实现，但本质是一类“基于梯度范数的多任务/多目标损失权重平衡”方法。

---

## 5. 结论

- 传统 NTK 是基于雅可比矩阵和内核矩阵的训练动力学理论；
- 代码中的 `update_loss_weights` 只是计算各个损失项的梯度范数，并据此自适应调整权重；
- 该方法更准确的称呼是“基于梯度范数的自适应损失权重更新”，而非 NTK。
