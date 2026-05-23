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
# Note: The uploaded code is related to Case 6 and Case 7. For further information please contact the corresponding author.

# In[1]:


import sciann as sn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[ ]:


# 设置随机种子，便于复现实验结果。
sn.set_random_seed(234)


# In[ ]:


case = 6
# 如需运行 Case 7，改为 case = 7


# In[ ]:


# 读取真实疫情数据，包含感染观测、住院观测和 Rt 数据
df = pd.read_table("./RealData.txt")
df


# In[ ]:


# 时间域
# timespan 保留真实日期用于绘图，t_data/t_test 使用从 0 开始的天数索引
timespan = df["Date"].values.astype('datetime64[D]')

t0 = 0.                 # (days) initial time
tf = len(timespan)      # (days) final time

t_data = np.arange(t0,tf)
t_test = np.arange(t0,tf,0.1)


# In[ ]:


# 真实数据
# H_obs 为住院新增/观测数据，I_obs 为感染观测数据，Rt_data 为参考再生数
H_obs = df["H data"].values
I_obs = df["I data"].values
Rt_data = df["Rt data"].values

# 参数
# delta 表示平均感染期的倒数
delta = 1/5  # (1/T) 5 = mean reproduction period 

# 缩放因子
# SI 和 SH 分别用于感染数据和住院数据归一化
SI = I_obs.max()
SH = H_obs.max()

# C 用于把 delta*sigma*I 与住院数据的尺度对应起来
C = SI*delta/SH

I_obs_sc = I_obs/SI
H_obs_sc = H_obs/SH
t_data_sc = t_data/tf
t_test_sc = t_test/tf


# In[ ]:


# 绘图结果
# 先检查真实感染观测、住院观测和 Rt 数据的时间变化
fig, ax = plt.subplots(3, 1, figsize=(10,9))

ax[0].plot(timespan, I_obs, 'xm', label='samples')
ax[0].legend(loc=6)
ax[0].set_xlabel('date')
ax[0].set_ylabel('individuals')
ax[0].set_title('Infectious Data')

ax[1].plot(timespan, H_obs, 'xm', label='samples')
ax[1].legend(loc=6)
ax[1].set_xlabel('date')
ax[1].set_ylabel('individuals')
ax[1].set_title('Hospitalizations Data')

ax[2].plot(timespan, Rt_data, label=r"$\mathcal{R}_t$")
ax[2].legend(loc=6)
ax[2].set_xlabel('date')
ax[2].set_title('Reproduction number')

fig.tight_layout()

plt.show()


# In[ ]:


# 训练参数
# loss_err='mse' 指定均方误差损失，optimizer='adam' 指定优化器
loss_err  = 'mse'
optimizer = 'adam'


# ### Joint

# In[ ]:


sn.reset_session()


# In[ ]:


# 构建神经网络 - joint 方法，同时学习 I(t) 和 Rt(t)
# Is 表示缩放后的感染人数，Rt 表示时变再生数
ts  = sn.Variable("ts")
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')


# In[ ]:


# Case 6 中 sigma 为常数参数；Case 7 中 sigma 为随时间变化的函数网络
if case==6:
    sigma = sn.Parameter(name="sigma", inputs=ts, non_neg=True)
if case==7:
    sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')


# In[ ]:


# 常微分方程残差
# L_dIdt 对应 dI/dt = delta*(Rt-1)*I，时间变量已按 tf 缩放
L_dIdt = sn.rename((sn.diff(Is,ts)-tf*delta*(Rt-1)*Is), "L_dIdt")

# deltaHs 和 deltaIs 分别用于拟合住院观测与感染观测
deltaHs = sn.rename(sigma*Is*C, "deltaHs")
deltaIs = sn.rename(delta*Rt*Is, "deltaIs")


# In[ ]:


# 构建 joint 损失函数：ODE 残差、Rt 正则项、感染数据项和住院数据项
loss_joint = [sn.PDE(L_dIdt), sn.Data(Rt*0.0),
              sn.Data(deltaIs), sn.Data(deltaHs)]


# In[ ]:


# 将时间变量和损失项封装为 SciANN 模型
m = sn.SciModel(ts, loss_joint, loss_err, optimizer)


# In[ ]:


# 训练点
# t_data_sc 对应真实观测时间，t_train_ode 是用于 ODE 残差的随机 collocation 点
Nc = 6000    # collocation points

I_obs_sc = I_obs_sc.reshape(-1,1)
H_obs_sc = H_obs_sc.reshape(-1,1)
t_train_ode = np.random.uniform(t0/tf, 1., Nc-1)
t_train_ode = np.insert(t_train_ode,0,0.0)
t_train = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
ids_data = np.arange(t_data_sc.size, dtype=np.intp)


# In[ ]:


# 前两个损失项目标为 0，后两个数据项分别使用 I_obs_sc 和 H_obs_sc
loss_train   = ['zeros']*2+[(ids_data,I_obs_sc),(ids_data,H_obs_sc)]
epochs_joint = 5000
batch_size   = 100


# In[ ]:


# 训练 joint 模型，并记录训练耗时
time1 = time.time()
h     = m.train(t_train,
                loss_train,
                epochs=epochs_joint,
                batch_size=batch_size,
                verbose=1
               )
time2 = time.time()


# In[ ]:


print(f'Training time: {time2-time1}')


# In[ ]:


# 获取 joint 模型预测结果
# deltaI/deltaH 乘以各自缩放因子还原到原始数据尺度
deltaI_pred_test = deltaIs.eval(m, t_test_sc)*SI
deltaH_pred_test = deltaHs.eval(m, t_test_sc)*SH
Rt_pred_test = Rt.eval(m, t_test_sc)
sigma_pred_test = sigma.eval(m, t_test_sc)

# 绘图结果
# 对比感染观测与 joint 模型预测
plt.plot(t_test,deltaI_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\hat{\Delta}_I$','samples'])
plt.show()

# 绘图结果
# 对比住院观测与 joint 模型预测
plt.plot(t_test,deltaH_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\hat{\Delta}_H$','samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()

if case==7:
    plt.plot(t_test, sigma_pred_test, '--', c='k',linewidth=4)
    plt.xlabel('days')
    plt.legend([r'$\hat{\sigma}$'])
    plt.show()


# In[ ]:


# 计算误差
# 使用相对 L2 误差评估 Rt 的识别质量；Case 6 额外输出常数 sigma
Rt_pred = Rt.eval(m, t_data_sc)

Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)

print(f'Rt error: {Rt_err:.3e}')

if case==6:
    print(f'Estimated sigma: {sigma_pred_test[0]:.4f}')


# ### Split

# In[ ]:


sn.reset_session()


# In[ ]:


# 构建神经网络 - split 第一阶段，仅回归住院数据 deltaH(t)
# deltaHsc 表示缩放后的住院观测函数
ts  = sn.Variable("ts")
deltaHsc = sn.Functional("deltaHsc", ts, 4*[50], output_activation='square')


# In[ ]:


# 构建 split 第一阶段数据回归模型
# 损失只包含 deltaHsc 与 H_obs_sc 的数据误差
loss_data = sn.Data(deltaHsc)

m_data = sn.SciModel(ts, loss_data, loss_err, optimizer)


# In[ ]:


# 设置 split 第一阶段训练参数
t_data_train = t_data_sc
epochs_data  = 1000
batch_data   = 10


# In[ ]:


# 训练模型 - split 第一阶段，仅做住院数据回归
time1_data = time.time()
h_data     = m_data.train(t_data_train, 
                          H_obs_sc, 
                          epochs=epochs_data,
                          batch_size=batch_data,
                          verbose=1)
time2_data = time.time()


# In[ ]:


print(f'Training time: {time2_data-time1_data}')


# In[ ]:


# 获取 split 第一阶段的住院数据预测结果
deltaHsc_pred = deltaHsc.eval(m_data, t_test_sc)

# 绘图结果
# 对比住院观测和第一阶段回归结果
plt.plot(t_test,deltaHsc_pred*SH, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$\hat{\Delta}_H$','samples'])
plt.show()


# In[ ]:


# 固定住院数据回归网络权重
# 第二阶段把 deltaHs 作为已知函数，不再训练该网络
deltaHsc_weights = deltaHsc.get_weights()

deltaHs = sn.Functional("deltaHs", ts, 4*[50], output_activation='square', trainable=False)
deltaHs.set_weights(deltaHsc_weights)


# In[ ]:


# 构建神经网络 - split 第二阶段，进入物理约束训练
# 在固定 deltaHs 的基础上学习 Rt 和 sigma
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')

if case==6:
    sigma = sn.Parameter(name="sigma", inputs=ts, non_neg=True)
if case==7:
    sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')+1e-3


# In[ ]:


# 常微分方程残差
# 由 deltaHs = sigma*Is*C 反推出 Is，再用感染动力学约束 Rt
Is=sn.rename(deltaHs/sigma/C, "Is")
deltaIs = sn.rename(delta*Rt*Is, "dSdt")

L_dIdt = sn.rename(sn.diff(Is,ts)-tf*delta*(Rt-1)*Is, "L_dIdt")


# In[ ]:


# 构建模型 - split 第二阶段物理约束训练
# 损失包含 ODE 残差、Rt 正则项和感染观测数据项
loss_ode = [sn.PDE(L_dIdt), sn.Data(Rt*0.0),
            sn.Data(deltaIs)]

m_ode = sn.SciModel(ts, loss_ode, loss_err, optimizer)

loss_train_ode = ['zeros']*2+[(ids_data,I_obs_sc)]

epochs_ode = 3000


# In[ ]:


# 训练模型 - split 第二阶段，物理约束训练
time1_ode = time.time()
h_ode     = m_ode.train(t_train,
                        loss_train_ode,
                        epochs=epochs_ode,
                        batch_size=batch_size,
                        verbose=1)
time2_ode = time.time()


# In[ ]:


print(f'Training time: {time2_ode-time1_ode}')


# In[ ]:


# 获取 split 第二阶段预测结果
deltaI_pred_test = deltaIs.eval(m_ode, t_test_sc)*SI
deltaH_pred_test = deltaHs.eval(m_ode, t_test_sc)*SH
Rt_pred_test = Rt.eval(m_ode, t_test_sc)
sigma_pred_test = sigma.eval(m_ode, t_test_sc)

# 绘图结果
# 对比感染观测和 split 模型预测
plt.plot(t_test,deltaI_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\hat{\Delta}_I$','samples'])
plt.show()

# 绘图结果
# 对比住院观测和 split 模型预测
plt.plot(t_test,deltaH_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\hat{\Delta}_H$','samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()

if case==7:
    plt.plot(t_test, sigma_pred_test, '--', c='k',linewidth=4)
    plt.xlabel('days')
    plt.legend([r'$\hat{\sigma}$'])
    plt.show()


# In[ ]:


# 计算误差
# 使用相对 L2 误差评估 split 方法下 Rt 的识别质量
Rt_pred = Rt.eval(m_ode, t_data_sc)

Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)

print(f'Rt error: {Rt_err:.3e}')

if case==6:
    print(f'Estimated sigma: {sigma_pred_test[0]:.4f}')


# In[ ]:



