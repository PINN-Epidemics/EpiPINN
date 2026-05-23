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


import sciann as sn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import odeint


# ## Case 1: constant trasmission rate

# In[ ]:


# SIR 模型，Case 1：常数传播率
# 参数 N 表示总人口，delta 表示平均感染期的倒数，r0 表示基本再生数
# beta = delta * r0，因此 r0 = beta / delta
# C 用于把 S/I/R 做尺度归一化，这里按 10^6 人口量级处理
N     = 56e6 # (-) population (Italy)
delta = 1/5  # (1/T) 5 = mean reproduction period 
r0    = 3.   # (-) basic reproduction number (estimate for Italy) 

beta = delta*r0 # (1/T) transmission rate 
t0   = 0.       # (days) initial time
tf = 90.        # (days) final time

C = 1e5
C1 = tf*C/N   
C2 = tf*delta


# In[ ]:


# 下面定义 SIR 常微分方程，对应论文 Eq. (1)-(2)
# dS/dt = -lambda*S, dI/dt = lambda*S - delta*I, dR/dt = delta*I
# lambda_val = beta * I / N，表示感染力项 beta*I/N
def SIR(x, t, delta, beta, N, t0):
    S, I, R = x
    lambda_val = beta*I/N;
    dSdt = -lambda_val*S
    dIdt = lambda_val*S - delta*I
    dRdt = delta*I
    return [dSdt, dIdt, dRdt]

# 初始条件设为 S(t0)=N-I0, I(t0)=I0, R(t0)=0，对应论文 Eq. (2)
# 这里取 I0=1，表示一次初始暴发
S0 = N-1
I0 = 1
R0 = 0
x0 = [S0, I0, R0]

# 先用 ODE 生成参考解
# PINN 的时间变量按 t_s = (t - t0)/(tf - t0) 归一化
timespan = np.arange('2020-02-01', '2020-05-01', dtype='datetime64[D]')
tspan = timespan.astype(int)

# 用 ODE 解生成时间序列和参考数据
# 先求解参考 ODE，再用于 PINN 对比
x = odeint(SIR, x0, tspan, args=(delta, beta, N, tspan[0]))

S_data = x[:, 0]
I_data = x[:, 1]
R_data = x[:, 2]

# 为观测感染人数加入泊松采样
# 画出 S/R 与 I/观测数据
I_obs = np.random.poisson(I_data)

# 画出 S/R 与 I/观测数据
# I_obs 是对 I_data 的泊松采样，用于模拟观测误差
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(timespan, S_data, 'b', label='Susceptible')
plt.plot(timespan, R_data, 'g', label='Recovered')
plt.legend()
plt.xlabel('date')
plt.ylabel('individuals')
plt.title('Susceptible and Recovered')
plt.subplot(2, 1, 2)
plt.plot(timespan, I_data, 'r', label='Infectious')
plt.plot(timespan, I_obs, 'xm', label='samples')
plt.legend()
plt.xlabel('date')
plt.ylabel('individuals')
plt.title('Infectious and Observed Data')
plt.show()


# In[ ]:


# 对变量做归一化：S=C*S_s, I=C*I_s, t_s=t/t_f，对应论文 Eq. (5)-(6)
# 这里是对 PINN 输入输出的尺度变换，便于 SciANN 训练
# 构造训练时间 t_data 和测试时间 t_test
t_data = np.arange(t0,tf)
t_test = np.arange(t0,tf,0.1)

I_obs_sc  = I_obs/C
I_data_sc = I_data/C
t_data_sc = t_data/tf
t_test_sc = t_test/tf

# 如果按周采样，则每 7 天取一个点
# weekly=True 时只保留每周观测，模拟稀疏数据
weekly = False
if weekly:
    I_obs_sc = I_obs_sc[::7]


# In[ ]:


# 损失函数使用 MSE，优化器使用 Adam，并启用 NTK 自适应权重
# loss_err='mse' 指定均方误差损失，optimizer='adam' 指定优化器
# adaptive_NTK 使用 NTK 方法动态调整各损失项权重
loss_err  = 'mse'
optimizer = 'adam'
adaptive_NTK = {'method':'NTK','freq':100}


# ### Joint

# In[ ]:


sn.reset_session()


# In[ ]:


# 构建神经网络 - joint 方法，同时学习 S、I 和 beta_s（对应论文 Section 2.2）
# ts 是归一化时间变量；Ss/Is 分别表示 S_s_hat(ts) 和 I_s_hat(ts)
# 4*[50] 表示 4 个隐藏层、每层 50 个神经元；square 保证输出非负
# Beta 是待识别的传播率参数 beta_s_hat(ts)，这里设为非负常数参数
ts  = sn.Variable('ts')
Ss = sn.Functional('Ss', ts, 4*[50], output_activation='square')
Is = sn.Functional('Is', ts, 4*[50], output_activation='square')

Beta = sn.Parameter(name='Beta', inputs=ts, non_neg=True)

# 根据守恒关系计算 R_s（对应论文中的 N=S+I+R）
# 即 R_s = N/C - I_s - S_s，用缩放后的变量保持总量守恒
Rs = N/C-Is-Ss


# In[ ]:


# 用 sign 构造 t=t0 处的初始条件约束（对应论文 Eq. (8)）
# 乘子 (1-sign(ts-t0/tf)) 只在初始时刻附近起作用
# 三个残差分别约束 S_s(0)、I_s(0)、R_s(0) 的初始值
L_S0 = sn.rename((Ss-S0/C)*(1-sn.sign(ts-t0/tf)), 'L_S0')
L_I0 = sn.rename((Is-I0/C)*(1-sn.sign(ts-t0/tf)), 'L_I0')
L_R0 = sn.rename((Rs-R0/C)*(1-sn.sign(ts-t0/tf)), 'L_R0')

# ODE 残差使用缩放后的 S/I/R 变量（对应论文 Eq. (7)-(9)）
# 分别约束 dS_s/dt_s + C1*beta_s*I_s*S_s、dI_s/dt_s - C1*beta_s*I_s*S_s + C2*I_s、dR_s/dt_s - C2*I_s
# 这些 PDE 项会在 collocation 点上参与物理约束训练
L_dSdt = sn.rename((sn.diff(Ss,ts)+C1*Beta*Is*Ss), 'L_dSdt')
L_dIdt = sn.rename((sn.diff(Is,ts)-C1*Beta*Is*Ss+C2*Is), 'L_dIdt')
L_dRdt = sn.rename((sn.diff(Rs,ts)-C2*Is), 'L_dRdt')


# In[ ]:


# 构建 joint 模型的损失函数，包含 ODE 残差、初始条件和数据项（对应论文 Eq. (10)）
# 整体形式为 L_joint = L_D + L_ODE + L_IC
# 最后的 sn.Data(Is) 用观测感染人数约束 I_s 的预测值
loss_joint = [sn.PDE(L_dSdt),  sn.PDE(L_dIdt),  sn.PDE(L_dRdt), 
              sn.PDE(L_S0),    sn.PDE(L_I0),    sn.PDE(L_R0),
              sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Is)]

m = sn.SciModel(ts, loss_joint, loss_err, optimizer)


# In[ ]:


# 组织训练数据和随机 collocation 点
# 前半部分使用 I_obs 观测数据，后半部分用于 ODE 物理残差约束
# np.log1p / np.exp 用于在 [0, tf] 上生成更偏向早期时间的采样点
Nc = 6000    # collocation points

I_obs_sc     = I_obs_sc.reshape(-1,1)
t_train_ode  = np.random.uniform(np.log1p(t0/tf), np.log1p(1.), Nc)
t_train_ode  = np.exp(t_train_ode) - 1.
if weekly:
    t_train  = np.concatenate([t_data_sc[::7].reshape(-1,1), t_train_ode.reshape(-1,1)])
    ids_data = np.arange(t_data_sc[::7].size,dtype=np.intp)
else:
    t_train  = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
    ids_data = np.arange(t_data_sc.size,dtype=np.intp)

loss_train   = ['zeros']*8+[(ids_data,I_obs_sc)]
epochs_joint = 5000
batch_size   = 100

log_params   = {'parameters': Beta,'freq':1}


# In[ ]:


# 训练 joint 模型，并记录训练耗时
# 通过 log_parameters 记录 Beta 的训练过程，便于后续和 split 方法对比
time1 = time.time()
h     = m.train(t_train,
                loss_train,
                epochs=epochs_joint,
                batch_size=batch_size,
                log_parameters=log_params,
                adaptive_weights=adaptive_NTK,
                verbose=1
               )
time2 = time.time()


# In[ ]:


print(f'Training time: {time2-time1}')


# In[ ]:


# 获取 joint 模型在测试时间上的 S/I/R 预测结果
# 预测值仍是缩放变量，绘图时乘以 C 还原到人数尺度
S_pred_test = Ss.eval(m, t_test_sc)
I_pred_test = Is.eval(m, t_test_sc)
R_pred_test = Rs.eval(m, t_test_sc)

# 绘图
plt.plot(t_data, S_data, c='b',linewidth=4)
plt.plot(t_test,S_pred_test*C, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$S$', '$\hat{S}$'])
plt.show()

plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test*C, '--', c='k',linewidth=4)
if weekly: plt.scatter(t_data[::7],I_obs[::7], marker='x', c='m', s=100)
else: plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$I$','$\hat{I}$','samples'])
plt.show()

plt.plot(t_data, R_data, c='g',linewidth=4)
plt.plot(t_test,R_pred_test*C, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$R$', '$\hat{R}$'])
plt.show()


# In[ ]:


# 计算相对 L2 误差，并评估 Beta 的识别误差
# 相对误差形式为 ||u-u_hat||_2 / ||u||_2；Beta 在本 case 中是常数，因此取 beta_pred[0]
S_pred = Ss.eval(m, t_data_sc)*C
I_pred = Is.eval(m, t_data_sc)*C
R_pred = Rs.eval(m, t_data_sc)*C
beta_pred = Beta.eval(m, t_data_sc)

S_err = np.linalg.norm(S_data-S_pred,2)/np.linalg.norm(S_data,2)
I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
R_err = np.linalg.norm(R_data-R_pred,2)/np.linalg.norm(R_data,2)
beta_err = abs(beta_pred[0]-beta)/beta

print(f'S error: {S_err:.3e}')
print(f'I error: {I_err:.3e}')
print(f'R error: {R_err:.3e}')
print(f'Beta error: {beta_err:.3e}')


# In[ ]:


# 保存 joint 阶段的训练历史和模型权重
# h.history 包含 loss 和 Beta 随 epoch 的变化，可用于后续分析
with open('hJoint.txt', 'wb') as myFile:
    pickle.dump(h.history, myFile)
myFile.close()

m.save_weights('mJoint.hdf5')


# ### Split

# In[ ]:


sn.reset_session()


# In[ ]:


# 构建神经网络 - split 方法第一步，仅回归 I(t)（对应论文 split approach first step）
# 这一阶段只根据观测数据拟合缩放后的 I_s(t_s)
ts  = sn.Variable('ts')
Isc = sn.Functional('Isc', ts, 4*[50], output_activation='square')


# In[ ]:


# 构建 split 第一阶段的数据回归模型
# 损失项对应 L_D(I_s)，即预测 I_s 与观测 I_s,obs 之间的均方误差
loss_data = sn.Data(Isc)

m_data = sn.SciModel(ts, loss_data, loss_err, optimizer)


# In[ ]:


# 设置 split 第一阶段训练参数；周采样时减少训练点和 batch size
if weekly: 
    t_data_train = t_data_sc[::7]
    epochs_data  = 1000
    batch_data   = 13
else: 
    t_data_train = t_data_sc
    epochs_data  = 3000
    batch_data   = 10


# In[ ]:


# 训练 split 第一阶段的数据回归模型
time1_data = time.time()
h_data     = m_data.train(t_data_train, 
                          I_obs_sc, 
                          epochs=epochs_data,
                          batch_size=batch_data,
                          verbose=1)
time2_data = time.time()


# In[ ]:


print(f'Training time: {time2_data-time1_data}')


# In[ ]:


# 获取 split 第一阶段的 I(t) 预测结果
Isc_pred = Isc.eval(m_data, t_test_sc)

# 绘图
plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,Isc_pred*C, '--', c='k', linewidth=4)
if weekly: plt.scatter(t_data[::7],I_obs[::7], marker='x', c='m', s=100)
else: plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$I$','$\hat{I}$','samples'])
plt.show()


# In[ ]:


# 计算回归得到的 I_s 与参考 I_data_sc 之间的相对误差
Isc_pred = Isc.eval(m_data,t_data_sc)
Isc_err = np.linalg.norm(I_data_sc-Isc_pred,2)/np.linalg.norm(I_data_sc,2)
print(f'Isc error: {Isc_err:.3e}')


# In[ ]:


# 冻结第一阶段学到的 I 网络权重，作为第二阶段的已知 I(t)
Isc_weights = Isc.get_weights()

Is = sn.Functional('Is', ts, 4*[50], output_activation='square', trainable=False)
Is.set_weights(Isc_weights)


# In[ ]:


# 构建神经网络 - split 方法第二步，在固定 I 后学习 S 和 beta（对应论文 split approach second step）
# 这里使用第一阶段拟合得到的 I(t)，继续识别 S(t) 和 beta(t)
Ss = sn.Functional('Ss', ts, 4*[50], output_activation='square')

Beta = sn.Parameter(name='Beta', inputs=ts, non_neg=True)

Rs = N/C-Is-Ss


# In[ ]:


# 构造 S 和 R 的初始条件约束；I 已由第一阶段网络固定
L_S0 = sn.rename((Ss-S0/C)*(1-sn.sign(ts-t0/tf)), 'L_S0')
L_R0 = sn.rename((Rs-R0/C)*(1-sn.sign(ts-t0/tf)), 'L_R0')

# 构造 ODE 残差，并把固定的 I 网络纳入物理约束
L_dSdt = sn.rename((sn.diff(Ss,ts)+C1*Beta*Is*Ss), 'L_dSdt')
L_dIdt = sn.rename((sn.diff(Is,ts)-C1*Beta*Is*Ss+C2*Is), 'L_dIdt')
L_dRdt = sn.rename((sn.diff(Rs,ts)-C2*Is), 'L_dRdt')


# In[ ]:


# 构建 split 第二阶段的物理约束模型
# 损失包含 L_ODE + L_IC，并额外保持固定 I 网络的一致性
loss_ode = [sn.PDE(L_dSdt),  sn.PDE(L_dIdt),  sn.PDE(L_dRdt), 
            sn.PDE(L_S0), sn.PDE(L_R0), 
            sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Is*0.0)]

m_ode = sn.SciModel(ts, loss_ode, loss_err, optimizer)


loss_train_ode = ['zeros']*8

epochs_ode = 1000
log_params   = {'parameters': Beta, 'freq':1}


# In[ ]:


# 训练 split 第二阶段模型，只使用 collocation 点施加物理约束
time1_ode = time.time()
h_ode     = m_ode.train(t_train_ode,
                        loss_train_ode,
                        epochs=epochs_ode,
                        batch_size=batch_size,
                        log_parameters=log_params,
                        adaptive_weights=adaptive_NTK,
                        verbose=1)
time2_ode = time.time()


# In[ ]:


print(f'Training time: {time2_ode-time1_ode}')


# In[ ]:


# 获取 split 模型在测试时间上的 S/I/R 预测结果
S_pred_test = Ss.eval(m_ode, t_test_sc)
I_pred_test = Is.eval(m_ode, t_test_sc)
R_pred_test = Rs.eval(m_ode, t_test_sc)

# 绘图
plt.plot(t_data, S_data, c='b',linewidth=4)
plt.plot(t_test,S_pred_test*C, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$S$', '$\hat{S}$'])
plt.show()

plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test*C, '--', c='k',linewidth=4)
if weekly: plt.scatter(t_data[::7],I_obs[::7], marker='x', c='m', s=100)
else: plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$I$','$\hat{I}$','samples'])
plt.show()

plt.plot(t_data, R_data, c='g',linewidth=4)
plt.plot(t_test,R_pred_test*C, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$R$', '$\hat{R}$'])
plt.show()


# In[ ]:


# 计算 split 模型的相对 L2 误差和 Beta 识别误差
S_pred = Ss.eval(m_ode, t_data_sc)*C
I_pred = Is.eval(m_ode, t_data_sc)*C
R_pred = Rs.eval(m_ode, t_data_sc)*C
beta_pred = Beta.eval(m_ode, t_data_sc)

S_err = np.linalg.norm(S_data-S_pred,2)/np.linalg.norm(S_data,2)
I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
R_err = np.linalg.norm(R_data-R_pred,2)/np.linalg.norm(R_data,2)
beta_err = abs(beta_pred[0]-beta)/beta

print(f'S error: {S_err:.3e}')
print(f'I error: {I_err:.3e}')
print(f'R error: {R_err:.3e}')
print(f'Beta error: {beta_err:.3e}')


# In[ ]:


# 保存 split 两个阶段的训练历史和模型权重
with open('hSplit_data.txt', 'wb') as myFile:
    pickle.dump(h_data.history, myFile)
myFile.close()

m_data.save_weights('mSplit_data.hdf5')

with open('hSplit_ode.txt', 'wb') as myFile:
    pickle.dump(h_ode.history, myFile)
myFile.close()

m_ode.save_weights('mSplit_ode.hdf5')


# In[ ]:


# 绘制 beta 估计值随 epoch 的变化，对比 joint 和 split 方法
plt.plot(h.history['Beta'], linewidth=2.5)
plt.plot(h_ode.history['Beta'],linewidth=2.5)
plt.legend(['Joint', 'Split'])
plt.ylabel(r'$\hat{\beta}_0$')
plt.xlabel('epochs')
plt.show()


# In[ ]:



