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
# Note: The uploaded code is related to Case 4 and Case 5. For further information please contact the corresponding author.

# In[ ]:


import sciann as sn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# ## Case 4: Synthetic $\mathcal{R}_t$ with large data errors

# In[2]:


# 读取 Case 4 合成数据：包含感染人数、住院增量、时变再生数 Rt 和住院率 sigma。
df_SIR = pd.read_table("./Case4.txt")
df_SIR


# In[3]:


# SIR 模型参数。
N     = 56e6 # (-) population (Italy)
delta = 1/5  # (1/T) 5 = mean reproduction period 


# In[4]:


# 时间域：把日期转换为从 0 开始的天数，并构造更密的测试网格用于画平滑曲线。
timespan = df_SIR["Date"].values.astype('datetime64[D]')

t0 = 0.                 # (days) initial time
tf = len(timespan)      # (days) final time

t_data = np.arange(t0,tf)
t_test = np.arange(t0,tf,0.1)


# In[5]:


# 缩放因子：分别缩放感染人数和住院增量，避免不同量级的数据主导损失。
SI = 1e5
SH = 1e3

C = SI*delta/SH


# In[6]:


# 感染数据、观测感染样本、真实 Rt 和真实 sigma。
I_data = df_SIR["Infectious"].values
I_obs  = df_SIR["I data"].values
Rt_data = df_SIR["Rt"].values
sigma_data = df_SIR["sigma"].values

# 住院增量数据：由 delta * sigma * I 生成真实值，并读取带误差的观测样本。
H_data = (delta*sigma_data*I_data).reshape(-1,)
H_obs = df_SIR["H data"].values

I_obs_sc  = I_obs/SI
I_data_sc = I_data/SI
H_obs_sc  = H_obs/SH
H_data_sc = H_data/SH
t_data_sc = t_data/tf
t_test_sc = t_test/tf


# In[7]:


# 绘制原始数据：检查 I、Delta_H、Rt 和 sigma 的时间变化及观测误差。
fig, ax = plt.subplots(4, 1, figsize=(10,12))

ax[0].plot(timespan, I_data, 'r', label='Infectious')
ax[0].plot(timespan, I_obs, 'xm', label='samples')
ax[0].legend(loc=6)
ax[0].set_xlabel('date')
ax[0].set_ylabel('individuals')
ax[0].set_title('Infectious and Observed Data')

ax[1].plot(timespan, H_data, 'b', label='Hospitalizations')
ax[1].plot(timespan, H_obs, 'xm', label='samples')
ax[1].legend(loc=6)
ax[1].set_xlabel('date')
ax[1].set_ylabel('individuals')
ax[1].set_title('Hospitalizations and Observed Data')

ax[2].plot(timespan, Rt_data, label=r"$\mathcal{R}_t$")
ax[2].legend(loc=6)
ax[2].set_xlabel('date')
ax[2].set_title('Reproduction number')

ax[3].plot(timespan,sigma_data, 'orange', label=r"$\sigma$")
ax[3].legend(loc=6)
ax[3].set_xlabel('date')
ax[3].set_title('Hospitalization rate')

fig.tight_layout()

plt.show()


# In[8]:


# 训练参数：所有阶段均使用均方误差和 Adam 优化器。
loss_err  = 'mse'
optimizer = 'adam'


# ### Joint

# In[9]:


sn.reset_session()


# In[10]:


# 设置随机种子，便于复现实验结果。
sn.set_random_seed(34)


# In[11]:


# 构建简化模型的联合训练网络：同时学习感染人数 I 和时变再生数 Rt。
ts  = sn.Variable("ts")
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')

Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')


# In[12]:


# 简化 SIR 约束：dI/dt = delta * (Rt - 1) * I，时间变量已缩放到 [0, 1]，因此乘以 tf。
L_dIdt = sn.rename((sn.diff(Is,ts)-tf*delta*(Rt-1)*Is), "L_dIdt")


# In[13]:


# 联合损失：同时约束 ODE 残差、Rt 的正则项和感染观测数据。
loss_joint = [sn.PDE(L_dIdt), sn.Data(Rt*0.0), sn.Data(Is)]


# In[14]:


m = sn.SciModel(ts, loss_joint, loss_err, optimizer)


# In[15]:


# 训练点：感染观测点用于数据项，随机配点用于 ODE 物理约束，并强制包含初始时刻。
Nc = 6000    # collocation points

I_obs_sc = I_obs_sc.reshape(-1,1)
t_train_ode = np.random.uniform(t0/tf, 1., Nc-1)
t_train_ode = np.insert(t_train_ode,0,0.0)
t_train = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
ids_data = np.arange(t_data_sc.size, dtype=np.intp)


# In[16]:


# 前两个损失目标为 0，最后一项只在观测时间点拟合 I_obs。
loss_train   = ['zeros']*2+[(ids_data,I_obs_sc)]
epochs_joint = 5000
batch_size   = 100


# In[ ]:


# 训练简化模型的联合 PINN。
time1 = time.time()
h     = m.train(t_train,
                loss_train,
                epochs=epochs_joint,
                batch_size=batch_size,
                verbose=1
               )
time2 = time.time()


# In[18]:


print(f'Training time: {time2-time1}')


# In[19]:


# 获取联合训练后的 I 和 Rt 预测结果，并恢复 I 的原始量纲。
I_pred_test = Is.eval(m, t_test_sc)*SI
Rt_pred_test = Rt.eval(m, t_test_sc)

# 绘图结果：比较真实曲线、PINN 预测和带噪声样本。
plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$','samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()


# In[20]:


# 计算相对 L2 误差；Rt_err100 只评估后 100 天，避开初期快速变化段。
I_pred = Is.eval(m, t_data_sc)*SI
Rt_pred = Rt.eval(m, t_data_sc)

I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)
Rt_err100 = np.linalg.norm(Rt_data[20:]-Rt_pred[20:],2)/np.linalg.norm(Rt_data[20:],2)


print(f'I error: {I_err:.3e}')
print(f'Rt error: {Rt_err:.3e}')
print(f'Rt error last 100 days: {Rt_err100:.3e}')


# ### Split

# In[21]:


sn.reset_session()


# In[22]:


# 设置随机种子，便于复现实验结果。
sn.set_random_seed(34)


# In[23]:


# Split 第一步：只用感染观测样本训练 Isc，先得到平滑的 I(t) 代理模型。
ts  = sn.Variable("ts")
Isc = sn.Functional("Isc", ts, 4*[50], output_activation='square')


# In[24]:


# 构建数据回归模型：该阶段不加入 ODE，仅拟合缩放后的 I_obs。
loss_data = sn.Data(Isc)

m_data = sn.SciModel(ts, loss_data, loss_err, optimizer)


# In[25]:


t_data_train = t_data_sc
epochs_data  = 1000
batch_data   = 10


# In[ ]:


# 训练 split 的数据回归阶段。
time1_data = time.time()
h_data     = m_data.train(t_data_train, 
                          I_obs_sc, 
                          epochs=epochs_data,
                          batch_size=batch_data,
                          verbose=1)
time2_data = time.time()


# In[27]:


print(f'Training time: {time2_data-time1_data}')


# In[28]:


# 获取 Isc 的预测结果，用于检查纯数据回归对感染曲线的拟合情况。
Isc_pred = Isc.eval(m_data, t_test_sc)

# 绘图结果：将 Isc 恢复到原始感染人数尺度后与真实值和观测样本对比。
plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,Isc_pred*SI, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$I$','$\hat{I}$','samples'])
plt.show()


# In[29]:


# 计算 Isc 相对 L2 误差，评估第一阶段 I(t) 代理模型质量。
Isc_pred = Isc.eval(m_data,t_data_sc)
Isc_err = np.linalg.norm(I_data_sc-Isc_pred,2)/np.linalg.norm(I_data_sc,2)
print(f'Isc error: {Isc_err:.3e}')


# In[30]:


# 固定 I 网络权重：把第一阶段学到的 Isc 复制为不可训练的 Is，第二阶段只反演 Rt。
Isc_weights = Isc.get_weights()

Is = sn.Functional("Is", ts, 4*[50], output_activation='square', trainable=False)
Is.set_weights(Isc_weights)


# In[31]:


# Split 第二步：在固定 I(t) 的基础上学习 Rt，使其满足简化感染方程。
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')


# In[32]:


# 简化 SIR 约束：由固定的 I(t) 和待学习的 Rt(t) 构成 ODE 残差。
L_dIdt = sn.rename((sn.diff(Is,ts)-tf*delta*(Rt-1)*Is), "L_dIdt")


# In[33]:


# 构建物理约束模型：损失包含 ODE 残差、Rt 正则项和固定 I 的一致性项。
loss_ode = [sn.PDE(L_dIdt), sn.Data(Rt*0.0), sn.Data(Is*0.0)]

m_ode = sn.SciModel(ts, loss_ode, loss_err, optimizer)

# 三个目标均为 0，表示只通过物理残差和正则项约束第二阶段训练。
loss_train_ode = ['zeros']*3

epochs_ode = 3000


# In[ ]:


# 训练 split 的物理约束阶段。
time1_ode = time.time()
h_ode     = m_ode.train(t_train_ode,
                        loss_train_ode,
                        epochs=epochs_ode,
                        batch_size=batch_size,
                        verbose=1)
time2_ode = time.time()


# In[35]:


print(f'Training time: {time2_ode-time1_ode}')


# In[36]:


# 获取 split 训练后的 I 和 Rt 预测结果；I 来自固定网络，Rt 来自第二阶段反演。
I_pred_test = Is.eval(m_ode, t_test_sc)*SI
Rt_pred_test = Rt.eval(m_ode, t_test_sc)

# 绘图结果：检查 split 方法下 I(t) 的保持效果和 Rt(t) 的反演效果。
plt.plot(t_data, I_data, c='r', linewidth=4)
plt.plot(t_test, I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data, I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$',r'samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()


# In[37]:


# 计算 split 简化模型误差；Rt_err100 只评估后 100 天。
I_pred = Is.eval(m_ode, t_data_sc)*SI
Rt_pred = Rt.eval(m_ode, t_data_sc)

I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)
Rt_err100 = np.linalg.norm(Rt_data[20:]-Rt_pred[20:],2)/np.linalg.norm(Rt_data[20:],2)

print(f'I error: {I_err:.3e}')
print(f'Rt error: {Rt_err:.3e}')
print(f'Rt error last 100 days: {Rt_err100:.3e}')


# ## Case 5: synthetic data of infections and hospitalizations

# ### Joint

# In[38]:


# 设置随机种子，便于复现实验结果。
sn.set_random_seed(34)


# In[39]:


sn.reset_session()


# In[40]:


# 构建扩展模型的联合训练网络：同时学习 I、Rt 和 sigma，并通过 sigma * I 得到住院增量。
ts  = sn.Variable("ts")
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')

Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')
sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')


# In[41]:


# 扩展模型约束：感染方程仍由 Rt 控制，住院增量 Delta_H 由 sigma * I 给出。
L_dIdt = sn.rename((sn.diff(Is,ts)-tf*delta*(Rt-1)*Is), "L_dIdt")
deltaHs = sn.rename(sigma*Is*C, "deltaHs")


# In[42]:


# 联合损失：约束 ODE 和 Rt 正则项，并同时拟合感染观测与住院增量观测。
loss_joint = [sn.PDE(L_dIdt), sn.Data(Rt*0.0), 
              sn.Data(Is), sn.Data(deltaHs)]


# In[43]:


m = sn.SciModel(ts, loss_joint, loss_err, optimizer)


# In[44]:


# 训练点：I 和 Delta_H 的观测点共享同一组日期，ODE 使用随机配点。
Nc = 6000    # collocation points

I_obs_sc = I_obs_sc.reshape(-1,1)
H_obs_sc = H_obs_sc.reshape(-1,1)
t_train_ode = np.random.uniform(t0/tf, 1., Nc-1)
t_train_ode = np.insert(t_train_ode,0,0.0)
t_train = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
ids_data = np.arange(t_data_sc.size, dtype=np.intp)


# In[45]:


# 前两项为物理/正则零目标，后两项在观测日期分别拟合 I_obs 和 H_obs。
loss_train   = ['zeros']*2+[(ids_data,I_obs_sc),(ids_data,H_obs_sc)]
epochs_joint = 5000
batch_size   = 100


# In[ ]:


# 训练扩展模型的联合 PINN。
time1 = time.time()
h     = m.train(t_train,
                loss_train,
                epochs=epochs_joint,
                batch_size=batch_size,
                verbose=1
               )
time2 = time.time()


# In[47]:


print(f'Training time: {time2-time1}')


# In[48]:


# 获取联合训练后的 I、Delta_H、Rt 和 sigma 预测，并恢复 I 与 Delta_H 的原始量纲。
I_pred_test = Is.eval(m, t_test_sc)*SI
deltaH_pred_test = deltaHs.eval(m, t_test_sc)*SH
Rt_pred_test = Rt.eval(m, t_test_sc)
sigma_pred_test = sigma.eval(m, t_test_sc)

# 绘图结果：分别比较感染、住院增量、Rt 和 sigma 的真实值与预测值。
plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$','samples'])
plt.show()

plt.plot(t_data,H_data, c='b', linewidth=4)
plt.plot(t_test,deltaH_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\Delta_H$',r'$\hat{\Delta}_H$','samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()

plt.plot(t_data, sigma_data, c='orange', linewidth=4)
plt.plot(t_test, sigma_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\sigma$', r'$\hat{\sigma}$'])
plt.show()


# In[49]:


# 计算扩展联合模型的相对 L2 误差；Rt 和 sigma 额外报告后 100 天误差。
I_pred = Is.eval(m, t_data_sc)*SI
deltaH_pred = deltaHs.eval(m, t_data_sc)*SH
Rt_pred = Rt.eval(m, t_data_sc)
sigma_pred = sigma.eval(m, t_data_sc)

I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
deltaH_err = np.linalg.norm(H_data-deltaH_pred,2)/np.linalg.norm(H_data,2)
Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)
Rt_err100 = np.linalg.norm(Rt_data[20:]-Rt_pred[20:],2)/np.linalg.norm(Rt_data[20:],2)
sigma_err = np.linalg.norm(sigma_data-sigma_pred,2)/np.linalg.norm(sigma_data,2)
sigma_err100 = np.linalg.norm(sigma_data[20:]-sigma_pred[20:],2)/np.linalg.norm(sigma_data[20:],2)


print(f'I error: {I_err:.3e}')
print(f'DeltaH error: {deltaH_err:.3e}')
print(f'Rt error: {Rt_err:.3e}')
print(f'Rt error last 100 days: {Rt_err100:.3e}')
print(f'sigma error: {sigma_err:.3e}')
print(f'sigma error last 100 days: {sigma_err100:.3e}')


# ### Split

# In[50]:


sn.reset_session()


# In[51]:


# 设置随机种子，便于复现实验结果。
sn.set_random_seed(34)


# In[52]:


# Split 第一步：只用住院增量观测样本训练 Hsc，先得到平滑的 Delta_H(t) 代理模型。
ts  = sn.Variable("ts")
Hsc = sn.Functional("Hsc", ts, 4*[100], output_activation='square')


# In[53]:


# 构建数据回归模型：该阶段不加入 ODE，仅拟合缩放后的 H_obs。
loss_data = sn.Data(Hsc)

m_data = sn.SciModel(ts, loss_data, loss_err, optimizer)


# In[54]:


t_data_train = t_data_sc
epochs_data  = 3000
batch_data   = 10


# In[ ]:


# 训练 split 的住院增量数据回归阶段。
time1_data = time.time()
h_data     = m_data.train(t_data_train, 
                          H_obs_sc, 
                          epochs=epochs_data,
                          batch_size=batch_data,
                          verbose=1)
time2_data = time.time()


# In[56]:


print(f'Training time: {time2_data-time1_data}')


# In[57]:


# 获取 Hsc 的预测结果，用于检查纯数据回归对住院增量曲线的拟合情况。
Hsc_pred = Hsc.eval(m_data, t_test_sc)

# 绘图结果：将 Hsc 恢复到原始住院增量尺度后与真实值和观测样本对比。
plt.plot(t_data,H_data, c='b', linewidth=4)
plt.plot(t_test,Hsc_pred*SH, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$\Delta_H$','$\hat{\Delta}_H$','samples'])
plt.show()


# In[58]:


# 计算 Hsc 相对 L2 误差，评估第一阶段 Delta_H(t) 代理模型质量。
Hsc_pred = Hsc.eval(m_data,t_data_sc)
Hsc_err = np.linalg.norm(H_data_sc-Hsc_pred,2)/np.linalg.norm(H_data_sc,2)
print(f'Hsc error: {Hsc_err:.3e}')


# In[60]:


# 固定 H 网络权重：把第一阶段学到的 Hsc 复制为不可训练的 deltaHs。
Hsc_weights = Hsc.get_weights()

deltaHs = sn.Functional("deltaHs", ts, 4*[100], output_activation='square', trainable=False)
deltaHs.set_weights(Hsc_weights)


# In[61]:


# Split 第二步：在固定 Delta_H(t) 的基础上学习 Rt 和 sigma，并由 Delta_H 与 sigma 重构 I。
Rt = sn.Functional("Rt", ts, 4*[100], output_activation='square')
sigma = sn.Functional("sigma", ts, 10*[5], output_activation='square')

# 由 Delta_H = C * sigma * I 得到 I = deltaHs * sigma / C；这里的 sigma 网络实际对应倒数关系，后处理中再取倒数。
c=1/C
Is = sn.rename(c*deltaHs*sigma, "Is")


# In[62]:


# 物理约束：将重构出的 I(t) 代入简化感染方程，反演 Rt(t) 与 sigma(t)。
L_dIdt = sn.rename((sn.diff(Is,ts)-tf*delta*(Rt-1)*Is), "L_dIdt")


# In[63]:


# 构建物理约束模型：约束 ODE、Rt/sigma 正则项，并继续在观测点匹配感染数据。
loss_ode = [sn.PDE(L_dIdt), sn.Data(Rt*0.0), sn.Data(sigma*0.0),# sn.Data(deltaHs*0.0),
            sn.Data(Is)]

m_ode = sn.SciModel(ts, loss_ode, loss_err, optimizer)

# 前三项为零目标，最后一项只在观测日期拟合 I_obs。
loss_train_ode = ['zeros']*3+[(ids_data,I_obs_sc)]

epochs_ode = 1000


# In[ ]:


# 训练 split 的扩展物理约束阶段。
time1_ode = time.time()
h_ode     = m_ode.train(t_train,
                        loss_train_ode,
                        epochs=epochs_ode,
                        batch_size=batch_size,
                        verbose=1)
time2_ode = time.time()


# In[65]:


print(f'Training time: {time2_ode-time1_ode}')


# In[66]:


# 获取 split 扩展模型预测结果；sigma 网络输出先取倒数再与真实 sigma 对比。
I_pred_test = Is.eval(m_ode, t_test_sc)*SI
Rt_pred_test = Rt.eval(m_ode, t_test_sc)
sigma_pred_test = sigma.eval(m_ode, t_test_sc)
# 取倒数还原为论文定义下的 sigma。
sigma_pred_test = 1/sigma_pred_test
deltaH_pred_test = deltaHs.eval(m_ode, t_test_sc)*SH


# In[67]:


# 绘图结果：检查 I、Delta_H、Rt 和 sigma 的 split 预测效果。
plt.plot(t_data, I_data, c='r', linewidth=4)
plt.plot(t_test, I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data, I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$',r'samples'])
plt.show()

plt.plot(t_data,H_data, c='b', linewidth=4)
plt.plot(t_test,deltaH_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,H_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$\Delta_H$',r'$\hat{\Delta}_H$','samples'])
plt.show()

plt.plot(t_data, Rt_data, linewidth=4)
plt.plot(t_test, Rt_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\mathcal{R}_t$', r'$\hat{\mathcal{R}}_t$'])
plt.show()

plt.plot(t_data, sigma_data, c='orange', linewidth=4)
plt.plot(t_test, sigma_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\sigma$', r'$\hat{\sigma}$'])
plt.show()


# In[68]:


# 计算 split 扩展模型误差；sigma 同样先取倒数再计算相对 L2 误差。
I_pred = Is.eval(m_ode, t_data_sc)*SI
Rt_pred = Rt.eval(m_ode, t_data_sc)
sigma_pred = sigma.eval(m_ode, t_data_sc)
# 取倒数还原为论文定义下的 sigma。
sigma_pred = 1/sigma_pred
deltaH_pred = deltaHs.eval(m_ode, t_data_sc)*SH

I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
deltaH_err = np.linalg.norm(H_data-deltaH_pred,2)/np.linalg.norm(H_data,2)
Rt_err = np.linalg.norm(Rt_data-Rt_pred,2)/np.linalg.norm(Rt_data,2)
Rt_err100 = np.linalg.norm(Rt_data[20:]-Rt_pred[20:],2)/np.linalg.norm(Rt_data[20:],2)
sigma_err = np.linalg.norm(sigma_data-sigma_pred,2)/np.linalg.norm(sigma_data,2)
sigma_err100 = np.linalg.norm(sigma_data[20:]-sigma_pred[20:],2)/np.linalg.norm(sigma_data[20:],2)


print(f'I error: {I_err:.3e}')
print(f'DeltaH error: {deltaH_err:.3e}')
print(f'Rt error: {Rt_err:.3e}')
print(f'Rt error last 100 days: {Rt_err100:.3e}')
print(f'sigma error: {sigma_err:.3e}')
print(f'sigma error last 100 days: {sigma_err100:.3e}')


# In[ ]:



