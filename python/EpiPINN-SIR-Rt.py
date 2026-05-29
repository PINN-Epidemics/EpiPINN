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
# Note: The uploaded code is related to Case 2 and Case 3. For further information please contact the corresponding author.

# In[ ]:


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


case = 2
# 如需运行 Case 3，将上一行改为 case = 3


# In[ ]:


# 根据 case 选择对应的模拟数据文件
# Case 2 和 Case 3 都是 SIR-Rt 形式，区别在于 Rt 的时变设定
# ./Case2.txt ./Case3.txt里的数据是怎么来的？
if case==2:
    df_SIR = pd.read_table("./Case2.txt")
if case==3:
    df_SIR = pd.read_table("./Case3.txt")
df_SIR


# In[5]:


# SIR 模型参数
# N 表示总人口，delta 表示平均感染期的倒数
N     = 56e6 # (-) population (Italy)
delta = 1/5  # (1/T) 5 = mean reproduction period 


# In[6]:


# 时间域
# timespan 保留真实日期，用于绘图；t_data/t_test 使用从 0 开始的天数索引
timespan = df_SIR["Date"].values.astype('datetime64[D]')

t0 = 0.                 # (days) initial time
tf = len(timespan)      # (days) final time

t_data = np.arange(t0,tf)
t_test = np.arange(t0,tf,0.1)


# In[7]:


# 缩放因子
# C 用于缩放 S/I/R，C1 和 C2 是归一化 ODE 中出现的系数
C = 1e5
C1 = tf*C/N   
C2 = tf*delta


# In[8]:


# 初始条件
# 假设初始时刻只有 1 个感染者，其余人均为易感者
S0 = N-1
I0 = 1
R0 = 0


# In[9]:


# SIR 数据
# 读取参考 S/I/R、带噪声的感染观测 I_obs，以及时变再生数 Rt
S_data = df_SIR["Susceptible"].values
I_data = df_SIR["Infectious"].values
R_data = df_SIR["Recovered"].values
I_obs  = df_SIR["I data"].values
Rt_data = df_SIR["Rt"].values
beta_data = Rt_data*delta # beta(t)=delta*Rt(t)，用于后续和网络识别结果对比

# 缩放数据
# 感染观测 I_obs 同样缩放；时间 t_data 和 t_test 也缩放到 [0,1] 区间
I_obs_sc  = I_obs/C
I_data_sc = I_data/C
t_data_sc = t_data/tf
t_test_sc = t_test/tf


# In[10]:


# 绘图结果
# 先检查参考 S/R、感染观测和 beta(t) 的时间变化
fig, ax = plt.subplots(3, 1, figsize=(10,10))
plotS = ax[0].plot(timespan, S_data, 'b', label='Susceptible')
ax[0].tick_params(axis='y', labelcolor='b')
ax[0].set_xlabel('date')
ax[0].set_ylabel('individuals')
ax[0].set_title('Susceptible and Recovered')

ax2 = ax[0].twinx()

plotR = ax2.plot(timespan, R_data, 'g', label='Recovered')
ax2.set_ylabel('individuals')
ax2.tick_params(axis='y', labelcolor='g')

lplot = plotS+plotR
ax[0].legend(lplot,[l.get_label() for l in lplot], loc=6)

ax[1].plot(timespan, I_data, 'r', label='Infectious')
ax[1].plot(timespan, I_obs, 'xm', label='samples')
ax[1].legend(loc=6)
ax[1].set_xlabel('date')
ax[1].set_ylabel('individuals')
ax[1].set_title('Infectious and Observed Data')

ax[2].plot(timespan,beta_data, label=r"$\beta$")
ax[2].legend(loc=6)
ax[2].set_xlabel('date')
ax[2].set_title('Transimission rate')

fig.tight_layout()

plt.show()


# In[11]:


# 训练参数
# loss_err='mse' 指定均方误差损失，adaptive_NTK 用于动态平衡不同损失项
loss_err  = 'mse'
optimizer = 'adam'
adaptive_NTK = {'method':'NTK','freq':100}


# ### Joint

# In[12]:


sn.reset_session()


# In[13]:


# 构建神经网络 - joint 方法，同时学习 S、I 和时变 beta(t)
# Ss/Is 分别表示缩放后的 S_s_hat(ts)、I_s_hat(ts)
ts  = sn.Variable("ts")
Ss = sn.Functional("Ss", ts, 4*[50], output_activation='square')
Is = sn.Functional("Is", ts, 4*[50], output_activation='square')

# Beta 用函数网络表示 beta_s_hat(ts)，square 输出保证非负
# Beta 的网络结构更大一些，便于学习时变传播率的复杂变化
Beta = sn.Functional("Beta", ts, 4*[100], output_activation='square')

# 根据守恒关系 R_s = N/C - I_s - S_s 计算恢复者变量
Rs = N/C-Is-Ss


# In[14]:


# 初始条件约束
# 使用 sign 构造只在 t=t0 附近起作用的初值残差
L_S0 = sn.rename((Ss-S0/C)*(1-sn.sign(ts-t0/tf)), "L_S0")
L_I0 = sn.rename((Is-I0/C)*(1-sn.sign(ts-t0/tf)), "L_I0")
L_R0 = sn.rename((Rs-R0/C)*(1-sn.sign(ts-t0/tf)), "L_R0")

# 常微分方程残差
# 对缩放后的 S/I/R 施加 SIR-Rt 物理约束
L_dSdt = sn.rename((sn.diff(Ss,ts)+C1*Beta*Is*Ss), "L_dSdt")
L_dIdt = sn.rename((sn.diff(Is,ts)-C1*Beta*Is*Ss+C2*Is), "L_dIdt")
L_dRdt = sn.rename((sn.diff(Rs,ts)-C2*Is), "L_dRdt")


# In[15]:


# 构建 joint 损失函数：ODE 残差、初始条件、beta 正则项和 I_obs 数据项
loss_joint = [sn.PDE(L_dSdt),  sn.PDE(L_dIdt),  sn.PDE(L_dRdt), 
              sn.PDE(L_S0),    sn.PDE(L_I0),    sn.PDE(L_R0),
              sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Beta*0.0),
              sn.Data(Is)]


# In[16]:


# 将时间变量和损失项封装为 SciANN 模型
m = sn.SciModel(ts, loss_joint, loss_err, optimizer)


# In[17]:


# 训练点
# t_data_sc 对应观测数据点，t_train_ode 是随机 collocation 点
Nc = 6000    # collocation points

I_obs_sc = I_obs_sc.reshape(-1,1)
t_train_ode = np.random.uniform(t0/tf, 1., Nc-1)
t_train_ode = np.insert(t_train_ode,0,0.0)
t_train = np.concatenate([t_data_sc.reshape(-1,1), t_train_ode.reshape(-1,1)])
ids_data = np.arange(t_data_sc.size, dtype=np.intp)


# In[18]:


# 前 9 个损失项目标为 0，最后一个数据项使用 I_obs_sc
loss_train   = ['zeros']*9+[(ids_data,I_obs_sc)]
epochs_joint = 5000
batch_size   = 100


# In[ ]:


# 训练 joint 模型，并记录训练耗时
time1 = time.time()
h     = m.train(t_train,
                loss_train,
                epochs=epochs_joint,
                batch_size=batch_size,
                adaptive_weights=adaptive_NTK,
                verbose=1
               )
time2 = time.time()


# In[20]:


print(f'Training time: {time2-time1}')


# In[21]:


# 获取 joint 模型预测结果
# S/I/R 预测值乘以 C 还原到人数尺度；Beta 保持传播率尺度
S_pred_test = Ss.eval(m, t_test_sc)*C
I_pred_test = Is.eval(m, t_test_sc)*C
R_pred_test = Rs.eval(m, t_test_sc)*C
beta_pred_test = Beta.eval(m, t_test_sc)

# 绘图结果
# 将网络预测与参考数据、观测样本逐项对比
plt.plot(t_data, S_data, c='b',linewidth=4)
plt.plot(t_test,S_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$S$', r'$\hat{S}$'])
plt.show()

plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$','samples'])
plt.show()

plt.plot(t_data, R_data, c='g',linewidth=4)
plt.plot(t_test,R_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$R$', r'$\hat{R}$'])
plt.show()

plt.plot(t_data, beta_data, linewidth=4)
plt.plot(t_test, beta_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\beta$', r'$\hat{\beta}$'])
plt.show()


# In[22]:


# 计算误差
# 使用相对 L2 误差评估 S/I/R 和 beta(t) 的拟合质量
S_pred = Ss.eval(m, t_data_sc)*C
I_pred = Is.eval(m, t_data_sc)*C
R_pred = Rs.eval(m, t_data_sc)*C
beta_pred = Beta.eval(m, t_data_sc)

S_err = np.linalg.norm(S_data-S _pred,2)/np.linalg.norm(S_data,2)
I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
R_err = np.linalg.norm(R_data-R_pred,2)/np.linalg.norm(R_data,2)
beta_err = np.linalg.norm(beta_data-beta_pred,2)/np.linalg.norm(beta_data,2)
# beta_err70 忽略前 20 天，单独评估后 70 天的传播率识别误差
beta_err70 = np.linalg.norm(beta_data[20:]-beta_pred[20:],2)/np.linalg.norm(beta_data[20:],2)


print(f'S error: {S_err:.3e}')
print(f'I error: {I_err:.3e}')
print(f'R error: {R_err:.3e}')
print(f'Beta error: {beta_err:.3e}')
print(f'Beta error last 70 days: {beta_err70:.3e}')


# ### Split

# In[23]:


sn.reset_session()


# In[24]:


# 构建神经网络 - split 第一阶段，仅用观测数据回归 I(t)
# Isc 表示缩放后的感染人数 I_s_hat(ts)
ts  = sn.Variable("ts")
Isc = sn.Functional("Isc", ts, 4*[50], output_activation='square')


# In[25]:


# 构建 split 第一阶段数据回归模型
# 损失只包含 I_s 与 I_obs_sc 的数据误差
loss_data = sn.Data(Isc)

m_data = sn.SciModel(ts, loss_data, loss_err, optimizer)


# In[26]:


# 设置 split 第一阶段训练参数
t_data_train = t_data_sc
epochs_data  = 3000
batch_data   = 10


# In[ ]:


# 训练模型 - split 第一阶段，仅做数据回归
time1_data = time.time()
h_data     = m_data.train(t_data_train, 
                          I_obs_sc, 
                          epochs=epochs_data,
                          batch_size=batch_data,
                          verbose=1)
time2_data = time.time()


# In[28]:


print(f'Training time: {time2_data-time1_data}')


# In[29]:


# 获取 split 第一阶段 I(t) 预测结果
Isc_pred = Isc.eval(m_data, t_test_sc)

# 绘图结果
# 对比回归得到的 I(t)、参考 I_data 和观测样本
plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,Isc_pred*C, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend(['$I$','$\hat{I}$','samples'])
plt.show()


# In[30]:


# 计算误差
# 评估第一阶段 I_s 回归结果和参考 I_data_sc 的相对偏差
Isc_pred = Isc.eval(m_data,t_data_sc)
Isc_err = np.linalg.norm(I_data_sc-Isc_pred,2)/np.linalg.norm(I_data_sc,2)
print(f'Isc error: {Isc_err:.3e}')


# In[31]:


# 固定 I 网络的权重
# 第二阶段将第一阶段学到的 I(t) 作为已知函数，不再训练
Isc_weights = Isc.get_weights()

Is = sn.Functional("Is", ts, 4*[50], output_activation='square', trainable=False)
Is.set_weights(Isc_weights)


# In[32]:


# 构建神经网络 - split 第二阶段，进入物理约束训练
# 在固定 I 的基础上继续学习 S(t)、R(t) 和时变 beta(t)
Ss = sn.Functional("Ss", ts, 4*[50], output_activation='square')

Beta = sn.Functional("Beta", ts, 4*[100], output_activation='square')

Rs = N/C-Is-Ss


# In[33]:


# 初始条件约束
# I 已冻结，因此这里只显式约束 S 和 R 的初值
L_S0 = sn.rename((Ss-S0/C)*(1-sn.sign(ts-t0/tf)), "L_S0")
L_R0 = sn.rename((Rs-R0/C)*(1-sn.sign(ts-t0/tf)), "L_R0")

# 常微分方程残差
# 使用固定 I 网络和待学习的 S、Beta 共同满足 SIR-Rt 方程
L_dSdt = sn.rename((sn.diff(Ss,ts)+C1*Beta*Is*Ss), "L_dSdt")
L_dIdt = sn.rename((sn.diff(Is,ts)-C1*Beta*Is*Ss+C2*Is), "L_dIdt")
L_dRdt = sn.rename((sn.diff(Rs,ts)-C2*Is), "L_dRdt")


# In[34]:


# 构建模型 - split 第二阶段物理约束训练
# 损失包含 ODE 残差、初始条件、beta 正则项和固定 I 的一致性约束
loss_ode = [sn.PDE(L_dSdt),  sn.PDE(L_dIdt),  sn.PDE(L_dRdt), 
            sn.PDE(L_S0), sn.PDE(L_R0), 
            sn.Data(Ss*0.0), sn.Data(Rs*0.0), sn.Data(Beta*0.0),
            sn.Data(Is*0.0)]

m_ode = sn.SciModel(ts, loss_ode, loss_err, optimizer)

loss_train_ode = ['zeros']*9

epochs_ode = 1000


# In[ ]:


# 训练模型 - split 第二阶段，物理约束训练
time1_ode = time.time()
h_ode     = m_ode.train(t_train_ode,
                        loss_train_ode,
                        epochs=epochs_ode,
                        batch_size=batch_size,
                        adaptive_weights=adaptive_NTK,
                        verbose=1)
time2_ode = time.time()


# In[36]:


print(f'Training time: {time2_ode-time1_ode}')


# In[37]:


# 获取 split 第二阶段预测结果
S_pred_test = Ss.eval(m_ode, t_test_sc)*C
I_pred_test = Is.eval(m_ode, t_test_sc)*C
R_pred_test = Rs.eval(m_ode, t_test_sc)*C
beta_pred_test = Beta.eval(m_ode, t_test_sc)

# 绘图结果
# 对比 split 方法得到的 S/I/R/beta 与参考数据
plt.plot(t_data, S_data, c='b',linewidth=4)
plt.plot(t_test,S_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$S$', r'$\hat{S}$'])
plt.show()

plt.plot(t_data,I_data, c='r', linewidth=4)
plt.plot(t_test,I_pred_test, '--', c='k', linewidth=4)
plt.scatter(t_data,I_obs, marker='x', c='m', s=100)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$I$',r'$\hat{I}$',r'samples'])
plt.show()

plt.plot(t_data, R_data, c='g',linewidth=4)
plt.plot(t_test,R_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.ylabel('individuals')
plt.legend([r'$R$', r'$\hat{R}$'])
plt.show()

plt.plot(t_data, beta_data, linewidth=4)
plt.plot(t_test, beta_pred_test, '--', c='k',linewidth=4)
plt.xlabel('days')
plt.legend([r'$\beta$', r'$\hat{\beta}$'])
plt.show()


# In[38]:


# 计算误差
# 统计 split 方法下 S/I/R 和 beta(t) 的相对 L2 误差
S_pred = Ss.eval(m_ode, t_data_sc)*C
I_pred = Is.eval(m_ode, t_data_sc)*C
R_pred = Rs.eval(m_ode, t_data_sc)*C
beta_pred = Beta.eval(m_ode, t_data_sc)

S_err = np.linalg.norm(S_data-S_pred,2)/np.linalg.norm(S_data,2)
I_err = np.linalg.norm(I_data-I_pred,2)/np.linalg.norm(I_data,2)
R_err = np.linalg.norm(R_data-R_pred,2)/np.linalg.norm(R_data,2)
beta_err = np.linalg.norm(beta_data-beta_pred,2)/np.linalg.norm(beta_data,2)
# beta_err70 忽略前 20 天，关注后 70 天的 beta 识别效果
beta_err70 = np.linalg.norm(beta_data[20:]-beta_pred[20:],2)/np.linalg.norm(beta_data[20:],2)

print(f'S error: {S_err:.3e}')
print(f'I error: {I_err:.3e}')
print(f'R error: {R_err:.3e}')
print(f'Beta error: {beta_err:.3e}')
print(f'Beta error last 70 days: {beta_err70:.3e}')


# In[ ]:



