"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""

import os
os.environ["DDE_BACKEND"] = "pytorch"   # 必须在 import deepxde 之前

import deepxde as dde
import numpy as np
import torch


print("=" * 60)
print("Environment check")
print("=" * 60)
print("DeepXDE backend:", dde.backend.backend_name)
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. The model will run on CPU.")


def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return x * np.sin(5 * x)


geom = dde.geometry.Interval(-1, 1)
num_train = 16
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [100] * 4 + [1], activation, initializer)

model = dde.Model(data, net)

# 如果是 PyTorch 后端，可以检查网络参数在哪个设备上
if dde.backend.backend_name == "pytorch":
    print("Network device before compile:", next(model.net.parameters()).device)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])

if dde.backend.backend_name == "pytorch":
    print("Network device after compile:", next(model.net.parameters()).device)

losshistory, train_state = model.train(iterations=10000)

if dde.backend.backend_name == "pytorch":
    print("Network device after train:", next(model.net.parameters()).device)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)