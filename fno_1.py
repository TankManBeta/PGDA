# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from fourier_1d import *
from util import *

ntrain = 1000
n_test = 100
batch_size = 100

f_train_h = generate(samples=ntrain,out_dim=2001)
f_test_h = generate(samples=n_test,out_dim=2001)
f0_h = torch.ones(1,2001)


learning_rate = 0.001
epochs = 500
step_size = 50
gamma = 0.5
modes = 16
width = 64
EP_list = [1/2**6, 1/2**8, 1/2**10]

for EP in EP_list:
    print("epsilon value : ",EP)
    u_train_h = FD_1(f_train_h,-EP,1,0,0,0)
    u_test_h = FD_1(f_test_h,-EP,1,0,0,0)
    u0_h = FD_1(f0_h,-EP,1,0,0,0)
    f_train = torch.Tensor(f_train_h[:,::40])
    u_train = torch.Tensor(u_train_h[:,::40])
    f_test = torch.Tensor(f_test_h[:,::40])
    u_test = torch.Tensor(u_test_h[:,::40])
    f0 = torch.Tensor(f0_h[:,::40])
    u0 = torch.Tensor(u0_h[:,::40])
    f_train = torch.reshape(f_train,(f_train.shape[0],f_train.shape[1],1))
    f_test = torch.reshape(f_test,(f_test.shape[0],f_test.shape[1],1))
    f0 = torch.reshape(f0, (f0.shape[0],f0.shape[1],1))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, u_train), batch_size=batch_size, shuffle=True)
    test_1_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, u_test), batch_size=1, shuffle=False)
    
    model = FNO1d(modes, width)
    print('Total parameters:',count_params(model))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start = default_timer()
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
        scheduler.step()
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        t2 = default_timer()
        print('\repoch {:d}/{:d} L2 = {:.6f}, MSE = {:.6f}, using {:.6f}s'.format(ep+1,epochs,train_l2,train_mse,t2-t1), end='', flush=True)
    
    print('Total training time:',default_timer()-start,'s')
    
# =============================================================================
#     pred = torch.zeros(u_test.shape)
#     index = 0
#     test_l2 = 0
#     test_mse = 0
#     with torch.no_grad():
#         for x, y in test_1_loader:
#             out = model(x).view(-1)
#             pred[index] = out
#             mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
#             test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#             test_mse += mse.item()
#             index += 1
#         test_mse /= len(test_1_loader)
#         test_l2 /= 100
#         print('test error: L2 =', test_l2,', MSE =',test_mse)
#     
#     residual = pred-u_test
#     fig = plt.figure()
#     x_grid = np.linspace(0, 1, 51)
#     for i in range(100):
#         plt.plot(x_grid,residual[i].detach().numpy())
#     plt.show()
# =============================================================================
    
    f_test_h0 = torch.reshape(torch.Tensor(f_test_h),(f_test_h.shape[0],f_test_h.shape[1],1))
    u_test_h = torch.Tensor(u_test_h)
    test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test_h0, u_test_h), batch_size=1, shuffle=False)
    
    pred_h = torch.zeros(u_test_h.shape)
    index = 0
    test_l2 = 0
    test_mse = 0
    with torch.no_grad():
        for x, y in test_h_loader:
            out = model(x).view(-1)
            pred_h[index] = out
            mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_mse += mse.item()
            index += 1
        test_mse /= len(test_h_loader)
        test_l2 /= 100
        print('test error on high resolution: L2 =', test_l2,', MSE =',test_mse)
    
    residual = pred_h-u_test_h
    fig = plt.figure()
    x_grid = np.linspace(0, 1, 2001)
    for i in range(n_test):
        plt.plot(x_grid,residual[i].detach().numpy())
    plt.show()
    
    f0_h0 = torch.reshape(f0_h,(1,2001,1))
    u0_pred = model(f0_h0).view(-1)
    u0 = torch.Tensor(u0_h).view(-1)
    fixed_l2 = 0
    for i in range(2001):
        fixed_l2 += (u0[i]-u0_pred[i])**2
    fixed_l2 /= 2000
    print('fixed L2 error on high resolution = ',fixed_l2)