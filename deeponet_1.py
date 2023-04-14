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
from deeponet import *
from util import *

ntrain = 1000
ntest = 100
#batch_size = 100
learning_rate = 0.001
epochs = 200
step_size = 50
gamma = 0.5
modes = 16
width = 64
EP_list = [1/2**6, 1/2**8, 1/2**10]

f_train_h = generate(samples=ntrain,out_dim=2001)
f_test_h = generate(samples=ntest,out_dim=2001)

for EP in EP_list:
    print("epsilon value : ",EP)
    u_train_h = FD_1(f_train_h,-EP,1,0,0,0)
    u_test_h = FD_1(f_test_h,-EP,1,0,0,0)
    
    f_train = torch.Tensor(f_train_h[:,::40])
    u_train = torch.Tensor(u_train_h[:,::40])
    f_test = torch.Tensor(f_test_h[:,::40])
    u_test = torch.Tensor(u_test_h[:,::40])
    
    dim = f_train.shape[-1]
    grid = np.linspace(0, 1, dim)
    N = f_train.shape[0]*dim
    loc = np.zeros((N,1))
    res = np.zeros((N,1))
    f = np.zeros((N,dim))
    for i in range(N):
        f[i] = f_train[i//dim]
        loc[i,0] = grid[i%dim]
        res[i,0] = u_train[i//dim,i%dim]
        
    f_train = torch.Tensor(f)
    loc_train = torch.Tensor(loc)
    res_train = torch.Tensor(res)    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, loc_train, res_train), batch_size=dim, shuffle=True)
    
    dim = f_test.shape[-1]
    grid = np.linspace(0, 1, dim)
    N = f_test.shape[0]*dim
    loc = np.zeros((N,1))
    res = np.zeros((N,1))
    f = np.zeros((N,dim))
    for i in range(N):
        f[i] = f_test[i//dim]
        loc[i,0] = grid[i%dim]
        res[i,0] = u_test[i//dim,i%dim]
    f_test = torch.Tensor(f)
    loc = torch.Tensor(loc)
    res = torch.Tensor(res)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, loc, res), batch_size=dim, shuffle=False)
    
    model = DeepONet(dim,1)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start = default_timer()
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        for x, l, y in train_loader:
            optimizer.zero_grad()
            out = model(x,l)
            mse = F.mse_loss(out.view(dim, -1), y.view(dim, -1), reduction='mean')
            mse.backward()
            optimizer.step()
            train_mse += mse.item()
        scheduler.step()
        train_mse /= len(train_loader)
        t2 = default_timer()
        print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep+1,epochs,train_mse,t2-t1), end='', flush=True)
        
    print('Total training time:',default_timer()-start,'s')
        
    test_mse = 0
    test_l2 = 0
    with torch.no_grad():
        for x, l, y in test_loader:
            out = model(x,l).view(-1)
            mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
            l2 = myloss(out.view(1, -1), y.view(1, -1))
            test_mse += mse.item()
            test_l2 += l2.item()
        test_mse /= len(test_loader)
        test_l2 /= ntest
        print('test error: L2 = ',test_l2,'MSE =',test_mse)
    
    f_test = torch.Tensor(f_test_h[:,::40])
    dim = f_test_h.shape[-1]
    grid = np.linspace(0, 1, dim)
    N = f_test_h.shape[0]*dim
    loc = np.zeros((N,1))
    res = np.zeros((N,1))
    f = np.zeros((N,f_test.shape[-1]))
    for i in range(N):
        f[i] = f_test[i//dim]
        loc[i,0] = grid[i%dim]
        res[i,0] = u_test_h[i//dim,i%dim]
    f_test = torch.Tensor(f)
    loc = torch.Tensor(loc)
    res = torch.Tensor(res)
    test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, loc, res), batch_size=dim, shuffle=False)
    
    pred_h = torch.zeros(u_test_h.shape)
    index = 0
    test_mse = 0
    test_l2 = 0
    with torch.no_grad():
        for x, l, y in test_h_loader:
            out = model(x,l).view(-1)
            pred_h[index] = out
            mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
            l2 = myloss(out.view(1, -1), y.view(1, -1))
            test_mse += mse.item()
            test_l2 += l2.item()
            index += 1
        test_mse /= len(test_h_loader)
        test_l2 /= ntest
        print('test error on high resolution: L2 = ',test_l2,'MSE =',test_mse)

    residual = pred_h-u_test_h
    fig = plt.figure()
    x_grid = np.linspace(0, 1, 2001)
    for i in range(ntest):
        plt.plot(x_grid,residual[i].detach().numpy())
    plt.show()