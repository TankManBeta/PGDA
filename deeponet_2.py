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
learning_rate = 0.001
epochs = 300
step_size = 50
gamma = 0.5

EP_list = [1/2**6, 1/2**7, 1/2**8, 1/2**9, 1/2**10, 1/2**11]
N_list = [2**6+1]#, 2**7+1, 2**8+1, 2**9+1, 2**10+1]
N_max = 2**12+1

f_train_h = generate(samples=ntrain,out_dim=N_max)
loss_history = dict()

for EP in EP_list:
    u_train_h = FD_1(f_train_h,-EP,1,0,0,0)
    for NS in N_list:    
        mse_history = []
        print("N value : ",NS-1,", epsilon value : ",EP)       
        sigma = min(1/2,2*EP*np.log(NS))
        
        dN = int((N_max-1)/(NS-1))
        f_train = torch.Tensor(f_train_h[:,::dN])
        
        gridS = np.linspace(0,1,NS)
        #gridS = np.hstack((np.linspace(0,1-sigma,int((NS-1)/2)+1),np.linspace(1-sigma,1,int((NS-1)/2)+1)[1:]))
        u_train = interpolate.interp1d(np.linspace(0,1,N_max),u_train_h)(gridS)
        
        dim = NS
        N = f_train.shape[0]*dim
        loc = np.zeros((N,1))
        res = np.zeros((N,1))
        f = np.zeros((N,dim))
        for i in range(N):
            f[i] = f_train[i//dim]
            loc[i,0] = gridS[i%dim]
            res[i,0] = u_train[i//dim,i%dim]
            
        f_train = torch.Tensor(f)
        loc_train = torch.Tensor(loc)
        res_train = torch.Tensor(res)    
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, loc_train, res_train), batch_size=dim, shuffle=True)
        
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
            mse_history.append(train_mse)
            print('\repoch {:d}/{:d} , MSE = {:.6f}, using {:.6f}s'.format(ep+1,epochs,train_mse,t2-t1), end='', flush=True)
            
        print('Total training time:',default_timer()-start,'s')
        loss_history["{} {}".format(NS,EP)] = mse_history
        
        f_test = torch.Tensor(f_train_h[:,::dN])
        dim = f_train_h.shape[-1]
        grid = np.linspace(0, 1, dim)
        N = f_train_h.shape[0]*dim
        loc = np.zeros((N,1))
        res = np.zeros((N,1))
        f = np.zeros((N,f_test.shape[-1]))
        for i in range(N):
            f[i] = f_test[i//dim]
            loc[i,0] = grid[i%dim]
            res[i,0] = u_train_h[i//dim,i%dim]
        f_test = torch.Tensor(f)
        loc = torch.Tensor(loc)
        res = torch.Tensor(res)
        test_h_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, loc, res), batch_size=dim, shuffle=False)
        
        pred_h = torch.zeros(u_train_h.shape)
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
            test_l2 /= ntrain
            print('test error on high resolution: L2 = ',test_l2,'MSE =',test_mse)
    
        #residual = pred_h-u_train_h
        #fig = plt.figure()
        #x_grid = np.linspace(0, 1, N_max)
        #for i in range(100):
        #    plt.plot(x_grid,residual[i].detach().numpy())
        #plt.show()