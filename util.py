# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

# Gaussian Random Field
class GRF(object):
    def __init__(self, begin=0, end=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n, A):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T+A

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])

        res = np.zeros((ys.shape[0],sensors.shape[0]))
        for i in range(ys.shape[0]):
            res[i,:] = interpolate.interp1d(np.ravel(self.x), ys[i], kind=self.interp, copy=False, assume_sorted=True)(sensors).T
        return res

# Gaussian Progress regression
class GP_regression:
    def __init__(self, num_x_samples):
        self.observations = {"x": list(), "y": list()}
        self.num_x_samples = num_x_samples
        self.x_samples = np.linspace(0, 1.0, self.num_x_samples).reshape(-1, 1)
        
        # prior
        self.mu = np.zeros_like(self.x_samples)
        self.cov = self.kernel(self.x_samples, self.x_samples)
        
        
    def update(self, observations):
        self.update_observation(observations)
        
        x = np.array(self.observations["x"]).reshape(-1, 1)
        y = np.array(self.observations["y"]).reshape(-1, 1)
        
        K11 = self.cov  # (N,N)
        K22 = self.kernel(x, x) # (k,k)
        K12 = self.kernel(self.x_samples, x)  # (N,k)
        K21 = self.kernel(x, self.x_samples)  # (k,N)
        K22_inv = np.linalg.inv(K22 + 1e-8 * np.eye(len(x)))  # (k,k)
        
        self.mu = K12.dot(K22_inv).dot(y)
        self.cov = self.kernel(self.x_samples, self.x_samples) - K12.dot(K22_inv).dot(K21)
        
    def visualize(self, num_gp_samples=3):
        gp_samples = np.random.multivariate_normal(
            mean=self.mu.ravel(), 
            cov=self.cov, 
            size=num_gp_samples)
        x_sample = self.x_samples.ravel()
        mu = self.mu.ravel()
        # uncertainty = 1.96 * np.sqrt(np.diag(self.cov))

        plt.figure()
        # plt.fill_between(x_sample, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(x_sample, mu, label='Mean')
        for i, gp_sample in enumerate(gp_samples):
            plt.plot(x_sample, gp_sample, lw=1, ls='-', label=f'Sample {i+1}')
            
        plt.plot(self.observations["x"], self.observations["y"], 'rx')
        # plt.legend()
        plt.grid()
        return gp_samples

    def update_observation(self, observations):
        for x, y in zip(observations["x"], observations["y"]):
            if x not in self.observations["x"]:
                self.observations["x"].append(x)
                self.observations["y"].append(y)
                
    @staticmethod
    def kernel(x1, x2, l=0.2, sigma_f=1):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)
    

def kernel_RBF(xs, ys, sigma=1, l=1):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return np.exp(-((dx/l)**2)/(sigma**2*2))

# Runge-Kutta
def RK4(y0, x):
    
    def equation(x,i,j,y):
        return x[j,i]
    
    dim = x.shape[1]
    N = x.shape[0]
    h = 1/(dim-1)*2
    res = np.zeros((N,(dim+1)//2))
    for j in range(N):
        res[j,0] = y0
        y = y0
        for i in range((dim-1)//2):
            k1 = equation(x,i,j,y)
            k2 = equation(x,2*i+1,j,y+h*k1/2)
            k3 = equation(x,2*i+1,j,y+h*k2/2)
            k4 = equation(x,2*i+2,j,y+h*k3)
            y = y+h*(k1+2*k2+2*k3+k4)/6
            res[j,i+1] = y
    return res


def FD_1(f,A,B,C,b_1,b_2):
    grid = np.linspace(0,1,f.shape[-1])
    dim = grid.shape[-1]
    h = (grid[-1]-grid[0])/(dim-1)
    N = f.shape[0]
    
    res = np.zeros((N,dim))
    p = A/(h**2)
    r = C-2*A/(h**2)-B/h
    q = A/(h**2)+B/h
    
    for k in range(N):
        U = np.zeros((dim-2,dim-2))
        U[0,:2] = np.array([r,q])
        U[-1,-2:] = np.array([p,r])
        
        j = 0
        for i in range(1,dim-3):
            U[i,j:j+3] = np.array([p,r,q])
            j += 1
        
        B = np.zeros(dim-2)
        B[:] = f[k,1:-1]
        B[0] -= p*b_1
        B[-1] -= q*b_2
        B = B.T
        
        res[k,0] = b_1
        res[k,1:-1] = np.linalg.solve(U,B).flatten()
        res[k,-1] = b_2
    return res

# generate random functions(1d) default dim=1001
def generate(samples=1000, begin=0, end=1, random_dim=101, out_dim=1001, length_scale=1,interp="cubic",A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples,A)
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid[:, None])
    return x_data

# generate random functions(2d)
def generate_2(d_1,d_2):
    res = np.zeros((d_2.shape[0],d_1.shape[0]))
    res[0] = d_1
    
    for i in range(1,d_2.shape[0]):
        res[i] = res[i-1]+(d_2[i]-d_2[i-1])
    return res

def FD_ib(f,grid_1,grid_2,eps=1):
    b_1 = f[0]
    b_2 = f[-1]
    dim_1 = grid_1.shape[0]
    dim_2 = grid_2.shape[0]+1
    h_1 = 1/(dim_1-1)
    h_2 = 1/(dim_2-2)
    
    p = -eps/(h_1**2)
    r = 2*eps/(h_1**2)-1/h_1+1/h_2
    q = -eps/(h_1**2)+1/h_1
    o = -1/h_2
    s = 0
    
    res = np.zeros((dim_2-1,dim_1))
    res[0,:] = f[:]
    for i in range(dim_2-1):
        res[i,0] = f[0]
        res[i,-1] = f[-1]
    
    U = np.zeros((dim_1-2,dim_1-2))
    U[0,:2] = np.array([r,q])
    U[-1,-2:] = np.array([p,r])
    
    j = 0
    for i in range(1,dim_1-3):
        U[i,j:j+3] = np.array([p,r,q])
        j += 1
        
    T = np.eye(dim_1-2)*o
    V = np.eye(dim_1-2)*s
    Z = np.zeros((dim_1-2,dim_1-2))
    
    A_blc = np.empty((dim_2-2,dim_2-2), dtype=object)
    for i in range(dim_2-2):
        for j in range(dim_2-2):
            if i==j:
                A_blc[i,j] = U
            elif i+1==j:
                A_blc[i,j] = V
            elif i-1==j:
                A_blc[i,j] = T
            else:
                A_blc[i,j] = Z
    A = np.vstack([np.hstack(A_i) for A_i in A_blc])
    
    B = np.zeros((dim_2-2,dim_1-2))
    B[0,:] = f[1:-1]
    B = np.reshape(B,((dim_2-2)*(dim_1-2),1))*(-o)
    
    C = np.zeros((dim_2-2,dim_1-2))
    for i in range(dim_2-2):
        C[i,0] = f[0]
    C = np.reshape(C,((dim_2-2)*(dim_1-2),1))*(-p)
    
    D = np.zeros((dim_2-2,dim_1-2))
    for i in range(dim_2-2):
        D[i,-1] = f[-1]
    D = np.reshape(D,((dim_2-2)*(dim_1-2),1))*(-q)
    
    sol = np.linalg.solve(A,B+C+D)
    sol = np.reshape(sol,(dim_2-2,dim_1-2))
    res[1:,1:-1] = sol
        
    return res

def FD_2d(f,grid):
    dim = grid.shape[0]
    h = 1/(dim-1)
    
    p = -1/(h**2)
    r = 4/(h**2)-2/h
    q = -1/(h**2)+1/h
    o = -1/(h**2)
    s = -1/(h**2)+1/h
    
    res = np.zeros_like(f)
    U = np.zeros((dim-2,dim-2))
    U[0,:2] = np.array([r,q])
    U[-1,-2:] = np.array([p,r])
    
    j = 0
    for i in range(1,dim-3):
        U[i,j:j+3] = np.array([p,r,q])
        j += 1
        
    T = np.eye(dim-2)*o
    V = np.eye(dim-2)*s
    Z = np.zeros((dim-2,dim-2))
    
    A_blc = np.empty((dim-2,dim-2), dtype=object)
    for i in range(dim-2):
        for j in range(dim-2):
            if i==j:
                A_blc[i,j] = U
            elif i+1==j:
                A_blc[i,j] = V
            elif i-1==j:
                A_blc[i,j] = T
            else:
                A_blc[i,j] = Z
    A = np.vstack([np.hstack(A_i) for A_i in A_blc])
    
    B = np.zeros((dim-2,dim-2))
    B[:,:] = f[1:-1,1:-1]
        
    B = np.reshape(B,((dim-2)*(dim-2),1))
    sol = np.linalg.solve(A,B)
    sol = np.reshape(sol,(dim-2,dim-2))
    res[1:-1,1:-1] = sol
        
    return res