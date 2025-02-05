#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:25:29 2025

@author: antontalankin
"""

import pandas as pd
import numpy as np 
import scipy
import matplotlib.pyplot as plt 
from scipy.optimize import nnls
from numpy import concatenate


path='35447_sw1.csv'
df=pd.read_csv(path)

t=df.t.astype(float)
a=df.A.astype(float)
# inversion
# set T2 limits & range
t2min=0.01
t2max=10000
t2range=200 # set numper of t2 bins 
T=np.linspace(np.log10(t2min),np.log10(t2max), t2range)
T2=[pow(10,i) for i in T]
x,y = np.meshgrid(t, T2)
X = np.exp(-x/y).T
Y = np.array(a)
# set regularization parameter (0.5 as base case)
lamb=.5
n_variables=X.shape[1]
A = concatenate([X, np.sqrt(lamb)*np.eye(n_variables)])
B = concatenate([Y, np.zeros(n_variables)])
A1=nnls(A,B)
amplitude=A1[0]

# plotting the data 
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5), dpi=200)
# Raw echoes
ax1.plot(t, a, color='r', linestyle='none', markersize=3, marker='o', mec='k', label='Raw echoes')
ax1.set_xlabel('Realization time, ms')
ax1.set_ylabel('Signal amplitude, calibrated')
ax1.set_title('Raw echoes')
ax1.set_xscale('log')
ax1.set_xlim(0.1,10000)
ax1.set_ylim(0,.25)
ax1.grid(which='both')
ax1.legend()
# T2 inversion
ax2.plot(T2, amplitude, color='r', marker='o', mec='k', label='T2 inversion')
ax2.set_title('1D T2 inversion - nnls')
ax2.set_xlabel('T2, ms')
ax2.set_ylabel('Porosity increment, fraction')
ax2.set_xlim(0.01, 10000)
ax2.set_xscale('log')
ax2.set_ylim(0,.01)
ax2.grid(which='both')
ax2.legend()
plt.tight_layout()
plt.show()
