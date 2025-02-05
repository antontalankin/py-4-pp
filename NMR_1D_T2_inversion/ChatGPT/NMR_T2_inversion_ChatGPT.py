#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:21:18 2025

@author: antontalankin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import pandas as pd

path='35447_sw1.csv'
df=pd.read_csv(path)


t=df['t'].to_numpy().astype('float')
a=df['A'].to_numpy().astype('float')


# Define the inversion function with Tikhonov regularization
def invert_t2_tikhonov(t, signal, t2_range, num_bins, reg_param):
    # Create a dictionary of exponential decays
    t2_values = np.logspace(np.log10(t2_range[0]), np.log10(t2_range[1]), num_bins)
    kernel = np.exp(-np.outer(t, 1 / t2_values))
    
    # Construct regularization matrix (identity matrix scaled by reg_param)
    L = reg_param * np.identity(num_bins)
    
    # Solve the regularized least squares problem
    amplitudes = lsq_linear(np.vstack([kernel, L]), np.hstack([signal, np.zeros(num_bins)]), bounds=(0, np.inf)).x
    
    return t2_values, amplitudes



# Inversion parameters
t2_range = (0.01, 10000)  # T2 range to search (s)
num_bins = 200  # Number of T2 bins
reg_param = .5  # Regularization parameter (Tikhonov)

# Perform the inversion
t2_values, amplitudes = invert_t2_tikhonov(t, a, t2_range, num_bins, reg_param)



# plotting the data 
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5), dpi=200)
# Raw echoes
ax1.plot(t, a, color='b', linestyle='none', markersize=3, marker='o', mec='k', label='Raw echoes')
ax1.set_xlabel('Realization time, ms')
ax1.set_ylabel('Signal amplitude, calibrated')
ax1.set_title('Raw echoes')
ax1.set_xscale('log')
ax1.set_xlim(0.1,10000)
ax1.set_ylim(0,.25)
ax1.grid(which='both')
ax1.legend()
# T2 inversion
ax2.plot(t2_values, amplitudes, color='b', marker='o', mec='k', label='T2 inversion')
ax2.set_title('1D T2 inversion - lsq_liear - chatGPT')
ax2.set_xlabel('T2, ms')
ax2.set_ylabel('Porosity increment, fraction')
ax2.set_xlim(0.01, 10000)
ax2.set_xscale('log')
ax2.set_ylim(0,.01)
ax2.grid(which='both')
ax2.legend()
plt.tight_layout()
plt.show()

