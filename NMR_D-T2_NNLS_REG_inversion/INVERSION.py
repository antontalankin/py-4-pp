#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:13:45 2025

@author: antontalankin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.sparse import diags
import matplotlib.gridspec as gridspec
import pandas as pd

# Constants
gamma = 2.675e8  # rad/s/T

# Setup parameters
G = 0.05  # T/m gradient strength
TEs_ms = np.array([0.2, 2, 4, 6, 8, 10])  # echo spacings in ms
TEs = TEs_ms * 1e-3  # convert to seconds
# n_echoes = 200  # echoes per train
# echo_numbers = np.arange(1, n_echoes + 1)

# Discretize T2 and D
T2_grid = np.logspace(0 ,4, 50)*1e-3  
D_grid = np.logspace(-11, -5, 50)   


file_names=['TE02.csv', 'TE2.csv', 'TE4.csv', 
            'TE6.csv', 'TE8.csv', 'TE10.csv']

num_echoes=[5000, 800, 400, 200, 200, 200]
signal_vectors=[]


for j,i in enumerate(file_names):
    df=pd.read_csv(i)
    signal_vectors.append(df['a'].to_numpy()[1:]) # fraction


S_full = np.concatenate(signal_vectors)


# Build kernel matrix
# def build_kernel(T2_grid, D_grid, TEs, echo_numbers, gamma, G):
#     M = len(T2_grid)
#     L = len(D_grid)
#     total_points = len(TEs) * len(echo_numbers)
#     K = np.zeros((total_points, M * L))
#     row = 0
#     for TE in TEs:
#         for n in echo_numbers:
#             T2_decay = np.exp(-n * TE / T2_grid[:, None])   # shape (M,1)
#             diff_decay = np.exp(-n * (gamma ** 2) * (G ** 2) * D_grid[None, :] * (TE ** 3) / 12)  # (1,L)
#             kernel_2d = T2_decay * diff_decay
#             K[row, :] = kernel_2d.flatten()
#             row += 1
#     return K


# K_full = build_kernel(T2_grid, D_grid, TEs, echo_numbers, gamma, G)



def build_variable_echo_kernel(T2_grid, D_grid, TEs, num_echoes, gamma, G):
    """
    Build the 2D T2–Diffusion inversion kernel for multi-TE sequences with varying echo counts.

    Parameters:
        T2_grid    : np.array of shape (M,)   - logarithmically spaced T2 relaxation times (s)
        D_grid     : np.array of shape (L,)   - logarithmically spaced diffusion coefficients (m²/s)
        TEs        : np.array of shape (N,)   - TE values in seconds
        num_echoes : list of length N         - number of echoes per TE
        gamma      : float                    - gyromagnetic ratio in rad/s/T
        G          : float                    - magnetic field gradient in T/m

    Returns:
        K          : np.array of shape (sum(num_echoes), M*L) - 2D inversion kernel matrix
    """
    M = len(T2_grid)
    L = len(D_grid)
    total_rows = sum(num_echoes)
    K = np.zeros((total_rows, M * L))
    
    row_idx = 0
    for i, TE in enumerate(TEs):
        echo_indices = np.arange(1, num_echoes[i] + 1)
        for n in echo_indices:
            # T2 exponential decay: shape (M, 1)
            E_T2 = np.exp(-n * TE / T2_grid[:, None])
            
            # Diffusion decay (Stejskal-Tanner formula approximation): shape (1, L)
            E_D = np.exp(-n * (gamma**2) * (G**2) * D_grid[None, :] * (TE**3) / 12)
            
            # Outer product to form 2D decay map: shape (M, L)
            decay_map = E_T2 * E_D
            
            # Flatten and store
            K[row_idx, :] = decay_map.flatten()
            row_idx += 1

    return K

K_full = build_variable_echo_kernel(T2_grid, D_grid, TEs, num_echoes, gamma, G)


# === Regularization ===
lambda_D = 1
lambda_T2 = 1
I_D = np.identity(50)
I_T2 = np.identity(50)
R_D = np.sqrt(lambda_D) * np.kron(I_D, np.identity(50))
R_T2 = np.sqrt(lambda_T2) * np.kron(np.identity(50), I_T2)
A = np.vstack([K_full, R_D, R_T2])
B = np.concatenate([S_full, np.zeros(R_D.shape[0] + R_T2.shape[0])])

# === Solve NNLS ===
# F_flat, _ = nnls(K_full, S_full)
F_flat, _ = nnls(A, B)
F_2D = F_flat.reshape((50, 50))
# F_2D /= F_2D.max()


# === Plot result ===
log_T2 = np.log10(T2_grid*1000)
log_D = np.log10(D_grid)

fig = plt.figure(figsize=(12, 9))
grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)

main_ax = fig.add_subplot(grid[1:4, 0:3])

contour = main_ax.contourf(log_T2, log_D, F_2D.T, levels=40)

# main_ax.
main_ax.set_xlabel('log10 T2 (ms)')
main_ax.set_ylabel('log10 Diffusion Coefficient (m²/s)')
main_ax.set_title('2D Inversion: D–T2 Map (NNLS)')
main_ax.axhline(y=-8.5, color='aqua', linestyle='dashed', linewidth=2)
# fig.colorbar(contour, ax=main_ax, label='Relative Amplitude')

top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
top_ax.plot(log_T2, F_2D.sum(axis=1), color='k', linestyle='dashed')
top_ax.set_ylabel('∑ T2 Amplitude')
top_ax.set_title('T2 Distribution')

side_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)
side_ax.plot(F_2D.sum(axis=0), log_D, color='k', linestyle='dashed')
side_ax.set_xlabel('∑ D Amplitude')
side_ax.set_title('Diffusion Distribution')




plt.tight_layout()
plt.show()