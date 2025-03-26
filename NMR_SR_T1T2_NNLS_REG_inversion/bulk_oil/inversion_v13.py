import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls

def t1_t2_inversion_nnls_sr(
    csv_path,
    n_T1_bins=50,
    n_T2_bins=50,
    T1_range=(1e-3, 100),
    T2_range=(1e-3, 10),
    lambda_T1=3,
    lambda_T2=2,
    normalize=False
):
    # --- Load data ---
    df = pd.read_csv(csv_path)
    tau1 = df["T1 (s)"].values
    tau2 = np.array(df.columns[1:]).astype(float)
    S_matrix = df.iloc[:, 1:].values

    # --- Normalize each T1 row (optional) ---
    if normalize:
        S_matrix = S_matrix / S_matrix.max(axis=1, keepdims=True)
    S_flat = S_matrix.flatten()

    # --- Inversion grid ---
    T1_bins = np.logspace(np.log10(T1_range[0]), np.log10(T1_range[1]), n_T1_bins)
    T2_bins = np.logspace(np.log10(T2_range[0]), np.log10(T2_range[1]), n_T2_bins)

    # --- Build kernel (SR model on T1) ---
    K1 = 1 - np.exp(-np.outer(tau1, 1 / T1_bins))   # SR recovery kernel
    K2 = np.exp(-np.outer(tau2, 1 / T2_bins))       # T2 decay kernel
    K = np.kron(K1, K2)

    # --- Asymmetric regularization ---
    I_T1 = np.identity(n_T1_bins)
    I_T2 = np.identity(n_T2_bins)
    R_T1 = np.sqrt(lambda_T1) * np.kron(I_T1, np.identity(n_T2_bins))
    R_T2 = np.sqrt(lambda_T2) * np.kron(np.identity(n_T1_bins), I_T2)

    # --- Augmented system for NNLS ---
    A = np.vstack([K, R_T1, R_T2])
    B = np.concatenate([S_flat, np.zeros(R_T1.shape[0] + R_T2.shape[0])])

    # --- Solve NNLS ---
    F_flat, _ = nnls(A, B)
    F_2D = F_flat.reshape((n_T1_bins, n_T2_bins))
    # F_2D[F_2D < 0] = 0
    # F_2D /= F_2D.max()

    # --- Plotting ---
    T1_bins_ms = T1_bins * 1000
    T2_bins_ms = T2_bins * 1000
    log_T1 = np.log10(T1_bins_ms)
    log_T2 = np.log10(T2_bins_ms)

    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

    main_ax = fig.add_subplot(grid[1:4, 0:3])
    contour = main_ax.contourf(log_T2, log_T1, F_2D, levels=40)
    main_ax.plot([min(log_T2), max(log_T2)], [min(log_T2), max(log_T2)], 'r--', label="T1/T2 = 1")
    main_ax.set_xlabel("log10 T2 (ms)")
    main_ax.set_ylabel("log10 T1 (ms)")
    main_ax.set_title("T1-T2 Distribution (NNLS + SR Kernel)")
    main_ax.legend(loc='upper left')
    # cbar = fig.colorbar(contour, ax=main_ax)
    # cbar.set_label("Relative Amplitude")

    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    top_ax.plot(log_T2, F_2D.sum(axis=0), color='k', linestyle='dashed')
    top_ax.set_ylabel("T2 Amplitude")
    top_ax.set_title("T2 Projection")

    side_ax = fig.add_subplot(grid[1:4, 3], sharey=main_ax)
    side_ax.plot(F_2D.sum(axis=1), log_T1, color='k', linestyle='dashed')
    side_ax.set_xlabel("T1 Amplitude")
    side_ax.set_title("T1 Projection")

    plt.tight_layout()
    plt.show()

    return F_2D, T1_bins_ms, T2_bins_ms

F_2D, T1_ms, T2_ms = t1_t2_inversion_nnls_sr("nmr_t1_t2_data.csv")