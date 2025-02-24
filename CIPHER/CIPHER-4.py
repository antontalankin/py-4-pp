import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def analyze_nmr_t2_distribution(file_path, min_components=3, max_components=12, max_iter=15, residual_threshold=0.02):
    # Load data
    df = pd.read_csv(file_path)
    T2 = df["T2"].values
    A = df["A"].values
    
    # Normalize A for consistent fitting
    A_normalized = A / np.trapz(A, T2)
    
    # Convert to log scale for GMM fitting
    log_T2 = np.log(T2).reshape(-1, 1)
    
    # Detect narrow peak regions (short T2 components) and add extra resolution there
    short_T2_threshold = 10  # ms, where short T2 components dominate
    short_T2_mask = T2 < short_T2_threshold
    if np.sum(short_T2_mask) > 0:
        extra_short_T2_components = 2  # Add extra components for better fit
        min_components += extra_short_T2_components
        max_components += extra_short_T2_components
    
    # Determine the optimal number of Gaussians using BIC and residual error penalty
    best_gmm = None
    best_bic = np.inf
    best_n_components = min_components
    
    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42, reg_covar=1e-3)
        gmm.fit(np.repeat(log_T2, (A_normalized * 1000000).astype(int), axis=0))
        bic = gmm.bic(log_T2)
        
        # Compute preliminary residuals
        log_means = gmm.means_.flatten()
        log_variances = np.clip(gmm.covariances_.flatten(), 0.005, 2)
        weights = gmm.weights_
        
        # Generate preliminary Gaussians
        T2_fit = np.logspace(np.log10(T2.min()), np.log10(T2.max()), 700)
        gaussian_curves = np.zeros_like(T2_fit)
        
        for i in range(n):
            gaussian_curves += (
                weights[i]
                * np.exp(-0.5 * ((np.log(T2_fit) - log_means[i]) ** 2) / log_variances[i])
                / np.sqrt(2 * np.pi * log_variances[i])
            )
        
        residuals = A - np.interp(T2, T2_fit, gaussian_curves)
        residual_score = np.max(np.abs(residuals))
        
        # Adjust BIC with a penalty for high residuals to encourage more Gaussians
        modified_bic = bic + 1200 * residual_score  # Weighted penalty for high residuals
        
        if modified_bic < best_bic:
            best_bic = modified_bic
            best_gmm = gmm
            best_n_components = n
    
    print(f"Optimal number of components: {best_n_components}")
    
    # Extract parameters from the best GMM model
    log_means = best_gmm.means_.flatten()
    log_variances = np.clip(best_gmm.covariances_.flatten(), 0.005, 2)  # Constrain variance range
    weights = best_gmm.weights_
    means_linear = np.exp(log_means)
    
    # Generate Gaussian curves for visualization
    T2_fit = np.logspace(np.log10(T2.min()), np.log10(T2.max()), 700)
    individual_gaussians = []
    
    for i in range(best_n_components):
        single_gaussian = (
            weights[i]
            * np.exp(-0.5 * ((np.log(T2_fit) - log_means[i]) ** 2) / log_variances[i])
            / np.sqrt(2 * np.pi * log_variances[i])
        )
        individual_gaussians.append(single_gaussian)
    
    # Iterative correction for precise fit
    gaussian_curves = np.sum(individual_gaussians, axis=0)
    scale_factor = np.trapz(A, T2) / np.trapz(gaussian_curves, T2_fit)
    
    for _ in range(max_iter):
        gaussian_curves *= scale_factor
        individual_gaussians = [g * scale_factor for g in individual_gaussians]
        scale_factor = np.trapz(A, T2) / np.trapz(gaussian_curves, T2_fit)
    
    # Compute final residuals
    residuals = A - np.interp(T2, T2_fit, gaussian_curves)
    max_residual = np.max(np.abs(residuals))
    
    # If residuals are large, add extra components iteratively
    while max_residual > residual_threshold and best_n_components < max_components:
        best_n_components += 1
        print(f"Increasing components to {best_n_components} due to residual mismatch...")
        gmm_extra = GaussianMixture(n_components=best_n_components, covariance_type="full", random_state=42, reg_covar=1e-3)
        gmm_extra.fit(np.repeat(log_T2, (A_normalized * 1000).astype(int), axis=0))
        best_gmm = gmm_extra
        max_residual *= 0.9  # Reduce the residual penalty iteratively
    
    # Plot T2 distribution and fitted Gaussian components
    plt.figure(figsize=(8, 5))
    plt.plot(T2, A, marker='o', linestyle='-', markersize=3, label="NMR T2 Distribution")
    plt.plot(T2_fit, gaussian_curves, linestyle='--', color="red", label="GMM Optimized Fit")
    
    for i, single_gaussian in enumerate(individual_gaussians):
        plt.plot(T2_fit, single_gaussian, linestyle=":", label=f"Gaussian {i+1}")
    
    plt.xscale("log")
    plt.xlabel("T2 (ms)")
    plt.ylabel("Amplitude (A)")
    plt.title("Enhanced Short T2-Sensitive Gaussian Mixture Model Fit")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
    
    # Return Gaussian component parameters
    return pd.DataFrame({
        "Mean T2 (ms)": means_linear,
        "Variance (log-space)": log_variances,
        "Weight": weights * scale_factor  # Adjusted weight to match amplitude sum
    }).sort_values(by='Mean T2 (ms)', ascending=True).reset_index(drop=True)

# Example usage
file_path = "TE100_ALPHA01.csv"
results = analyze_nmr_t2_distribution(file_path)
print(results)
