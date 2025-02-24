import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def analyze_nmr_t2_distribution(file_path, num_components=5, max_iter=20):
    # Load data
    df = pd.read_csv(file_path)
    T2 = df["T2"].values
    A = df["A"].values
    
    # Normalize A for consistent fitting and rescale after fitting
    A_normalized = A / np.trapz(A, T2)
    
    # Convert to log scale for GMM fitting
    log_T2 = np.log(T2).reshape(-1, 1)
    
    # Expand dataset based on amplitude weighting
    expanded_log_T2 = np.repeat(log_T2, (A_normalized * 1000000).astype(int), axis=0)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, covariance_type="full", random_state=42, reg_covar=1e-3)
    gmm.fit(expanded_log_T2)
    
    # Extract parameters in log space
    log_means = gmm.means_.flatten()
    log_variances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    # Convert back to linear scale
    means_linear = np.exp(log_means)
    
    # Generate Gaussian curves for visualization
    T2_fit = np.logspace(np.log10(T2.min()), np.log10(T2.max()), 500)
    individual_gaussians = []
    
    for i in range(num_components):
        single_gaussian = (
            weights[i]
            * np.exp(-0.5 * ((np.log(T2_fit) - log_means[i]) ** 2) / log_variances[i])
            / np.sqrt(2 * np.pi * log_variances[i])
        )
        individual_gaussians.append(single_gaussian)
    
    # Iteratively scale Gaussian sum to match original A
    gaussian_curves = np.sum(individual_gaussians, axis=0)
    scale_factor = np.trapz(A, T2) / np.trapz(gaussian_curves, T2_fit)
    
    for _ in range(max_iter):
        gaussian_curves *= scale_factor
        individual_gaussians = [g * scale_factor for g in individual_gaussians]
        scale_factor = np.trapz(A, T2) / np.trapz(gaussian_curves, T2_fit)
    
    # Plot T2 distribution and fitted Gaussian components
    plt.figure(figsize=(8, 5))
    plt.plot(T2, A, marker='o', linestyle='-', markersize=3, label="NMR T2 Distribution")
    plt.plot(T2_fit, gaussian_curves, linestyle='--', color="red", label="GMM Iteratively Scaled Fit")
    
    for i, single_gaussian in enumerate(individual_gaussians):
        plt.plot(T2_fit, single_gaussian, linestyle=":", label=f"Gaussian {i+1}")
    
    plt.xscale("log")
    plt.xlabel("T2 (ms)")
    plt.ylabel("Amplitude (A)")
    plt.title("Amplitude-Weighted Gaussian Mixture Model Fit to NMR T2 Distribution")
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
file_path = "T2_inversion_22884.csv"
results = analyze_nmr_t2_distribution(file_path)
print(results)
