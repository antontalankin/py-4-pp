#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:04:03 2025

@author: antontalankin
"""

# Re-load the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file again
file_path = "NMR_T2_inversion.csv"
df = pd.read_csv(file_path)

# Convert T2 values to seconds for calculations
df["T2"] = df["T2"] / 1000  # Convert from ms to s

# Set T2 cutoff in milliseconds
T2_cutoff = 10  # in ms

# Calculate T2 geometric mean using only non-zero amplitude values
valid_indices = df["Amplitude"] > 0  # Ignore zero amplitude values
T2_geom_mean = np.exp(
    np.sum(np.log(df["T2"][valid_indices]) * df["Amplitude"][valid_indices]) /
    np.sum(df["Amplitude"][valid_indices])
)

# Convert T2 geometric mean back to ms
T2_geom_mean_ms = T2_geom_mean * 1000  # Convert from s to ms

# Compute volume below T2 cutoff (T2 <= 10 ms)
volume_below_cutoff = df.loc[df["T2"] * 1000 <= T2_cutoff, "Amplitude"].sum()

# Compute total NMR porosity
total_nmr_porosity = df["Amplitude"].sum()

# Plot T2 distribution with updated legend
plt.figure(figsize=(8, 5))
plt.semilogx(df["T2"] * 1000, df["Amplitude"], marker="o", linestyle="-", color="b", label="T2 Distribution")
plt.axvline(T2_geom_mean_ms, color="r", linestyle="--", label=f"T2 Geom Mean: {T2_geom_mean_ms:.3f} ms")
plt.axvline(T2_cutoff, color="g", linestyle="--", label=f"T2 Cutoff: {T2_cutoff:.1f} ms\nVolume below cutoff: {volume_below_cutoff:.3f}\nTotal NMR Porosity: {total_nmr_porosity:.3f}")

plt.xlabel("T2 (ms)")
plt.ylabel("Amplitude (Porosity Increment)")
plt.title("NMR T2 Distribution")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()