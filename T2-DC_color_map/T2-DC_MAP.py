import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.ndimage

# Load NMR data from a text file
file_path = "MRIL_NmrAdv_Output_FIN.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract depth values, porosity, and distributions
depth_values = []
phit_values = []
t2_values, t2dist_values = [], []
t1_values, t1dist_values = [], []
dc_values, dcdist_values = [], []

data_section = False
for line in lines:
    split_line = line.strip().split("\t")
    if len(split_line) == 2 and not data_section:
        try:
            depth_values.append(float(split_line[0]))
            phit_values.append(float(split_line[1]))
        except ValueError:
            pass
    elif len(split_line) == 6:
        data_section = True
        try:
            t2_values.append(float(split_line[0]))
            t2dist_values.append(float(split_line[1]))
            t1_values.append(float(split_line[2]))
            t1dist_values.append(float(split_line[3]))
            dc_values.append(float(split_line[4]))
            dcdist_values.append(float(split_line[5]))
        except ValueError:
            pass

# Convert lists to numpy arrays
t2_array = np.array(t2_values)
t2_dist_array = np.array(t2dist_values)
t1_array = np.array(t1_values)
t1_dist_array = np.array(t1dist_values)
dc_array = np.array(dc_values)
dc_dist_array = np.array(dcdist_values)

# Normalize T1 distribution amplitude for better visibility
t1_dist_scaled = t1_dist_array * (max(t2_dist_array) / max(t1_dist_array))

# Create a 2D grid for T2-D mapping
t2_grid, dc_grid = np.meshgrid(t2_array, dc_array, indexing="ij")
t2_d_map = np.outer(t2_dist_array, dc_dist_array)

# Apply Gaussian smoothing for a refined visualization
t2_d_map_smoothed = scipy.ndimage.gaussian_filter(t2_d_map, sigma=1)

# Limit T2 values to max of 5 seconds
t2_mask = t2_array <= 5
t2_filtered = t2_array[t2_mask]
t2_dist_filtered = t2_dist_array[t2_mask]
t2_grid_filtered, dc_grid_filtered = np.meshgrid(t2_filtered, dc_array, indexing="ij")
t2_d_map_filtered = t2_d_map_smoothed[t2_mask, :]

# Define colormap scale
vmin, vmax = np.min(t2_d_map_filtered), np.max(t2_d_map_filtered)
water_line_new = 10 ** -4.7  # Adjusted water line position

# Plot refined T2-D Map
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[0.5, 4], wspace=0.4, hspace=0.1)

# Main T2-D Map plot with smoothed contours
ax_main = plt.subplot(gs[1, 0])
contour = ax_main.contourf(t2_grid_filtered, dc_grid_filtered, t2_d_map_filtered, levels=100, cmap="rainbow", vmin=vmin, vmax=vmax)
ax_main.axhline(y=water_line_new, color="cyan", linestyle="--", linewidth=2, label="Water Line (10⁻⁴.⁷)")
ax_main.set_xscale("log")
ax_main.set_yscale("log")
ax_main.set_xlim([min(t2_filtered), 5])
ax_main.set_xlabel("T₂ (s)", fontsize=12, fontweight="bold")
ax_main.set_ylabel("Diffusion Coefficient (cm²/s)", fontsize=12, fontweight="bold")
ax_main.legend(fontsize=10, loc="upper right")
ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Colorbar for the main plot
cbar = fig.colorbar(contour, ax=ax_main, orientation="vertical", fraction=0.05, pad=0.02)
cbar.set_label("Amplitude", fontsize=12)

# T2 and Scaled T1 Distribution (Top subplot)
ax_t2 = plt.subplot(gs[0, 0], sharex=ax_main)
ax_t2.semilogx(t2_filtered, t2_dist_filtered, color="black", linewidth=2, label="T₂ Distribution")
ax_t2.semilogx(t1_array, t1_dist_scaled, color="red", linewidth=2, linestyle="--", label="T₁ Distribution (Scaled)")
ax_t2.set_xlim(ax_main.get_xlim())
ax_t2.set_ylabel("Amplitude", fontsize=12)
ax_t2.set_title("T₂ & T₁ Distributions", fontsize=12, fontweight="bold")
ax_t2.legend(fontsize=10, loc="upper right")
ax_t2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
ax_t2.tick_params(axis="both", which="major", labelsize=10)
plt.setp(ax_t2.get_xticklabels(), visible=False)

# Diffusion Distribution (Right subplot)
ax_dc = plt.subplot(gs[1, 1], sharey=ax_main)
ax_dc.semilogy(dc_dist_array, dc_array, color="black", linewidth=2)
ax_dc.set_ylim(ax_main.get_ylim())
ax_dc.set_xlabel("Amplitude", fontsize=12)
ax_dc.set_title("Diffusion Distribution", fontsize=12, fontweight="bold")
ax_dc.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
ax_dc.tick_params(axis="both", which="major", labelsize=10)
plt.setp(ax_dc.get_yticklabels(), visible=False)

# Show the final refined figure

plt.show()
