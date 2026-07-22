import numpy as np
import matplotlib.pyplot as plt

# Load saved 3D dataset
data = np.load("output/3d_tumor_volume_patient.npz")

print("=== NPZ File Inspector ===")
for key in data.files:
    arr = data[key]
    print(f"Key: '{key}' | Shape: {arr.shape} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")

# Extract main density array
if 'u_adaptive' in data:
    u_raw = data['u_adaptive']
elif 'u_3d' in data:
    u_raw = data['u_3d']
else:
    u_raw = data[data.files[0]]

# Extract the final time step if 4D (Time, Z, Y, X) or 3D (Time, Y, X)
if u_raw.ndim == 4:  # (T, Z, Y, X)
    u_final = u_raw[-1]
elif u_raw.ndim == 3 and u_raw.shape[0] != u_raw.shape[1]:  # (T, Y, X)
    u_final = u_raw[-1]
else:
    u_final = u_raw

print(f"\nExtracted slice/volume shape: {u_final.shape}")
print(f"Max cell density in frame: {u_final.max():.6f}")

# ==========================================
# 3D VOLUMETRIC SCATTER PLOT
# ==========================================
if u_final.ndim == 3:
    max_val = u_final.max()
    
    if max_val == 0:
        print("\n⚠️ Warning: All cell density values in this frame are 0.0 (tumor eliminated or starting frame).")
        # Try finding a frame with non-zero tumor density if u_raw has a time dimension
        if u_raw.ndim == 4:
            for t in range(u_raw.shape[0] - 1, -1, -1):
                if u_raw[t].max() > 0:
                    u_final = u_raw[t]
                    max_val = u_final.max()
                    print(f"Using non-zero frame at timestep t={t} (Max density: {max_val:.4f})")
                    break

    # Dynamic threshold: 5% of maximum density present
    threshold = max_val * 0.05 if max_val > 0 else 0.001
    print(f"Applying dynamic visibility threshold: > {threshold:.6f}")

    z, y, x = np.where(u_final > threshold)
    weights = u_final[u_final > threshold]

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    if len(x) > 0:
        p = ax1.scatter(x, y, z, c=weights, cmap='YlOrRd', alpha=0.6, edgecolors='none', s=15)
        fig1.colorbar(p, ax=ax1, label='Tumor Cell Density')
    else:
        print("No voxels above threshold.")

    ax1.set_title("3D Glioblastoma Tumor Density")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    
    # Match axes bounds to actual grid size
    ax1.set_xlim([0, u_final.shape[2]])
    ax1.set_ylim([0, u_final.shape[1]])
    ax1.set_zlim([0, u_final.shape[0]])

    plt.show()

    # ==========================================
    # 2D ORTHOGONAL SLICES
    # ==========================================
    cz, cy, cx = u_final.shape[0] // 2, u_final.shape[1] // 2, u_final.shape[2] // 2

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(u_final[cz, :, :], cmap='hot', origin='lower')
    axes[0].set_title(f"Axial Slice (Z={cz} mm)")

    axes[1].imshow(u_final[:, cy, :], cmap='hot', origin='lower')
    axes[1].set_title(f"Coronal Slice (Y={cy} mm)")

    axes[2].imshow(u_final[:, :, cx], cmap='hot', origin='lower')
    axes[2].set_title(f"Sagittal Slice (X={cx} mm)")

    for ax in axes:
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")

    plt.tight_layout()
    plt.show()

else:
    # 2D Heatmap fallback
    plt.figure(figsize=(8, 6))
    plt.imshow(u_final, cmap='hot', origin='lower')
    plt.colorbar(label='Tumor Cell Density')
    plt.title("2D Tumor Density Slice")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.show()

    # Save the 3D render directly to your output folder
fig1.savefig("output/3d_tumor_render.png", dpi=300, bbox_inches='tight')
fig2.savefig("output/3d_orthogonal_slices.png", dpi=300, bbox_inches='tight')
print("\nSaved 3D visual assets to output/3d_tumor_render.png and output/3d_orthogonal_slices.png!")