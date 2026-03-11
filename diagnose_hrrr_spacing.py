"""
Diagnose actual HRRR grid spacing variations on Lambert Conformal projection
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_adaptive_grid import DataLoader
import config
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in meters using Haversine formula"""
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

print("Loading HRRR grid...")
loader = DataLoader()
loader.load_all()

lats = loader.hrrr_grid['lats']
lons = loader.hrrr_grid['lons']
ny, nx = lats.shape

print(f"HRRR grid shape: {ny} × {nx}")

# Calculate great-circle distances to eastern neighbor (j direction)
print("Calculating X-direction spacing...")
dx_distances = haversine_distance(
    lats[:, :-1], lons[:, :-1],
    lats[:, 1:], lons[:, 1:]
)

# Calculate great-circle distances to northern neighbor (i direction)
print("Calculating Y-direction spacing...")
dy_distances = haversine_distance(
    lats[:-1, :], lons[:-1, :],
    lats[1:, :], lons[1:, :]
)

print(f"\nX-direction (east-west) spacing:")
print(f"  Min: {dx_distances.min():.1f}m")
print(f"  Max: {dx_distances.max():.1f}m")
print(f"  Mean: {dx_distances.mean():.1f}m")
print(f"  Std: {dx_distances.std():.1f}m")
print(f"  Range: {dx_distances.max() - dx_distances.min():.1f}m ({100*(dx_distances.max()-dx_distances.min())/dx_distances.mean():.1f}% of mean)")

print(f"\nY-direction (north-south) spacing:")
print(f"  Min: {dy_distances.min():.1f}m")
print(f"  Max: {dy_distances.max():.1f}m")
print(f"  Mean: {dy_distances.mean():.1f}m")
print(f"  Std: {dy_distances.std():.1f}m")
print(f"  Range: {dy_distances.max() - dy_distances.min():.1f}m ({100*(dy_distances.max()-dy_distances.min())/dy_distances.mean():.1f}% of mean)")

# Visualize the variation
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# X-spacing map
im0 = axes[0, 0].imshow(dx_distances, cmap='viridis', aspect='auto')
axes[0, 0].set_title('X-direction (E-W) cell spacing (meters)', fontsize=12)
axes[0, 0].set_xlabel('Grid index (j)')
axes[0, 0].set_ylabel('Grid index (i)')
plt.colorbar(im0, ax=axes[0, 0], label='meters')

# Y-spacing map
im1 = axes[0, 1].imshow(dy_distances, cmap='viridis', aspect='auto')
axes[0, 1].set_title('Y-direction (N-S) cell spacing (meters)', fontsize=12)
axes[0, 1].set_xlabel('Grid index (j)')
axes[0, 1].set_ylabel('Grid index (i)')
plt.colorbar(im1, ax=axes[0, 1], label='meters')

# Histograms
axes[1, 0].hist(dx_distances.ravel(), bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(3000, color='red', linestyle='--', linewidth=2, label='Assumed 3000m')
axes[1, 0].axvline(dx_distances.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean {dx_distances.mean():.1f}m')
axes[1, 0].set_xlabel('Cell spacing (meters)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('X-direction spacing distribution', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(dy_distances.ravel(), bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(3000, color='red', linestyle='--', linewidth=2, label='Assumed 3000m')
axes[1, 1].axvline(dy_distances.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean {dy_distances.mean():.1f}m')
axes[1, 1].set_xlabel('Cell spacing (meters)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Y-direction spacing distribution', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
output_file = os.path.join(config.OUTPUT_DIR, 'hrrr_spacing_diagnosis.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Diagnostic saved: {output_file}")

# Focus on Catalina region
cat_lat_center = 33.4
cat_lon_center = -118.4

# Find nearest HRRR cell
lat_diffs = np.abs(lats - cat_lat_center)
lon_diffs = np.abs(lons - cat_lon_center)
total_diff = lat_diffs + lon_diffs
cat_i, cat_j = np.unravel_index(np.argmin(total_diff), lats.shape)

print(f"\nCatalina region (i={cat_i}, j={cat_j}):")
if cat_j < nx-1:
    print(f"  X-spacing: {dx_distances[cat_i, cat_j]:.2f}m")
if cat_i < ny-1:
    print(f"  Y-spacing: {dy_distances[cat_i, cat_j]:.2f}m")
print(f"  Difference from 3000m: X={dx_distances[cat_i, cat_j]-3000:.2f}m, Y={dy_distances[cat_i, cat_j]-3000:.2f}m")
