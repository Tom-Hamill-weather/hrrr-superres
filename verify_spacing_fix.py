"""
Verify that the spacing fix eliminated HRRR cell boundary artifacts
"""
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import config
import os

print("Loading regenerated grid...")
nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_BINARY.nc')
with nc.Dataset(nc_file, 'r') as ds:
    all_lats = ds.variables['latitude'][:]
    all_lons = ds.variables['longitude'][:]

print(f"Total points: {len(all_lats):,}")

# Focus on Catalina area where boundary artifacts were visible
lat_center = 33.4
lon_center = -118.4
box_size = 0.3  # degrees

mask = (
    (all_lats >= lat_center - box_size) & (all_lats <= lat_center + box_size) &
    (all_lons >= lon_center - box_size) & (all_lons <= lon_center + box_size)
)

lats = all_lats[mask]
lons = all_lons[mask]

print(f"Points in test region: {len(lats):,}")

# Compute nearest neighbor distances
coords = np.column_stack([lats, lons])
tree = cKDTree(coords)
distances, indices = tree.query(coords, k=2)
nn_dist = distances[:, 1] * 111000  # Convert to meters

print(f"\n{'='*70}")
print("NEAREST NEIGHBOR DISTANCE STATISTICS")
print(f"{'='*70}")
print(f"  Min: {nn_dist.min():.2f}m")
print(f"  Max: {nn_dist.max():.2f}m")
print(f"  Mean: {nn_dist.mean():.2f}m")
print(f"  Median: {np.median(nn_dist):.2f}m")
print(f"  Std dev: {nn_dist.std():.2f}m")
print(f"  25th percentile: {np.percentile(nn_dist, 25):.2f}m")
print(f"  75th percentile: {np.percentile(nn_dist, 75):.2f}m")

# Check for anomalies
very_close = np.sum(nn_dist < 50)
too_far = np.sum(nn_dist > 3500)

print(f"\n{'='*70}")
print("ANOMALY CHECK")
print(f"{'='*70}")
print(f"  Points closer than 50m: {very_close} ({100*very_close/len(nn_dist):.3f}%)")
print(f"  Points farther than 3500m: {too_far} ({100*too_far/len(nn_dist):.3f}%)")

# Analyze spacing uniformity at HRRR cell boundaries
hrrr_cell_size_deg = 3.0 / 111.0

# For each point, compute its position within its HRRR cell
lat_cell_pos = (lats % hrrr_cell_size_deg) / hrrr_cell_size_deg
lon_cell_pos = (lons % hrrr_cell_size_deg) / hrrr_cell_size_deg

# Points near cell boundaries (within 5% of cell size from edge)
near_boundary = (
    (lat_cell_pos < 0.05) | (lat_cell_pos > 0.95) |
    (lon_cell_pos < 0.05) | (lon_cell_pos > 0.95)
)

# Points in interior (between 20% and 80% of cell)
interior = (
    (lat_cell_pos >= 0.2) & (lat_cell_pos <= 0.8) &
    (lon_cell_pos >= 0.2) & (lon_cell_pos <= 0.8)
)

boundary_spacing = nn_dist[near_boundary]
interior_spacing = nn_dist[interior]

print(f"\n{'='*70}")
print("BOUNDARY vs INTERIOR SPACING")
print(f"{'='*70}")
print(f"  Boundary points: {np.sum(near_boundary):,}")
print(f"    Mean spacing: {boundary_spacing.mean():.2f}m")
print(f"    Std dev: {boundary_spacing.std():.2f}m")
print(f"    Min: {boundary_spacing.min():.2f}m")
print(f"    Max: {boundary_spacing.max():.2f}m")

print(f"\n  Interior points: {np.sum(interior):,}")
print(f"    Mean spacing: {interior_spacing.mean():.2f}m")
print(f"    Std dev: {interior_spacing.std():.2f}m")
print(f"    Min: {interior_spacing.min():.2f}m")
print(f"    Max: {interior_spacing.max():.2f}m")

# Statistical test: are boundary and interior distributions similar?
mean_diff = abs(boundary_spacing.mean() - interior_spacing.mean())
std_diff = abs(boundary_spacing.std() - interior_spacing.std())

print(f"\n  Difference (Boundary - Interior):")
print(f"    Mean difference: {boundary_spacing.mean() - interior_spacing.mean():.2f}m")
print(f"    Std dev difference: {boundary_spacing.std() - interior_spacing.std():.2f}m")

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

if mean_diff < 5.0 and std_diff < 5.0:
    print("✓ PASS: Boundary and interior spacing are nearly identical")
    print("  The spacing fix successfully eliminated HRRR cell boundary artifacts!")
elif mean_diff < 10.0 and std_diff < 10.0:
    print("⚠ MARGINAL: Small differences remain between boundary and interior")
    print("  Most artifacts should be eliminated, but minor variations persist")
else:
    print("✗ FAIL: Significant differences remain between boundary and interior")
    print("  HRRR cell boundary artifacts may still be visible")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Histogram comparison
axes[0, 0].hist(boundary_spacing, bins=50, alpha=0.7, label='Boundary', edgecolor='black', density=True)
axes[0, 0].hist(interior_spacing, bins=50, alpha=0.7, label='Interior', edgecolor='black', density=True)
axes[0, 0].set_xlabel('Nearest Neighbor Distance (m)', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].set_title('Spacing Distribution: Boundary vs Interior', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(93.75, color='red', linestyle='--', alpha=0.5, label='Expected min')

# 2D position plot
sc = axes[0, 1].scatter(lon_cell_pos, lat_cell_pos, s=1, alpha=0.3, c=nn_dist,
                        cmap='viridis', vmin=90, vmax=200)
axes[0, 1].set_xlabel('Longitude position within HRRR cell', fontsize=11)
axes[0, 1].set_ylabel('Latitude position within HRRR cell', fontsize=11)
axes[0, 1].set_title('Point spacing vs position within HRRR cell', fontsize=12)
axes[0, 1].axhline(0.05, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[0, 1].axhline(0.95, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[0, 1].axvline(0.05, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[0, 1].axvline(0.95, color='red', linestyle='--', alpha=0.3, linewidth=1)
plt.colorbar(sc, ax=axes[0, 1], label='NN distance (m)')

# Overall histogram
axes[1, 0].hist(nn_dist, bins=100, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Nearest Neighbor Distance (m)', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Overall Spacing Distribution', fontsize=12)
axes[1, 0].axvline(93.75, color='red', linestyle='--', linewidth=2, label='Expected min (93.75m)')
axes[1, 0].axvline(nn_dist.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Mean ({nn_dist.mean():.1f}m)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Quantile-Quantile plot to compare distributions
from scipy import stats
axes[1, 1].remove()
axes[1, 1] = fig.add_subplot(2, 2, 4)

# Sample to same size for comparison
n_samples = min(len(boundary_spacing), len(interior_spacing))
boundary_sample = np.random.choice(boundary_spacing, n_samples, replace=False)
interior_sample = np.random.choice(interior_spacing, n_samples, replace=False)

boundary_sorted = np.sort(boundary_sample)
interior_sorted = np.sort(interior_sample)

axes[1, 1].scatter(interior_sorted, boundary_sorted, s=1, alpha=0.5)
axes[1, 1].plot([90, 200], [90, 200], 'r--', linewidth=2, label='y=x (perfect match)')
axes[1, 1].set_xlabel('Interior spacing quantiles (m)', fontsize=11)
axes[1, 1].set_ylabel('Boundary spacing quantiles (m)', fontsize=11)
axes[1, 1].set_title('Q-Q Plot: Boundary vs Interior', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_aspect('equal')

plt.tight_layout()
output_file = os.path.join(config.OUTPUT_DIR, 'spacing_fix_verification.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Verification plot saved: {output_file}")
