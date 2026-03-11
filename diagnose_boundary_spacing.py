"""
Diagnose spacing issues at HRRR cell boundaries
"""
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import config
import os

# Load the grid
nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_BINARY.nc')
with nc.Dataset(nc_file, 'r') as ds:
    all_lats = ds.variables['latitude'][:]
    all_lons = ds.variables['longitude'][:]

print(f"Total points: {len(all_lats):,}")

# Focus on Catalina area
lat_center = 33.4
lon_center = -118.4
box_size = 0.1  # degrees

mask = (
    (all_lats >= lat_center - box_size) & (all_lats <= lat_center + box_size) &
    (all_lons >= lon_center - box_size) & (all_lons <= lon_center + box_size)
)

lats = all_lats[mask]
lons = all_lons[mask]

print(f"Points in test region: {len(lats):,}")

# Check for near-duplicates
coords = np.column_stack([lats, lons])
tree = cKDTree(coords)

# Find nearest neighbor distances
distances, indices = tree.query(coords, k=2)
nn_dist = distances[:, 1] * 111000  # Convert to meters

# Statistics
print(f"\nNearest neighbor distance statistics (meters):")
print(f"  Min: {nn_dist.min():.2f}")
print(f"  Max: {nn_dist.max():.2f}")
print(f"  Mean: {nn_dist.mean():.2f}")
print(f"  Median: {np.median(nn_dist):.2f}")
print(f"  Std: {nn_dist.std():.2f}")

# Check for suspiciously close points
very_close = nn_dist < 50  # Less than 50m
if np.any(very_close):
    print(f"\n⚠ Found {np.sum(very_close)} points closer than 50m to their neighbor!")
    print(f"  Distances: {nn_dist[very_close]}")

# Check HRRR cell boundary alignment
# HRRR cell size is 3km = ~0.027 degrees
hrrr_cell_size_deg = 3.0 / 111.0

# For each point, compute its position within its HRRR cell
lat_cell_pos = (lats % hrrr_cell_size_deg) / hrrr_cell_size_deg
lon_cell_pos = (lons % hrrr_cell_size_deg) / hrrr_cell_size_deg

# Points near cell boundaries (within 10% of cell size from edge)
near_boundary = (
    (lat_cell_pos < 0.1) | (lat_cell_pos > 0.9) |
    (lon_cell_pos < 0.1) | (lon_cell_pos > 0.9)
)

print(f"\nPoints near HRRR cell boundaries: {np.sum(near_boundary):,} ({100*np.sum(near_boundary)/len(lats):.1f}%)")

# Compare spacing at boundaries vs interior
boundary_spacing = nn_dist[near_boundary]
interior_spacing = nn_dist[~near_boundary]

print(f"\nSpacing comparison:")
print(f"  At boundaries - Mean: {boundary_spacing.mean():.2f}m, Median: {np.median(boundary_spacing):.2f}m")
print(f"  In interior - Mean: {interior_spacing.mean():.2f}m, Median: {np.median(interior_spacing):.2f}m")

# Look for anomalously close pairs at boundaries
close_at_boundary = near_boundary & (nn_dist < 80)
if np.any(close_at_boundary):
    print(f"\n⚠ Found {np.sum(close_at_boundary)} boundary points with spacing < 80m")
    print(f"  Min boundary spacing: {nn_dist[near_boundary].min():.2f}m")

# Create histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(boundary_spacing, bins=50, alpha=0.7, label='Boundary', edgecolor='black')
axes[0].hist(interior_spacing, bins=50, alpha=0.7, label='Interior', edgecolor='black')
axes[0].set_xlabel('Nearest Neighbor Distance (m)')
axes[0].set_ylabel('Count')
axes[0].set_title('Spacing Distribution: Boundary vs Interior')
axes[0].legend()
axes[0].axvline(93.75, color='red', linestyle='--', label='Expected minimum (93.75m)')
axes[0].grid(alpha=0.3)

axes[1].scatter(lon_cell_pos, lat_cell_pos, s=1, alpha=0.3, c=nn_dist, cmap='viridis', vmin=0, vmax=300)
axes[1].set_xlabel('Longitude position within HRRR cell')
axes[1].set_ylabel('Latitude position within HRRR cell')
axes[1].set_title('Point spacing vs position within HRRR cell')
axes[1].axhline(0.1, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(0.9, color='red', linestyle='--', alpha=0.5)
axes[1].axvline(0.1, color='red', linestyle='--', alpha=0.5)
axes[1].axvline(0.9, color='red', linestyle='--', alpha=0.5)
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('NN distance (m)')

plt.tight_layout()
output_file = os.path.join(config.OUTPUT_DIR, 'boundary_spacing_diagnosis.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Diagnostic plot saved: {output_file}")
