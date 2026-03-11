"""
Diagnose where the very close point pairs are coming from
"""
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import config
import os

print("Loading grid...")
nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_BINARY.nc')
with nc.Dataset(nc_file, 'r') as ds:
    all_lats = ds.variables['latitude'][:]
    all_lons = ds.variables['longitude'][:]

# Focus on a smaller region for detailed analysis
lat_center = 33.4
lon_center = -118.4
box_size = 0.05  # Much smaller box

mask = (
    (all_lats >= lat_center - box_size) & (all_lats <= lat_center + box_size) &
    (all_lons >= lon_center - box_size) & (all_lons <= lon_center + box_size)
)

lats = all_lats[mask]
lons = all_lons[mask]

print(f"Points in test region: {len(lats):,}")

# Find nearest neighbors
coords = np.column_stack([lats, lons])
tree = cKDTree(coords)
distances, indices = tree.query(coords, k=2)
nn_dist = distances[:, 1] * 111000  # Convert to meters

# Find very close pairs
very_close_mask = nn_dist < 50
n_close = np.sum(very_close_mask)

print(f"\nVery close pairs (< 50m): {n_close} ({100*n_close/len(nn_dist):.2f}%)")

if n_close > 0:
    # Analyze the close pairs
    close_distances = nn_dist[very_close_mask]
    close_lats = lats[very_close_mask]
    close_lons = lons[very_close_mask]
    close_neighbor_idx = indices[very_close_mask, 1]

    print(f"\nClose pair statistics:")
    print(f"  Min distance: {close_distances.min():.2f}m")
    print(f"  Max distance: {close_distances.max():.2f}m")
    print(f"  Mean distance: {close_distances.mean():.2f}m")
    print(f"  Median distance: {np.median(close_distances):.2f}m")

    # Check if close pairs form patterns
    hrrr_cell_size_deg = 3.0 / 111.0

    # For close pairs, check their position within HRRR cells
    lat_cell_pos = (close_lats % hrrr_cell_size_deg) / hrrr_cell_size_deg
    lon_cell_pos = (close_lons % hrrr_cell_size_deg) / hrrr_cell_size_deg

    # Are they near HRRR cell boundaries?
    near_hrrr_boundary = (
        (lat_cell_pos < 0.02) | (lat_cell_pos > 0.98) |
        (lon_cell_pos < 0.02) | (lon_cell_pos > 0.98)
    )

    print(f"\n  Close pairs near HRRR boundaries: {np.sum(near_hrrr_boundary)} ({100*np.sum(near_hrrr_boundary)/len(near_hrrr_boundary):.1f}%)")

    # Check for patch boundaries (5×5 HRRR cells = patches)
    patch_size_hrrr = 5
    patch_size_deg = patch_size_hrrr * hrrr_cell_size_deg

    lat_patch_pos = (close_lats % patch_size_deg) / patch_size_deg
    lon_patch_pos = (close_lons % patch_size_deg) / patch_size_deg

    near_patch_boundary = (
        (lat_patch_pos < 0.02) | (lat_patch_pos > 0.98) |
        (lon_patch_pos < 0.02) | (lon_patch_pos > 0.98)
    )

    print(f"  Close pairs near patch boundaries: {np.sum(near_patch_boundary)} ({100*np.sum(near_patch_boundary)/len(near_patch_boundary):.1f}%)")

    # Sample some close pairs to examine
    print(f"\nSample of closest pairs:")
    closest_idx = np.argsort(close_distances)[:10]
    for i in closest_idx:
        idx1 = np.where(very_close_mask)[0][i]
        idx2 = indices[idx1, 1]
        dist_m = nn_dist[idx1]
        lat1, lon1 = lats[idx1], lons[idx1]
        lat2, lon2 = lats[idx2], lons[idx2]
        print(f"  {dist_m:5.2f}m: ({lat1:.6f}, {lon1:.6f}) <-> ({lat2:.6f}, {lon2:.6f})")
        print(f"         Δlat={abs(lat2-lat1)*111000:.2f}m, Δlon={abs(lon2-lon1)*111000*np.cos(np.radians(lat1)):.2f}m")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Position within HRRR cell
    axes[0].scatter(lon_cell_pos, lat_cell_pos, c=close_distances, s=20, alpha=0.6,
                    cmap='hot', vmin=0, vmax=50)
    axes[0].axhline(0.02, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].axhline(0.98, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].axvline(0.02, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].axvline(0.98, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[0].set_xlabel('Lon position within HRRR cell')
    axes[0].set_ylabel('Lat position within HRRR cell')
    axes[0].set_title('Close pairs (<50m) vs HRRR cell position')
    axes[0].grid(alpha=0.3)

    # Position within patch
    axes[1].scatter(lon_patch_pos, lat_patch_pos, c=close_distances, s=20, alpha=0.6,
                    cmap='hot', vmin=0, vmax=50)
    axes[1].axhline(0.02, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axhline(0.98, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axvline(0.02, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axvline(0.98, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].set_xlabel('Lon position within patch (5×5 HRRR cells)')
    axes[1].set_ylabel('Lat position within patch (5×5 HRRR cells)')
    axes[1].set_title('Close pairs (<50m) vs patch position')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(config.OUTPUT_DIR, 'close_pairs_diagnosis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved: {output_file}")
