"""
Create a detailed scatter plot showing exact point positions
to visually verify gap elimination
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import config
import os

def visualize_point_grid(lat_center, lon_center, box_size_km=3):
    """Create detailed scatter plot of point positions"""

    # Check for trails version first, fall back to regular
    nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_SPARSE_trails.nc')
    if not os.path.exists(nc_file):
        nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_SPARSE.nc')

    # Load points
    lat_half = (box_size_km / 2.0) / 111.0
    lon_half = (box_size_km / 2.0) / (111.0 * np.cos(np.radians(lat_center)))

    lat_min = lat_center - lat_half
    lat_max = lat_center + lat_half
    lon_min = lon_center - lon_half
    lon_max = lon_center + lon_half

    with nc.Dataset(nc_file, 'r') as ds:
        all_lats = ds.variables['latitude'][:]
        all_lons = ds.variables['longitude'][:]

        # Check if tier information exists
        has_tiers = 'tier' in ds.variables
        if has_tiers:
            all_tiers = ds.variables['tier'][:]
        else:
            all_tiers = None

        in_box = (
            (all_lats >= lat_min) & (all_lats <= lat_max) &
            (all_lons >= lon_min) & (all_lons <= lon_max)
        )

        lats = all_lats[in_box]
        lons = all_lons[in_box]
        tiers = all_tiers[in_box] if has_tiers else None

    print(f"\nPoints in {box_size_km}×{box_size_km}km box: {len(lats):,}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color by tier
    if has_tiers:
        tier_colors = {
            0: 'red',
            1: 'orange',
            2: 'yellow',
            3: 'green',
            4: 'blue',
            5: 'purple'
        }

        for tier in sorted(np.unique(tiers)):
            mask = (tiers == tier)
            tier_lats = lats[mask]
            tier_lons = lons[mask]
            resolution = config.TIER_RESOLUTIONS[tier]

            ax.scatter(tier_lons, tier_lats,
                      c=tier_colors.get(tier, 'black'),
                      s=10, alpha=0.6,
                      label=f'Tier {tier} ({resolution}m): {len(tier_lats)} pts')
    else:
        # No tier information - plot all points in one color
        ax.scatter(lons, lats, c='blue', s=10, alpha=0.6,
                  label=f'All points: {len(lats)}')

    # Add grid lines at HRRR cell boundaries
    # HRRR cells are 3km = 0.027027 degrees
    hrrr_cell_size_deg = 3.0 / 111.0

    # Find cell boundaries in lat/lon
    lat_cells = np.arange(
        np.floor(lat_min / hrrr_cell_size_deg) * hrrr_cell_size_deg,
        np.ceil(lat_max / hrrr_cell_size_deg) * hrrr_cell_size_deg + hrrr_cell_size_deg/2,
        hrrr_cell_size_deg
    )
    lon_cells = np.arange(
        np.floor(lon_min / hrrr_cell_size_deg) * hrrr_cell_size_deg,
        np.ceil(lon_max / hrrr_cell_size_deg) * hrrr_cell_size_deg + hrrr_cell_size_deg/2,
        hrrr_cell_size_deg
    )

    # Draw HRRR cell boundaries
    for lat_line in lat_cells:
        ax.axhline(lat_line, color='gray', linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)
    for lon_line in lon_cells:
        ax.axvline(lon_line, color='gray', linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)

    # Mark center
    ax.plot(lon_center, lat_center, 'k+', markersize=20, markeredgewidth=2, label='Center', zorder=10)

    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(
        f'Adaptive Grid Point Positions (FIXED)\n'
        f'Center: ({lat_center:.4f}°N, {lon_center:.4f}°W) | '
        f'Box: {box_size_km}×{box_size_km}km\n'
        f'Gray dashed lines = HRRR cell boundaries (3km)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    output_file = os.path.join(config.OUTPUT_DIR, f'point_grid_detail_{box_size_km}km.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    return output_file


if __name__ == '__main__':
    lat_center = 33.4
    lon_center = -118.4

    print(f"\n{'='*70}")
    print(f" DETAILED POINT GRID VISUALIZATION")
    print(f"{'='*70}")

    # Create 3km box
    visualize_point_grid(lat_center, lon_center, box_size_km=3)

    # Also create 1.5km box for extreme zoom
    visualize_point_grid(lat_center, lon_center, box_size_km=1.5)

    print(f"\n{'='*70}")
    print(f" COMPLETE")
    print(f"{'='*70}\n")
