"""
python visualize_point_grid_binary.py clat clon
Visualize point grid for BINARY adaptive grid
Works with files that don't have tier information
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap
import config
import os
import argparse
import sys

def visualize_point_grid(lat_center, lon_center, box_size_deg=0.4, input_file='adaptive_grid_BINARY.nc'):
    """Create detailed scatter plot of point positions with geographic context"""

    nc_file = os.path.join(config.OUTPUT_DIR, input_file)

    # Load points - use degrees directly for larger domain
    lat_half = box_size_deg / 2.0
    lon_half = box_size_deg / 2.0

    lat_min = lat_center - lat_half
    lat_max = lat_center + lat_half
    lon_min = lon_center - lon_half
    lon_max = lon_center + lon_half

    with nc.Dataset(nc_file, 'r') as ds:
        all_lats = ds.variables['latitude'][:]
        all_lons = ds.variables['longitude'][:]

        in_box = (
            (all_lats >= lat_min) & (all_lats <= lat_max) &
            (all_lons >= lon_min) & (all_lons <= lon_max)
        )

        lats = all_lats[in_box]
        lons = all_lons[in_box]

    box_size_km = box_size_deg * 111  # approximate km
    print(f"\nPoints in {box_size_deg}° (~{box_size_km:.0f}km) box: {len(lats):,}")

    # Estimate local resolution by nearest neighbor distance
    if len(lats) > 1:
        coords = np.column_stack([lats, lons])
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)  # k=2 to get nearest neighbor (not self)
        nn_distances = distances[:, 1]  # Second column is nearest neighbor

        # Convert to meters (approximate)
        nn_distances_m = nn_distances * 111000  # degrees to meters

        # Assign colors based on spacing
        colors = []
        for dist_m in nn_distances_m:
            if dist_m < 150:  # ~93.75m
                colors.append('red')
            elif dist_m < 250:  # ~187.5m
                colors.append('orange')
            elif dist_m < 450:  # ~375m
                colors.append('gray')
            elif dist_m < 900:  # ~750m
                colors.append('green')
            elif dist_m < 1800:  # ~1500m
                colors.append('blue')
            else:  # ~3000m
                colors.append('purple')
    else:
        colors = ['black'] * len(lats)

    # Create figure with Basemap
    fig, ax = plt.subplots(figsize=(16, 14))

    # Create basemap
    m = Basemap(
        projection='merc',
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        resolution='f',  
        ax=ax
    )

    # Draw land/water background first
    m.fillcontinents(color='lightgray', lake_color='aqua', alpha=0.3, zorder=1)

    # Plot points colored by local spacing
    x, y = m(lons, lats)
    m.scatter(x, y, c=colors, s=2, alpha=0.7, zorder=2)

    # Draw geographic features last for visibility
    m.drawcoastlines(linewidth=2.0, color='black', zorder=5)
    m.drawstates(linewidth=1.0, color='black', zorder=4)
    m.drawcounties(linewidth=0.5, color='black', zorder=3)

    # Mark center
    x_center, y_center = m(lon_center, lat_center)
    ax.plot(x_center, y_center, 'k+', markersize=20, markeredgewidth=3, label='Center', zorder=10)

    ax.set_title(
        f'Adaptive Grid Point Positions\n'
        f'Center: ({lat_center:.4f}°N, {abs(lon_center):.4f}°W) | '
        f'Box: {box_size_deg}° (~{box_size_km:.0f}km)',
        fontsize=24, pad=20
    )

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='~94m spacing'),
        Patch(facecolor='orange', label='~188m spacing'),
        Patch(facecolor='gray', label='~375m spacing'),
        Patch(facecolor='green', label='~750m spacing'),
        Patch(facecolor='blue', label='~1.5km spacing'),
        Patch(facecolor='purple', label='~3km spacing'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, framealpha=0.9)

    plt.tight_layout()

    # Format lat/lon for filename
    lat_str = f"{abs(lat_center):.1f}{'N' if lat_center >= 0 else 'S'}"
    lon_str = f"{abs(lon_center):.1f}{'E' if lon_center >= 0 else 'W'}"

    output_file = os.path.join(config.OUTPUT_DIR,
                               f'point_grid_binary_{lat_str}_{lon_str}_{box_size_deg}deg.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize point grid for BINARY adaptive grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_point_grid_binary.py 33.4 -118.4
  python visualize_point_grid_binary.py 40.7 -74.0  # New York City
        '''
    )
    parser.add_argument('lat', type=float, nargs='?', default=33.4,
                        help='Center latitude (default: 33.4 for Catalina Island)')
    parser.add_argument('lon', type=float, nargs='?', default=-118.4,
                        help='Center longitude (default: -118.4 for Catalina Island)')
    parser.add_argument('--box-sizes', type=float, nargs='+',
                        default=[0.4, 0.2, 0.05],
                        help='Box sizes in degrees (default: 0.4 0.2 0.05)')
    parser.add_argument('--input-file', type=str, default='adaptive_grid_SPARSE.nc',
                        help='Input NetCDF file (default: adaptive_grid_SPARSE.nc)')

    args = parser.parse_args()

    lat_center = args.lat
    lon_center = args.lon

    print(f"\n{'='*70}")
    print(f" DETAILED POINT GRID VISUALIZATION (BINARY)")
    print(f" Center: {lat_center}°N, {abs(lon_center)}°W")
    print(f"{'='*70}")

    # Create visualizations for each box size
    for box_size in args.box_sizes:
        visualize_point_grid(lat_center, lon_center, box_size_deg=box_size,
                            input_file=args.input_file)

    print(f"\n{'='*70}")
    print(f" COMPLETE")
    print(f"{'='*70}\n")
