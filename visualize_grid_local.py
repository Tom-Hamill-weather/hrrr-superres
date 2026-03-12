"""
Local visualization of adaptive grid points

Usage:
    python visualize_grid_local.py <latitude> <longitude> [--netcdf <file>]

Creates a ~20x20 km map centered on the input coordinates showing:
- Individual adaptive grid points as small black dots
- Coastlines and state boundaries (lw=1, color='Black')
- County borders (lw=0.4, color='Gray')

Example:
    python visualize_grid_local.py 40.7 -111.9  # Salt Lake City area
"""

import os
import sys
import argparse
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import config


def load_adaptive_grid_points(nc_file, lat_center, lon_center, box_size_km=20):
    """
    Load adaptive grid points within a box around the center coordinates

    Args:
        nc_file: Path to netCDF file with adaptive grid points
        lat_center: Center latitude
        lon_center: Center longitude
        box_size_km: Size of box in km (default 20 km)

    Returns:
        lats, lons, tiers: Arrays of points within the box
    """
    print(f"\nLoading adaptive grid from {nc_file}...")
    print(f"  Center: ({lat_center:.4f}, {lon_center:.4f})")
    print(f"  Box size: {box_size_km} km × {box_size_km} km")

    # Approximate degrees for box size
    # At mid-latitudes: ~111 km per degree latitude, ~111*cos(lat) km per degree longitude
    lat_half = (box_size_km / 2.0) / 111.0
    lon_half = (box_size_km / 2.0) / (111.0 * np.cos(np.radians(lat_center)))

    lat_min = lat_center - lat_half
    lat_max = lat_center + lat_half
    lon_min = lon_center - lon_half
    lon_max = lon_center + lon_half

    print(f"  Lat range: {lat_min:.4f} to {lat_max:.4f}")
    print(f"  Lon range: {lon_min:.4f} to {lon_max:.4f}")

    # Load all points from netCDF
    with nc.Dataset(nc_file, 'r') as ds:
        all_lats = ds.variables['latitude'][:]
        all_lons = ds.variables['longitude'][:]

        # Check if tier information exists
        has_tiers = 'tier' in ds.variables
        if has_tiers:
            all_tiers = ds.variables['tier'][:]
        else:
            print("  ⚠ Warning: No tier information in file")
            all_tiers = None

        # Filter to box
        in_box = (
            (all_lats >= lat_min) & (all_lats <= lat_max) &
            (all_lons >= lon_min) & (all_lons <= lon_max)
        )

        lats = all_lats[in_box]
        lons = all_lons[in_box]
        tiers = all_tiers[in_box] if has_tiers else None

    print(f"  Points in box: {len(lats):,}")

    if len(lats) == 0:
        print("  ⚠ WARNING: No points found in this region!")
        return None, None, None

    # Tier distribution
    if has_tiers:
        print(f"  Tier distribution:")
        for tier in range(6):  # SPARSE has tiers 0-5
            count = (tiers == tier).sum()
            if count > 0:
                pct = 100 * count / len(tiers)
                res = config.TIER_RESOLUTIONS.get(tier, 'N/A')
                print(f"    Tier {tier} ({res}m): {count:,} points ({pct:.1f}%)")

    return lats, lons, tiers


def create_local_map(lats, lons, lat_center, lon_center, box_size_km, output_file):
    """
    Create local map visualization showing individual points

    Args:
        lats, lons: Arrays of point coordinates
        lat_center, lon_center: Center coordinates
        box_size_km: Size of box in km
        output_file: Output filename for figure
    """
    print("\nCreating local map visualization...")

    # Calculate map extent (add small margin)
    margin_km = 2  # 2 km margin on each side
    total_km = box_size_km + 2 * margin_km
    lat_half = (total_km / 2.0) / 111.0
    lon_half = (total_km / 2.0) / (111.0 * np.cos(np.radians(lat_center)))

    lat_min = lat_center - lat_half
    lat_max = lat_center + lat_half
    lon_min = lon_center - lon_half
    lon_max = lon_center + lon_half

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Create basemap
    # Use Lambert Conformal Conic projection centered on the region
    m = Basemap(
        projection='lcc',
        resolution='h',  # High resolution for local area
        area_thresh=10.,  # Show small features
        lat_0=lat_center,
        lon_0=lon_center,
        lat_1=lat_center - 0.5,
        lat_2=lat_center + 0.5,
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        ax=ax
    )

    # Draw map boundaries
    print("  Drawing map boundaries...")
    m.drawcoastlines(linewidth=1.0, color='Black')
    m.drawstates(linewidth=1.0, color='Black')
    m.drawcounties(linewidth=0.4, color='Gray')

    # Plot individual points as small black dots
    print(f"  Plotting {len(lats):,} points...")
    x, y = m(lons, lats)
    m.plot(x, y, 'k.', markersize=1.5, alpha=0.7)

    # Add grid lines for reference
    parallels = np.arange(np.floor(lat_min), np.ceil(lat_max), 0.1)
    meridians = np.arange(np.floor(lon_min), np.ceil(lon_max), 0.1)
    m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.3, color='LightGray', fontsize=8)
    m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.3, color='LightGray', fontsize=8)

    # Add center point marker
    x_center, y_center = m(lon_center, lat_center)
    m.plot(x_center, y_center, 'r+', markersize=15, markeredgewidth=2, label='Center')

    # Title
    ax.set_title(
        f'Adaptive Grid Points - Local View\n'
        f'Center: ({lat_center:.4f}°N, {lon_center:.4f}°W) | '
        f'Box: {box_size_km}×{box_size_km} km | '
        f'Points: {len(lats):,}',
        fontsize=14, fontweight='bold', pad=20
    )

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Save figure
    print(f"\n  Saving figure to {output_file}...")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✓ Local visualization saved: {output_file} ({file_size_mb:.2f} MB)")


def main():
    """Main visualization routine"""
    parser = argparse.ArgumentParser(
        description='Visualize adaptive grid points in a local area',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_grid_local.py 40.7 -111.9              # Salt Lake City
  python visualize_grid_local.py 39.7 -105.0              # Denver
  python visualize_grid_local.py 33.7 -84.4               # Atlanta
  python visualize_grid_local.py 47.6 -122.3              # Seattle
  python visualize_grid_local.py 40.7 -111.9 --box 30     # 30×30 km box
  python visualize_grid_local.py 40.7 -111.9 --netcdf output/adaptive_grid_SPARSE.nc
        """
    )

    parser.add_argument('latitude', type=float, help='Center latitude (degrees North)')
    parser.add_argument('longitude', type=float, help='Center longitude (degrees, negative for West)')
    parser.add_argument('--box', type=float, default=20.0, help='Box size in km (default: 20)')
    parser.add_argument('--netcdf', type=str, default=None,
                       help='Path to netCDF file (default: output/adaptive_grid_SPARSE_trails.nc)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: auto-generated)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID LOCAL VISUALIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Determine netCDF file
    if args.netcdf:
        nc_file = args.netcdf
    else:
        # Try trails version first, then SPARSE, then fall back to GEN2, then original
        nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_SPARSE_trails.nc')
        if not os.path.exists(nc_file):
            nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_SPARSE.nc')
            if not os.path.exists(nc_file):
                nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_GEN2.nc')
                if not os.path.exists(nc_file):
                    nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_points.nc')

    if not os.path.exists(nc_file):
        print(f"✗ Error: NetCDF file not found: {nc_file}")
        print("  Run generate_adaptive_grid_SPARSE.py first to create the grid")
        print("  Or specify a file with --netcdf <path>")
        return 1

    # Load points in local area
    lats, lons, tiers = load_adaptive_grid_points(
        nc_file,
        args.latitude,
        args.longitude,
        args.box
    )

    if lats is None:
        print("\n✗ Error: No points found in the specified region")
        print("  Try a different location or larger box size (--box)")
        return 1

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Auto-generate filename
        lat_str = f"{abs(args.latitude):.2f}{'N' if args.latitude >= 0 else 'S'}"
        lon_str = f"{abs(args.longitude):.2f}{'E' if args.longitude >= 0 else 'W'}"
        output_file = os.path.join(
            config.OUTPUT_DIR,
            f'adaptive_grid_local_{lat_str}_{lon_str}_{int(args.box)}km.png'
        )

    # Create visualization
    create_local_map(lats, lons, args.latitude, args.longitude, args.box, output_file)

    print("\n" + "="*70)
    print(" ✓ LOCAL VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
