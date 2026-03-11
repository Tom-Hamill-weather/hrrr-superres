"""
Standalone visualization script for adaptive grid
Uses HRRR native Lambert Conformal projection with discrete color bins
"""

import os
import sys
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree
from datetime import datetime
import config


def load_hrrr_grid():
    """Load HRRR grid"""
    print("Loading HRRR grid...")
    hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')

    lats = np.load(os.path.join(hrrr_dir, 'hrrr_lats.npy'))
    lons = np.load(os.path.join(hrrr_dir, 'hrrr_lons.npy'))

    print(f"  HRRR grid: {lats.shape}")
    print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")

    return lats, lons


def load_adaptive_grid(nc_file):
    """Load adaptive grid points from netCDF"""
    print(f"\nLoading adaptive grid from {nc_file}...")

    with nc.Dataset(nc_file, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]

        # Check if tier information exists
        has_tiers = 'tier' in ds.variables
        if has_tiers:
            tiers = ds.variables['tier'][:]
        else:
            print("  ⚠ Warning: No tier information in file - all points will be treated equally")
            tiers = None

        print(f"  Total points: {len(lats):,}")
        print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
        print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")

        # Tier distribution
        if has_tiers:
            for tier in [0, 1, 2, 3, 4, 5]:
                count = (tiers == tier).sum()
                if count > 0:
                    pct = 100 * count / len(tiers)
                    print(f"    Tier {tier}: {count:,} points ({pct:.1f}%)")

    return lats, lons, tiers


def filter_points_to_hrrr_domain(point_lats, point_lons, point_tiers, hrrr_lats, hrrr_lons):
    """Filter adaptive grid points to only those within HRRR domain"""
    print("\nFiltering points to HRRR domain...")

    # Define HRRR domain bounds with small buffer
    lat_min, lat_max = hrrr_lats.min() - 0.1, hrrr_lats.max() + 0.1
    lon_min, lon_max = hrrr_lons.min() - 0.1, hrrr_lons.max() + 0.1

    # Filter points
    mask = ((point_lats >= lat_min) & (point_lats <= lat_max) &
            (point_lons >= lon_min) & (point_lons <= lon_max))

    filtered_lats = point_lats[mask]
    filtered_lons = point_lons[mask]
    filtered_tiers = point_tiers[mask] if point_tiers is not None else None

    n_filtered = (~mask).sum()
    print(f"  Filtered out {n_filtered:,} points outside HRRR domain")
    print(f"  Kept {mask.sum():,} points within domain")

    return filtered_lats, filtered_lons, filtered_tiers


def compute_point_density(point_lats, point_lons, hrrr_lats, hrrr_lons):
    """Compute number of adaptive grid points per HRRR cell"""
    print("\nComputing point density per HRRR cell...")

    shape = hrrr_lats.shape

    # Build KDTree for HRRR grid centers
    hrrr_points = np.column_stack([hrrr_lats.ravel(), hrrr_lons.ravel()])
    tree = cKDTree(hrrr_points)

    # Find nearest HRRR cell for each adaptive grid point
    adaptive_points = np.column_stack([point_lats, point_lons])
    _, indices = tree.query(adaptive_points, k=1)

    # VECTORIZED counting using bincount
    point_count_flat = np.bincount(indices, minlength=hrrr_points.shape[0])
    point_count = point_count_flat.reshape(shape)

    print(f"  Point density computed")
    print(f"    Max points per cell: {point_count.max():,}")
    print(f"    Mean points per cell: {point_count.mean():.1f}")
    print(f"    Median points per cell: {np.median(point_count):.1f}")
    print(f"    Cells with points: {(point_count > 0).sum():,} / {point_count.size:,}")

    return point_count


def create_hrrr_basemap(hrrr_lats, hrrr_lons):
    """Create Basemap using HRRR native Lambert Conformal projection"""
    print("\nCreating HRRR native projection map...")

    # HRRR uses Lambert Conformal Conic projection
    # Actual HRRR parameters from GRIB2
    lat_0 = 38.5  # Central latitude
    lon_0 = -97.5  # Central longitude
    lat_1 = 38.5  # First standard parallel
    lat_2 = 38.5  # Second standard parallel

    # First create a temporary projection to get grid extent
    temp_proj = Basemap(
        projection='lcc',
        lat_0=lat_0,
        lon_0=lon_0,
        lat_1=lat_1,
        lat_2=lat_2,
        llcrnrlat=21, urcrnrlat=53,
        llcrnrlon=-135, urcrnrlon=-60,
        resolution=None
    )

    # Convert HRRR grid corners to projection coordinates
    x_hrrr, y_hrrr = temp_proj(hrrr_lons, hrrr_lats)

    # Get actual extent in projection coordinates
    x_min, x_max = x_hrrr.min(), x_hrrr.max()
    y_min, y_max = y_hrrr.min(), y_hrrr.max()

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    # Create final Basemap using width and height to exactly match HRRR grid
    m = Basemap(
        projection='lcc',
        lat_0=lat_0,
        lon_0=lon_0,
        lat_1=lat_1,
        lat_2=lat_2,
        width=width,
        height=height,
        resolution='l',
        area_thresh=1000
    )

    print(f"  Projection: Lambert Conformal Conic")
    print(f"  Center: ({lat_0}°N, {abs(lon_0)}°W)")
    print(f"  Grid extent: {width/1000:.0f} km × {height/1000:.0f} km")

    return m


def define_density_bins():
    """Define discrete density bins for visualization with log scale"""
    # Use log-scale bins that match the actual data distribution
    # Density <= 1 is white, removed one intermediate blue-green color
    bins = [0, 2, 5, 10, 50, 100, 200, 1100]
    labels = ['≤1', '2-4', '5-9', '10-49', '50-99', '100-199', '200+']

    # Use diverse color palette - better spread across actual data range
    colors = [
        '#ffffff',  # 0-1 - white (no/minimal points)
        '#fde724',  # 2-4 - bright yellow
        '#b5de2b',  # 5-9 - yellow-green
        '#6ece58',  # 10-49 - green
        '#26828e',  # 50-99 - blue-teal
        '#3e4989',  # 100-199 - blue
        '#d62728',  # 200+ - red
    ]

    return bins, labels, colors


def create_discrete_colormap(colors, bins):
    """Create discrete colormap and normalization"""
    cmap = mcolors.ListedColormap(colors)

    # Create boundaries for discrete bins
    norm = mcolors.BoundaryNorm(bins, cmap.N)

    return cmap, norm


def visualize_adaptive_grid(hrrr_lats, hrrr_lons, point_density, output_file):
    """Create visualization with discrete color bins"""
    print("\nCreating visualization...")

    # Define bins and colors
    bins, labels, colors = define_density_bins()
    cmap, norm = create_discrete_colormap(colors, bins)

    # Create figure with better aspect ratio to reduce whitespace
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_axes([0.05, 0.05, 0.85, 0.90])  # [left, bottom, width, height]

    # Create basemap
    m = create_hrrr_basemap(hrrr_lats, hrrr_lons)

    # Draw map features
    print("  Drawing map features...")
    m.drawcoastlines(linewidth=0.5, color='black', zorder=5)
    m.drawcountries(linewidth=0.5, color='black', zorder=5)
    m.drawstates(linewidth=0.3, color='gray', zorder=5)

    # Draw parallels and meridians
    parallels = np.arange(20, 55, 5)
    meridians = np.arange(-130, -60, 10)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=13.5, linewidth=0.3, color='gray')
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=13.5, linewidth=0.3, color='gray')

    # Convert HRRR grid to map projection coordinates
    print("  Converting to map coordinates...")
    x, y = m(hrrr_lons, hrrr_lats)

    # Mask cells with zero points
    point_density_masked = np.ma.masked_where(point_density == 0, point_density)

    # Plot with pcolormesh using discrete colors
    print("  Plotting point density...")
    pcm = m.pcolormesh(
        x, y, point_density_masked,
        cmap=cmap,
        norm=norm,
        shading='auto',
        zorder=1,
        rasterized=True  # Faster rendering for large grids
    )

    # Create colorbar with discrete labels
    print("  Adding colorbar...")
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])  # [left, bottom, width, height]

    cbar = plt.colorbar(
        pcm,
        cax=cax,
        spacing='uniform',  # Equal visual space for each bin (log scale appearance)
        extend='max'  # Indicate values can exceed top bin
    )

    # Calculate tick positions at center of each color band
    # For BoundaryNorm with N boundaries, we have N-1 colors
    # Centers are at (bins[i] + bins[i+1]) / 2
    tick_positions = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(labels))]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(labels, fontsize=13)

    cbar.set_label('Points per HRRR Cell (log scale)', fontsize=14)

    # Title with statistics
    total_points = point_density.sum()
    cells_with_points = (point_density > 0).sum()
    total_cells = point_density.size

    ax.set_title(
        f'Adaptive Grid Point Density for XGBoost Weather Downscaling\n'
        f'Total Points: {total_points:,} | '
        f'HRRR Cells with Points: {cells_with_points:,} / {total_cells:,} '
        f'({100*cells_with_points/total_cells:.1f}%)',
        fontsize=20,
        pad=10
    )

    # Save figure
    print(f"  Saving to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✓ Visualization saved: {output_file} ({file_size_mb:.1f} MB)")


def main():
    """Main visualization routine"""
    print("\n" + "="*70)
    print(" ADAPTIVE GRID VISUALIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check for input file
    nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_SPARSE.nc')

    if not os.path.exists(nc_file):
        print(f"✗ Error: NetCDF file not found: {nc_file}")
        print("  Run generate_adaptive_grid_SPARSE.py first to create the grid")
        return 1

    # Load HRRR grid
    hrrr_lats, hrrr_lons = load_hrrr_grid()

    # Load adaptive grid
    point_lats, point_lons, point_tiers = load_adaptive_grid(nc_file)

    # Filter points to HRRR domain
    point_lats, point_lons, point_tiers = filter_points_to_hrrr_domain(
        point_lats, point_lons, point_tiers, hrrr_lats, hrrr_lons
    )

    # Compute point density
    point_density = compute_point_density(point_lats, point_lons, hrrr_lats, hrrr_lons)

    # Create visualization
    output_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_density_discrete.png')
    visualize_adaptive_grid(hrrr_lats, hrrr_lons, point_density, output_file)

    print("\n" + "="*70)
    print(" ✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
