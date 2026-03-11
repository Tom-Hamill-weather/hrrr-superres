"""
Fixed visualization using GRAF plotting style
- Shows full CONUS using HRRR grid corners
- Better color palette with discrete bins
- Masks ocean areas
"""

import os
import sys
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import matplotlib.pyplot as plt
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
    print(f"  SW corner: ({lats[0,0]:.2f}, {lons[0,0]:.2f})")
    print(f"  NE corner: ({lats[-1,-1]:.2f}, {lons[-1,-1]:.2f})")

    return lats, lons


def load_adaptive_grid(nc_file):
    """Load adaptive grid points from netCDF"""
    print(f"\nLoading adaptive grid from {nc_file}...")

    with nc.Dataset(nc_file, 'r') as ds:
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
        tiers = ds.variables['tier'][:]

        print(f"  Total points: {len(lats):,}")

        # Tier distribution
        for tier in [1, 2, 3, 4, 5]:
            count = (tiers == tier).sum()
            pct = 100 * count / len(tiers)
            print(f"    Tier {tier}: {count:,} points ({pct:.1f}%)")

    return lats, lons, tiers


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

    # Vectorized counting
    point_count_flat = np.bincount(indices, minlength=hrrr_points.shape[0])
    point_count = point_count_flat.reshape(shape)

    print(f"  Point density computed")
    print(f"    Max points per cell: {point_count.max():,}")
    print(f"    Mean points per cell: {point_count.mean():.1f}")
    print(f"    Cells with points: {(point_count > 0).sum():,} / {point_count.size:,}")

    return point_count


def visualize_adaptive_grid(hrrr_lats, hrrr_lons, point_density, output_file, ny, nx):
    """Create visualization using GRAF style"""
    print("\nCreating visualizations (GRAF style)...")

    # Define discrete bins and colors (GRAF-inspired)
    # Bins for point density
    clevs = [0.99, 4, 10, 50, 100, 200, 500, 1000]
    colorst = [
        '#C4E8FF',  
        '#8FB3FF',  
        '#D8F9D8',  
        '#A6ECA6',  
        '#42F742',  
        'Gold',
        'Orange',
        '#FA5257'   
    ]

    # HRRR projection parameters (standard CONUS HRRR)
    lat_0 = 38.5
    lon_0 = -97.5
    lat_1 = 38.5
    lat_2 = 38.5

    # Mask cells with zero points (will show as white/background)
    point_density_masked = ma.masked_where(point_density == 0, point_density)

    # Statistics for titles
    total_points = int(point_density.sum())
    cells_with_points = (point_density > 0).sum()
    total_cells = point_density.size

    # ========== PLOT 1: Full CONUS ==========
    print("  Creating full CONUS map...")

    fig1 = plt.figure(figsize=(10, 8))
    axloc = [0.02, 0.02, 0.96, 0.96]
    ax1 = fig1.add_axes(axloc)

    m1 = Basemap(
        rsphere=(6378137.00, 6356752.3142),
        resolution='l',
        area_thresh=1000.,
        projection='lcc',
        lat_1=lat_1,
        lat_2=lat_2,
        lat_0=lat_0,
        lon_0=lon_0,
        llcrnrlon=hrrr_lons[0, 0],
        llcrnrlat=hrrr_lats[0, 0],
        urcrnrlon=hrrr_lons[-1, -1],
        urcrnrlat=hrrr_lats[-1, -1]
    )

    x1, y1 = m1(hrrr_lons, hrrr_lats)

    print ('point_density_masked[ny//2, 0:nx:10] = ', \
        point_density_masked[ny//2, 0:nx:10] )
    CS1 = m1.contourf(
        x1, y1, point_density_masked,
        clevs,
        cmap=None,
        colors=colorst,
        extend='max'
    )

    m1.drawcoastlines(linewidth=0.3, color='LightGray')
    m1.drawcountries(linewidth=0.4, color='Gray')
    m1.drawstates(linewidth=0.2, color='Gray')

    # Colorbar for CONUS
    cax1 = fig1.add_axes([0.02, 0.06, 0.96, 0.03])
    cb1 = plt.colorbar(
        CS1,
        orientation='horizontal',
        cax=cax1,
        drawedges=True,
        ticks=clevs,
        format='%g'
    )
    cb1.ax.tick_params(labelsize=9)
    cb1.set_label('Adaptive Grid Points per HRRR Cell', fontsize=11)

    # Title for CONUS
    ax1.set_title(
        f'Adaptive Grid Point Density for XGBoost Weather Downscaling\n'
        f'Total Points: {total_points:,} | '
        f'HRRR Cells: {cells_with_points:,}/{total_cells:,} '
        f'({100*cells_with_points/total_cells:.1f}% coverage)',
        fontsize=14, color='Black'
    )

    # Save CONUS figure
    print(f"  Saving CONUS map to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✓ CONUS visualization saved: {output_file} ({file_size_mb:.1f} MB)")

    # ========== PLOT 2: Utah/Colorado Zoom ==========
    print("  Creating Utah/Colorado zoom map...")

    fig2 = plt.figure(figsize=(10, 8))
    axloc = [0.02, 0.12, 0.96, 0.78]
    ax2 = fig2.add_axes(axloc)

    # Define zoom region (Utah + Colorado)
    zoom_lat_min, zoom_lat_max = 36.5, 42.5
    zoom_lon_min, zoom_lon_max = -114.5, -101.5

    m2 = Basemap(
        rsphere=(6378137.00, 6356752.3142),
        resolution='h',  # Higher resolution for zoom
        area_thresh=100.,
        projection='lcc',
        lat_1=lat_1,
        lat_2=lat_2,
        lat_0=39.5,  # Center on region
        lon_0=-108,
        llcrnrlon=zoom_lon_min,
        llcrnrlat=zoom_lat_min,
        urcrnrlon=zoom_lon_max,
        urcrnrlat=zoom_lat_max
    )

    x2, y2 = m2(hrrr_lons, hrrr_lats)

    CS2 = m2.contourf(
        x2, y2, point_density_masked,
        clevs,
        cmap=None,
        colors=colorst,
        extend='max'
    )

    m2.drawcoastlines(linewidth=0.3, color='LightGray')
    m2.drawcountries(linewidth=0.4, color='Gray')
    m2.drawstates(linewidth=0.9, color='Gray')  # 3x wider for zoom
    m2.drawcounties(linewidth=0.3, color='LightGray')  # Add county boundaries

    # Colorbar for zoom
    cax2 = fig2.add_axes([0.02, 0.06, 0.96, 0.03])
    cb2 = plt.colorbar(
        CS2,
        orientation='horizontal',
        cax=cax2,
        drawedges=True,
        ticks=clevs,
        format='%g'
    )
    cb2.ax.tick_params(labelsize=9)
    cb2.set_label('Adaptive Grid Points per HRRR Cell', fontsize=11)

    # Title for zoom
    ax2.set_title(
        'Adaptive Grid Point Density - Utah & Colorado',
        fontsize=12,
        color='Black'
    )

    # Save zoom figure
    zoom_file = output_file.replace('.png', '_utah_colorado.png')
    print(f"  Saving Utah/Colorado zoom to {zoom_file}...")
    plt.savefig(zoom_file, dpi=300, bbox_inches='tight')
    plt.close()

    file_size_mb = os.path.getsize(zoom_file) / (1024**2)
    print(f"✓ Utah/Colorado zoom saved: {zoom_file} ({file_size_mb:.1f} MB)")


def main():
    """Main visualization routine"""
    print("\n" + "="*70)
    print(" ADAPTIVE GRID VISUALIZATION (GRAF STYLE)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check for input file
    nc_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_points.nc')

    if not os.path.exists(nc_file):
        print(f"✗ Error: NetCDF file not found: {nc_file}")
        print("  Run generate_adaptive_grid_SUPERFAST.py first to create the grid")
        return 1

    # Load HRRR grid
    hrrr_lats, hrrr_lons = load_hrrr_grid()
    ny, nx = np.shape(hrrr_lats)

    # Load adaptive grid
    point_lats, point_lons, point_tiers = load_adaptive_grid(nc_file)

    # Compute point density
    point_density = compute_point_density(point_lats, point_lons, hrrr_lats, hrrr_lons)

    # Create visualization
    output_file = os.path.join(config.OUTPUT_DIR, 'adaptive_grid_density_GRAF_style.png')
    visualize_adaptive_grid(hrrr_lats, hrrr_lons, point_density, output_file, ny, nx)

    print("\n" + "="*70)
    print(" ✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
