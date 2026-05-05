"""
python visualize_hires_grid.py <nc_file> <lat> <lon> [--box-sizes ...]

Visualize point grid from a hi-res adaptive grid NetCDF file.
Works with hires_points.nc (output of generate_hires_points.py)
and also with the older adaptive_grid_points.nc format.
"""
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap
import config
import os
import argparse
import sys


def load_terrain(lat_min, lat_max, lon_min, lon_max):
    """Load terrain elevation subset from WRF_CONUS_terrain_info.nc for a bounding box.

    The WRF grid is Lambert Conformal (curvilinear), so lat/lon vary in 2D.
    We use a coarse-then-fine two-step search to avoid loading the full 3167×5387
    arrays while still finding the correct index region for any CONUS location.
    """
    terrain_file = os.path.join(config.BASE_DIR, 'WRF_CONUS_terrain_info.nc')
    buf = 0.05  # degree margin added around the target box

    with nc.Dataset(terrain_file, 'r') as ds:
        nrows, ncols = ds.variables['lats'].shape

        # Step 1: coarse search — sample every 20th point (~170 k values per variable)
        step = 20
        lats_c = ds.variables['lats'][::step, ::step]
        lons_c = ds.variables['lons'][::step, ::step]
        coarse_mask = (
            (lats_c >= lat_min - 0.2) & (lats_c <= lat_max + 0.2) &
            (lons_c >= lon_min - 0.2) & (lons_c <= lon_max + 0.2)
        )
        if not coarse_mask.any():
            return None, None, None

        ci_rows, ci_cols = np.where(coarse_mask)
        r0 = max(0,      ci_rows.min() * step - 2 * step)
        r1 = min(nrows, (ci_rows.max() + 1) * step + 2 * step)
        c0 = max(0,      ci_cols.min() * step - 2 * step)
        c1 = min(ncols, (ci_cols.max() + 1) * step + 2 * step)

        # Step 2: fine search within the candidate slab
        tlats = ds.variables['lats'][r0:r1, c0:c1]
        tlons = ds.variables['lons'][r0:r1, c0:c1]
        fine_mask = (
            (tlats >= lat_min - buf) & (tlats <= lat_max + buf) &
            (tlons >= lon_min - buf) & (tlons <= lon_max + buf)
        )
        if not fine_mask.any():
            return None, None, None

        fm_rows, fm_cols = np.where(fine_mask)
        fr0 = r0 + fm_rows.min()
        fr1 = r0 + fm_rows.max() + 1
        fc0 = c0 + fm_cols.min()
        fc1 = c0 + fm_cols.max() + 1

        terrain_lats = ds.variables['lats'][fr0:fr1, fc0:fc1]
        terrain_lons = ds.variables['lons'][fr0:fr1, fc0:fc1]
        terrain_elev = ds.variables['terrain_height'][fr0:fr1, fc0:fc1]

    return terrain_lats, terrain_lons, np.ma.filled(terrain_elev, np.nan)


def visualize_point_grid(lat_center, lon_center, box_size_deg=0.4, input_file='adaptive_grid_points.nc'):
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

    # Load terrain for this bounding box
    terrain_lats, terrain_lons, terrain_elev = load_terrain(lat_min, lat_max, lon_min, lon_max)

    # constrained_layout handles colorbar placement without extra whitespace
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)

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

    # Underplot terrain elevation
    if terrain_elev is not None:
        tx, ty = m(terrain_lons, terrain_lats)
        # Mask cells whose centers project outside the map extent to avoid edge artifacts
        outside = (
            (tx < m.llcrnrx) | (tx > m.urcrnrx) |
            (ty < m.llcrnry) | (ty > m.urcrnry)
        )
        terrain_elev[outside] = np.nan
        valid = terrain_elev[np.isfinite(terrain_elev)]
        if valid.size > 0:
            vmin = np.percentile(valid, 2)   # actual plot-area minimum
            vmax = np.percentile(valid, 98)
        else:
            vmin, vmax = 0, 3000
        # White at low elevations, muted/desaturated tones at high elevations
        # (avoids dark colors that obscure the point symbols)
        from matplotlib.colors import LinearSegmentedColormap
        _topo_cmap = LinearSegmentedColormap.from_list(
            'topo_white_muted',
            [(0.00, 'white'),
             (0.25, '#e8ede4'),
             (0.50, '#ccd9c0'),
             (0.75, '#b0c49a'),
             (1.00, '#93ae7e')]
        )
        terrain_im = ax.pcolormesh(
            tx, ty, terrain_elev,
            cmap=_topo_cmap, vmin=vmin, vmax=vmax,
            shading='nearest', zorder=1
        )
        ax.set_xlim(m.llcrnrx, m.urcrnrx)
        ax.set_ylim(m.llcrnry, m.urcrnry)
        cbar = fig.colorbar(terrain_im, ax=ax, fraction=0.03, pad=0.01, aspect=35)
        cbar.set_label('Elevation (m)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        m.fillcontinents(color='lightgray', lake_color='aqua', alpha=0.3, zorder=1)

    # Scale marker size: inversely with box size AND inversely with local spacing
    # so sparse (large-spacing) points are drawn bigger and remain visible.
    base_size = max(2, int(1.6 / box_size_deg))
    color_size = {
        'red':    base_size * 1,     # ~94m  — densest, keep small
        'orange': base_size * 1,     # ~188m
        'gray':   base_size * 2,     # ~375m
        'green':  base_size * 4,     # ~750m
        'blue':   base_size * 7,     # ~1.5km
        'purple': base_size * 12,    # ~3km  — sparsest, largest
    }
    sizes = np.array([color_size.get(c, base_size) for c in colors], dtype=float)

    # Plot points colored and sized by local spacing
    x, y = m(lons, lats)
    m.scatter(x, y, c=colors, s=sizes, alpha=0.8, zorder=4)

    # Draw geographic features on top of terrain
    m.drawcoastlines(linewidth=1.5, color='black', zorder=5)
    m.drawstates(linewidth=1.0, color='black', zorder=5)
    m.drawcounties(linewidth=0.5, color='dimgray', zorder=5)

    # Mark center
    x_center, y_center = m(lon_center, lat_center)
    ax.plot(x_center, y_center, 'k+', markersize=18, markeredgewidth=2.5, zorder=10)

    ax.set_title(
        f'Adaptive Grid Point Positions  —  {input_file}\n'
        f'Center: ({lat_center:.4f}°N, {abs(lon_center):.4f}°W) | '
        f'Box: {box_size_deg}° (~{box_size_km:.0f}km)',
        fontsize=22, pad=12
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red',    label='~94m spacing'),
        Patch(facecolor='orange', label='~188m spacing'),
        Patch(facecolor='gray',   label='~375m spacing'),
        Patch(facecolor='green',  label='~750m spacing'),
        Patch(facecolor='blue',   label='~1.5km spacing'),
        Patch(facecolor='purple', label='~3km spacing'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)

    # Format lat/lon for filename
    lat_str = f"{abs(lat_center):.1f}{'N' if lat_center >= 0 else 'S'}"
    lon_str = f"{abs(lon_center):.1f}{'E' if lon_center >= 0 else 'W'}"

    output_file = os.path.join(config.OUTPUT_DIR,
                               f'hires_grid_{lat_str}_{lon_str}_{box_size_deg}deg.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close(fig)

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize hi-res adaptive grid points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_hires_grid.py hires_points.nc 36.6 -121.9   # Monterey coast
  python visualize_hires_grid.py hires_points.nc 39.5 -106.0   # Summit County CO
  python visualize_hires_grid.py hires_points.nc 40.7 -74.0    # New York City
        '''
    )
    parser.add_argument('input_file',
                        help='NetCDF file in the output/ directory (e.g. hires_points.nc)')
    parser.add_argument('lat', type=float,
                        help='Center latitude')
    parser.add_argument('lon', type=float,
                        help='Center longitude')
    parser.add_argument('--box-sizes', type=float, nargs='+',
                        default=[1.6, 0.8, 0.2],
                        help='Box sizes in degrees (default: 1.6 0.8 0.2)')

    args = parser.parse_args()

    lat_center = args.lat
    lon_center = args.lon

    print(f"\n{'='*70}")
    print(f" DETAILED POINT GRID VISUALIZATION")
    print(f" File:   {args.input_file}")
    print(f" Center: {lat_center}°N, {abs(lon_center)}°W")
    print(f"{'='*70}")

    # Create visualizations for each box size
    for box_size in args.box_sizes:
        visualize_point_grid(lat_center, lon_center, box_size_deg=box_size,
                             input_file=args.input_file)

    print(f"\n{'='*70}")
    print(f" COMPLETE")
    print(f"{'='*70}\n")

