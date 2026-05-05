"""
Fast visualization of OSM trail coverage using rasterization
Similar approach to visualize_grid.py - show trail density per HRRR cell
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree
from shapely.geometry import Point
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


def create_hrrr_basemap(hrrr_lats, hrrr_lons):
    """Create Basemap using HRRR native Lambert Conformal projection"""
    print("\nCreating HRRR native projection map...")

    lat_0 = 38.5
    lon_0 = -97.5
    lat_1 = 38.5
    lat_2 = 38.5

    temp_proj = Basemap(
        projection='lcc',
        lat_0=lat_0, lon_0=lon_0, lat_1=lat_1, lat_2=lat_2,
        llcrnrlat=21, urcrnrlat=53,
        llcrnrlon=-135, urcrnrlon=-60,
        resolution=None
    )

    x_hrrr, y_hrrr = temp_proj(hrrr_lons, hrrr_lats)
    x_min, x_max = x_hrrr.min(), x_hrrr.max()
    y_min, y_max = y_hrrr.min(), y_hrrr.max()
    width = x_max - x_min
    height = y_max - y_min

    m = Basemap(
        projection='lcc',
        lat_0=lat_0, lon_0=lon_0, lat_1=lat_1, lat_2=lat_2,
        width=width, height=height,
        resolution='l', area_thresh=1000
    )

    print(f"  Projection: Lambert Conformal Conic")
    print(f"  Center: ({lat_0}°N, {abs(lon_0)}°W)")
    print(f"  Grid extent: {width/1000:.0f} km × {height/1000:.0f} km")

    return m


def load_osm_trails():
    """Load OSM trail sample data"""
    print("\nLoading OSM trail data...")

    trail_file = os.path.join(config.DATA_DIR, 'trails', 'osm_trails_sample.gpkg')

    if not os.path.exists(trail_file):
        print(f"✗ Error: Trail file not found: {trail_file}")
        return None

    gdf = gpd.read_file(trail_file)
    print(f"  Loaded {len(gdf):,} trail segments")

    total_km = gdf.to_crs('EPSG:3857').length.sum() / 1000
    print(f"  Total length: {total_km:,.0f} km")

    return gdf


def sample_trail_points(trails_gdf, spacing_m=500):
    """
    Sample points along trails at regular intervals
    This converts trails to point cloud for easier rasterization
    """
    print(f"\nSampling points along trails (spacing={spacing_m}m)...")

    all_points = []
    total_sampled = 0

    for idx, row in trails_gdf.iterrows():
        geom = row.geometry
        state = row.get('state', 'unknown')

        # Handle both LineString and MultiLineString
        lines = [geom] if geom.geom_type == 'LineString' else geom.geoms

        for line in lines:
            # Get line length in meters
            line_3857 = gpd.GeoSeries([line], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
            length_m = line_3857.length

            # Sample points along line
            n_points = max(2, int(length_m / spacing_m))

            for i in range(n_points):
                frac = i / (n_points - 1) if n_points > 1 else 0.5
                point = line.interpolate(frac, normalized=True)
                all_points.append({
                    'lat': point.y,
                    'lon': point.x,
                    'state': state
                })

        total_sampled += 1
        if total_sampled % 100000 == 0:
            print(f"    Processed {total_sampled:,} / {len(trails_gdf):,} segments...")

    points_gdf = gpd.GeoDataFrame(all_points,
                                   geometry=[Point(p['lon'], p['lat']) for p in all_points],
                                   crs='EPSG:4326')

    print(f"✓ Sampled {len(points_gdf):,} points from {len(trails_gdf):,} trail segments")

    return points_gdf


def compute_trail_density(trail_points, hrrr_lats, hrrr_lons):
    """Compute number of trail sample points per HRRR cell"""
    print("\nComputing trail density per HRRR cell...")

    shape = hrrr_lats.shape

    # Build KDTree for HRRR grid centers
    hrrr_points = np.column_stack([hrrr_lats.ravel(), hrrr_lons.ravel()])
    tree = cKDTree(hrrr_points)

    # Find nearest HRRR cell for each trail point
    trail_coords = np.column_stack([trail_points['lat'], trail_points['lon']])
    _, indices = tree.query(trail_coords, k=1)

    # Count points per cell
    point_count_flat = np.bincount(indices, minlength=hrrr_points.shape[0])
    point_count = point_count_flat.reshape(shape)

    print(f"  Trail coverage computed")
    print(f"    Max points per cell: {point_count.max():,}")
    print(f"    Mean points per cell: {point_count.mean():.1f}")
    print(f"    Cells with trails: {(point_count > 0).sum():,} / {point_count.size:,}")

    return point_count


def visualize_trail_coverage(hrrr_lats, hrrr_lons, trail_density, output_file):
    """Create visualization with discrete color bins"""
    print("\nCreating visualization...")

    # Define bins (simpler than point density)
    bins = [0, 1, 5, 10, 20, 50, 200]
    labels = ['0', '1-4', '5-9', '10-19', '20-49', '50+']

    # Colors
    colors = [
        '#ffffff',  # 0 - white
        '#ffffcc',  # 1-4 - very light yellow
        '#c7e9b4',  # 5-9 - light green
        '#7fcdbb',  # 10-19 - cyan-green
        '#41b6c4',  # 20-49 - cyan
        '#2c7fb8',  # 50+ - blue
    ]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bins, cmap.N)

    # Create figure
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_axes([0.05, 0.05, 0.85, 0.90])

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

    # Convert HRRR grid to map coordinates
    print("  Converting to map coordinates...")
    x, y = m(hrrr_lons, hrrr_lats)

    # Mask cells with zero trails
    trail_density_masked = np.ma.masked_where(trail_density == 0, trail_density)

    # Plot
    print("  Plotting trail coverage...")
    pcm = m.pcolormesh(x, y, trail_density_masked,
                       cmap=cmap, norm=norm,
                       shading='auto', zorder=1, rasterized=True)

    # Colorbar
    print("  Adding colorbar...")
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])

    cbar = plt.colorbar(pcm, cax=cax, spacing='uniform', extend='max')

    tick_positions = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(labels))]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(labels, fontsize=13)
    cbar.set_label('Trail Sample Points per HRRR Cell\n(500m spacing)', fontsize=14)

    # Title
    cells_with_trails = (trail_density > 0).sum()
    total_cells = trail_density.size

    ax.set_title(
        f'OpenStreetMap Trail Coverage (Sample: CA, CO, NC, WA, AZ)\n'
        f'HRRR Cells with Trails: {cells_with_trails:,} / {total_cells:,} '
        f'({100*cells_with_trails/total_cells:.1f}%)',
        fontsize=20, pad=10
    )

    # Add note
    note_text = (
        "NOTE: Sample from 5 states only.\n"
        "Shows where named hiking trails exist.\n"
        "Full CONUS would have all 48 states."
    )
    ax.text(
        0.98, 0.02, note_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # Save
    print(f"\n  Saving to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✓ Visualization saved: {output_file} ({file_size_mb:.1f} MB)")


def main():
    """Main visualization routine"""
    print("\n" + "="*70)
    print(" OSM TRAIL COVERAGE VISUALIZATION (FAST)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load HRRR grid
    hrrr_lats, hrrr_lons = load_hrrr_grid()

    # Load OSM trails
    trails_gdf = load_osm_trails()
    if trails_gdf is None:
        return 1

    # Sample points along trails
    trail_points = sample_trail_points(trails_gdf, spacing_m=500)

    # Compute trail density per HRRR cell
    trail_density = compute_trail_density(trail_points, hrrr_lats, hrrr_lons)

    # Create visualization
    output_file = os.path.join(config.OUTPUT_DIR, 'osm_trail_coverage_sample.png')
    visualize_trail_coverage(hrrr_lats, hrrr_lons, trail_density, output_file)

    print("\n" + "="*70)
    print(" ✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
