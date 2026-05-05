"""
Visualize OSM trail coverage on CONUS map
Uses same projection as visualize_grid.py for consistency
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import config


def load_hrrr_grid():
    """Load HRRR grid for projection setup"""
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

    # HRRR uses Lambert Conformal Conic projection
    lat_0 = 38.5
    lon_0 = -97.5
    lat_1 = 38.5
    lat_2 = 38.5

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


def load_osm_trails():
    """Load OSM trail sample data"""
    print("\nLoading OSM trail data...")

    trail_file = os.path.join(config.DATA_DIR, 'trails', 'osm_trails_sample.gpkg')

    if not os.path.exists(trail_file):
        print(f"✗ Error: Trail file not found: {trail_file}")
        print("  Run sample_osm_trails.py first")
        return None

    gdf = gpd.read_file(trail_file)

    print(f"  Loaded {len(gdf):,} trail segments")

    # Calculate stats
    total_km = gdf.to_crs('EPSG:3857').length.sum() / 1000
    print(f"  Total length: {total_km:,.0f} km")

    # Show state distribution
    if 'state' in gdf.columns:
        print(f"\n  Trails by state:")
        for state, count in gdf['state'].value_counts().items():
            state_km = gdf[gdf['state'] == state].to_crs('EPSG:3857').length.sum() / 1000
            print(f"    {state:15s}: {count:7,} segments ({state_km:8,.0f} km)")

    return gdf


def visualize_trails(hrrr_lats, hrrr_lons, trails_gdf, output_file):
    """Create visualization showing trail coverage"""
    print("\nCreating visualization...")

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

    # Plot trails by state with different colors
    print("  Plotting trails...")

    state_colors = {
        'california': '#e41a1c',
        'colorado': '#377eb8',
        'north-carolina': '#4daf4a',
        'washington': '#984ea3',
        'arizona': '#ff7f00'
    }

    for state, color in state_colors.items():
        state_trails = trails_gdf[trails_gdf['state'] == state]
        if len(state_trails) == 0:
            continue

        print(f"    Plotting {state}: {len(state_trails):,} segments...", end=' ', flush=True)

        # Extract coordinates and plot
        for idx, row in state_trails.iterrows():
            geom = row.geometry

            # Handle both LineString and MultiLineString
            if geom.geom_type == 'LineString':
                lons = [c[0] for c in geom.coords]
                lats = [c[1] for c in geom.coords]

                # Convert to map projection
                x, y = m(lons, lats)

                # Plot
                m.plot(x, y, color=color, linewidth=0.3, alpha=0.6, zorder=2)

            elif geom.geom_type == 'MultiLineString':
                # Plot each line in the multi-part geometry
                for line in geom.geoms:
                    lons = [c[0] for c in line.coords]
                    lats = [c[1] for c in line.coords]

                    # Convert to map projection
                    x, y = m(lons, lats)

                    # Plot
                    m.plot(x, y, color=color, linewidth=0.3, alpha=0.6, zorder=2)

        print("Done")

    # Title with statistics
    total_segments = len(trails_gdf)
    total_km = trails_gdf.to_crs('EPSG:3857').length.sum() / 1000

    ax.set_title(
        f'OpenStreetMap Trail Coverage (Sample from 5 States)\n'
        f'Total: {total_segments:,} trail segments | '
        f'Total Length: {total_km:,.0f} km',
        fontsize=20,
        pad=10
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=state_colors['california'], label='California'),
        Patch(facecolor=state_colors['colorado'], label='Colorado'),
        Patch(facecolor=state_colors['north-carolina'], label='North Carolina'),
        Patch(facecolor=state_colors['washington'], label='Washington'),
        Patch(facecolor=state_colors['arizona'], label='Arizona'),
    ]

    cax = fig.add_axes([0.92, 0.40, 0.02, 0.20])
    cax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=True)
    cax.axis('off')

    # Add note
    note_text = (
        "NOTE: This is a sample from 5 states only.\n"
        "Full CONUS coverage would include trails\n"
        "from all 48 continental states."
    )
    ax.text(
        0.98, 0.02, note_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # Save figure
    print(f"\n  Saving to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"✓ Visualization saved: {output_file} ({file_size_mb:.1f} MB)")


def main():
    """Main visualization routine"""
    print("\n" + "="*70)
    print(" OSM TRAIL COVERAGE VISUALIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load HRRR grid
    hrrr_lats, hrrr_lons = load_hrrr_grid()

    # Load OSM trails
    trails_gdf = load_osm_trails()

    if trails_gdf is None:
        return 1

    # Create visualization
    output_file = os.path.join(config.OUTPUT_DIR, 'osm_trail_coverage_sample.png')
    visualize_trails(hrrr_lats, hrrr_lons, trails_gdf, output_file)

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
