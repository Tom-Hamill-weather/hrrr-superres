"""
Adaptive Grid Point Generation for Golf Courses Only

Generates an irregular grid with high-resolution points over golf courses:
- Tier 0 (93.75m): Golf course areas with buffer
- Tier 5 (3km): Background (HRRR native resolution)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep
from scipy.spatial import cKDTree
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
from pyproj import Proj
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import config

class DataLoader:
    """Load HRRR grid and golf course data"""

    def __init__(self):
        self.data = {}
        self.hrrr_grid = None

    def load_hrrr_grid(self):
        """Load HRRR grid and terrain"""
        print("\nLoading HRRR grid and terrain...")
        hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')

        try:
            lats = np.load(os.path.join(hrrr_dir, 'hrrr_lats.npy'))
            lons = np.load(os.path.join(hrrr_dir, 'hrrr_lons.npy'))
            terrain = np.load(os.path.join(hrrr_dir, 'hrrr_terrain.npy'))

            self.hrrr_grid = {
                'lats': lats,
                'lons': lons,
                'terrain': terrain,
                'shape': lats.shape
            }

            print(f"✓ HRRR grid loaded: {lats.shape}")
            print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
            print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")
            print(f"  Terrain range: {terrain.min():.1f} to {terrain.max():.1f} m")

        except Exception as e:
            print(f"✗ Error loading HRRR grid: {e}")
            raise

    def load_golf_courses(self):
        """Load golf course locations from preprocessed OSM data"""
        print("\nLoading golf courses...")
        try:
            golf_file = os.path.join(config.DATA_DIR, 'golf', 'us_golf_courses_osm.geojson')

            if os.path.exists(golf_file):
                self.data['golf_courses'] = gpd.read_file(golf_file)
                print(f"✓ Golf courses loaded: {len(self.data['golf_courses'])} features")
            else:
                print(f"✗ Golf courses file not found: {golf_file}")
                print("  Run: python preprocess_golf_courses.py")
                raise FileNotFoundError(f"Golf course data not found: {golf_file}")
        except Exception as e:
            print(f"✗ Error loading golf courses: {e}")
            raise

    def load_all(self):
        """Load all required datasets"""
        print("="*70)
        print(" LOADING GEOSPATIAL DATASETS (GOLF COURSES ONLY)")
        print("="*70)

        self.load_hrrr_grid()
        self.load_golf_courses()

        print("\n" + "="*70)
        print(" DATA LOADING COMPLETE")
        print("="*70)


class TierClassifier:
    """Classify HRRR grid cells into tiers (golf courses only)"""

    def __init__(self, data_loader):
        self.loader = data_loader
        self.tier_map = None

    def create_tier_map(self):
        """Create tier classification for entire HRRR grid"""
        print("\n" + "="*70)
        print(" STEP 1/4: TIER CLASSIFICATION (GOLF COURSES ONLY)")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize to Tier 5 (background - HRRR native resolution)
        tier_map = np.full(shape, 5, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)  # Bitfield for criteria met

        # Metadata bit 5: Golf course
        GOLF_COURSE_BIT = 5

        # Process golf courses with buffer (Tier 0)
        print("\n[1/1] Processing Golf Courses...")
        golf_courses = self.loader.data.get('golf_courses')

        if golf_courses is not None and len(golf_courses) > 0:
            # Project to equal area for accurate buffering
            golf_proj = golf_courses.to_crs('EPSG:5070')  # Albers Equal Area
            # 1500m = half HRRR cell; marks cells that overlap any course as Tier 0
            # (excludes them from Tier 5 background — actual points come from polygons)
            buffer_m = 1500
            golf_proj['geometry'] = golf_proj.geometry.buffer(buffer_m)
            golf_buffered = golf_proj.to_crs('EPSG:4326')

            # Classify using buffered geometries
            count = 0
            for geom in tqdm(golf_buffered.geometry, desc="  Classifying cells", leave=False):
                bounds = geom.bounds
                lat_mask = (lats >= bounds[1]) & (lats <= bounds[3])
                lon_mask = (lons >= bounds[0]) & (lons <= bounds[2])
                mask = lat_mask & lon_mask

                tier_map[mask] = np.minimum(tier_map[mask], 0)
                metadata[mask] |= (1 << GOLF_COURSE_BIT)
                count += mask.sum()

            print(f"    ✓ Classified {count:,} cells as Tier 0 (golf courses)")
        else:
            print("    ✗ No golf course data available")

        self.tier_map = tier_map
        self.metadata_map = metadata

        # Summary statistics
        print("\n" + "="*70)
        print(" TIER CLASSIFICATION SUMMARY")
        print("="*70)
        for tier in [0, 5]:
            count = (tier_map == tier).sum()
            pct = 100 * count / tier_map.size
            if count > 0:
                print(f"  Tier {tier}: {count:,} cells ({pct:.1f}%)")

        return tier_map, metadata


class AdaptiveGridGenerator:
    """Generate adaptive grid points based on tier classification"""

    def __init__(self, data_loader, tier_map, metadata_map):
        self.loader = data_loader
        self.tier_map = tier_map
        self.metadata_map = metadata_map
        self.points = None

    def generate_points(self):
        """Generate points with tier-appropriate spacing.

        Tier 5 (background): one point per HRRR cell center.
        Tier 0 (golf):       globally-aligned 93.75m grid, points kept only
                             within each course's actual polygon boundary.
                             For poorly-defined geometries (points/lines) a
                             small approximate perimeter is buffered in.
        """
        print("\n" + "="*70)
        print(" STEP 2/4: GENERATING ADAPTIVE GRID POINTS")
        print("="*70)

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        all_points = []
        all_metadata = []

        start_time = time.time()

        hrrr_proj = Proj(proj='lcc', lat_0=38.5, lon_0=-97.5,
                         lat_1=38.5, lat_2=38.5, R=6371229)
        hrrr_crs_str = ('+proj=lcc +lat_0=38.5 +lon_0=-97.5 '
                        '+lat_1=38.5 +lat_2=38.5 +units=m +R=6371229 +no_defs')

        # -------------------------------------------------------
        # TIER 5: Background — one point per HRRR cell center
        # -------------------------------------------------------
        print(f"\n[1/2] Generating Tier 5 background points (3000m)...")
        tier5_mask = self.tier_map == 5
        tier5_indices = np.argwhere(tier5_mask)
        print(f"      {len(tier5_indices):,} HRRR cells in Tier 5")

        for idx in tqdm(tier5_indices, desc="      Tier 5", unit="cells", leave=True):
            i, j = idx
            all_points.append([lats_hrrr[i, j], lons_hrrr[i, j], 5])
            all_metadata.append(int(self.metadata_map[i, j]))

        print(f"      ✓ {len(tier5_indices):,} Tier 5 background points")

        # -------------------------------------------------------
        # TIER 0: Golf course polygons — globally-aligned 93.75m grid
        # -------------------------------------------------------
        print(f"\n[2/2] Generating Tier 0 golf course points (93.75m)...")

        resolution_m = config.TIER_RESOLUTIONS[0]   # 93.75 m
        n_per_side   = round(3000 / resolution_m)   # 32

        # Global aligned grid origin: every grid point is at
        #   x = grid_x0 + k * resolution_m  for integer k
        # anchored so the first HRRR cell centre falls on a grid point.
        x00, y00 = hrrr_proj(lons_hrrr[0, 0], lats_hrrr[0, 0])
        grid_x0 = x00 - (n_per_side - 1) / 2.0 * resolution_m
        grid_y0 = y00 - (n_per_side - 1) / 2.0 * resolution_m

        golf_courses = self.loader.data.get('golf_courses')
        if golf_courses is None or len(golf_courses) == 0:
            print("      ✗ No golf course data available")
        else:
            print(f"      Projecting {len(golf_courses):,} features to HRRR LCC...")
            golf_lcc = golf_courses.to_crs(hrrr_crs_str)

            GOLF_BIT = 1 << 5   # bit 5 = golf course metadata flag
            tier0_start = len(all_points)

            for geom in tqdm(golf_lcc.geometry,
                             desc="      Processing courses",
                             unit="courses", leave=True):
                if geom is None or geom.is_empty:
                    continue

                # Normalize geometry for solid interior coverage:
                #
                # 1. Point/line geometries have no area — give them a rough
                #    perimeter before proceeding.
                if geom.geom_type in ('Point', 'MultiPoint'):
                    geom = geom.buffer(150)        # ~150m radius circle
                elif geom.geom_type in ('LineString', 'MultiLineString'):
                    geom = geom.buffer(50)         # 50m corridor
                #
                # 2. Drop interior holes (OSM often punches bunkers / ponds /
                #    water hazards out as interior rings, leaving a hollow
                #    polygon that plots as just an outline).
                if geom.geom_type == 'Polygon':
                    if list(geom.interiors):
                        geom = Polygon(geom.exterior)
                elif geom.geom_type == 'MultiPolygon':
                    # Drop holes from every sub-polygon
                    geom = geom.__class__([Polygon(p.exterior) for p in geom.geoms])
                #
                # 3. Expand by half the grid spacing.  This does two things:
                #    • Thin fairways (< 93.75m wide) become wide enough for
                #      at least one column of grid points.
                #    • Adjacent MultiPolygon sub-parts within one grid spacing
                #      of each other merge into a single filled region.
                geom = geom.buffer(resolution_m / 2.0)

                bounds = geom.bounds  # (minx, miny, maxx, maxy)

                k_x0 = math.ceil( (bounds[0] - grid_x0) / resolution_m)
                k_x1 = math.floor((bounds[2] - grid_x0) / resolution_m)
                k_y0 = math.ceil( (bounds[1] - grid_y0) / resolution_m)
                k_y1 = math.floor((bounds[3] - grid_y0) / resolution_m)

                if k_x1 < k_x0 or k_y1 < k_y0:
                    continue

                xs = grid_x0 + np.arange(k_x0, k_x1 + 1) * resolution_m
                ys = grid_y0 + np.arange(k_y0, k_y1 + 1) * resolution_m
                xx, yy = np.meshgrid(xs, ys)
                pts_x = xx.ravel()
                pts_y = yy.ravel()

                prepared = prep(geom)
                within = np.array([prepared.contains(Point(x, y))
                                   for x, y in zip(pts_x, pts_y)])

                kept_x = pts_x[within]
                kept_y = pts_y[within]

                if len(kept_x) == 0:
                    continue

                lons_out, lats_out = hrrr_proj(kept_x, kept_y, inverse=True)
                for lat, lon in zip(lats_out, lons_out):
                    all_points.append([lat, lon, 0])
                    all_metadata.append(GOLF_BIT)

            tier0_count = len(all_points) - tier0_start
            print(f"      ✓ {tier0_count:,} Tier 0 golf course points")

        # Convert to arrays
        points_array  = np.array(all_points)
        metadata_array = np.array(all_metadata)

        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f" POINT GENERATION COMPLETE")
        print("="*70)
        print(f"  Total points:   {len(points_array):,}")
        print(f"  Time elapsed:   {elapsed:.1f} s ({elapsed/60:.1f} min)")

        self.points         = points_array
        self.point_metadata = metadata_array

        return points_array, metadata_array


class OutputWriter:
    """Write output netCDF and visualization"""

    def __init__(self, points, metadata, hrrr_grid):
        self.points = points
        self.metadata = metadata
        self.hrrr_grid = hrrr_grid

    def write_netcdf(self, filename):
        """Write adaptive grid to netCDF file"""
        print("\n" + "="*70)
        print(" STEP 3/4: WRITING OUTPUT NETCDF")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nOutput file: {output_path}")
        print(f"Writing {len(self.points):,} points...")

        start_time = time.time()

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            # Dimensions
            npoints = len(self.points)
            ncfile.createDimension('npoints', npoints)

            # Global attributes
            ncfile.title = 'Adaptive Grid Points for Golf Courses - XGBoost Weather Downscaling'
            ncfile.institution = 'TWC/IBM'
            ncfile.source = 'generate_adaptive_grid_golf.py'
            ncfile.history = f'Created {datetime.now().isoformat()}'
            ncfile.conventions = 'CF-1.8'
            ncfile.total_points = npoints
            ncfile.description = 'High-resolution grid points over golf courses only'
            ncfile.tier0_resolution_m = config.TIER_RESOLUTIONS[0]
            ncfile.tier5_resolution_m = config.TIER_RESOLUTIONS[5]
            ncfile.golf_course_method = 'polygon-interior 93.75m globally-aligned grid'

            # Variables
            lat_var = ncfile.createVariable('latitude', 'f4', ('npoints',))
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'Latitude'
            lat_var.standard_name = 'latitude'
            lat_var[:] = self.points[:, 0]

            lon_var = ncfile.createVariable('longitude', 'f4', ('npoints',))
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'Longitude'
            lon_var.standard_name = 'longitude'
            lon_var[:] = self.points[:, 1]

            tier_var = ncfile.createVariable('tier', 'i1', ('npoints',))
            tier_var.long_name = 'Grid Tier (0=highest resolution)'
            tier_var.description = 'Tier 0: 93.75m (golf courses), Tier 5: 3km (background)'
            tier_var[:] = self.points[:, 2].astype(np.int8)

            meta_var = ncfile.createVariable('metadata', 'u2', ('npoints',))
            meta_var.long_name = 'Point Classification Metadata (bitfield)'
            meta_var.description = 'Bit 5: Golf Course'
            meta_var[:] = self.metadata

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ NetCDF file written: {output_path}")
        print(f"  Points: {npoints:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Write time: {elapsed:.1f} seconds")
        return output_path

    def create_visualization(self, filename):
        """Create HRRR-domain point density map using native LCC projection.

        Matches the plotting conventions of visualize_grid.py:
          - HRRR LCC parameters (lat_0=38.5, lon_0=-97.5, lat_1=lat_2=38.5)
          - Map extent set via width/height computed from actual HRRR grid
          - KDTree built in projection space (metres), not lat/lon degrees
          - np.bincount for fast cell assignment
          - Discrete colour bins with ListedColormap
          - Parallels/meridians labelled
        """
        print("\n" + "="*70)
        print(" STEP 4/4: CREATING VISUALIZATION")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nOutput file: {output_path}")

        lats_hrrr = self.hrrr_grid['lats']
        lons_hrrr = self.hrrr_grid['lons']
        shape     = lats_hrrr.shape

        start_time = time.time()

        # ── 1. Build HRRR native LCC projection sized to the actual grid ──
        print("\n[1/4] Building HRRR LCC projection...")
        lat_0, lon_0, lat_1, lat_2 = 38.5, -97.5, 38.5, 38.5

        temp_proj = Basemap(
            projection='lcc', lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            llcrnrlat=21, urcrnrlat=53, llcrnrlon=-135, urcrnrlon=-60,
            resolution=None
        )
        x_hrrr, y_hrrr = temp_proj(lons_hrrr, lats_hrrr)
        width  = x_hrrr.max() - x_hrrr.min()
        height = y_hrrr.max() - y_hrrr.min()

        m = Basemap(
            projection='lcc', lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            width=width, height=height, resolution='l', area_thresh=1000
        )

        # ── 2. Count adaptive grid points per HRRR cell in projection space ──
        print("[2/4] Computing point density per HRRR cell...")
        hrrr_pts_proj = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
        tree = cKDTree(hrrr_pts_proj)

        x_adap, y_adap = temp_proj(self.points[:, 1], self.points[:, 0])
        _, indices = tree.query(np.column_stack([x_adap, y_adap]), k=1)

        point_density = np.bincount(
            indices, minlength=hrrr_pts_proj.shape[0]
        ).reshape(shape)

        print(f"    Max points per cell:  {point_density.max():,}")
        print(f"    Mean points per cell: {point_density.mean():.1f}")

        # ── 3. Discrete colour bins (same as visualize_grid.py) ──────────────
        bins   = [0, 5, 10, 50, 100, 200, 400, 800, 1000, 1100]
        labels = ['≤4','5-9','10-49','50-99','100-199',
                  '200-399','400-799','800-999','1000+']
        colors = [
            'White',   '#C4E8FF', '#8FB3FF', '#42F742',
            'Yellow',  'Gold',    'Orange',  '#F6A3AE', 'Orchid',
        ]
        cmap = matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)

        # ── 4. Plot ────────────────────────────────────────────────────────────
        print("[3/4] Drawing map...")
        fig = plt.figure(figsize=(18, 11))
        ax  = fig.add_axes([0.05, 0.05, 0.85, 0.90])

        m.drawcoastlines(linewidth=0.5, color='black', zorder=5)
        m.drawcountries(linewidth=0.5, color='black', zorder=5)
        m.drawstates(linewidth=0.3,   color='gray',  zorder=5)
        m.drawparallels(np.arange(20, 55, 5),    labels=[1,0,0,0],
                        fontsize=13.5, linewidth=0.3, color='gray')
        m.drawmeridians(np.arange(-130, -60, 10), labels=[0,0,0,1],
                        fontsize=13.5, linewidth=0.3, color='gray')

        x_map, y_map = m(lons_hrrr, lats_hrrr)
        density_masked = np.ma.masked_where(point_density == 0, point_density)

        pcm = m.pcolormesh(
            x_map, y_map, density_masked,
            cmap=cmap, norm=norm, shading='auto', zorder=1, rasterized=True
        )

        cax  = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        cbar = plt.colorbar(pcm, cax=cax, spacing='uniform', extend='max')
        tick_pos = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(labels))]
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(labels, fontsize=13)
        cbar.set_label('Points per HRRR Cell', fontsize=14)

        total_pts = int(point_density.sum())
        n_cells   = int((point_density > 0).sum())
        ax.set_title(
            f'Adaptive Grid Point Density — Golf Courses Only\n'
            f'Total Points: {total_pts:,} | '
            f'HRRR Cells with Points: {n_cells:,} / {point_density.size:,} '
            f'({100*n_cells/point_density.size:.1f}%)',
            fontsize=20, pad=10
        )

        print(f"[4/4] Saving {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"\n✓ Visualization saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB  |  Time: {elapsed:.1f}s")
        return output_path


def main():
    """Main execution"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION - GOLF COURSES ONLY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Golf course resolution: {config.TIER_RESOLUTIONS[0]}m (Tier 0, within polygon)")
    print(f"  Background resolution: {config.TIER_RESOLUTIONS[5]}m (Tier 5)")
    print(f"  Cell-exclusion buffer: 1.5 km (half HRRR cell)")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = DataLoader()
    loader.load_all()

    # Classify tiers (golf courses only)
    classifier = TierClassifier(loader)
    tier_map, metadata_map = classifier.create_tier_map()

    # Generate points
    generator = AdaptiveGridGenerator(loader, tier_map, metadata_map)
    points, metadata = generator.generate_points()

    # Write outputs with unique names
    writer = OutputWriter(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_points_golf.nc')
    png_file = writer.create_visualization('adaptive_grid_density_golf.png')

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print(" ✓ ADAPTIVE GRID GENERATION COMPLETE (GOLF COURSES)")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  NetCDF: {nc_file}")
    print(f"  Visualization: {png_file}")
    print(f"\nSummary:")
    print(f"  Total points: {len(points):,}")
    print(f"  Golf course points (Tier 0): {(points[:, 2] == 0).sum():,}")
    print(f"  Background points (Tier 5): {(points[:, 2] == 5).sum():,}")
    print(f"  Total runtime: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
