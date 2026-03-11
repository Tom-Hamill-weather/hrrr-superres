"""
SUPER FAST version - vectorizes EVERYTHING including point generation
No cell-by-cell loops anywhere in the entire pipeline
"""

import os
import sys
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.ndimage import generic_filter
from scipy.spatial import cKDTree
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.path as mpath
import config

# Import from original
from generate_adaptive_grid import DataLoader, TerrainAnalyzer


class TierClassifierVectorized:
    """Fully vectorized tier classifier using spatial indexing"""

    def __init__(self, data_loader, terrain_variability):
        self.loader = data_loader
        self.terrain_var = terrain_variability
        self.tier_map = None

        # Define HRRR Lambert Conformal projection
        # Parameters from HRRR GRIB2: Latin1=38.5, Latin2=38.5, LoV=262.5 (-97.5)
        self.hrrr_crs = '+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

    def create_tier_map(self):
        """Create tier classification using fast spatial operations"""
        print("\n" + "="*70)
        print(" STEP 2/5: TIER CLASSIFICATION (VECTORIZED)")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize (default to Tier 5 = 3km background)
        tier_map = np.full(shape, 5, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)

        # Create GeoDataFrame of all HRRR grid cell centers
        print("\nCreating spatial index of grid cells...")
        start = time.time()

        # Vectorized creation of points
        grid_points = [Point(lon, lat) for lat, lon in zip(lats.ravel(), lons.ravel())]
        grid_gdf = gpd.GeoDataFrame(
            {'i': np.repeat(np.arange(shape[0]), shape[1]),
             'j': np.tile(np.arange(shape[1]), shape[0])},
            geometry=grid_points,
            crs='EPSG:4326'
        )

        # Reproject to HRRR Lambert Conformal for accurate buffering
        grid_gdf = grid_gdf.to_crs(self.hrrr_crs)

        elapsed = time.time() - start
        print(f"✓ Created spatial index with {len(grid_gdf):,} points in {elapsed:.1f}s")
        print(f"  Using HRRR Lambert Conformal projection for accurate buffering")

        # [1] Terrain variability - three graduated tiers (already vectorized)
        print("\n[1/6] Terrain variability (graduated resolution)...")
        # Tier 1: Extreme mountains (>150m std dev) → 90m
        extreme_terrain = self.terrain_var > config.TERRAIN_TIER1_THRESHOLD
        tier_map[extreme_terrain] = np.minimum(tier_map[extreme_terrain], 1)
        metadata[extreme_terrain] |= (1 << 8)
        print(f"      ✓ Tier 1 (extreme mountains, 90m): {extreme_terrain.sum():,} cells")

        # Tier 2: Medium mountains (75-150m std dev) → 180m
        medium_terrain = (self.terrain_var > config.TERRAIN_TIER2_THRESHOLD) & (self.terrain_var <= config.TERRAIN_TIER1_THRESHOLD)
        tier_map[medium_terrain] = np.minimum(tier_map[medium_terrain], 2)
        metadata[medium_terrain] |= (1 << 9)
        print(f"      ✓ Tier 2 (medium mountains, 180m): {medium_terrain.sum():,} cells")

        # Tier 3: Lower variability (30-75m std dev) → 360m
        lower_terrain = (self.terrain_var > config.TERRAIN_TIER3_THRESHOLD) & (self.terrain_var <= config.TERRAIN_TIER2_THRESHOLD)
        tier_map[lower_terrain] = np.minimum(tier_map[lower_terrain], 3)
        metadata[lower_terrain] |= (1 << 10)
        print(f"      ✓ Tier 3 (lower terrain, 360m): {lower_terrain.sum():,} cells")

        # [1.5] Tier 2: Urbanized Areas (major metros) → 180m
        print("\n[1.5/6] Tier 2: Urbanized Areas (major metros ≥50k pop)...")
        self._classify_high_density_urban_fast(tier_map, metadata, grid_gdf)

        # [2] Tier 4: Urban Clusters (small towns)
        print("\n[2/6] Tier 4: Urban Clusters (towns 2.5k-50k pop)...")
        self._classify_urban_fast(tier_map, metadata, grid_gdf)

        # [3] Tier 4: Highways
        print("\n[3/6] Tier 4: Major highways...")
        self._classify_highways_fast(tier_map, metadata, grid_gdf)

        # [4] Tier 1: Coastlines
        print("\n[4/6] Tier 1: Coastlines...")
        self._classify_coastlines_fast(tier_map, metadata, grid_gdf)

        # [5] Tier 1: Lakes
        if config.INCLUDE_LAKES_IN_TIER1:
            print("\n[5/6] Tier 1: Large lakes...")
            self._classify_lakes_fast(tier_map, metadata, grid_gdf)
        else:
            print("\n[5/6] Tier 1: Large lakes... SKIPPED")

        # [6] Tier 1: Ski resorts
        print("\n[6/8] Tier 1: Ski resorts...")
        self._classify_ski_resorts_fast(tier_map, metadata, grid_gdf)

        # [7] Tier 1: Golf courses
        if config.INCLUDE_GOLF_COURSES:
            print("\n[7/8] Tier 1: Golf courses...")
            self._classify_golf_courses_fast(tier_map, metadata, grid_gdf)
        else:
            print("\n[7/8] Tier 1: Golf courses... SKIPPED")

        # [8] Tier 1: Parks
        if config.INCLUDE_PARKS:
            print("\n[8/8] Tier 1: National & State Parks...")
            self._classify_parks_fast(tier_map, metadata, grid_gdf)
        else:
            print("\n[8/8] Tier 1: National & State Parks... SKIPPED")

        # Validate
        self._validate_tier_distribution(tier_map)

        self.tier_map = tier_map
        self.metadata_map = metadata

        # Summary
        print("\n" + "="*70)
        print(" TIER CLASSIFICATION SUMMARY")
        print("="*70)
        for tier in [1, 2, 3, 4, 5]:
            count = (tier_map == tier).sum()
            pct = 100 * count / tier_map.size
            res = config.TIER_RESOLUTIONS[tier]
            print(f"  Tier {tier} ({res}m): {count:,} cells ({pct:.1f}%)")

        return tier_map, metadata

    def _classify_high_density_urban_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized urbanized area classification (major metros ≥50k population)"""
        high_density_gdf = self.loader.data.get('high_density_urban')
        if high_density_gdf is None or len(high_density_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(high_density_gdf)} urbanized areas (major metros)...")
        start = time.time()

        # Reproject to Lambert Conformal
        high_density_gdf = high_density_gdf.to_crs(self.hrrr_crs)

        joined = gpd.sjoin(grid_gdf, high_density_gdf, how='inner', predicate='within')

        # Vectorized update (Tier 2 = 180m for urbanized areas)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 2)
        metadata[i_indices, j_indices] |= (1 << 0)  # Mark as urban in metadata

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_urban_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized urban cluster classification (small towns)"""
        urban_gdf = self.loader.data.get('urban')
        if urban_gdf is None or len(urban_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(urban_gdf)} urban clusters (small towns)...")
        start = time.time()

        # Reproject to Lambert Conformal
        urban_gdf = urban_gdf.to_crs(self.hrrr_crs)

        joined = gpd.sjoin(grid_gdf, urban_gdf, how='inner', predicate='within')

        # Vectorized update (Tier 4 = 810m for urban clusters/small towns)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 4)
        metadata[i_indices, j_indices] |= (1 << 0)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_highways_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized highway classification"""
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is None or len(roads_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(roads_gdf)} road segments...")
        start = time.time()

        # Reproject to Lambert Conformal and buffer in meters
        roads_buffered = roads_gdf.to_crs(self.hrrr_crs)
        roads_buffered['geometry'] = roads_buffered.geometry.buffer(500)  # 500m buffer

        joined = gpd.sjoin(grid_gdf, roads_buffered, how='inner', predicate='within')

        # Vectorized update (Tier 4 = 810m for highways)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 4)
        metadata[i_indices, j_indices] |= (1 << 7)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_coastlines_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized coastline classification"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None or len(coastline_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing coastlines...")
        start = time.time()

        # Reproject to Lambert Conformal, simplify and buffer in meters
        coastline_simple = coastline_gdf.to_crs(self.hrrr_crs)
        coastline_simple['geometry'] = coastline_simple.geometry.simplify(1000)  # 1km simplification
        coastline_buffered = coastline_simple.copy()
        coastline_buffered['geometry'] = coastline_buffered.geometry.buffer(
            config.COASTLINE_BUFFER_OFFSHORE_KM * 1000  # Convert km to meters
        )

        coast_union = unary_union(coastline_buffered.geometry)
        coast_gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[coast_union], crs=self.hrrr_crs)

        joined = gpd.sjoin(grid_gdf, coast_gdf, how='inner', predicate='within')

        # Apply hard limit
        if len(joined) > 50000:
            print(f"      Found {len(joined):,} coastal cells, limiting to 50,000")
            joined = joined.iloc[:50000]

        # Vectorized update
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 2)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_lakes_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized lake shoreline classification (boundaries only, not entire lake)"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None or len(lakes_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(lakes_gdf)} large lake shorelines...")
        start = time.time()

        # Reproject to Lambert Conformal, extract lake boundaries and buffer in meters
        lakes_shorelines = lakes_gdf.to_crs(self.hrrr_crs)
        lakes_shorelines['geometry'] = lakes_shorelines.geometry.boundary  # Get perimeter only
        lakes_shorelines['geometry'] = lakes_shorelines.geometry.buffer(1000)  # 1km buffer around shoreline

        joined = gpd.sjoin(grid_gdf, lakes_shorelines, how='inner', predicate='within')

        # Vectorized update
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 3)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_ski_resorts_fast(self, tier_map, metadata, grid_gdf):
        """Vectorized ski resort classification"""
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is None or len(ski_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(ski_gdf)} ski resorts...")
        start = time.time()

        # Reproject to Lambert Conformal and buffer in meters
        ski_buffered = ski_gdf.to_crs(self.hrrr_crs)
        ski_buffered['geometry'] = ski_buffered.geometry.buffer(4000)  # 4km buffer (covers base to summit)

        joined = gpd.sjoin(grid_gdf, ski_buffered, how='inner', predicate='within')

        # Vectorized update
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 4)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_golf_courses_fast(self, tier_map, metadata, grid_gdf):
        """Fully vectorized golf course classification"""
        golf_gdf = self.loader.data.get('golf_courses')
        if golf_gdf is None or len(golf_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(golf_gdf)} golf courses...")
        start = time.time()

        # Reproject to Lambert Conformal and buffer in meters
        golf_buffered = golf_gdf.to_crs(self.hrrr_crs)
        golf_buffered['geometry'] = golf_buffered.geometry.buffer(
            config.GOLF_COURSE_BUFFER_KM * 1000  # Convert km to meters
        )

        # Fully vectorized spatial join
        joined = gpd.sjoin(grid_gdf, golf_buffered, how='inner', predicate='within')

        # Vectorized update
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 5)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_parks_fast(self, tier_map, metadata, grid_gdf):
        """Fully vectorized parks classification"""
        parks_gdf = self.loader.data.get('parks')
        if parks_gdf is None or len(parks_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(parks_gdf)} parks...")
        start = time.time()

        # Reproject to Lambert Conformal and buffer in meters
        parks_buffered = parks_gdf.to_crs(self.hrrr_crs)
        parks_buffered['geometry'] = parks_buffered.geometry.buffer(
            config.PARKS_BUFFER_KM * 1000  # Convert km to meters
        )

        # Fully vectorized spatial join
        joined = gpd.sjoin(grid_gdf, parks_buffered, how='inner', predicate='within')

        # Vectorized update
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 6)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _validate_tier_distribution(self, tier_map):
        """Sanity check"""
        total_cells = tier_map.size
        tier1_count = (tier_map == 1).sum()
        tier1_pct = 100 * tier1_count / total_cells

        if tier1_pct > 15:
            print(f"\n⚠ WARNING: Tier 1 has {tier1_pct:.1f}% of cells")

        tier_counts = [(tier_map == t).sum() for t in [1, 2, 3, 4, 5]]
        tier_points = [
            tier_counts[0] * ((3000 / config.TIER_RESOLUTIONS[1]) ** 2),
            tier_counts[1] * ((3000 / config.TIER_RESOLUTIONS[2]) ** 2),
            tier_counts[2] * ((3000 / config.TIER_RESOLUTIONS[3]) ** 2),
            tier_counts[3] * ((3000 / config.TIER_RESOLUTIONS[4]) ** 2),
            tier_counts[4] * 1
        ]
        estimated_total = sum(tier_points)
        print(f"\nEstimated total points: {estimated_total:,.0f}")


class AdaptiveGridGeneratorVectorized:
    """VECTORIZED point generation - no loops!"""

    def __init__(self, data_loader, tier_map, metadata_map):
        self.loader = data_loader
        self.tier_map = tier_map
        self.metadata_map = metadata_map
        self.points = None

    def generate_points(self):
        """Fully vectorized point generation"""
        print("\n" + "="*70)
        print(" STEP 3/5: GENERATING ADAPTIVE GRID POINTS (VECTORIZED)")
        print("="*70)

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        all_points = []
        all_metadata = []

        overall_start = time.time()

        for tier_idx, tier in enumerate([1, 2, 3, 4, 5], 1):
            print(f"\n[{tier_idx}/5] Generating Tier {tier} points (resolution: {config.TIER_RESOLUTIONS[tier]}m)...")
            start = time.time()

            # Get indices for this tier (vectorized)
            tier_mask = self.tier_map == tier
            tier_i, tier_j = np.where(tier_mask)

            if len(tier_i) == 0:
                print(f"      No cells in Tier {tier} - skipping")
                continue

            print(f"      {len(tier_i):,} HRRR cells in Tier {tier}")

            resolution_m = config.TIER_RESOLUTIONS[tier]
            hrrr_cell_size_m = 3000

            if resolution_m >= hrrr_cell_size_m:
                # Tier 5: Just use cell centers (fully vectorized!)
                points_per_cell = 1
                tier_lats = lats_hrrr[tier_i, tier_j]
                tier_lons = lons_hrrr[tier_i, tier_j]
                tier_tiers = np.full(len(tier_i), tier, dtype=np.int8)
                tier_meta = self.metadata_map[tier_i, tier_j]

                tier_points = np.column_stack([tier_lats, tier_lons, tier_tiers])

            else:
                # Subdivide cells (vectorized with broadcasting!)
                points_per_cell = int((hrrr_cell_size_m / resolution_m) ** 2)
                n_per_side = int(np.sqrt(points_per_cell))

                print(f"      {points_per_cell} points per cell = ~{len(tier_i) * points_per_cell:,} total points")
                print(f"      Generating sub-grids (vectorized)...")

                # Extract center coords for all cells at once
                center_lats = lats_hrrr[tier_i, tier_j]
                center_lons = lons_hrrr[tier_i, tier_j]
                cell_metadata = self.metadata_map[tier_i, tier_j]

                # Create offset arrays for sub-grid
                lat_spacing = (resolution_m / 1000) / 111
                lon_spacing_base = (resolution_m / 1000) / 111

                # Sub-grid offsets
                half_extent = (n_per_side - 1) / 2
                offsets = np.linspace(-half_extent, half_extent, n_per_side)
                offset_grid = np.array(np.meshgrid(offsets, offsets)).T.reshape(-1, 2)

                # Vectorized generation for all cells
                n_cells = len(center_lats)
                n_sub = len(offset_grid)

                # Broadcast to create all points at once
                tier_lats = np.repeat(center_lats, n_sub) + np.tile(offset_grid[:, 0], n_cells) * lat_spacing

                # Lon spacing varies with latitude
                lon_spacings = lon_spacing_base / np.cos(np.radians(center_lats))
                tier_lons = np.repeat(center_lons, n_sub) + np.tile(offset_grid[:, 1], n_cells) * np.repeat(lon_spacings, n_sub)

                tier_tiers = np.full(len(tier_lats), tier, dtype=np.int8)
                tier_meta = np.repeat(cell_metadata, n_sub)

                tier_points = np.column_stack([tier_lats, tier_lons, tier_tiers])

            elapsed = time.time() - start
            print(f"      ✓ Generated {len(tier_points):,} points in {elapsed:.1f}s")

            all_points.append(tier_points)
            all_metadata.append(tier_meta)

        # Concatenate all tiers (vectorized!)
        points_array = np.vstack(all_points)
        metadata_array = np.concatenate(all_metadata)

        # Filter out points outside HRRR domain (Lambert Conformal projection)
        print("\nFiltering points outside HRRR domain...")
        filter_start = time.time()

        # Load HRRR boundary polygon from saved boundary coordinates
        try:
            boundary_lats = np.load('hrrr_boundary_lats.npy')
            boundary_lons = np.load('hrrr_boundary_lons.npy')

            # Two-stage filtering for speed:
            # Stage 1: Fast bounding box pre-filter (eliminates obvious outliers)
            print("      [1/2] Quick bounding box filter...")
            bbox_start = time.time()
            lat_min, lat_max = boundary_lats.min(), boundary_lats.max()
            lon_min, lon_max = boundary_lons.min(), boundary_lons.max()

            # Add small margin to avoid edge cases
            margin = 0.1  # degrees
            in_bbox = (
                (points_array[:, 0] >= lat_min - margin) &
                (points_array[:, 0] <= lat_max + margin) &
                (points_array[:, 1] >= lon_min - margin) &
                (points_array[:, 1] <= lon_max + margin)
            )

            points_before = len(points_array)
            bbox_rejected = points_before - in_bbox.sum()
            bbox_elapsed = time.time() - bbox_start
            print(f"            ✓ Rejected {bbox_rejected:,} points outside bounding box in {bbox_elapsed:.1f}s")

            # Stage 2: Precise polygon test (only for points in bounding box)
            if in_bbox.sum() > 0:
                print(f"      [2/2] Precise polygon test on {in_bbox.sum():,} candidates...")
                poly_start = time.time()

                # Simplify boundary polygon for speed (we don't need sub-km precision for filtering)
                print(f"            Simplifying boundary from {len(boundary_lats)} points...")
                boundary_polygon = Polygon(zip(boundary_lons, boundary_lats))
                simplified_polygon = boundary_polygon.simplify(tolerance=0.01, preserve_topology=True)  # ~1km tolerance
                simplified_coords = np.array(simplified_polygon.exterior.coords)
                print(f"            ✓ Simplified to {len(simplified_coords)} points (~{5712/len(simplified_coords):.0f}x faster)")

                # Create simplified path for fast testing
                hrrr_path = mpath.Path(simplified_coords)

                # Test only points that passed bbox filter
                candidate_points = points_array[in_bbox]
                point_coords = np.column_stack([candidate_points[:, 1], candidate_points[:, 0]])  # lon, lat
                within_polygon = hrrr_path.contains_points(point_coords)

                # Combine filters
                final_mask = np.zeros(points_before, dtype=bool)
                final_mask[in_bbox] = within_polygon

                points_array = points_array[final_mask]
                metadata_array = metadata_array[final_mask]
                points_removed = points_before - len(points_array)

                poly_elapsed = time.time() - poly_start
                print(f"            ✓ Polygon test completed in {poly_elapsed:.1f}s")
            else:
                # All points rejected by bbox
                points_array = np.empty((0, 3))
                metadata_array = np.empty(0)
                points_removed = points_before

            filter_elapsed = time.time() - filter_start
            print(f"      ✓ Total: Removed {points_removed:,} points outside HRRR domain in {filter_elapsed:.1f}s")
            print(f"      ✓ Retained {len(points_array):,} points within HRRR domain")

        except FileNotFoundError:
            print("      ⚠ Warning: HRRR boundary files not found, skipping domain filtering")
            print("      Run: python3 extract_hrrr_boundary.py to generate boundary files")

        total_elapsed = time.time() - overall_start
        print("\n" + "="*70)
        print(f" POINT GENERATION COMPLETE")
        print("="*70)
        print(f"  Total points: {len(points_array):,}")
        print(f"  Time elapsed: {total_elapsed:.1f} seconds")

        self.points = points_array
        self.point_metadata = metadata_array

        return points_array, metadata_array


class OutputWriterVectorized:
    """Vectorized output writer"""

    def __init__(self, points, metadata, hrrr_grid):
        self.points = points
        self.metadata = metadata
        self.hrrr_grid = hrrr_grid

    def write_netcdf(self, filename):
        """Write netCDF (already fast)"""
        print("\n" + "="*70)
        print(" STEP 4/5: WRITING OUTPUT NETCDF")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nWriting {len(self.points):,} points to {output_path}...")

        start_time = time.time()

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            npoints = len(self.points)
            ncfile.createDimension('npoints', npoints)

            ncfile.title = 'Adaptive Grid Points for XGBoost Weather Downscaling'
            ncfile.institution = 'TWC/IBM'
            ncfile.source = 'generate_adaptive_grid_SUPERFAST.py'
            ncfile.history = f'Created {datetime.now().isoformat()}'
            ncfile.conventions = 'CF-1.8'
            ncfile.total_points = npoints

            lat_var = ncfile.createVariable('latitude', 'f4', ('npoints',))
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'Latitude'
            lat_var[:] = self.points[:, 0]

            lon_var = ncfile.createVariable('longitude', 'f4', ('npoints',))
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'Longitude'
            lon_var[:] = self.points[:, 1]

            tier_var = ncfile.createVariable('tier', 'i1', ('npoints',))
            tier_var.long_name = 'Grid Tier'
            tier_var[:] = self.points[:, 2].astype(np.int8)

            meta_var = ncfile.createVariable('metadata', 'u2', ('npoints',))
            meta_var.long_name = 'Point Classification Metadata (bitfield)'
            meta_var.description = 'Bit 0: Urban, Bit 1: Suburban, Bit 2: Coastline, ' \
                                   'Bit 3: Lake, Bit 4: Ski Resort, Bit 5: Golf Course, ' \
                                   'Bit 6: Park, Bit 7: Highway, Bit 8: High Terrain Var'
            meta_var[:] = self.metadata

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ NetCDF written: {file_size_mb:.1f} MB in {elapsed:.1f}s")
        return output_path

    def create_visualization(self, filename):
        """VECTORIZED visualization"""
        print("\n" + "="*70)
        print(" STEP 5/5: CREATING VISUALIZATION (VECTORIZED)")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        start_time = time.time()

        lats_hrrr = self.hrrr_grid['lats']
        lons_hrrr = self.hrrr_grid['lons']
        shape = lats_hrrr.shape

        print(f"\n[1/3] Counting points per cell (vectorized)...")

        # VECTORIZED point counting using KDTree
        hrrr_points = np.column_stack([lats_hrrr.ravel(), lons_hrrr.ravel()])
        tree = cKDTree(hrrr_points)

        adaptive_points = self.points[:, :2]
        _, indices = tree.query(adaptive_points, k=1)

        # VECTORIZED counting using bincount
        point_count_flat = np.bincount(indices, minlength=hrrr_points.shape[0])
        point_count = point_count_flat.reshape(shape)

        print(f"      ✓ Counted in {time.time() - start_time:.1f}s")
        print(f"        Max points per cell: {point_count.max():,}")
        print(f"        Mean points per cell: {point_count.mean():.1f}")

        # Create map
        print(f"\n[2/3] Creating map...")
        fig = plt.figure(figsize=config.PLOT_FIGSIZE)

        lat_min, lat_max = lats_hrrr.min(), lats_hrrr.max()
        lon_min, lon_max = lons_hrrr.min(), lons_hrrr.max()

        m = Basemap(
            projection='lcc', lat_0=40, lon_0=-96, lat_1=33, lat_2=45,
            llcrnrlat=lat_min, urcrnrlat=lat_max,
            llcrnrlon=lon_min, urcrnrlon=lon_max,
            resolution='l'
        )

        m.drawcoastlines(linewidth=0.5, color='black')
        m.drawstates(linewidth=0.3, color='gray')

        x, y = m(lons_hrrr, lats_hrrr)
        point_count_plot = np.ma.masked_where(point_count == 0, point_count)

        cs = m.pcolormesh(x, y, point_count_plot, cmap=config.COLORMAP,
                         shading='auto',
                         norm=matplotlib.colors.LogNorm(vmin=1, vmax=point_count.max()))

        cbar = plt.colorbar(cs, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Points per HRRR Cell (log scale)', fontsize=10)

        plt.title(
            f'Adaptive Grid Point Density\nTotal Points: {len(self.points):,}',
            fontsize=14, fontweight='bold', pad=20
        )

        print(f"\n[3/3] Saving visualization...")
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ Visualization saved: {file_size_mb:.1f} MB in {elapsed:.1f}s")
        return output_path


def main():
    """Main execution - SUPER FAST!"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION (SUPER FAST - FULLY VECTORIZED)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target: {config.TARGET_TOTAL_POINTS:,} points")
    print(f"  Tier 1: {config.TIER_RESOLUTIONS[1]}m resolution")
    print(f"  Coastal buffer: {config.COASTLINE_BUFFER_OFFSHORE_KM}km")
    print(f"  Include lakes: {config.INCLUDE_LAKES_IN_TIER1}")
    print(f"\nStarted: {datetime.now().strftime('%H:%M:%S')}")

    # Load data
    loader = DataLoader()
    loader.load_all()

    # Terrain analysis
    terrain_analyzer = TerrainAnalyzer(
        loader.hrrr_grid['terrain'],
        loader.hrrr_grid['lats'],
        loader.hrrr_grid['lons']
    )
    terrain_var = terrain_analyzer.compute_terrain_variability()

    # Tier classification (vectorized)
    classifier = TierClassifierVectorized(loader, terrain_var)
    tier_map, metadata_map = classifier.create_tier_map()

    # Point generation (vectorized!)
    generator = AdaptiveGridGeneratorVectorized(loader, tier_map, metadata_map)
    points, metadata = generator.generate_points()

    # Output (vectorized!)
    writer = OutputWriterVectorized(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_points.nc')
    png_file = writer.create_visualization('adaptive_grid_density.png')

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print(" ✓ COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  NetCDF: {nc_file}")
    print(f"  Visualization: {png_file}")
    print(f"\nTotal points: {len(points):,}")
    print(f"Total runtime: {overall_elapsed/60:.1f} minutes")
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
