"""
FAST version - uses spatial indexing and vectorization
Orders of magnitude faster than iterating through cells
"""

import os
import sys
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
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
import config

# Import from original
from generate_adaptive_grid import DataLoader, TerrainAnalyzer, OutputWriter, AdaptiveGridGenerator


class TierClassifierFast:
    """Ultra-fast tier classifier using spatial indexing"""

    def __init__(self, data_loader, terrain_variability):
        self.loader = data_loader
        self.terrain_var = terrain_variability
        self.tier_map = None

    def create_tier_map(self):
        """Create tier classification using fast spatial operations"""
        print("\n" + "="*70)
        print(" STEP 2/5: TIER CLASSIFICATION (FAST VERSION)")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize
        tier_map = np.full(shape, 4, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)

        # Create GeoDataFrame of all HRRR grid cell centers
        print("\nCreating spatial index of grid cells...")
        grid_points = []
        grid_indices = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid_points.append(Point(lons[i, j], lats[i, j]))
                grid_indices.append((i, j))

        grid_gdf = gpd.GeoDataFrame(
            {'grid_idx': grid_indices},
            geometry=grid_points,
            crs='EPSG:4326'
        )
        print(f"✓ Created spatial index with {len(grid_gdf):,} points")

        # [1] Tier 2: Terrain variability
        print("\n[1/6] Tier 2: Terrain variability...")
        high_terrain_var = self.terrain_var > config.TERRAIN_STDDEV_THRESHOLD
        tier_map[high_terrain_var] = np.minimum(tier_map[high_terrain_var], 2)
        metadata[high_terrain_var] |= (1 << 8)
        print(f"      ✓ Classified {high_terrain_var.sum():,} cells")

        # [2] Tier 3: Urban areas
        print("\n[2/6] Tier 3: Urban areas...")
        self._classify_urban_fast(tier_map, metadata, grid_gdf)

        # [3] Tier 3: Highways
        print("\n[3/6] Tier 3: Major highways...")
        self._classify_highways_fast(tier_map, metadata, grid_gdf)

        # [4] Tier 1: Coastlines (FAST!)
        print("\n[4/6] Tier 1: Coastlines...")
        self._classify_coastlines_fast(tier_map, metadata, grid_gdf)

        # [5] Tier 1: Lakes (skip per config)
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
        for tier in [1, 2, 3, 4]:
            count = (tier_map == tier).sum()
            pct = 100 * count / tier_map.size
            print(f"  Tier {tier}: {count:,} cells ({pct:.1f}%)")

        return tier_map, metadata

    def _classify_urban_fast(self, tier_map, metadata, grid_gdf):
        """Fast urban classification using spatial join"""
        urban_gdf = self.loader.data.get('urban')
        if urban_gdf is None or len(urban_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(urban_gdf)} urban areas...")
        start = time.time()

        # Spatial join - finds all grid points within urban areas
        joined = gpd.sjoin(grid_gdf, urban_gdf, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 3)
            metadata[i, j] |= (1 << 0)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_highways_fast(self, tier_map, metadata, grid_gdf):
        """Fast highway classification using buffer and spatial join"""
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is None or len(roads_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(roads_gdf)} road segments...")
        start = time.time()

        # Buffer roads by 0.5km
        roads_buffered = roads_gdf.copy()
        roads_buffered['geometry'] = roads_buffered.geometry.buffer(0.5 / 111)

        # Spatial join
        joined = gpd.sjoin(grid_gdf, roads_buffered, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 3)
            metadata[i, j] |= (1 << 7)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_coastlines_fast(self, tier_map, metadata, grid_gdf):
        """FAST coastline classification using simplified geometry and spatial join"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None or len(coastline_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing coastlines...")
        print(f"      Step 1: Simplifying coastline geometry...")
        start = time.time()

        # Simplify coastline to speed up buffer operation
        coastline_simple = coastline_gdf.copy()
        coastline_simple['geometry'] = coastline_simple.geometry.simplify(0.01)  # ~1km tolerance

        print(f"      Step 2: Creating {config.COASTLINE_BUFFER_OFFSHORE_KM}km buffer...")
        # Create buffer
        coastline_buffered = coastline_simple.copy()
        coastline_buffered['geometry'] = coastline_buffered.geometry.buffer(
            config.COASTLINE_BUFFER_OFFSHORE_KM / 111
        )

        # Merge all buffered coastlines
        print(f"      Step 3: Merging coastline buffers...")
        coast_union = unary_union(coastline_buffered.geometry)
        coast_gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[coast_union], crs='EPSG:4326')

        print(f"      Step 4: Spatial join to find coastal grid cells...")
        # Spatial join - this is MUCH faster than iterating
        joined = gpd.sjoin(grid_gdf, coast_gdf, how='inner', predicate='within')

        # Apply hard limit
        if len(joined) > 50000:
            print(f"      ⚠ Found {len(joined):,} coastal cells, limiting to 50,000")
            joined = joined.iloc[:50000]

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 1)
            metadata[i, j] |= (1 << 2)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_lakes_fast(self, tier_map, metadata, grid_gdf):
        """Fast lake classification"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None or len(lakes_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(lakes_gdf)} large lakes...")
        start = time.time()

        # Buffer lakes slightly
        lakes_buffered = lakes_gdf.copy()
        lakes_buffered['geometry'] = lakes_buffered.geometry.buffer(1 / 111)  # 1km buffer

        # Spatial join
        joined = gpd.sjoin(grid_gdf, lakes_buffered, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 1)
            metadata[i, j] |= (1 << 3)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_ski_resorts_fast(self, tier_map, metadata, grid_gdf):
        """Fast ski resort classification"""
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is None or len(ski_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(ski_gdf)} ski resorts...")
        start = time.time()

        # Buffer ski resorts
        ski_buffered = ski_gdf.copy()
        ski_buffered['geometry'] = ski_buffered.geometry.buffer(2 / 111)  # 2km buffer

        # Spatial join
        joined = gpd.sjoin(grid_gdf, ski_buffered, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 1)
            metadata[i, j] |= (1 << 4)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_golf_courses_fast(self, tier_map, metadata, grid_gdf):
        """Fast golf course classification using spatial join"""
        golf_gdf = self.loader.data.get('golf_courses')
        if golf_gdf is None or len(golf_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(golf_gdf)} golf courses...")
        start = time.time()

        # Buffer
        golf_buffered = golf_gdf.copy()
        golf_buffered['geometry'] = golf_buffered.geometry.buffer(
            config.GOLF_COURSE_BUFFER_KM / 111  # Convert km to degrees
        )

        # Spatial join
        joined = gpd.sjoin(grid_gdf, golf_buffered, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 1)
            metadata[i, j] |= (1 << 5)  # Bit 5 = golf course

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_parks_fast(self, tier_map, metadata, grid_gdf):
        """Fast parks classification using spatial join"""
        parks_gdf = self.loader.data.get('parks')
        if parks_gdf is None or len(parks_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(parks_gdf)} parks...")
        start = time.time()

        # Buffer
        parks_buffered = parks_gdf.copy()
        parks_buffered['geometry'] = parks_buffered.geometry.buffer(
            config.PARKS_BUFFER_KM / 111
        )

        # Spatial join
        joined = gpd.sjoin(grid_gdf, parks_buffered, how='inner', predicate='within')

        # Update tier map
        for idx in joined.index:
            i, j = grid_gdf.loc[idx, 'grid_idx']
            tier_map[i, j] = np.minimum(tier_map[i, j], 1)
            metadata[i, j] |= (1 << 6)  # Bit 6 = park

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _validate_tier_distribution(self, tier_map):
        """Sanity check on tier distribution"""
        total_cells = tier_map.size
        tier1_count = (tier_map == 1).sum()
        tier1_pct = 100 * tier1_count / total_cells

        if tier1_pct > 10:
            print(f"\n⚠ WARNING: Tier 1 has {tier1_pct:.1f}% of cells (expected <10%)")

        # Estimate total points
        tier_counts = [(tier_map == t).sum() for t in [1, 2, 3, 4]]
        tier_points = [
            tier_counts[0] * ((3000 / config.TIER_RESOLUTIONS[1]) ** 2),
            tier_counts[1] * ((3000 / config.TIER_RESOLUTIONS[2]) ** 2),
            tier_counts[2] * ((3000 / config.TIER_RESOLUTIONS[3]) ** 2),
            tier_counts[3] * 1
        ]
        estimated_total = sum(tier_points)

        print(f"\nEstimated total points: {estimated_total:,.0f}")
        if estimated_total > config.TARGET_TOTAL_POINTS * 2:
            print(f"⚠ WARNING: Estimated {estimated_total:,.0f} >> target {config.TARGET_TOTAL_POINTS:,}")


def main():
    """Main execution with FAST classifier"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION (FAST VERSION)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target points: {config.TARGET_TOTAL_POINTS:,}")
    print(f"  Tier 1 resolution: {config.TIER_RESOLUTIONS[1]}m")
    print(f"  Coastal buffer: {config.COASTLINE_BUFFER_OFFSHORE_KM}km")
    print(f"  Include lakes: {config.INCLUDE_LAKES_IN_TIER1}")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = DataLoader()
    loader.load_all()

    # Analyze terrain
    terrain_analyzer = TerrainAnalyzer(
        loader.hrrr_grid['terrain'],
        loader.hrrr_grid['lats'],
        loader.hrrr_grid['lons']
    )
    terrain_var = terrain_analyzer.compute_terrain_variability()

    # Classify tiers (FAST!)
    classifier = TierClassifierFast(loader, terrain_var)
    tier_map, metadata_map = classifier.create_tier_map()

    # Generate points
    generator = AdaptiveGridGenerator(loader, tier_map, metadata_map)
    points, metadata = generator.generate_points()

    # Write outputs
    writer = OutputWriter(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_points.nc')
    png_file = writer.create_visualization('adaptive_grid_density.png')

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print(" ✓ COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  NetCDF: {nc_file}")
    print(f"  Visualization: {png_file}")
    print(f"\nSummary:")
    print(f"  Total points: {len(points):,}")
    print(f"  Runtime: {overall_elapsed/60:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
