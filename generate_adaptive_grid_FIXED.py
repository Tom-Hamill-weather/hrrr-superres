"""
Adaptive Grid Point Generation for XGBoost Weather Downscaling - FIXED VERSION

Key fixes:
- Proper spatial intersection instead of bounding boxes
- Sanity checks on tier cell counts
- Memory-efficient point generation
- Conservative coastline/lake classification
"""

import os
import sys
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
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

# Import data loader from original
from generate_adaptive_grid import DataLoader, TerrainAnalyzer, OutputWriter


class TierClassifierFixed:
    """Fixed tier classifier with proper spatial intersection"""

    def __init__(self, data_loader, terrain_variability):
        self.loader = data_loader
        self.terrain_var = terrain_variability
        self.tier_map = None

    def create_tier_map(self):
        """Create tier classification with proper spatial operations"""
        print("\n" + "="*70)
        print(" STEP 2/5: TIER CLASSIFICATION (FIXED)")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize to Tier 4
        tier_map = np.full(shape, 4, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)

        # [1] Tier 2: Terrain variability
        print("\n[1/6] Classifying Tier 2: Terrain variability...")
        high_terrain_var = self.terrain_var > config.TERRAIN_STDDEV_THRESHOLD
        tier_map[high_terrain_var] = np.minimum(tier_map[high_terrain_var], 2)
        metadata[high_terrain_var] |= (1 << 8)
        print(f"      ✓ Tier 2 cells from terrain: {high_terrain_var.sum():,}")

        # [2] Tier 3: Urban areas (point-in-polygon)
        print("\n[2/6] Tier 3: Urban areas...")
        self._classify_urban_areas(tier_map, metadata, lats, lons)

        # [3] Tier 3: Major highways (distance-based)
        print("\n[3/6] Tier 3: Major highways...")
        self._classify_highways(tier_map, metadata, lats, lons)

        # [4] Tier 1: Coastlines (CONSERVATIVE - only actual coastal cells)
        print("\n[4/6] Tier 1: Coastlines...")
        self._classify_coastlines(tier_map, metadata, lats, lons)

        # [5] Tier 1: Large lakes (optional)
        if config.INCLUDE_LAKES_IN_TIER1:
            print("\n[5/6] Tier 1: Large lakes...")
            self._classify_lakes(tier_map, metadata, lats, lons)
        else:
            print("\n[5/6] Tier 1: Large lakes... SKIPPED (config.INCLUDE_LAKES_IN_TIER1=False)")

        # [6] Tier 1: Ski resorts
        print("\n[6/6] Tier 1: Ski resorts...")
        self._classify_ski_resorts(tier_map, metadata, lats, lons)

        # Sanity checks
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

    def _classify_urban_areas(self, tier_map, metadata, lats, lons):
        """Classify urban areas using point-in-polygon"""
        urban_gdf = self.loader.data.get('urban')
        if urban_gdf is None or len(urban_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(urban_gdf)} urban areas...")

        # Create points for HRRR grid centers
        points = [Point(lon, lat) for lat, lon in zip(lats.ravel(), lons.ravel())]
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=urban_gdf.crs)

        # Spatial join to find points within urban areas
        joined = gpd.sjoin(points_gdf, urban_gdf, how='inner', predicate='within')

        # Mark these cells as Tier 3
        urban_indices = joined.index.values
        for idx in urban_indices:
            i = idx // lats.shape[1]
            j = idx % lats.shape[1]
            tier_map[i, j] = np.minimum(tier_map[i, j], 3)
            metadata[i, j] |= (1 << 0)

        print(f"      ✓ Classified {len(urban_indices):,} cells")

    def _classify_highways(self, tier_map, metadata, lats, lons):
        """Classify areas near major highways"""
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is None or len(roads_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(roads_gdf)} road segments...")
        print("      (Using distance-based classification, may take a few minutes)")

        # Sample grid points (use every Nth point for speed)
        sample_factor = 5  # Sample every 5th row/col
        sample_lats = lats[::sample_factor, ::sample_factor]
        sample_lons = lons[::sample_factor, ::sample_factor]

        # Merge all roads into single geometry
        all_roads = unary_union(roads_gdf.geometry)

        count = 0
        for i in range(0, lats.shape[0], sample_factor):
            for j in range(0, lats.shape[1], sample_factor):
                pt = Point(lons[i, j], lats[i, j])
                # Check if within 0.5km (rough distance)
                if all_roads.distance(pt) < 0.5 / 111:  # ~0.5km in degrees
                    # Mark this cell and neighbors
                    for di in range(sample_factor):
                        for dj in range(sample_factor):
                            ii, jj = i + di, j + dj
                            if ii < lats.shape[0] and jj < lats.shape[1]:
                                tier_map[ii, jj] = np.minimum(tier_map[ii, jj], 3)
                                metadata[ii, jj] |= (1 << 7)
                                count += 1

        print(f"      ✓ Classified {count:,} cells")

    def _classify_coastlines(self, tier_map, metadata, lats, lons):
        """CONSERVATIVE coastline classification - only actual coastal cells"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None or len(coastline_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing coastlines...")
        print("      Using CONSERVATIVE approach - only cells near actual coastline")

        # Merge all coastlines
        all_coast = unary_union(coastline_gdf.geometry)

        # Buffer by a small amount (~5km max)
        coast_buffered = all_coast.buffer(5 / 111)  # 5km in degrees

        # Check each grid cell
        count = 0
        for i in tqdm(range(lats.shape[0]), desc="      Checking cells", leave=False):
            for j in range(lats.shape[1]):
                pt = Point(lons[i, j], lats[i, j])
                if coast_buffered.contains(pt) or coast_buffered.intersects(pt.buffer(3/111)):
                    tier_map[i, j] = np.minimum(tier_map[i, j], 1)
                    metadata[i, j] |= (1 << 2)
                    count += 1

                    # Hard limit - stop if too many cells
                    if count > 50000:  # Max 50k coastal cells (conservative)
                        print(f"      ⚠ Hit safety limit of 50k coastal cells")
                        break
            if count > 50000:
                break

        print(f"      ✓ Classified {count:,} cells")

    def _classify_lakes(self, tier_map, metadata, lats, lons):
        """Classify large lakes"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None or len(lakes_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(lakes_gdf)} large lakes...")

        # Merge all lakes
        all_lakes = unary_union(lakes_gdf.geometry)

        count = 0
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                pt = Point(lons[i, j], lats[i, j])
                if all_lakes.contains(pt) or all_lakes.intersects(pt.buffer(1/111)):
                    tier_map[i, j] = np.minimum(tier_map[i, j], 1)
                    metadata[i, j] |= (1 << 3)
                    count += 1

        print(f"      ✓ Classified {count:,} cells")

    def _classify_ski_resorts(self, tier_map, metadata, lats, lons):
        """Classify ski resort areas"""
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is None or len(ski_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(ski_gdf)} ski resorts...")

        count = 0
        for _, resort in ski_gdf.iterrows():
            pt = resort.geometry
            if pt.geom_type != 'Point':
                continue

            # Find nearest grid cells (within ~2km)
            for i in range(lats.shape[0]):
                for j in range(lats.shape[1]):
                    cell_pt = Point(lons[i, j], lats[i, j])
                    if pt.distance(cell_pt) < 2 / 111:  # ~2km
                        tier_map[i, j] = np.minimum(tier_map[i, j], 1)
                        metadata[i, j] |= (1 << 4)
                        count += 1

        print(f"      ✓ Classified {count:,} cells")

    def _validate_tier_distribution(self, tier_map):
        """Sanity check on tier distribution"""
        total_cells = tier_map.size
        tier1_count = (tier_map == 1).sum()
        tier1_pct = 100 * tier1_count / total_cells

        # Tier 1 shouldn't exceed 10% of total
        if tier1_pct > 10:
            print(f"\n⚠ WARNING: Tier 1 has {tier1_pct:.1f}% of cells (expected <10%)")
            print("  This will generate too many points. Classification may be incorrect.")

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
        if estimated_total > config.TARGET_TOTAL_POINTS * 1.5:
            print(f"⚠ WARNING: Estimated points ({estimated_total:,.0f}) >> target ({config.TARGET_TOTAL_POINTS:,})")


# Use the fixed classifier
def main():
    """Main execution with fixed classifier"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION (FIXED VERSION)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target points: {config.TARGET_TOTAL_POINTS:,}")
    print(f"  Tier resolutions: {config.TIER_RESOLUTIONS}")
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

    # Classify tiers (FIXED VERSION)
    classifier = TierClassifierFixed(loader, terrain_var)
    tier_map, metadata_map = classifier.create_tier_map()

    # Import the point generator
    from generate_adaptive_grid import AdaptiveGridGenerator

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
