"""
GEN2 Adaptive Grid Generation with Gradual Transitions

Key improvements over SUPERFAST:
- Gradual resolution transitions (no abrupt tier boundaries)
- 6-tier system (0.1, 0.3, 0.5, 0.75, 1.5, 3.0 km)
- 3-pass hybrid algorithm:
  1. Core region classification
  2. Distance-based transition zones
  3. Tier constraint enforcement
- Target: ~20 million points
- Fully vectorized for performance
"""

import os
import sys
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.ndimage import generic_filter, distance_transform_edt
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

# Import from original for reuse
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

# Special value to indicate unassigned cells in core classification
NO_ASSIGNMENT = -1


class TierClassifierGEN2:
    """
    GEN2 tier classifier with gradual transitions

    Implements 3-pass algorithm:
    1. Core region classification (discrete assignments based on features)
    2. Distance-based transition zones (smooth distance fields)
    3. Tier constraint enforcement (ensure no large jumps)
    """

    def __init__(self, data_loader, terrain_variability):
        self.loader = data_loader
        self.terrain_var = terrain_variability
        self.tier_map = None
        self.metadata_map = None

        # Define HRRR Lambert Conformal projection for accurate buffering
        # Parameters from HRRR GRIB2: Latin1=38.5, Latin2=38.5, LoV=262.5 (-97.5)
        self.hrrr_crs = '+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

    def create_tier_map(self):
        """
        Three-pass tier classification with gradual transitions

        Returns:
            tier_map: 2D array of tier assignments (0-5)
            metadata_map: 2D array of feature metadata (bitfield)
        """
        print("\n" + "="*70)
        print(" GEN2 TIER CLASSIFICATION (3-PASS HYBRID ALGORITHM)")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        print(f"\nGrid size: {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR cells")

        # PASS 1: Core region classification
        print("\n" + "="*70)
        print(" PASS 1/3: CORE REGION CLASSIFICATION")
        print("="*70)
        core_tier_map, metadata_map = self._classify_core_regions()

        # PASS 2: Distance-based transition zones
        print("\n" + "="*70)
        print(" PASS 2/3: DISTANCE-BASED TRANSITION ZONES")
        print("="*70)
        transition_tier_map = self._create_transition_zones(core_tier_map, metadata_map)

        # PASS 3: Tier constraint enforcement
        print("\n" + "="*70)
        print(" PASS 3/3: TIER CONSTRAINT ENFORCEMENT")
        print("="*70)
        final_tier_map = self._enforce_tier_constraints(transition_tier_map)

        # Validate
        self._validate_tier_distribution(final_tier_map)

        self.tier_map = final_tier_map
        self.metadata_map = metadata_map

        # Summary
        print("\n" + "="*70)
        print(" FINAL TIER DISTRIBUTION")
        print("="*70)
        for tier in [0, 1, 2, 3, 4, 5]:
            count = (final_tier_map == tier).sum()
            pct = 100 * count / final_tier_map.size
            res = config.TIER_RESOLUTIONS[tier]
            print(f"  Tier {tier} ({res}m): {count:,} cells ({pct:.1f}%)")

        return final_tier_map, metadata_map

    def _classify_core_regions(self):
        """
        PASS 1: Classify core regions based on features

        Directly assigns tiers to cells based on geographic features:
        - Tier 0: Coastlines, lake shorelines, ski resorts
        - Tier 1: Most rugged terrain, golf courses, parks, major urban
        - Tier 2: Somewhat rugged terrain, suburban, highways, small urban
        - Tier 3: Slightly rugged terrain
        - Tier 5: Background (default)
        - Tier 4: NOT assigned here (transition only)

        Returns:
            tier_map: 2D array with discrete tier assignments
            metadata_map: 2D array with feature metadata (bitfield)
        """
        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        # Initialize (default to Tier 5 = background)
        tier_map = np.full(shape, 5, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)

        # Create GeoDataFrame with POLYGON geometries for HRRR cells (not just center points!)
        # This is CRITICAL - using cell polygons allows intersects to catch all touching cells
        print("\nCreating spatial index of grid cells as POLYGONS...")
        start = time.time()

        # First create points and reproject to get projected coordinates
        grid_points = [Point(lon, lat) for lat, lon in zip(lats.ravel(), lons.ravel())]
        grid_gdf_temp = gpd.GeoDataFrame(
            {'i': np.repeat(np.arange(shape[0]), shape[1]),
             'j': np.tile(np.arange(shape[1]), shape[0])},
            geometry=grid_points,
            crs='EPSG:4326'
        )
        grid_gdf_temp = grid_gdf_temp.to_crs(self.hrrr_crs)

        # HRRR cells are 3000m × 3000m - create polygon boxes around each center
        from shapely.geometry import box
        cell_size = 3000.0  # meters
        half_cell = cell_size / 2.0

        cell_polygons = []
        for geom in grid_gdf_temp.geometry:
            x, y = geom.x, geom.y
            # Create box: (minx, miny, maxx, maxy)
            cell_box = box(x - half_cell, y - half_cell, x + half_cell, y + half_cell)
            cell_polygons.append(cell_box)

        grid_gdf = gpd.GeoDataFrame(
            {'i': grid_gdf_temp['i'].values,
             'j': grid_gdf_temp['j'].values},
            geometry=cell_polygons,
            crs=self.hrrr_crs
        )

        elapsed = time.time() - start
        print(f"✓ Created spatial index with {len(grid_gdf):,} points in {elapsed:.1f}s")

        # Classify features in priority order (finest resolution first)
        # Note: Using np.minimum() ensures "finest resolution wins" rule

        # [1] TERRAIN VARIABILITY (3 tiers)
        print("\n[1/11] Terrain-based classification...")
        self._classify_terrain(tier_map, metadata)

        # [2] TIER 0: Coastlines (HARD CONSTRAINT)
        print("\n[2/11] Tier 0: Coastlines...")
        self._classify_coastlines(tier_map, metadata, grid_gdf)

        # [3] TIER 0: Large lake shorelines
        if config.INCLUDE_LAKES_IN_TIER0:
            print("\n[3/11] Tier 0: Large lake shorelines...")
            self._classify_lakes(tier_map, metadata, grid_gdf)
        else:
            print("\n[3/11] Tier 0: Large lake shorelines... SKIPPED")

        # [4] TIER 0: Ski resorts
        print("\n[4/11] Tier 0: Ski resorts...")
        self._classify_ski_resorts(tier_map, metadata, grid_gdf)

        # [5] TIER 1: Golf courses
        if config.INCLUDE_GOLF_COURSES:
            print("\n[5/11] Tier 1: Golf courses...")
            self._classify_golf_courses(tier_map, metadata, grid_gdf)
        else:
            print("\n[5/11] Tier 1: Golf courses... SKIPPED")

        # [6] TIER 1: National & State Parks
        if config.INCLUDE_PARKS:
            print("\n[6/11] Tier 1: National & State Parks...")
            self._classify_parks(tier_map, metadata, grid_gdf)
        else:
            print("\n[6/11] Tier 1: National & State Parks... SKIPPED")

        # [7] TIER 2: Major urban areas (≥50k population)
        print("\n[7/11] Tier 2: Major urbanized areas...")
        self._classify_high_density_urban(tier_map, metadata, grid_gdf)

        # [8] TIER 2: Small urban clusters (2.5k-50k population)
        print("\n[8/11] Tier 2: Small urban clusters...")
        self._classify_urban(tier_map, metadata, grid_gdf)

        # [9] TIER 2: Suburban areas
        print("\n[9/11] Tier 2: Suburban areas...")
        self._classify_suburban(tier_map, metadata, grid_gdf)

        # [10] TIER 2: Major highways
        if config.INCLUDE_HIGHWAYS:
            print("\n[10/11] Tier 2: Major highways...")
            self._classify_highways(tier_map, metadata, grid_gdf)
        else:
            print("\n[10/11] Tier 2: Major highways... SKIPPED")

        # [11] Mark unassigned cells for transition processing
        print("\n[11/11] Marking unassigned cells for transition processing...")
        unassigned_mask = (tier_map == 5) & (self.terrain_var < config.TERRAIN_TIER3_THRESHOLD)
        unassigned_count = unassigned_mask.sum()
        print(f"      {unassigned_count:,} cells without explicit tier assignment")
        print(f"      These will be assigned via distance-based transitions in Pass 2")

        return tier_map, metadata

    def _classify_terrain(self, tier_map, metadata):
        """Classify terrain variability into 3 tiers (conservative, budget available after narrow coastal buffer)"""
        # Tier 1: Most extreme terrain (>600m std dev) → 187.5m
        extreme = self.terrain_var > config.TERRAIN_TIER1_THRESHOLD
        tier_map[extreme] = np.minimum(tier_map[extreme], 1)
        metadata[extreme] |= (1 << 7)
        print(f"      ✓ Tier 1 (most extreme, >{config.TERRAIN_TIER1_THRESHOLD}m std dev): {extreme.sum():,} cells")

        # Tier 2: Very rugged (>400m std dev) → 375m
        very_rugged = (self.terrain_var > config.TERRAIN_TIER2_THRESHOLD) & \
                      (self.terrain_var <= config.TERRAIN_TIER1_THRESHOLD)
        tier_map[very_rugged] = np.minimum(tier_map[very_rugged], 2)
        metadata[very_rugged] |= (1 << 8)
        print(f"      ✓ Tier 2 (very rugged, {config.TERRAIN_TIER2_THRESHOLD}-{config.TERRAIN_TIER1_THRESHOLD}m): {very_rugged.sum():,} cells")

        # Tier 3: Moderate terrain (>200m std dev) → 750m
        moderate = (self.terrain_var > config.TERRAIN_TIER3_THRESHOLD) & \
                   (self.terrain_var <= config.TERRAIN_TIER2_THRESHOLD)
        tier_map[moderate] = np.minimum(tier_map[moderate], 3)
        metadata[moderate] |= (1 << 9)
        print(f"      ✓ Tier 3 (moderate, {config.TERRAIN_TIER3_THRESHOLD}-{config.TERRAIN_TIER2_THRESHOLD}m): {moderate.sum():,} cells")

    def _classify_coastlines(self, tier_map, metadata, grid_gdf):
        """Classify coastlines as Tier 0 (93.75m resolution) - HARD CONSTRAINT"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None or len(coastline_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing coastlines...")
        start = time.time()

        # Reproject and simplify
        coastline_simple = coastline_gdf.to_crs(self.hrrr_crs)
        # Reduce simplification to 100m to preserve island details
        coastline_simple['geometry'] = coastline_simple.geometry.simplify(100)

        # DON'T union - use intersects on individual segments for speed
        coastline_buffered = coastline_simple.copy()
        coastline_buffered['geometry'] = coastline_buffered.geometry.buffer(
            config.COASTLINE_BUFFER_KM * 1000  # Convert km to meters
        )

        # Use intersects on individual geometries (faster than union)
        # This catches ALL cells that touch the buffered coastline
        joined = gpd.sjoin(grid_gdf, coastline_buffered, how='inner', predicate='intersects')

        # NO LIMIT - coastlines are a hard constraint
        # Vectorized update (Tier 0 = 93.75m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 0)
        metadata[i_indices, j_indices] |= (1 << 2)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_lakes(self, tier_map, metadata, grid_gdf):
        """Classify lake shorelines as Tier 0 (100m resolution)"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None or len(lakes_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(lakes_gdf)} large lake shorelines...")
        start = time.time()

        # Reproject, extract boundaries, and buffer
        lakes_shorelines = lakes_gdf.to_crs(self.hrrr_crs)
        lakes_shorelines['geometry'] = lakes_shorelines.geometry.boundary
        lakes_shorelines['geometry'] = lakes_shorelines.geometry.buffer(
            config.LAKE_BUFFER_KM * 1000  # Convert km to meters
        )

        # Use intersects to catch all cells touching the buffer
        joined = gpd.sjoin(grid_gdf, lakes_shorelines, how='inner', predicate='intersects')

        # Vectorized update (Tier 0 = 100m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 0)
        metadata[i_indices, j_indices] |= (1 << 3)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_ski_resorts(self, tier_map, metadata, grid_gdf):
        """Classify ski resorts as Tier 0 (100m resolution)"""
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is None or len(ski_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(ski_gdf)} ski resorts...")
        start = time.time()

        # Reproject and buffer (5km covers resort area)
        ski_buffered = ski_gdf.to_crs(self.hrrr_crs)
        ski_buffered['geometry'] = ski_buffered.geometry.buffer(
            config.SKI_RESORT_BUFFER_KM * 1000
        )

        joined = gpd.sjoin(grid_gdf, ski_buffered, how='inner', predicate='within')

        # Vectorized update (Tier 0 = 100m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 0)
        metadata[i_indices, j_indices] |= (1 << 4)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_golf_courses(self, tier_map, metadata, grid_gdf):
        """Classify golf courses as Tier 1 (300m resolution)"""
        golf_gdf = self.loader.data.get('golf_courses')
        if golf_gdf is None or len(golf_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(golf_gdf)} golf courses...")
        start = time.time()

        # Reproject and buffer
        golf_buffered = golf_gdf.to_crs(self.hrrr_crs)
        golf_buffered['geometry'] = golf_buffered.geometry.buffer(
            config.GOLF_COURSE_BUFFER_KM * 1000
        )

        joined = gpd.sjoin(grid_gdf, golf_buffered, how='inner', predicate='within')

        # Vectorized update (Tier 1 = 300m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 5)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_parks(self, tier_map, metadata, grid_gdf):
        """Classify national & state parks as Tier 1 (300m resolution)"""
        parks_gdf = self.loader.data.get('parks')
        if parks_gdf is None or len(parks_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(parks_gdf)} parks...")
        start = time.time()

        # Reproject and buffer
        parks_buffered = parks_gdf.to_crs(self.hrrr_crs)
        parks_buffered['geometry'] = parks_buffered.geometry.buffer(
            config.PARKS_BUFFER_KM * 1000
        )

        joined = gpd.sjoin(grid_gdf, parks_buffered, how='inner', predicate='within')

        # Vectorized update (Tier 1 = 300m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 1)
        metadata[i_indices, j_indices] |= (1 << 6)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_high_density_urban(self, tier_map, metadata, grid_gdf):
        """Classify major urban areas (≥50k pop) as Tier 2 (375m resolution)"""
        high_density_gdf = self.loader.data.get('high_density_urban')
        if high_density_gdf is None or len(high_density_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(high_density_gdf)} major urbanized areas...")
        start = time.time()

        # Reproject to Lambert Conformal (no buffer - use polygon boundaries directly)
        high_density_gdf = high_density_gdf.to_crs(self.hrrr_crs)

        joined = gpd.sjoin(grid_gdf, high_density_gdf, how='inner', predicate='within')

        # Vectorized update (Tier 2 = 375m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 2)
        metadata[i_indices, j_indices] |= (1 << 0)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_urban(self, tier_map, metadata, grid_gdf):
        """Classify small urban clusters (2.5k-50k pop) as Tier 2 (500m resolution)"""
        urban_gdf = self.loader.data.get('urban')
        if urban_gdf is None or len(urban_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(urban_gdf)} small urban clusters...")
        start = time.time()

        # Reproject to Lambert Conformal (no buffer - use polygon boundaries directly)
        urban_gdf = urban_gdf.to_crs(self.hrrr_crs)

        joined = gpd.sjoin(grid_gdf, urban_gdf, how='inner', predicate='within')

        # Vectorized update (Tier 2 = 500m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 2)
        metadata[i_indices, j_indices] |= (1 << 1)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _classify_suburban(self, tier_map, metadata, grid_gdf):
        """
        Classify suburban areas as Tier 2 (500m resolution)

        Note: Requires suburban area data. If not available, this is skipped.
        Suburban areas can be identified from Census data or derived from
        urban area classifications.
        """
        # TODO: Implement suburban classification if data becomes available
        # For now, we rely on urban cluster data to implicitly cover suburban areas
        print("      Skipping (suburban data not yet integrated)")
        print("      Note: Suburban coverage handled implicitly by urban cluster classification")

    def _classify_highways(self, tier_map, metadata, grid_gdf):
        """Classify major highways as Tier 2 (500m resolution)"""
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is None or len(roads_gdf) == 0:
            print("      Skipping (no data)")
            return

        print(f"      Processing {len(roads_gdf)} highway segments...")
        start = time.time()

        # Reproject and buffer
        roads_buffered = roads_gdf.to_crs(self.hrrr_crs)
        roads_buffered['geometry'] = roads_buffered.geometry.buffer(
            config.HIGHWAY_BUFFER_KM * 1000
        )

        joined = gpd.sjoin(grid_gdf, roads_buffered, how='inner', predicate='within')

        # Vectorized update (Tier 2 = 500m)
        i_indices = joined['i'].values
        j_indices = joined['j'].values
        tier_map[i_indices, j_indices] = np.minimum(tier_map[i_indices, j_indices], 2)
        metadata[i_indices, j_indices] |= (1 << 7)

        elapsed = time.time() - start
        print(f"      ✓ Classified {len(joined):,} cells in {elapsed:.1f}s")

    def _create_transition_zones(self, core_tier_map, metadata_map):
        """
        PASS 2: Create smooth transition zones using distance fields

        Algorithm:
        1. Compute distance from each cell to nearest special feature (not terrain)
        2. For cells not in core regions, assign transition tier based on distance
        3. Terrain-based tiers don't create transitions (they ARE the natural resolution)

        Args:
            core_tier_map: Tier map from Pass 1 with discrete assignments
            metadata_map: Feature metadata to distinguish terrain from special features

        Returns:
            transition_tier_map: Tier map with smooth transitions
        """
        print("\nComputing distance fields for transition zones...")
        start_time = time.time()

        shape = core_tier_map.shape
        cell_size_km = 3.0  # HRRR cell size in km

        # Step 1: Compute distance fields for transitions
        # IMPORTANT: Only compute distance to SPECIAL FEATURES (coast, lakes, ski, urban)
        # NOT terrain-based tiers - those are natural resolutions for their locations
        # and shouldn't create transition zones

        # Get special feature cores from metadata (not terrain-based)
        # Metadata bits: 0=urban, 1=small urban, 2=coastline, 3=lake, 4=ski, 5=golf, 6=park, 7=highway
        # Bits 8-10 are terrain tiers - we EXCLUDE those
        special_features_mask = (metadata_map & 0xFF) > 0  # Any of bits 0-7 set

        print(f"  Computing distance field from special features (coast/lakes/ski/urban)...")
        print(f"    Special feature cells: {special_features_mask.sum():,}")
        print(f"    (Excluding {(~special_features_mask & (core_tier_map < 5)).sum():,} terrain-only cells from transitions)")

        tier_start = time.time()
        distance_to_features = distance_transform_edt(~special_features_mask)
        tier_elapsed = time.time() - tier_start
        print(f"    ✓ Completed in {tier_elapsed:.1f}s")

        distance_elapsed = time.time() - start_time
        print(f"✓ All distance fields computed in {distance_elapsed:.1f}s")

        # Step 2: Assign transition tiers based on distance to special features
        print("\nAssigning transition tiers based on distance from special features...")
        assign_start = time.time()

        transition_map = core_tier_map.copy()

        # Process only unassigned cells (Tier 5 background)
        # These are cells with no terrain variability and no special features
        unassigned_mask = (core_tier_map == 5)
        unassigned_cells = np.argwhere(unassigned_mask)

        print(f"  Processing {len(unassigned_cells):,} unassigned cells...")

        # Vectorized distance-based assignment
        for idx, (i, j) in enumerate(unassigned_cells):
            # Get distance to nearest special feature (in km)
            dist_km = distance_to_features[i, j] * cell_size_km

            # Apply transition thresholds from config
            # These define gradual resolution decrease away from special features
            assigned_tier = 5  # Default to background
            for tier in sorted(config.TRANSITION_THRESHOLDS.keys()):
                min_km, max_km = config.TRANSITION_THRESHOLDS[tier]
                if min_km <= dist_km < max_km:
                    assigned_tier = tier
                    break

            transition_map[i, j] = assigned_tier

        assign_elapsed = time.time() - assign_start
        print(f"✓ Transition assignment completed in {assign_elapsed:.1f}s")

        total_elapsed = time.time() - start_time
        print(f"\nPass 2 total time: {total_elapsed:.1f}s")

        return transition_map

    def _enforce_tier_constraints(self, tier_map):
        """
        PASS 3: Enforce tier constraints to prevent large jumps

        Iteratively smooth the tier map to ensure no adjacent cells
        differ by more than MAX_TIER_JUMP tiers.

        Args:
            tier_map: Tier map from Pass 2 with transitions

        Returns:
            smoothed_tier_map: Final tier map with enforced constraints
        """
        print("\nEnforcing tier constraints (iterative smoothing)...")
        start_time = time.time()

        smoothed = tier_map.astype(np.float32)

        # Define smoothing function for generic_filter
        def smooth_cell(values):
            """
            Smooth a cell based on its neighbors

            Args:
                values: Flattened 3×3 window (9 values)

            Returns:
                Smoothed value for center cell
            """
            center = values[4]  # Center of 3×3 window
            neighbors = np.concatenate([values[:4], values[5:]])  # 8-connected neighbors

            # Check if any neighbor violates constraint
            max_jump = np.max(np.abs(center - neighbors))

            if max_jump > config.MAX_TIER_JUMP:
                # Smooth: use minimum to favor finer resolution
                neighbor_avg = np.mean(neighbors)
                return min(center, neighbor_avg)

            return center

        # Iterative smoothing
        for iteration in range(config.CONSTRAINT_MAX_ITERATIONS):
            iter_start = time.time()
            prev = smoothed.copy()

            # Apply smoothing filter
            smoothed = generic_filter(
                smoothed,
                smooth_cell,
                size=3,
                mode='nearest'  # Use nearest for boundary handling
            )

            # Check convergence
            max_change = np.max(np.abs(smoothed - prev))
            cells_changed = np.sum(~np.isclose(smoothed, prev, atol=config.CONSTRAINT_CONVERGENCE_TOLERANCE))

            iter_elapsed = time.time() - iter_start
            print(f"  Iteration {iteration + 1}/{config.CONSTRAINT_MAX_ITERATIONS}: "
                  f"{cells_changed:,} cells changed (max Δ={max_change:.3f}) "
                  f"in {iter_elapsed:.1f}s")

            # Early stopping if converged
            if max_change < config.CONSTRAINT_CONVERGENCE_TOLERANCE:
                print(f"  ✓ Converged after {iteration + 1} iterations")
                break

        # Round to nearest valid tier
        final_map = np.round(smoothed).astype(np.int8)

        # Clamp to valid tier range [0, 5]
        final_map = np.clip(final_map, 0, 5)

        total_elapsed = time.time() - start_time
        print(f"\nPass 3 total time: {total_elapsed:.1f}s")

        # Verify constraints
        self._verify_tier_constraints(final_map)

        return final_map

    def _verify_tier_constraints(self, tier_map):
        """Verify that tier constraints are satisfied"""
        print("\nVerifying tier constraints...")

        shape = tier_map.shape
        violations = 0
        max_jump_found = 0

        # Check all interior cells (avoid boundary issues)
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                center = tier_map[i, j]

                # Check 8-connected neighbors
                neighbors = [
                    tier_map[i-1, j-1], tier_map[i-1, j], tier_map[i-1, j+1],
                    tier_map[i, j-1],                      tier_map[i, j+1],
                    tier_map[i+1, j-1], tier_map[i+1, j], tier_map[i+1, j+1]
                ]

                max_jump = max(abs(center - n) for n in neighbors)
                max_jump_found = max(max_jump_found, max_jump)

                if max_jump > config.MAX_TIER_JUMP:
                    violations += 1

        total_cells = (shape[0] - 2) * (shape[1] - 2)
        violation_pct = 100 * violations / total_cells if total_cells > 0 else 0

        print(f"  Max tier jump found: {max_jump_found}")
        print(f"  Constraint violations: {violations:,} / {total_cells:,} ({violation_pct:.3f}%)")

        if violations == 0:
            print("  ✓ All constraints satisfied!")
        elif violation_pct < 0.1:
            print("  ⚠ Very few violations (acceptable)")
        else:
            print("  ⚠ WARNING: Significant constraint violations detected")

    def _validate_tier_distribution(self, tier_map):
        """Sanity check on tier distribution"""
        total_cells = tier_map.size

        # Check Tier 0 percentage
        tier0_count = (tier_map == 0).sum()
        tier0_pct = 100 * tier0_count / total_cells

        if tier0_pct > 5:
            print(f"\n⚠ WARNING: Tier 0 has {tier0_pct:.1f}% of cells (target: <5%)")

        # Estimate total points
        tier_counts = [(tier_map == t).sum() for t in [0, 1, 2, 3, 4, 5]]
        tier_points = [
            tier_counts[0] * ((3000 / config.TIER_RESOLUTIONS[0]) ** 2),
            tier_counts[1] * ((3000 / config.TIER_RESOLUTIONS[1]) ** 2),
            tier_counts[2] * ((3000 / config.TIER_RESOLUTIONS[2]) ** 2),
            tier_counts[3] * ((3000 / config.TIER_RESOLUTIONS[3]) ** 2),
            tier_counts[4] * ((3000 / config.TIER_RESOLUTIONS[4]) ** 2),
            tier_counts[5] * 1  # Background = 1 point per cell
        ]
        estimated_total = sum(tier_points)

        target = config.TARGET_TOTAL_POINTS
        diff_pct = 100 * (estimated_total - target) / target

        print(f"\nEstimated total points: {estimated_total:,.0f}")
        print(f"Target: {target:,} ({diff_pct:+.1f}% difference)")

        if abs(diff_pct) > 50:
            print(f"⚠ WARNING: Point count differs from target by {abs(diff_pct):.1f}%")


class AdaptiveGridGeneratorGEN2:
    """
    GEN2 point generator for 6-tier system

    Fully vectorized point generation for all 6 tiers (0-5).
    Based on SUPERFAST algorithm with extended tier support.
    """

    def __init__(self, data_loader, tier_map, metadata_map):
        self.loader = data_loader
        self.tier_map = tier_map
        self.metadata_map = metadata_map
        self.points = None
        self.point_metadata = None

    def generate_points(self):
        """
        Fully vectorized point generation for tiers 0-5

        Returns:
            points_array: Nx3 array of [lat, lon, tier]
            metadata_array: N-length array of metadata bitfields
        """
        print("\n" + "="*70)
        print(" STEP 4/6: GENERATING ADAPTIVE GRID POINTS (VECTORIZED)")
        print("="*70)

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        all_points = []
        all_metadata = []

        overall_start = time.time()

        # Process all 6 tiers (0-5)
        for tier_idx, tier in enumerate([0, 1, 2, 3, 4, 5], 1):
            print(f"\n[{tier_idx}/6] Generating Tier {tier} points (resolution: {config.TIER_RESOLUTIONS[tier]}m)...")
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
                # HIERARCHICAL SUBDIVISION: Subdivide HRRR cells uniformly in native projection
                # This ensures perfect nesting and no gaps/duplicates
                print(f"      Hierarchical subdivision (resolution: {resolution_m}m)...")

                from mpl_toolkits.basemap import Basemap

                # Create projection matching HRRR
                lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
                lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2

                proj = Basemap(
                    projection='lcc',
                    lat_0=lat_ref,
                    lon_0=lon_ref,
                    lat_1=lat_ref - 5,
                    lat_2=lat_ref + 5,
                    llcrnrlat=lats_hrrr.min(),
                    urcrnrlat=lats_hrrr.max(),
                    llcrnrlon=lons_hrrr.min(),
                    urcrnrlon=lons_hrrr.max(),
                    resolution=None
                )

                # Convert tier cells to projected coordinates
                cell_lats = lats_hrrr[tier_i, tier_j]
                cell_lons = lons_hrrr[tier_i, tier_j]
                cell_x, cell_y = proj(cell_lons, cell_lats)

                # HRRR cell size in meters
                hrrr_cell_size_m = 3000.0

                # Number of subdivisions per side (must be power of 2)
                n_sub = int(np.round(hrrr_cell_size_m / resolution_m))
                points_per_cell = n_sub * n_sub

                print(f"      {n_sub}×{n_sub} = {points_per_cell} points per HRRR cell")

                # For each cell, generate sub-grid in projected space
                all_lats = []
                all_lons = []
                all_meta = []

                # Vectorize over cells
                for idx in range(len(tier_i)):
                    cx, cy = cell_x[idx], cell_y[idx]
                    cell_meta = self.metadata_map[tier_i[idx], tier_j[idx]]

                    # Cell boundaries (SW corner at cx-1500, cy-1500)
                    x_start = cx - hrrr_cell_size_m / 2
                    y_start = cy - hrrr_cell_size_m / 2

                    # Generate uniform grid starting from SW corner
                    # Points at: start, start+spacing, start+2*spacing, ..., start+(n-1)*spacing
                    xs = x_start + np.arange(n_sub) * resolution_m
                    ys = y_start + np.arange(n_sub) * resolution_m

                    # Meshgrid
                    xx, yy = np.meshgrid(xs, ys)
                    pts_x = xx.ravel()
                    pts_y = yy.ravel()

                    # Convert to lat/lon
                    pts_lon, pts_lat = proj(pts_x, pts_y, inverse=True)

                    all_lats.append(pts_lat)
                    all_lons.append(pts_lon)
                    all_meta.extend([cell_meta] * len(pts_lat))

                tier_lats = np.concatenate(all_lats)
                tier_lons = np.concatenate(all_lons)
                tier_meta = np.array(all_meta)
                tier_tiers = np.full(len(tier_lats), tier, dtype=np.int8)

                tier_points = np.column_stack([tier_lats, tier_lons, tier_tiers])

                print(f"      Generated {len(tier_points):,} points")

            elapsed = time.time() - start
            print(f"      ✓ Generated {len(tier_points):,} points in {elapsed:.1f}s")

            all_points.append(tier_points)
            all_metadata.append(tier_meta)

        # Concatenate all tiers (vectorized!)
        if len(all_points) > 0:
            points_array = np.vstack(all_points)
            metadata_array = np.concatenate(all_metadata)
        else:
            print("\n⚠ WARNING: No points generated!")
            return np.array([]), np.array([])

        # Remove duplicate points (from shared cell edges)
        print("\nRemoving duplicate points from shared cell boundaries...")
        dup_start = time.time()
        initial_count = len(points_array)

        # Round coordinates to ~1m precision to identify duplicates
        # (coordinates in degrees, ~0.00001 deg ≈ 1m)
        lat_lon_rounded = np.round(points_array[:, :2], decimals=5)

        # Find unique points (keep first occurrence, preserve tier assignment)
        _, unique_indices = np.unique(lat_lon_rounded, axis=0, return_index=True)

        points_array = points_array[unique_indices]
        metadata_array = metadata_array[unique_indices]

        duplicates_removed = initial_count - len(points_array)
        dup_elapsed = time.time() - dup_start
        print(f"✓ Removed {duplicates_removed:,} duplicate points ({100*duplicates_removed/initial_count:.1f}%) in {dup_elapsed:.1f}s")
        print(f"  Retained {len(points_array):,} unique points")

        # Filter out points outside HRRR domain
        print("\nFiltering points outside HRRR domain...")
        filter_start = time.time()

        # Load HRRR boundary polygon from saved boundary coordinates
        try:
            boundary_lats = np.load('hrrr_boundary_lats.npy')
            boundary_lons = np.load('hrrr_boundary_lons.npy')

            # Two-stage filtering for speed:
            # Stage 1: Fast bounding box pre-filter
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

                # Simplify boundary polygon for speed
                print(f"            Simplifying boundary from {len(boundary_lats)} points...")
                boundary_polygon = Polygon(zip(boundary_lons, boundary_lats))
                simplified_polygon = boundary_polygon.simplify(tolerance=0.01, preserve_topology=True)
                simplified_coords = np.array(simplified_polygon.exterior.coords)
                print(f"            ✓ Simplified to {len(simplified_coords)} points")

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


class OutputWriterGEN2:
    """Vectorized output writer for GEN2"""

    def __init__(self, points, metadata, hrrr_grid):
        self.points = points
        self.metadata = metadata
        self.hrrr_grid = hrrr_grid

    def write_netcdf(self, filename):
        """Write netCDF output file"""
        print("\n" + "="*70)
        print(" STEP 5/6: WRITING OUTPUT NETCDF")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nWriting {len(self.points):,} points to {output_path}...")

        start_time = time.time()

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            npoints = len(self.points)
            ncfile.createDimension('npoints', npoints)

            ncfile.title = 'GEN2 Adaptive Grid Points for Weather Downscaling'
            ncfile.institution = 'TWC/IBM'
            ncfile.source = 'generate_adaptive_grid_GEN2.py'
            ncfile.history = f'Created {datetime.now().isoformat()}'
            ncfile.conventions = 'CF-1.8'
            ncfile.total_points = npoints
            ncfile.tier_system = 'GEN2 (6 tiers: 0.1, 0.3, 0.5, 0.75, 1.5, 3.0 km)'
            ncfile.transition_distance_km = config.TRANSITION_DISTANCE_KM
            ncfile.max_tier_jump = config.MAX_TIER_JUMP

            lat_var = ncfile.createVariable('latitude', 'f4', ('npoints',))
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'Latitude'
            lat_var[:] = self.points[:, 0]

            lon_var = ncfile.createVariable('longitude', 'f4', ('npoints',))
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'Longitude'
            lon_var[:] = self.points[:, 1]

            tier_var = ncfile.createVariable('tier', 'i1', ('npoints',))
            tier_var.long_name = 'Grid Tier (0-5)'
            tier_var.description = 'Tier 0=100m, 1=300m, 2=500m, 3=750m, 4=1500m, 5=3000m'
            tier_var[:] = self.points[:, 2].astype(np.int8)

            meta_var = ncfile.createVariable('metadata', 'u2', ('npoints',))
            meta_var.long_name = 'Point Classification Metadata (bitfield)'
            meta_var.description = 'Bit 0: Urban, Bit 1: Small Urban, Bit 2: Coastline, ' \
                                   'Bit 3: Lake, Bit 4: Ski Resort, Bit 5: Golf Course, ' \
                                   'Bit 6: Park, Bit 7: Highway, Bit 8-10: Terrain Tier'
            meta_var[:] = self.metadata

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ NetCDF written: {file_size_mb:.1f} MB in {elapsed:.1f}s")
        return output_path

    def create_visualization(self, filename):
        """Create visualization of point density"""
        print("\n" + "="*70)
        print(" STEP 6/6: CREATING VISUALIZATION (VECTORIZED)")
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
            f'GEN2 Adaptive Grid Point Density (Gradual Transitions)\nTotal Points: {len(self.points):,}',
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
    """Main execution for GEN2 adaptive grid generation"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION GEN2")
    print(" Gradual Transitions | 6-Tier System | ~20M Points")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target: {config.TARGET_TOTAL_POINTS:,} points")
    print(f"  Transition distance: {config.TRANSITION_DISTANCE_KM}km")
    print(f"  Max tier jump: {config.MAX_TIER_JUMP}")
    print(f"  Transition algorithm: {config.TRANSITION_ALGORITHM}")
    print(f"\nTier Resolutions:")
    for tier, res in config.TIER_RESOLUTIONS.items():
        print(f"  Tier {tier}: {res}m ({res/1000:.1f} km)")
    print(f"\nStarted: {datetime.now().strftime('%H:%M:%S')}")

    # STEP 1: Load data
    print("\n" + "="*70)
    print(" STEP 1/6: LOADING DATA")
    print("="*70)
    loader = DataLoader()
    loader.load_all()

    # STEP 2: Terrain analysis
    print("\n" + "="*70)
    print(" STEP 2/6: TERRAIN ANALYSIS")
    print("="*70)
    terrain_analyzer = TerrainAnalyzer(
        loader.hrrr_grid['terrain'],
        loader.hrrr_grid['lats'],
        loader.hrrr_grid['lons']
    )
    terrain_var = terrain_analyzer.compute_terrain_variability()

    # STEP 3: Tier classification (GEN2 3-pass algorithm)
    classifier = TierClassifierGEN2(loader, terrain_var)
    tier_map, metadata_map = classifier.create_tier_map()

    # STEP 4: Point generation
    generator = AdaptiveGridGeneratorGEN2(loader, tier_map, metadata_map)
    points, metadata = generator.generate_points()

    # Check if point generation succeeded
    if len(points) == 0:
        print("\n⚠ ERROR: No points generated. Exiting.")
        return

    # STEP 5 & 6: Output
    writer = OutputWriterGEN2(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_GEN2.nc')
    png_file = writer.create_visualization('adaptive_grid_GEN2_density.png')

    overall_elapsed = time.time() - overall_start

    # Final summary
    print("\n" + "="*70)
    print(" ✓ GEN2 GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  NetCDF: {nc_file}")
    print(f"  Visualization: {png_file}")
    print(f"\nResults:")
    print(f"  Total points: {len(points):,}")
    print(f"  Target points: {config.TARGET_TOTAL_POINTS:,}")
    diff_pct = 100 * (len(points) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
    print(f"  Difference: {diff_pct:+.1f}%")
    print(f"\nPerformance:")
    print(f"  Total runtime: {overall_elapsed/60:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
