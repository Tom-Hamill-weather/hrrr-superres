"""
Sparse Index Approach - Efficient Grid Generation
==================================================

Algorithm:
1. Process features individually (no massive unions)
2. Store sparse (i,j) index sets where mask = 1
3. Use bounding boxes to skip irrelevant features
4. Merge index sets at the end with tier priority
5. Apply stride patterns and convert to lat/lon only at final step

Memory efficient: Only stores indices of masked points (~5-10% of domain)
"""

import numpy as np
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, box
from shapely.prepared import prep
from shapely import vectorized
try:
    from shapely import contains_xy
    HAS_CONTAINS_XY = True
except ImportError:
    HAS_CONTAINS_XY = False
from shapely.strtree import STRtree
import time
import os
from generate_adaptive_grid import DataLoader, TerrainAnalyzer
import config

class SparseGridGenerator:
    def __init__(self, test_region=None):
        self.loader = DataLoader()
        self.n_fine = 32  # points per HRRR cell dimension
        self.test_region = test_region

    def generate(self):
        print("="*70)
        print(" SPARSE INDEX APPROACH - EFFICIENT GRID GENERATION")
        print("="*70)

        start_time = time.time()

        # Load data
        print("\nLoading data...")
        self.loader.load_all()

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        terrain = self.loader.hrrr_grid['terrain']
        shape = lats_hrrr.shape

        print(f"HRRR shape: {shape[0]}×{shape[1]} = {shape[0]*shape[1]:,} cells")

        n_fine_i = shape[0] * self.n_fine
        n_fine_j = shape[1] * self.n_fine
        print(f"Fine grid: {n_fine_i}×{n_fine_j} = {n_fine_i*n_fine_j:,} points")

        # Create projection
        lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
        lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2
        self.proj = Basemap(
            projection='lcc',
            lat_0=lat_ref, lon_0=lon_ref,
            lat_1=lat_ref - 5, lat_2=lat_ref + 5,
            llcrnrlat=lats_hrrr.min(), urcrnrlat=lats_hrrr.max(),
            llcrnrlon=lons_hrrr.min(), urcrnrlon=lons_hrrr.max(),
            resolution=None
        )
        self.hrrr_crs = self.proj.proj4string

        # Convert to projected coordinates
        print("\nConverting to projected coordinates...")
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)

        # Precompute cell geometry
        print("Precomputing cell geometry...")
        dx_east, dy_east, dx_north, dy_north = self._compute_cell_vectors(x_hrrr, y_hrrr)

        # Initialize sparse index sets for each mask
        print("\nInitializing sparse index sets...")
        index_sets = self._initialize_index_sets()

        # Process terrain (fast - already on HRRR grid)
        print("\nProcessing terrain thresholds...")
        self._process_terrain(terrain, index_sets, n_fine_i, n_fine_j)

        # Process features in patches (avoids creating full 2B point grid)
        print("\nProcessing features in patches...")
        self._process_features_patched(lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                                       dx_east, dy_east, dx_north, dy_north,
                                       shape, index_sets)

        # Apply tier priority and stride patterns
        print("\nApplying tier assignments and stride patterns...")
        index_tier_map = self._apply_tier_logic(index_sets, n_fine_i, n_fine_j)

        # Convert indices to lat/lon
        print("\nConverting indices to lat/lon...")
        lats_out, lons_out, tiers_out = self._indices_to_latlon(index_tier_map, lats_hrrr, lons_hrrr,
                                                                  x_hrrr, y_hrrr, dx_east, dy_east,
                                                                  dx_north, dy_north)

        print(f"\nTotal points: {len(lats_out):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_out) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # Write output
        print("\nWriting output...")
        if self.test_region:
            output_path = f'output/adaptive_grid_SPARSE_{self.test_region}.nc'
        else:
            output_path = 'output/adaptive_grid_SPARSE.nc'
        self._write_output(lats_out, lons_out, tiers_out, output_path)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ Complete in {elapsed/60:.1f} minutes")
        print(f"  Output: {output_path}")
        print(f"{'='*70}")

        return lats_out, lons_out, tiers_out

    def _compute_cell_vectors(self, x_hrrr, y_hrrr):
        """Compute orientation vectors for HRRR cells"""
        dx_east = np.zeros_like(x_hrrr)
        dy_east = np.zeros_like(x_hrrr)
        dx_north = np.zeros_like(x_hrrr)
        dy_north = np.zeros_like(x_hrrr)

        dx_east[:, :-1] = x_hrrr[:, 1:] - x_hrrr[:, :-1]
        dy_east[:, :-1] = y_hrrr[:, 1:] - y_hrrr[:, :-1]
        dx_east[:, -1] = dx_east[:, -2]
        dy_east[:, -1] = dy_east[:, -2]

        dx_north[:-1, :] = x_hrrr[1:, :] - x_hrrr[:-1, :]
        dy_north[:-1, :] = y_hrrr[1:, :] - y_hrrr[:-1, :]
        dx_north[-1, :] = dx_north[-2, :]
        dy_north[-1, :] = dy_north[-2, :]

        return dx_east, dy_east, dx_north, dy_north

    def _generate_fine_grid(self, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                           dx_east, dy_east, dx_north, dy_north, shape):
        """Generate full fine grid coordinates"""
        # Repeat to fine grid
        x_repeated = np.repeat(np.repeat(x_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north, self.n_fine, axis=0), self.n_fine, axis=1)

        n_fine_i = shape[0] * self.n_fine
        n_fine_j = shape[1] * self.n_fine

        i_fine_idx = np.arange(n_fine_i) % self.n_fine
        j_fine_idx = np.arange(n_fine_j) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        frac_east = jj_fine_idx / self.n_fine - 0.5
        frac_north = ii_fine_idx / self.n_fine - 0.5

        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        # Convert to lat/lon
        lons_fine, lats_fine = self.proj(x_fine, y_fine, inverse=True)

        return x_fine, y_fine, lats_fine, lons_fine

    def _initialize_index_sets(self):
        """Initialize sparse index sets for each mask type"""
        return {
            'coastline_750m': set(),
            'coastline_1500m': set(),
            'coastline_3000m': set(),
            'coastline_6000m': set(),
            'coastline_12000m': set(),
            'lakes_750m': set(),
            'lakes_1500m': set(),
            'lakes_3000m': set(),
            'lakes_6000m': set(),
            'lakes_12000m': set(),
            'ski_resorts': set(),
            'urban_high': set(),
            'urban_suburban': set(),
            'roads': set(),
            'parks': set(),
            'forests': set(),
            'terrain_gt800': set(),
            'terrain_gt600': set(),
            'terrain_gt300': set(),
            'terrain_gt150': set(),
        }

    def _process_terrain(self, terrain, index_sets, n_fine_i, n_fine_j):
        """Process terrain thresholds efficiently"""
        analyzer = TerrainAnalyzer(terrain, self.loader.hrrr_grid['lats'], self.loader.hrrr_grid['lons'])
        terrain_var = analyzer.compute_terrain_variability()

        # Expand to fine grid
        terrain_fine = np.repeat(np.repeat(terrain_var, self.n_fine, axis=0), self.n_fine, axis=1)

        thresholds = [(800, 'terrain_gt800'), (600, 'terrain_gt600'),
                      (300, 'terrain_gt300'), (150, 'terrain_gt150')]

        for thresh, key in thresholds:
            indices = np.where(terrain_fine > thresh)
            index_sets[key] = set(zip(indices[0], indices[1]))
            print(f"  {key}: {len(index_sets[key]):,} indices")

    def _bounds_intersect(self, bounds1, bounds2, margin=0):
        """Check if bounding boxes intersect"""
        return not (bounds1[2] + margin < bounds2[0] or
                    bounds1[0] - margin > bounds2[2] or
                    bounds1[3] + margin < bounds2[1] or
                    bounds1[1] - margin > bounds2[3])

    def _process_features_patched(self, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                                   dx_east, dy_east, dx_north, dy_north,
                                   shape, index_sets):
        """Process features in patches to avoid memory issues"""
        patch_size = 10  # HRRR cells per patch dimension
        n_patches_i = (shape[0] + patch_size - 1) // patch_size
        n_patches_j = (shape[1] + patch_size - 1) // patch_size
        total_patches = n_patches_i * n_patches_j

        print(f"  Patch size: {patch_size}×{patch_size} HRRR cells = {patch_size*self.n_fine}×{patch_size*self.n_fine} fine points")
        print(f"  Total patches: {total_patches}")

        # Pre-buffer all features and build spatial indices
        buffered_features, spatial_indices = self._preproject_features()

        # Process patches in single-core loop (but with pre-buffered geometries for speed)
        print(f"  Processing patches with pre-buffered geometries...")

        for pi in range(n_patches_i):
            for pj in range(n_patches_j):
                patch_idx = pi * n_patches_j + pj
                if patch_idx % 500 == 0:
                    print(f"  Progress: {patch_idx}/{total_patches} patches ({100*patch_idx/total_patches:.1f}%), {sum(len(s) for s in index_sets.values()):,} total indices")

                # Define patch boundaries
                i_start = pi * patch_size
                i_end = min(i_start + patch_size, shape[0])
                j_start = pj * patch_size
                j_end = min(j_start + patch_size, shape[1])

                # Generate fine grid for this patch
                x_patch, y_patch, patch_bounds, i_offset, j_offset = self._generate_patch_grid(
                    x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                    i_start, i_end, j_start, j_end
                )

                # Process features for this patch
                self._process_patch_features(buffered_features, spatial_indices, x_patch, y_patch,
                                             patch_bounds, index_sets, i_offset, j_offset)

        print(f"  Complete: {total_patches}/{total_patches} patches, {sum(len(s) for s in index_sets.values()):,} total indices")

    def _generate_patch_grid(self, x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                            i_start, i_end, j_start, j_end):
        """Generate fine grid for a single patch"""
        # Extract patch from HRRR grid
        x_patch_hrrr = x_hrrr[i_start:i_end, j_start:j_end]
        y_patch_hrrr = y_hrrr[i_start:i_end, j_start:j_end]
        dx_east_patch = dx_east[i_start:i_end, j_start:j_end]
        dy_east_patch = dy_east[i_start:i_end, j_start:j_end]
        dx_north_patch = dx_north[i_start:i_end, j_start:j_end]
        dy_north_patch = dy_north[i_start:i_end, j_start:j_end]

        # Repeat to fine grid
        x_repeated = np.repeat(np.repeat(x_patch_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_patch_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_east_repeated = np.repeat(np.repeat(dx_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        # Fine grid indices within patch
        n_fine_i_patch = (i_end - i_start) * self.n_fine
        n_fine_j_patch = (j_end - j_start) * self.n_fine

        i_fine_idx = np.arange(n_fine_i_patch) % self.n_fine
        j_fine_idx = np.arange(n_fine_j_patch) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        frac_east = jj_fine_idx / self.n_fine - 0.5
        frac_north = ii_fine_idx / self.n_fine - 0.5

        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        # Calculate patch bounds
        patch_bounds = (x_fine.min(), y_fine.min(), x_fine.max(), y_fine.max())

        # Calculate global offsets
        i_offset = i_start * self.n_fine
        j_offset = j_start * self.n_fine

        return x_fine, y_fine, patch_bounds, i_offset, j_offset

    def _preproject_features(self):
        """Pre-project features and PRE-COMPUTE all buffered geometries for speed"""
        print("  Pre-computing buffered geometries (this will take a few minutes but saves hours later)...")

        buffered_features = {}
        spatial_indices = {}

        # Coastlines - buffer at 5 distances
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            coast_proj = significant.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(coast_proj)} coastline segments...")

            for dist, key in [(750, 'coastline_750m'), (1500, 'coastline_1500m'),
                             (3000, 'coastline_3000m'), (6000, 'coastline_6000m'),
                             (12000, 'coastline_12000m')]:
                buffered = [geom.buffer(dist) for geom in coast_proj.geometry]
                buffered_features[key] = buffered
                spatial_indices[key] = STRtree(buffered)

            del coast_proj, significant  # Free memory

        # Lakes - buffer at 5 distances
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(lakes_proj)} lake boundaries...")

            for dist, key in [(750, 'lakes_750m'), (1500, 'lakes_1500m'),
                             (3000, 'lakes_3000m'), (6000, 'lakes_6000m'),
                             (12000, 'lakes_12000m')]:
                buffered = [geom.boundary.buffer(dist) for geom in lakes_proj.geometry]
                buffered_features[key] = buffered
                spatial_indices[key] = STRtree(buffered)

            del lakes_proj  # Free memory

        # Ski resorts - buffer at 2km
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None and len(ski_gdf) > 0:
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(ski_proj)} ski resorts...")

            buffered = [geom.buffer(2000) for geom in ski_proj.geometry]
            buffered_features['ski_resorts'] = buffered
            spatial_indices['ski_resorts'] = STRtree(buffered)

            del ski_proj  # Free memory

        # High-density urban - no buffer
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            print(f"    Processing {len(urban_proj)} high-density urban boundaries...")

            boundaries = [geom.boundary for geom in urban_proj.geometry]
            buffered_features['urban_high'] = boundaries
            spatial_indices['urban_high'] = STRtree(boundaries)

            del urban_proj  # Free memory

        # Suburban - no buffer
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            suburban_proj = urban_low.to_crs(self.hrrr_crs)
            print(f"    Processing {len(suburban_proj)} suburban boundaries...")

            boundaries = [geom.boundary for geom in suburban_proj.geometry]
            buffered_features['urban_suburban'] = boundaries
            spatial_indices['urban_suburban'] = STRtree(boundaries)

            del suburban_proj  # Free memory

        # Roads - buffer at 500m
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(roads_proj)} road segments...")

            buffered = [geom.buffer(500) for geom in roads_proj.geometry]
            buffered_features['roads'] = buffered
            spatial_indices['roads'] = STRtree(buffered)

            del roads_proj  # Free memory

        # Parks - no buffer
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None and len(parks_gdf) > 0:
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(parks_proj)} national parks...")

            geoms = list(parks_proj.geometry)
            buffered_features['parks'] = geoms
            spatial_indices['parks'] = STRtree(geoms)

            del parks_proj  # Free memory

        # Forests - no buffer
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(forests_proj)} national forests...")

            geoms = list(forests_proj.geometry)
            buffered_features['forests'] = geoms
            spatial_indices['forests'] = STRtree(geoms)

            del forests_proj  # Free memory

        print("  ✓ All geometries buffered and indexed")
        return buffered_features, spatial_indices

    def _process_patch_features(self, buffered_features, spatial_indices, x_patch, y_patch,
                                 patch_bounds, index_sets, i_offset, j_offset):
        """Process all feature types within a patch using pre-buffered geometries"""
        # Create a box for the patch bounds with maximum buffer margin
        max_margin = 12000  # Maximum buffer distance
        patch_box = box(patch_bounds[0] - max_margin, patch_bounds[1] - max_margin,
                       patch_bounds[2] + max_margin, patch_bounds[3] + max_margin)

        # Process coastlines (5 buffer distances)
        for key in ['coastline_750m', 'coastline_1500m', 'coastline_3000m',
                    'coastline_6000m', 'coastline_12000m']:
            if key in spatial_indices:
                nearby_indices = spatial_indices[key].query(patch_box)
                for idx in nearby_indices:
                    buffered_geom = buffered_features[key][idx]
                    self._process_buffered_feature(buffered_geom, key, x_patch, y_patch,
                                                   patch_bounds, index_sets, i_offset, j_offset)

        # Process lakes (5 buffer distances)
        for key in ['lakes_750m', 'lakes_1500m', 'lakes_3000m',
                    'lakes_6000m', 'lakes_12000m']:
            if key in spatial_indices:
                nearby_indices = spatial_indices[key].query(patch_box)
                for idx in nearby_indices:
                    buffered_geom = buffered_features[key][idx]
                    self._process_buffered_feature(buffered_geom, key, x_patch, y_patch,
                                                   patch_bounds, index_sets, i_offset, j_offset)

        # Process ski resorts
        if 'ski_resorts' in spatial_indices:
            nearby_indices = spatial_indices['ski_resorts'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['ski_resorts'][idx]
                self._process_buffered_feature(buffered_geom, 'ski_resorts', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

        # Process high-density urban
        if 'urban_high' in spatial_indices:
            nearby_indices = spatial_indices['urban_high'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['urban_high'][idx]
                self._process_buffered_feature(buffered_geom, 'urban_high', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

        # Process suburban
        if 'urban_suburban' in spatial_indices:
            nearby_indices = spatial_indices['urban_suburban'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['urban_suburban'][idx]
                self._process_buffered_feature(buffered_geom, 'urban_suburban', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

        # Process roads
        if 'roads' in spatial_indices:
            nearby_indices = spatial_indices['roads'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['roads'][idx]
                self._process_buffered_feature(buffered_geom, 'roads', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

        # Process parks
        if 'parks' in spatial_indices:
            nearby_indices = spatial_indices['parks'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['parks'][idx]
                self._process_buffered_feature(buffered_geom, 'parks', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

        # Process forests
        if 'forests' in spatial_indices:
            nearby_indices = spatial_indices['forests'].query(patch_box)
            for idx in nearby_indices:
                buffered_geom = buffered_features['forests'][idx]
                self._process_buffered_feature(buffered_geom, 'forests', x_patch, y_patch,
                                               patch_bounds, index_sets, i_offset, j_offset)

    def _process_buffered_feature(self, buffered_geom, mask_key, x_patch, y_patch,
                                   patch_bounds, index_sets, i_offset, j_offset):
        """Process a single PRE-BUFFERED feature using hierarchical multigrid approach"""
        # Get feature bounds
        feature_bounds = buffered_geom.bounds

        # Check if feature intersects patch
        if not self._bounds_intersect(feature_bounds, patch_bounds, margin=0):
            return  # Feature doesn't overlap patch

        # HIERARCHICAL APPROACH: Check coarse grid first
        coarse_stride = 5  # Check every 5th point (320×320 → 64×64 coarse grid)

        # Extract coarse grid points
        x_coarse = x_patch[::coarse_stride, ::coarse_stride]
        y_coarse = y_patch[::coarse_stride, ::coarse_stride]

        # Check coarse points using contains_xy (faster than vectorized.contains)
        if HAS_CONTAINS_XY:
            coarse_inside = contains_xy(buffered_geom, x_coarse.ravel(), y_coarse.ravel())
        else:
            coarse_inside = vectorized.contains(buffered_geom, x_coarse.ravel(), y_coarse.ravel())

        # Early exit: if NO coarse points are inside, skip this entire feature
        if not coarse_inside.any():
            return

        # Identify which coarse cells have hits
        coarse_inside_2d = coarse_inside.reshape(x_coarse.shape)

        # Expand to identify fine-grid regions to check
        fine_mask = np.zeros(x_patch.shape, dtype=bool)

        for ci in range(coarse_inside_2d.shape[0]):
            for cj in range(coarse_inside_2d.shape[1]):
                if coarse_inside_2d[ci, cj]:
                    # Mark this coarse cell and neighbors for fine checking
                    i_start = max(0, ci * coarse_stride - coarse_stride)
                    i_end = min(x_patch.shape[0], (ci + 1) * coarse_stride + coarse_stride)
                    j_start = max(0, cj * coarse_stride - coarse_stride)
                    j_end = min(x_patch.shape[1], (cj + 1) * coarse_stride + coarse_stride)

                    fine_mask[i_start:i_end, j_start:j_end] = True

        # Extract fine points to check
        x_fine_check = x_patch[fine_mask]
        y_fine_check = y_patch[fine_mask]

        if len(x_fine_check) == 0:
            return

        # Check fine points using contains_xy
        if HAS_CONTAINS_XY:
            fine_inside = contains_xy(buffered_geom, x_fine_check, y_fine_check)
        else:
            fine_inside = vectorized.contains(buffered_geom, x_fine_check, y_fine_check)

        # Get indices of points that are inside
        fine_indices_2d = np.where(fine_mask)
        i_indices_local = fine_indices_2d[0][fine_inside]
        j_indices_local = fine_indices_2d[1][fine_inside]

        # Convert to global indices
        i_indices_global = i_indices_local + i_offset
        j_indices_global = j_indices_local + j_offset

        # Add to index set
        index_sets[mask_key].update(zip(i_indices_global, j_indices_global))

    def _process_features_sparse(self, x_fine, y_fine, index_sets):
        """Process features individually with sparse index sets"""
        x_flat = x_fine.ravel()
        y_flat = y_fine.ravel()

        # Domain bounds
        domain_bounds = (x_fine.min(), y_fine.min(), x_fine.max(), y_fine.max())

        # Coastlines
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            coast_proj = significant.to_crs(self.hrrr_crs)
            print(f"  Processing {len(coast_proj)} coastline segments...")

            for idx, geom in enumerate(coast_proj.geometry):
                if idx % 100 == 0:
                    print(f"    Coastline {idx}/{len(coast_proj)}...")
                self._process_feature(geom, [750, 1500, 3000, 6000, 12000],
                                    ['coastline_750m', 'coastline_1500m', 'coastline_3000m',
                                     'coastline_6000m', 'coastline_12000m'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Lakes
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            print(f"  Processing {len(lakes_proj)} lakes...")

            for idx, geom in enumerate(lakes_proj.geometry):
                if idx % 50 == 0:
                    print(f"    Lake {idx}/{len(lakes_proj)}...")
                self._process_feature(geom.boundary, [750, 1500, 3000, 6000, 12000],
                                    ['lakes_750m', 'lakes_1500m', 'lakes_3000m',
                                     'lakes_6000m', 'lakes_12000m'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Ski resorts
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None and len(ski_gdf) > 0:
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            print(f"  Processing {len(ski_proj)} ski resorts...")

            for geom in ski_proj.geometry:
                self._process_feature(geom, [2000], ['ski_resorts'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # High-density urban
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            print(f"  Processing {len(urban_proj)} high-density urban areas...")

            for idx, geom in enumerate(urban_proj.geometry):
                if idx % 50 == 0:
                    print(f"    Urban {idx}/{len(urban_proj)}...")
                self._process_feature(geom.boundary, [0], ['urban_high'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Suburban
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            cluster_proj = urban_low.to_crs(self.hrrr_crs)
            print(f"  Processing {len(cluster_proj)} suburban areas...")

            for idx, geom in enumerate(cluster_proj.geometry):
                if idx % 300 == 0:
                    print(f"    Suburban {idx}/{len(cluster_proj)}...")
                self._process_feature(geom.boundary, [0], ['urban_suburban'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Roads
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            print(f"  Processing {len(roads_proj)} road segments...")

            for idx, geom in enumerate(roads_proj.geometry):
                if idx % 1000 == 0:
                    print(f"    Road {idx}/{len(roads_proj)}...")
                self._process_feature(geom, [500], ['roads'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Parks
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None and len(parks_gdf) > 0:
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            print(f"  Processing {len(parks_proj)} national parks...")

            for geom in parks_proj.geometry:
                self._process_feature(geom, [0], ['parks'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

        # Forests
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            print(f"  Processing {len(forests_proj)} national forests...")

            for geom in forests_proj.geometry:
                self._process_feature(geom, [0], ['forests'],
                                    x_fine, y_fine, x_flat, y_flat, index_sets, domain_bounds)

    def _process_feature(self, geom, buffer_dists, mask_keys, x_fine, y_fine,
                        x_flat, y_flat, index_sets, domain_bounds):
        """Process a single feature geometry"""
        # Get feature bounds
        feature_bounds = geom.bounds

        # Check if feature intersects domain (with max buffer margin)
        max_buffer = max(buffer_dists)
        if not self._bounds_intersect(feature_bounds, domain_bounds, margin=max_buffer):
            return  # Feature too far from domain

        # Find indices within expanded bounding box
        margin = max_buffer * 1.1  # 10% safety margin
        in_bbox = (
            (x_flat >= feature_bounds[0] - margin) & (x_flat <= feature_bounds[2] + margin) &
            (y_flat >= feature_bounds[1] - margin) & (y_flat <= feature_bounds[3] + margin)
        )

        if not in_bbox.any():
            return  # No points near this feature

        # Get candidate points
        x_candidates = x_flat[in_bbox]
        y_candidates = y_flat[in_bbox]
        candidate_indices = np.where(in_bbox)[0]

        # Process each buffer distance
        for buffer_dist, mask_key in zip(buffer_dists, mask_keys):
            if buffer_dist > 0:
                buffered = geom.buffer(buffer_dist)
            else:
                buffered = geom

            # Check which candidates are inside
            try:
                inside = vectorized.contains(buffered, x_candidates, y_candidates)
            except:
                prepared = prep(buffered)
                inside = np.array([prepared.contains(Point(x, y))
                                 for x, y in zip(x_candidates, y_candidates)])

            # Convert flat indices to (i, j)
            flat_indices = candidate_indices[inside]
            i_indices = flat_indices // x_fine.shape[1]
            j_indices = flat_indices % x_fine.shape[1]

            # Add to index set
            index_sets[mask_key].update(zip(i_indices, j_indices))

    def _apply_tier_logic(self, index_sets, n_fine_i, n_fine_j):
        """Apply tier priority and stride patterns, returning indices with tier assignments"""
        # Tier 0: Ski resorts (stride=1, 93.75m)
        tier0_indices = index_sets['ski_resorts']

        # Tier 1: Coastlines/lakes 750m OR extreme terrain (stride=2, 187.5m)
        tier1_full = (index_sets['coastline_750m'] | index_sets['lakes_750m'] |
                     index_sets['terrain_gt800'])
        tier1_indices = tier1_full - tier0_indices

        # Tier 2: Coastlines/lakes 1500m OR high-density urban OR very rugged (stride=4, 375m)
        tier2_full = (index_sets['coastline_1500m'] | index_sets['lakes_1500m'] |
                     index_sets['urban_high'] | index_sets['terrain_gt600'])
        tier2_indices = tier2_full - tier1_full - tier0_indices

        # Tier 3: Coastlines/lakes 3km OR suburban OR roads OR parks OR forests OR rugged (stride=8, 750m)
        tier3_full = (index_sets['coastline_3000m'] | index_sets['lakes_3000m'] |
                     index_sets['urban_suburban'] | index_sets['roads'] |
                     index_sets['parks'] | index_sets['forests'] | index_sets['terrain_gt300'])
        tier3_indices = tier3_full - tier2_full - tier1_full - tier0_indices

        # Tier 4: Coastlines/lakes 6km OR moderate terrain (stride=16, 1.5km)
        tier4_full = (index_sets['coastline_6000m'] | index_sets['lakes_6000m'] |
                     index_sets['terrain_gt150'])
        tier4_indices = tier4_full - tier3_full - tier2_full - tier1_full - tier0_indices

        print(f"  Tier 0 (93.75m): {len(tier0_indices):,} indices")
        print(f"  Tier 1 (187.5m): {len(tier1_indices):,} indices")
        print(f"  Tier 2 (375m): {len(tier2_indices):,} indices")
        print(f"  Tier 3 (750m): {len(tier3_indices):,} indices")
        print(f"  Tier 4 (1.5km): {len(tier4_indices):,} indices")

        # Apply stride patterns and track tier assignments
        print("  Applying stride decimation...")
        index_tier_map = {}  # Maps (i,j) -> tier number

        # Tier 0: stride=1 (keep all)
        print(f"    Processing Tier 0 (stride=1)...")
        for idx in tier0_indices:
            index_tier_map[idx] = 0

        # Tier 1: stride=2
        print(f"    Processing Tier 1 (stride=2)...")
        for i, j in tier1_indices:
            if (i & 1) == 0 and (j & 1) == 0:
                index_tier_map[(i, j)] = 1

        # Tier 2: stride=4
        print(f"    Processing Tier 2 (stride=4)...")
        for i, j in tier2_indices:
            if (i & 3) == 0 and (j & 3) == 0:
                index_tier_map[(i, j)] = 2

        # Tier 3: stride=8
        print(f"    Processing Tier 3 (stride=8)...")
        for i, j in tier3_indices:
            if (i & 7) == 0 and (j & 7) == 0:
                index_tier_map[(i, j)] = 3

        # Tier 4: stride=16
        print(f"    Processing Tier 4 (stride=16)...")
        for i, j in tier4_indices:
            if (i & 15) == 0 and (j & 15) == 0:
                index_tier_map[(i, j)] = 4

        # Tier 5: Background grid (stride=32, 3km) - generate directly with stride pattern
        print("  Generating Tier 5 background grid...")
        tier5_count = 0
        for i in range(0, n_fine_i, 32):
            for j in range(0, n_fine_j, 32):
                if (i, j) not in index_tier_map:
                    index_tier_map[(i, j)] = 5
                    tier5_count += 1

        print(f"  Tier 5 (3km): {tier5_count:,} background points")
        print(f"  Final indices after stride: {len(index_tier_map):,}")

        return index_tier_map

    def _indices_to_latlon(self, index_tier_map, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                           dx_east, dy_east, dx_north, dy_north):
        """Convert (i,j) indices to lat/lon coordinates by reconstructing from HRRR grid"""
        lats = []
        lons = []
        tiers = []

        for (i, j), tier in index_tier_map.items():
            # Find parent HRRR cell
            i_hrrr = i // self.n_fine
            j_hrrr = j // self.n_fine

            # Find position within cell
            i_local = i % self.n_fine
            j_local = j % self.n_fine

            # Get fractions within cell (-0.5 to +0.5)
            frac_east = (j_local / self.n_fine) - 0.5
            frac_north = (i_local / self.n_fine) - 0.5

            # Reconstruct projected coordinates
            x = x_hrrr[i_hrrr, j_hrrr] + frac_east * dx_east[i_hrrr, j_hrrr] + frac_north * dx_north[i_hrrr, j_hrrr]
            y = y_hrrr[i_hrrr, j_hrrr] + frac_east * dy_east[i_hrrr, j_hrrr] + frac_north * dy_north[i_hrrr, j_hrrr]

            # Convert to lat/lon
            lon, lat = self.proj(x, y, inverse=True)

            lats.append(lat)
            lons.append(lon)
            tiers.append(tier)

        return np.array(lats), np.array(lons), np.array(tiers, dtype=np.int8)

    def _write_output(self, lats, lons, tiers, output_path):
        """Write output NetCDF file"""
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))
        tier_var = ds.createVariable('tier', 'i1', ('points',))

        lat_var.units = 'degrees_north'
        lat_var.long_name = 'Latitude'
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'Longitude'
        tier_var.long_name = 'Resolution tier (0=93.75m, 1=187.5m, 2=375m, 3=750m, 4=1.5km, 5=3km)'

        lat_var[:] = lats
        lon_var[:] = lons
        tier_var[:] = tiers

        ds.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate adaptive grid with sparse index approach')
    parser.add_argument('--test-region', choices=['nw_us', 'full'], default='full',
                       help='Region to process')

    args = parser.parse_args()

    test_region = args.test_region if args.test_region != 'full' else None

    generator = SparseGridGenerator(test_region=test_region)
    lats, lons = generator.generate()
