"""
STAGE 1: Generate Boolean Masks for Features (PATCH-BASED)
============================================================

Generate and save individual boolean masks at 93.75m resolution using
a memory-efficient patch-based approach.

Features:
- Coastline buffers (5 distance tiers)
- Lake buffers (5 distance tiers)
- Ski resorts (2km buffer)
- High-density urban areas
- Suburban areas (low-density urban)
- Major roads (500m buffer)
- National parks (full areas)
- National forests (full areas)
- Terrain variability (4 thresholds)

Each mask is saved as a compressed numpy array for fast loading.
"""

import numpy as np
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point
from shapely.prepared import prep
from shapely import vectorized
import time
import os
from generate_adaptive_grid import DataLoader, TerrainAnalyzer
import config

class MaskGeneratorPatched:
    def __init__(self):
        self.loader = DataLoader()
        self.n_fine = 32  # points per HRRR cell dimension
        self.patch_size = 10  # HRRR cells per patch
        self.output_dir = 'output/masks'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_masks(self):
        """Generate all feature masks using patch-based approach"""
        print("="*70)
        print(" STAGE 1: GENERATING BOOLEAN MASKS (PATCH-BASED)")
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

        # Create fine grid dimensions
        n_fine_i = shape[0] * self.n_fine
        n_fine_j = shape[1] * self.n_fine
        print(f"Fine grid: {n_fine_i}×{n_fine_j} = {n_fine_i*n_fine_j:,} points")
        print(f"Memory per mask: ~{n_fine_i*n_fine_j/8/1024/1024:.1f} MB (boolean)")
        print(f"Patch size: {self.patch_size}×{self.patch_size} HRRR cells")

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

        # Precompute HRRR cell geometry vectors
        print("Precomputing cell geometry...")
        dx_east, dy_east, dx_north, dy_north = self._compute_cell_vectors(x_hrrr, y_hrrr)

        # Prepare feature geometries
        print("\nPreparing feature geometries...")
        feature_geoms, feature_bounds = self._prepare_feature_geometries()

        # TERRAIN MASKS
        print("\n" + "="*70)
        print("GENERATING TERRAIN MASKS")
        print("="*70)
        self._generate_terrain_masks(terrain, n_fine_i, n_fine_j)

        # FEATURE MASKS (patch-based)
        print("\n" + "="*70)
        print("GENERATING FEATURE MASKS (PATCH-BASED)")
        print("="*70)

        # Initialize global masks
        global_masks = self._initialize_global_masks(feature_geoms, n_fine_i, n_fine_j)

        # Store feature bounds for spatial filtering
        self.feature_bounds = feature_bounds

        # Process in patches
        n_patches_i = int(np.ceil(shape[0] / self.patch_size))
        n_patches_j = int(np.ceil(shape[1] / self.patch_size))
        total_patches = n_patches_i * n_patches_j
        print(f"Total patches: {total_patches}")

        patch_count = 0
        for i_patch in range(n_patches_i):
            for j_patch in range(n_patches_j):
                i_start = i_patch * self.patch_size
                i_end = min(i_start + self.patch_size, shape[0])
                j_start = j_patch * self.patch_size
                j_end = min(j_start + self.patch_size, shape[1])

                # Process patch
                self._process_patch(
                    i_start, i_end, j_start, j_end,
                    x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                    feature_geoms, global_masks
                )

                patch_count += 1
                if patch_count % 100 == 0:
                    pct = 100 * patch_count / total_patches
                    print(f"  Progress: {patch_count}/{total_patches} patches ({pct:.1f}%)")

        # Save all masks
        print("\nSaving masks...")
        self._save_masks(global_masks)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ All masks generated in {elapsed/60:.1f} minutes")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}")

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

    def _prepare_feature_geometries(self):
        """Prepare all feature geometries with buffers and bounds"""
        feature_geoms = {}
        feature_bounds = {}

        # Coastlines
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant_coastlines = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            print(f"  Coastlines: {len(significant_coastlines)} segments")
            coast_proj = significant_coastlines.to_crs(self.hrrr_crs)
            geom_union = coast_proj.geometry.union_all()
            for dist in [750, 1500, 3000, 6000, 12000]:
                buffered = geom_union.buffer(dist).simplify(1.0)
                feature_geoms[f'coastline_{dist}m'] = buffered
                feature_bounds[f'coastline_{dist}m'] = buffered.bounds

        # Lakes
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            print(f"  Lakes: {len(lakes_gdf)} features")
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            geom_union = lakes_proj.geometry.boundary.union_all()
            for dist in [750, 1500, 3000, 6000, 12000]:
                buffered = geom_union.buffer(dist).simplify(1.0)
                feature_geoms[f'lakes_{dist}m'] = buffered
                feature_bounds[f'lakes_{dist}m'] = buffered.bounds

        # Ski resorts
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None and len(ski_gdf) > 0:
            print(f"  Ski resorts: {len(ski_gdf)} locations")
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            buffered = ski_proj.geometry.buffer(2000).union_all()
            feature_geoms['ski_resorts'] = buffered
            feature_bounds['ski_resorts'] = buffered.bounds

        # High-density urban
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            print(f"  High-density urban: {len(urban_high)} areas")
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            geom_union = urban_proj.geometry.boundary.union_all()
            feature_geoms['urban_high'] = geom_union
            feature_bounds['urban_high'] = geom_union.bounds

        # Suburban
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            print(f"  Suburban: {len(urban_low)} areas")
            cluster_proj = urban_low.to_crs(self.hrrr_crs)
            geom_union = cluster_proj.geometry.boundary.union_all()
            feature_geoms['urban_suburban'] = geom_union
            feature_bounds['urban_suburban'] = geom_union.bounds

        # Roads
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            print(f"  Major roads: {len(roads_gdf)} segments")
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            buffered = roads_proj.geometry.buffer(500).union_all().simplify(1.0)
            feature_geoms['roads'] = buffered
            feature_bounds['roads'] = buffered.bounds

        # National parks
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None and len(parks_gdf) > 0:
            print(f"  National parks: {len(parks_gdf)} units")
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            geom_union = parks_proj.geometry.union_all().simplify(1.0)
            feature_geoms['parks'] = geom_union
            feature_bounds['parks'] = geom_union.bounds

        # National forests
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            print(f"  National forests: {len(forests_gdf)} units")
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            geom_union = forests_proj.geometry.union_all().simplify(1.0)
            feature_geoms['forests'] = geom_union
            feature_bounds['forests'] = geom_union.bounds

        return feature_geoms, feature_bounds

    def _initialize_global_masks(self, feature_geoms, n_fine_i, n_fine_j):
        """Initialize global boolean masks"""
        global_masks = {}
        for feature_name in feature_geoms.keys():
            global_masks[feature_name] = np.zeros((n_fine_i, n_fine_j), dtype=bool)
        return global_masks

    def _bounds_intersect(self, bounds1, bounds2):
        """Check if two bounding boxes intersect

        bounds format: (minx, miny, maxx, maxy)
        """
        return not (bounds1[2] < bounds2[0] or  # bounds1 is left of bounds2
                    bounds1[0] > bounds2[2] or  # bounds1 is right of bounds2
                    bounds1[3] < bounds2[1] or  # bounds1 is below bounds2
                    bounds1[1] > bounds2[3])    # bounds1 is above bounds2

    def _process_patch(self, i_start, i_end, j_start, j_end,
                      x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                      feature_geoms, global_masks):
        """Process a single patch and update global masks"""
        # Extract patch data
        x_patch = x_hrrr[i_start:i_end, j_start:j_end]
        y_patch = y_hrrr[i_start:i_end, j_start:j_end]

        dx_east_patch = dx_east[i_start:i_end, j_start:j_end]
        dy_east_patch = dy_east[i_start:i_end, j_start:j_end]
        dx_north_patch = dx_north[i_start:i_end, j_start:j_end]
        dy_north_patch = dy_north[i_start:i_end, j_start:j_end]

        # Generate fine grid for patch
        x_fine, y_fine = self._generate_patch_fine_grid(
            x_patch, y_patch, dx_east_patch, dy_east_patch,
            dx_north_patch, dy_north_patch
        )

        # Compute patch bounds for spatial filtering
        patch_bounds = (x_fine.min(), y_fine.min(), x_fine.max(), y_fine.max())

        # Flatten for geometric operations
        x_flat = x_fine.ravel()
        y_flat = y_fine.ravel()

        # Check each feature (with spatial filtering)
        for feature_name, geometry in feature_geoms.items():
            # SPATIAL FILTER: Only check if feature bounds intersect patch bounds
            feature_bound = self.feature_bounds[feature_name]
            if not self._bounds_intersect(patch_bounds, feature_bound):
                # Skip this feature - it doesn't overlap with this patch
                continue

            # Feature overlaps - do geometric check
            mask_patch = self._geometric_contains_flat(geometry, x_flat, y_flat, x_fine.shape)

            # Update global mask
            i_fine_start = i_start * self.n_fine
            i_fine_end = i_end * self.n_fine
            j_fine_start = j_start * self.n_fine
            j_fine_end = j_end * self.n_fine

            global_masks[feature_name][i_fine_start:i_fine_end, j_fine_start:j_fine_end] |= mask_patch

    def _generate_patch_fine_grid(self, x_patch, y_patch, dx_east_patch, dy_east_patch,
                                 dx_north_patch, dy_north_patch):
        """Generate fine grid for a patch"""
        # Repeat to fine grid
        x_repeated = np.repeat(np.repeat(x_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        n_fine_i, n_fine_j = x_repeated.shape

        # Fine grid indices
        i_fine_idx = np.arange(n_fine_i) % self.n_fine
        j_fine_idx = np.arange(n_fine_j) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        # Fractional offsets
        frac_east = jj_fine_idx / self.n_fine - 0.5
        frac_north = ii_fine_idx / self.n_fine - 0.5

        # Apply offsets
        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        return x_fine, y_fine

    def _geometric_contains_flat(self, geometry, x_flat, y_flat, shape):
        """Check which points are contained in a geometry"""
        # Pre-filter with bounding box
        bounds = geometry.bounds
        in_bounds = (
            (x_flat >= bounds[0]) & (x_flat <= bounds[2]) &
            (y_flat >= bounds[1]) & (y_flat <= bounds[3])
        )

        mask_flat = np.zeros(len(x_flat), dtype=bool)

        if in_bounds.any():
            x_check = x_flat[in_bounds]
            y_check = y_flat[in_bounds]

            # Use vectorized contains
            try:
                mask_check = vectorized.contains(geometry, x_check, y_check)
            except:
                # Fallback
                prepared_geom = prep(geometry)
                mask_check = np.array([prepared_geom.contains(Point(x, y))
                                     for x, y in zip(x_check, y_check)])

            mask_flat[in_bounds] = mask_check

        return mask_flat.reshape(shape)

    def _generate_terrain_masks(self, terrain, n_fine_i, n_fine_j):
        """Generate terrain variability masks"""
        print("\nComputing terrain variability...")
        analyzer = TerrainAnalyzer(terrain, self.loader.hrrr_grid['lats'], self.loader.hrrr_grid['lons'])
        terrain_var = analyzer.compute_terrain_variability()

        # Expand to fine grid
        terrain_fine = np.repeat(np.repeat(terrain_var, self.n_fine, axis=0), self.n_fine, axis=1)

        thresholds = [800, 600, 300, 150]
        for thresh in thresholds:
            mask = terrain_fine > thresh
            filename = f'{self.output_dir}/terrain_gt{thresh}m.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"  Terrain >{thresh}m: {count:,} points ({pct:.2f}%) → {filename}")

    def _save_masks(self, global_masks):
        """Save all global masks"""
        for feature_name, mask in global_masks.items():
            filename = f'{self.output_dir}/{feature_name}.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"  {feature_name}: {count:,} points ({pct:.2f}%) → {filename}")


if __name__ == '__main__':
    generator = MaskGeneratorPatched()
    generator.generate_all_masks()
