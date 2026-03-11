"""
STAGE 1: Generate Boolean Masks - MEMORY EFFICIENT
===================================================

Uses on-the-fly buffering and spatial filtering to avoid creating
massive unified geometries that don't fit in memory.

Strategy:
1. Keep features as individual geometries (no union)
2. Use spatial index (bounds) to find features near each patch
3. Buffer only nearby features on-the-fly for each patch
4. Much lower memory footprint
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

class MaskGeneratorEfficient:
    def __init__(self):
        self.loader = DataLoader()
        self.n_fine = 32
        self.patch_size = 10
        self.output_dir = 'output/masks'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_masks(self):
        """Generate all feature masks using memory-efficient approach"""
        print("="*70)
        print(" STAGE 1: GENERATING BOOLEAN MASKS (MEMORY EFFICIENT)")
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

        # Precompute cell geometry
        print("Precomputing cell geometry...")
        dx_east, dy_east, dx_north, dy_north = self._compute_cell_vectors(x_hrrr, y_hrrr)

        # Prepare feature lists (no union!)
        print("\nPreparing feature spatial index...")
        feature_lists = self._prepare_feature_lists()

        # TERRAIN MASKS
        print("\n" + "="*70)
        print("GENERATING TERRAIN MASKS")
        print("="*70)
        self._generate_terrain_masks(terrain, n_fine_i, n_fine_j)

        # FEATURE MASKS (patch-based with on-the-fly buffering)
        print("\n" + "="*70)
        print("GENERATING FEATURE MASKS (PATCH-BASED)")
        print("="*70)

        # Initialize global masks
        mask_configs = self._get_mask_configs()
        global_masks = {}
        for mask_name in mask_configs.keys():
            global_masks[mask_name] = np.zeros((n_fine_i, n_fine_j), dtype=bool)

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

                self._process_patch(
                    i_start, i_end, j_start, j_end,
                    x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                    feature_lists, mask_configs, global_masks
                )

                patch_count += 1
                if patch_count % 500 == 0:
                    pct = 100 * patch_count / total_patches
                    print(f"  Progress: {patch_count}/{total_patches} patches ({pct:.1f}%)")

        # Save masks
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

    def _prepare_feature_lists(self):
        """Prepare lists of individual features with bounds (no union!)"""
        feature_lists = {}

        # Coastlines
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            coast_proj = significant.to_crs(self.hrrr_crs)
            feature_lists['coastline'] = [(geom, geom.bounds) for geom in coast_proj.geometry]
            print(f"  Coastlines: {len(feature_lists['coastline'])} segments")

        # Lakes
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            feature_lists['lakes'] = [(geom.boundary, geom.boundary.bounds) for geom in lakes_proj.geometry]
            print(f"  Lakes: {len(feature_lists['lakes'])} features")

        # Ski resorts
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None and len(ski_gdf) > 0:
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            feature_lists['ski_resorts'] = [(geom, geom.bounds) for geom in ski_proj.geometry]
            print(f"  Ski resorts: {len(feature_lists['ski_resorts'])} locations")

        # High-density urban
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            feature_lists['urban_high'] = [(geom.boundary, geom.boundary.bounds) for geom in urban_proj.geometry]
            print(f"  High-density urban: {len(feature_lists['urban_high'])} areas")

        # Suburban
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            cluster_proj = urban_low.to_crs(self.hrrr_crs)
            feature_lists['urban_suburban'] = [(geom.boundary, geom.boundary.bounds) for geom in cluster_proj.geometry]
            print(f"  Suburban: {len(feature_lists['urban_suburban'])} areas")

        # Roads
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            feature_lists['roads'] = [(geom, geom.bounds) for geom in roads_proj.geometry]
            print(f"  Major roads: {len(feature_lists['roads'])} segments")

        # National parks
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None and len(parks_gdf) > 0:
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            feature_lists['parks'] = [(geom, geom.bounds) for geom in parks_proj.geometry]
            print(f"  National parks: {len(feature_lists['parks'])} units")

        # National forests
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            feature_lists['forests'] = [(geom, geom.bounds) for geom in forests_proj.geometry]
            print(f"  National forests: {len(feature_lists['forests'])} units")

        return feature_lists

    def _get_mask_configs(self):
        """Define which features go into which masks with buffer distances"""
        return {
            'coastline_750m': ('coastline', 750),
            'coastline_1500m': ('coastline', 1500),
            'coastline_3000m': ('coastline', 3000),
            'coastline_6000m': ('coastline', 6000),
            'coastline_12000m': ('coastline', 12000),
            'lakes_750m': ('lakes', 750),
            'lakes_1500m': ('lakes', 1500),
            'lakes_3000m': ('lakes', 3000),
            'lakes_6000m': ('lakes', 6000),
            'lakes_12000m': ('lakes', 12000),
            'ski_resorts': ('ski_resorts', 2000),
            'urban_high': ('urban_high', 0),
            'urban_suburban': ('urban_suburban', 0),
            'roads': ('roads', 500),
            'parks': ('parks', 0),
            'forests': ('forests', 0),
        }

    def _bounds_intersect(self, bounds1, bounds2, margin=0):
        """Check if two bounding boxes intersect (with margin)"""
        return not (bounds1[2] + margin < bounds2[0] or
                    bounds1[0] - margin > bounds2[2] or
                    bounds1[3] + margin < bounds2[1] or
                    bounds1[1] - margin > bounds2[3])

    def _process_patch(self, i_start, i_end, j_start, j_end,
                      x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                      feature_lists, mask_configs, global_masks):
        """Process a single patch"""
        # Extract patch data
        x_patch = x_hrrr[i_start:i_end, j_start:j_end]
        y_patch = y_hrrr[i_start:i_end, j_start:j_end]

        dx_east_patch = dx_east[i_start:i_end, j_start:j_end]
        dy_east_patch = dy_east[i_start:i_end, j_start:j_end]
        dx_north_patch = dx_north[i_start:i_end, j_start:j_end]
        dy_north_patch = dy_north[i_start:i_end, j_start:j_end]

        # Generate fine grid
        x_fine, y_fine = self._generate_patch_fine_grid(
            x_patch, y_patch, dx_east_patch, dy_east_patch,
            dx_north_patch, dy_north_patch
        )

        # Patch bounds
        patch_bounds = (x_fine.min(), y_fine.min(), x_fine.max(), y_fine.max())
        x_flat = x_fine.ravel()
        y_flat = y_fine.ravel()

        # Process each mask configuration
        for mask_name, (feature_key, buffer_dist) in mask_configs.items():
            if feature_key not in feature_lists:
                continue

            # Find nearby features using spatial filter
            nearby_features = []
            for geom, bounds in feature_lists[feature_key]:
                # Check if feature bounds (+ buffer) intersect patch
                if self._bounds_intersect(bounds, patch_bounds, margin=buffer_dist):
                    nearby_features.append(geom)

            if not nearby_features:
                continue  # No features nearby

            # Buffer nearby features and combine
            if buffer_dist > 0:
                buffered = [f.buffer(buffer_dist) for f in nearby_features]
            else:
                buffered = nearby_features

            # Union nearby features (much smaller than full CONUS!)
            if len(buffered) == 1:
                combined = buffered[0]
            else:
                from shapely.ops import unary_union
                combined = unary_union(buffered)

            # Check patch points against combined geometry
            mask_patch = self._geometric_contains_flat(combined, x_flat, y_flat, x_fine.shape)

            # Update global mask
            i_fine_start = i_start * self.n_fine
            i_fine_end = i_end * self.n_fine
            j_fine_start = j_start * self.n_fine
            j_fine_end = j_end * self.n_fine

            global_masks[mask_name][i_fine_start:i_fine_end, j_fine_start:j_fine_end] |= mask_patch

    def _generate_patch_fine_grid(self, x_patch, y_patch, dx_east_patch, dy_east_patch,
                                 dx_north_patch, dy_north_patch):
        """Generate fine grid for a patch"""
        x_repeated = np.repeat(np.repeat(x_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        n_fine_i, n_fine_j = x_repeated.shape

        i_fine_idx = np.arange(n_fine_i) % self.n_fine
        j_fine_idx = np.arange(n_fine_j) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        frac_east = jj_fine_idx / self.n_fine - 0.5
        frac_north = ii_fine_idx / self.n_fine - 0.5

        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        return x_fine, y_fine

    def _geometric_contains_flat(self, geometry, x_flat, y_flat, shape):
        """Check which points are contained in a geometry"""
        bounds = geometry.bounds
        in_bounds = (
            (x_flat >= bounds[0]) & (x_flat <= bounds[2]) &
            (y_flat >= bounds[1]) & (y_flat <= bounds[3])
        )

        mask_flat = np.zeros(len(x_flat), dtype=bool)

        if in_bounds.any():
            x_check = x_flat[in_bounds]
            y_check = y_flat[in_bounds]

            try:
                mask_check = vectorized.contains(geometry, x_check, y_check)
            except:
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
        for mask_name, mask in global_masks.items():
            filename = f'{self.output_dir}/{mask_name}.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"  {mask_name}: {count:,} points ({pct:.2f}%) → {filename}")


if __name__ == '__main__':
    generator = MaskGeneratorEfficient()
    generator.generate_all_masks()
