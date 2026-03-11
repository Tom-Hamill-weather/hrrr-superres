"""
STAGE 1: Generate Boolean Masks for Features
==============================================

Generate and save individual boolean masks at 93.75m resolution for:
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

class MaskGenerator:
    def __init__(self):
        self.loader = DataLoader()
        self.n_fine = 32  # points per HRRR cell dimension
        self.output_dir = 'output/masks'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_masks(self):
        """Generate all feature masks"""
        print("="*70)
        print(" STAGE 1: GENERATING BOOLEAN MASKS")
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

        # Generate fine grid coordinates (all at once, in memory)
        print("\nGenerating fine grid coordinates...")
        x_fine, y_fine = self._generate_fine_grid(
            x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north, shape
        )

        print(f"Fine grid generated: {x_fine.shape}")

        # Flatten for geometric operations
        x_flat = x_fine.ravel()
        y_flat = y_fine.ravel()

        # TERRAIN MASKS
        print("\n" + "="*70)
        print("GENERATING TERRAIN MASKS")
        print("="*70)
        self._generate_terrain_masks(terrain, n_fine_i, n_fine_j)

        # FEATURE MASKS (geometric)
        print("\n" + "="*70)
        print("GENERATING FEATURE MASKS")
        print("="*70)

        # Coastlines
        self._generate_coastline_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # Lakes
        self._generate_lake_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # Ski resorts
        self._generate_ski_resort_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # Urban areas
        self._generate_urban_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # Roads
        self._generate_road_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # National parks
        self._generate_park_masks(x_flat, y_flat, n_fine_i, n_fine_j)

        # National forests
        self._generate_forest_masks(x_flat, y_flat, n_fine_i, n_fine_j)

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

        # East vectors
        dx_east[:, :-1] = x_hrrr[:, 1:] - x_hrrr[:, :-1]
        dy_east[:, :-1] = y_hrrr[:, 1:] - y_hrrr[:, :-1]
        dx_east[:, -1] = dx_east[:, -2]
        dy_east[:, -1] = dy_east[:, -2]

        # North vectors
        dx_north[:-1, :] = x_hrrr[1:, :] - x_hrrr[:-1, :]
        dy_north[:-1, :] = y_hrrr[1:, :] - y_hrrr[:-1, :]
        dx_north[-1, :] = dx_north[-2, :]
        dy_north[-1, :] = dy_north[-2, :]

        return dx_east, dy_east, dx_north, dy_north

    def _generate_fine_grid(self, x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north, shape):
        """Generate the full fine grid at once"""
        # Repeat HRRR values to fine grid
        x_repeated = np.repeat(np.repeat(x_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_hrrr, self.n_fine, axis=0), self.n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north, self.n_fine, axis=0), self.n_fine, axis=1)

        n_fine_i = shape[0] * self.n_fine
        n_fine_j = shape[1] * self.n_fine

        # Create fine grid indices
        i_fine_idx = np.arange(n_fine_i) % self.n_fine
        j_fine_idx = np.arange(n_fine_j) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        # Compute fractional offsets
        frac_east = jj_fine_idx / self.n_fine - 0.5
        frac_north = ii_fine_idx / self.n_fine - 0.5

        # Apply offsets
        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        return x_fine, y_fine

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

    def _generate_coastline_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate coastline buffer masks at multiple distances"""
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is None:
            print("\n⚠ No coastline data")
            return

        significant_coastlines = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
        print(f"\nCoastlines: {len(significant_coastlines)} segments (>{50}km)")

        coast_proj = significant_coastlines.to_crs(self.hrrr_crs)
        geom_union = coast_proj.geometry.union_all()

        distances = [750, 1500, 3000, 6000, 12000]
        for dist in distances:
            print(f"  Computing {dist}m buffer...")
            buffer_geom = geom_union.buffer(dist).simplify(tolerance=1.0)

            mask = self._geometric_contains(buffer_geom, x_flat, y_flat, n_fine_i, n_fine_j)

            filename = f'{self.output_dir}/coastline_{dist}m.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"    → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_lake_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate lake buffer masks"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None or len(lakes_gdf) == 0:
            print("\n⚠ No lake data")
            return

        print(f"\nLakes: {len(lakes_gdf)} features")

        lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
        geom_union = lakes_proj.geometry.boundary.union_all()

        distances = [750, 1500, 3000, 6000, 12000]
        for dist in distances:
            print(f"  Computing {dist}m buffer...")
            buffer_geom = geom_union.buffer(dist).simplify(tolerance=1.0)

            mask = self._geometric_contains(buffer_geom, x_flat, y_flat, n_fine_i, n_fine_j)

            filename = f'{self.output_dir}/lakes_{dist}m.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"    → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_ski_resort_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate ski resort masks (2km buffer)"""
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is None or len(ski_gdf) == 0:
            print("\n⚠ No ski resort data")
            return

        print(f"\nSki resorts: {len(ski_gdf)} locations")

        ski_proj = ski_gdf.to_crs(self.hrrr_crs)
        # Buffer by 2km
        ski_buffered = ski_proj.geometry.buffer(2000).union_all()

        print(f"  Computing 2km buffer...")
        mask = self._geometric_contains(ski_buffered, x_flat, y_flat, n_fine_i, n_fine_j)

        filename = f'{self.output_dir}/ski_resorts_2km.npz'
        np.savez_compressed(filename, mask=mask)
        count = mask.sum()
        pct = 100 * count / mask.size
        print(f"    → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_urban_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate high-density and suburban (low-density) urban masks"""
        # High-density urban
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            print(f"\nHigh-density urban: {len(urban_high)} areas")
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            geom_union = urban_proj.geometry.boundary.union_all()

            mask = self._geometric_contains(geom_union, x_flat, y_flat, n_fine_i, n_fine_j)

            filename = f'{self.output_dir}/urban_high_density.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"  → {count:,} points ({pct:.2f}%) saved to {filename}")

        # Suburban (low-density urban)
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            print(f"\nSuburban areas: {len(urban_low)} areas")
            cluster_proj = urban_low.to_crs(self.hrrr_crs)
            geom_union = cluster_proj.geometry.boundary.union_all()

            mask = self._geometric_contains(geom_union, x_flat, y_flat, n_fine_i, n_fine_j)

            filename = f'{self.output_dir}/urban_suburban.npz'
            np.savez_compressed(filename, mask=mask)
            count = mask.sum()
            pct = 100 * count / mask.size
            print(f"  → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_road_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate major road masks (500m buffer)"""
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is None or len(roads_gdf) == 0:
            print("\n⚠ No road data")
            return

        print(f"\nMajor roads: {len(roads_gdf)} segments")

        roads_proj = roads_gdf.to_crs(self.hrrr_crs)
        geom_union = roads_proj.geometry.union_all()

        print(f"  Computing 500m buffer...")
        buffer_geom = geom_union.buffer(500).simplify(tolerance=1.0)

        mask = self._geometric_contains(buffer_geom, x_flat, y_flat, n_fine_i, n_fine_j)

        filename = f'{self.output_dir}/roads_500m.npz'
        np.savez_compressed(filename, mask=mask)
        count = mask.sum()
        pct = 100 * count / mask.size
        print(f"  → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_park_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate national park masks (full areas)"""
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is None or len(parks_gdf) == 0:
            print("\n⚠ No national park data")
            return

        print(f"\nNational parks: {len(parks_gdf)} units")

        parks_proj = parks_gdf.to_crs(self.hrrr_crs)
        geom_union = parks_proj.geometry.union_all().simplify(tolerance=1.0)

        mask = self._geometric_contains(geom_union, x_flat, y_flat, n_fine_i, n_fine_j)

        filename = f'{self.output_dir}/national_parks.npz'
        np.savez_compressed(filename, mask=mask)
        count = mask.sum()
        pct = 100 * count / mask.size
        print(f"  → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _generate_forest_masks(self, x_flat, y_flat, n_fine_i, n_fine_j):
        """Generate national forest masks (full areas)"""
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is None or len(forests_gdf) == 0:
            print("\n⚠ No national forest data")
            return

        print(f"\nNational forests: {len(forests_gdf)} units")

        forests_proj = forests_gdf.to_crs(self.hrrr_crs)
        geom_union = forests_proj.geometry.union_all().simplify(tolerance=1.0)

        mask = self._geometric_contains(geom_union, x_flat, y_flat, n_fine_i, n_fine_j)

        filename = f'{self.output_dir}/national_forests.npz'
        np.savez_compressed(filename, mask=mask)
        count = mask.sum()
        pct = 100 * count / mask.size
        print(f"  → {count:,} points ({pct:.2f}%) saved to {filename}")

    def _geometric_contains(self, geometry, x_flat, y_flat, n_fine_i, n_fine_j):
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

        return mask_flat.reshape(n_fine_i, n_fine_j)


if __name__ == '__main__':
    generator = MaskGenerator()
    generator.generate_all_masks()
