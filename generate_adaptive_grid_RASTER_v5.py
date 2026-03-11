"""
Raster-based adaptive grid generation v5
Fast distance-based classification using vectorized operations
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class RasterGridGeneratorV5:
    def __init__(self):
        self.loader = DataLoader()

    def generate(self):
        print("=" * 70)
        print(" RASTER-BASED ADAPTIVE GRID v5")
        print(" Fast distance-based classification")
        print("=" * 70)

        start_time = time.time()

        # STEP 1: Load data
        print("\nSTEP 1: Loading data...")
        self.loader.load_all()

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        terrain = self.loader.hrrr_grid['terrain']
        shape = lats_hrrr.shape

        # Initialize terrain analyzer
        terrain_analyzer = TerrainAnalyzer(terrain, lats_hrrr, lons_hrrr)
        terrain_var = terrain_analyzer.compute_terrain_variability()

        # Create projection
        lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
        lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2
        self.proj = Basemap(
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
        self.hrrr_crs = self.proj.proj4string

        # Convert HRRR grid to projected coordinates (vectorized)
        print("\nSTEP 2: Converting HRRR grid to projected coordinates...")
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)
        hrrr_coords = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])

        # STEP 3: Classify cells into tiers
        print("\nSTEP 3: Classifying cells...")
        tier_map = np.full(shape, 5, dtype=np.int8)  # Default: Tier 5

        # Coastlines -> Tier 0
        print("  Processing coastlines...")
        self._classify_by_distance(
            tier_map, hrrr_coords, shape, 'coastline', 0,
            buffer_m=config.COASTLINE_BUFFER_KM * 1000
        )

        # Lakes -> Tier 0
        print("  Processing lakes...")
        self._classify_lakes_by_distance(tier_map, hrrr_coords, shape)

        # Terrain -> Tiers 1-3
        print("  Processing terrain...")
        self._apply_terrain(tier_map, terrain_var)

        # Urban -> Tier 2
        print("  Processing urban areas...")
        self._classify_urban_by_containment(tier_map, hrrr_coords, shape)

        # Print tier distribution
        print("\nTier Distribution:")
        for tier in range(6):
            count = np.sum(tier_map == tier)
            pct = 100 * count / tier_map.size
            res = config.TIER_RESOLUTIONS[tier]
            print(f"  Tier {tier} ({res:6.2f}m): {count:7,} cells ({pct:5.2f}%)")

        # STEP 4: Generate points using stride patterns
        print("\nSTEP 4: Generating points with stride patterns...")
        all_lats = []
        all_lons = []

        for tier in range(6):
            tier_lats, tier_lons = self._generate_tier_points(
                tier, tier_map, lats_hrrr, lons_hrrr
            )
            all_lats.append(tier_lats)
            all_lons.append(tier_lons)
            print(f"  Tier {tier}: {len(tier_lats):,} points")

        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points: {len(lats_all):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # STEP 5: Write output
        print("\nSTEP 5: Writing output...")
        output_path = 'output/adaptive_grid_RASTER.nc'
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats_all))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))

        lat_var[:] = lats_all
        lon_var[:] = lons_all

        ds.close()

        elapsed = time.time() - start_time
        print(f"\n✓ Complete in {elapsed/60:.1f} minutes")
        print(f"  Output: {output_path}")

        return lats_all, lons_all

    def _classify_by_distance(self, tier_map, hrrr_coords, shape,
                              feature_name, target_tier, buffer_m):
        """Classify cells based on distance to features using KDTree"""
        feature_gdf = self.loader.data.get(feature_name)
        if feature_gdf is None:
            print(f"    No {feature_name} data")
            return

        # Reproject features
        feature_proj = feature_gdf.to_crs(self.hrrr_crs)

        # Extract feature coordinates
        feature_coords = []
        for geom in feature_proj.geometry:
            if geom.geom_type == 'LineString':
                feature_coords.extend(geom.coords)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    feature_coords.extend(line.coords)
            elif geom.geom_type == 'Polygon':
                feature_coords.extend(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    feature_coords.extend(poly.exterior.coords)

        if len(feature_coords) == 0:
            print(f"    No {feature_name} coordinates")
            return

        feature_coords = np.array(feature_coords)

        # Build KDTree for feature points
        tree = cKDTree(feature_coords)

        # Query distances from HRRR cells to nearest feature
        distances, _ = tree.query(hrrr_coords)

        # Mark cells within buffer distance
        within_buffer = distances <= buffer_m
        tier_map_flat = tier_map.ravel()
        tier_map_flat[within_buffer] = np.minimum(tier_map_flat[within_buffer], target_tier)
        tier_map[:] = tier_map_flat.reshape(shape)

        count = np.sum(within_buffer)
        print(f"    Tier {target_tier}: {count:,} cells")

    def _classify_lakes_by_distance(self, tier_map, hrrr_coords, shape):
        """Classify lake boundaries using distance-based approach"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None:
            print("    No lakes data")
            return

        # Reproject
        lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)

        # Extract boundary coordinates
        boundary_coords = []
        for geom in lakes_proj.geometry:
            if geom.geom_type == 'Polygon':
                boundary_coords.extend(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    boundary_coords.extend(poly.exterior.coords)

        if len(boundary_coords) == 0:
            print("    No lake boundary coordinates")
            return

        boundary_coords = np.array(boundary_coords)

        # Build KDTree
        tree = cKDTree(boundary_coords)

        # Query distances
        distances, _ = tree.query(hrrr_coords)

        # Mark cells within buffer
        buffer_m = config.LAKE_BUFFER_KM * 1000
        within_buffer = distances <= buffer_m
        tier_map_flat = tier_map.ravel()
        tier_map_flat[within_buffer] = np.minimum(tier_map_flat[within_buffer], 0)
        tier_map[:] = tier_map_flat.reshape(shape)

        count = np.sum(within_buffer)
        print(f"    Tier 0: {count:,} cells")

    def _classify_urban_by_containment(self, tier_map, hrrr_coords, shape):
        """Classify urban cells by checking if cell centers are inside urban polygons"""
        urban_gdf = self.loader.data.get('high_density_urban')
        if urban_gdf is None:
            print("    No urban data")
            return

        # Reproject
        urban_proj = urban_gdf.to_crs(self.hrrr_crs)

        # Create union of all urban polygons
        urban_union = unary_union(urban_proj.geometry)

        # Check which HRRR cell centers are inside urban areas
        from shapely.geometry import Point
        hrrr_points = [Point(x, y) for x, y in hrrr_coords]

        # Check containment (vectorized using list comprehension)
        inside_urban = np.array([urban_union.contains(pt) for pt in hrrr_points])

        # Mark cells
        tier_map_flat = tier_map.ravel()
        tier_map_flat[inside_urban] = np.minimum(tier_map_flat[inside_urban], 2)
        tier_map[:] = tier_map_flat.reshape(shape)

        count = np.sum(inside_urban)
        print(f"    Tier 2: {count:,} cells")

    def _apply_terrain(self, tier_map, terrain_var):
        """Apply terrain thresholds"""
        # Tier 1: Most extreme
        extreme = terrain_var > config.TERRAIN_TIER1_THRESHOLD
        tier_map[extreme] = np.minimum(tier_map[extreme], 1)
        print(f"    Tier 1 (>{config.TERRAIN_TIER1_THRESHOLD}m): {extreme.sum():,} cells")

        # Tier 2: Rugged
        rugged = (terrain_var > config.TERRAIN_TIER2_THRESHOLD) & \
                 (terrain_var <= config.TERRAIN_TIER1_THRESHOLD)
        tier_map[rugged] = np.minimum(tier_map[rugged], 2)
        print(f"    Tier 2 ({config.TERRAIN_TIER2_THRESHOLD}-{config.TERRAIN_TIER1_THRESHOLD}m): {rugged.sum():,} cells")

        # Tier 3: Moderate
        moderate = (terrain_var > config.TERRAIN_TIER3_THRESHOLD) & \
                   (terrain_var <= config.TERRAIN_TIER2_THRESHOLD)
        tier_map[moderate] = np.minimum(tier_map[moderate], 3)
        print(f"    Tier 3 ({config.TERRAIN_TIER3_THRESHOLD}-{config.TERRAIN_TIER2_THRESHOLD}m): {moderate.sum():,} cells")

    def _generate_tier_points(self, tier, tier_map, lats_hrrr, lons_hrrr):
        """Generate points for a tier using stride patterns - VECTORIZED WITH PATCHING"""
        tier_i, tier_j = np.where(tier_map == tier)

        if len(tier_i) == 0:
            return np.array([]), np.array([])

        # Stride = 2^tier
        stride = 2 ** tier
        base_res = config.TIER_RESOLUTIONS[0]  # 93.75m
        n_finest = 32  # 3000m / 93.75m
        n_points = n_finest // stride

        # Generate offset pattern once
        offsets = np.arange(n_points) * base_res * stride

        # Process in patches to avoid memory issues
        patch_size = 5000  # Process 5k cells at a time
        n_cells = len(tier_i)

        all_lats_patches = []
        all_lons_patches = []

        for start_idx in range(0, n_cells, patch_size):
            end_idx = min(start_idx + patch_size, n_cells)

            # Get cell centers for this patch
            patch_i = tier_i[start_idx:end_idx]
            patch_j = tier_j[start_idx:end_idx]

            lats_centers = lats_hrrr[patch_i, patch_j]
            lons_centers = lons_hrrr[patch_i, patch_j]
            x_centers, y_centers = self.proj(lons_centers, lats_centers)

            # Create meshgrid once for the pattern
            xx_offsets, yy_offsets = np.meshgrid(offsets, offsets)  # Both (n_points, n_points)

            # Broadcast cell starts to create full grid for all cells in patch
            # Shape: (n_patch_cells, n_points, n_points)
            x_pts = (x_centers - 1500)[:, None, None] + xx_offsets[None, :, :]
            y_pts = (y_centers - 1500)[:, None, None] + yy_offsets[None, :, :]

            # Flatten spatial dimensions
            x_pts_flat = x_pts.ravel()
            y_pts_flat = y_pts.ravel()

            # Convert back to lat/lon (vectorized)
            lons_patch, lats_patch = self.proj(x_pts_flat, y_pts_flat, inverse=True)

            all_lats_patches.append(lats_patch)
            all_lons_patches.append(lons_patch)

        # Concatenate all patches
        lats_flat = np.concatenate(all_lats_patches)
        lons_flat = np.concatenate(all_lons_patches)

        return lats_flat, lons_flat

if __name__ == '__main__':
    generator = RasterGridGeneratorV5()
    lats, lons = generator.generate()
