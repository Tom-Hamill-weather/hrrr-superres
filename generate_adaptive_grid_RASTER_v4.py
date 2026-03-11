"""
Raster-based adaptive grid generation v4
Vectorized spatial operations without rasterio dependency
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import box
from shapely.ops import unary_union
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class RasterGridGeneratorV4:
    def __init__(self):
        self.loader = DataLoader()

    def generate(self):
        print("=" * 70)
        print(" RASTER-BASED ADAPTIVE GRID v4")
        print(" Vectorized spatial operations")
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
        terrain_analyzer.compute_terrain_variability()

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
        hrrr_crs = self.proj.proj4string

        # STEP 2: Create HRRR cell grid (vectorized)
        print("\nSTEP 2: Creating HRRR cell geometries...")
        cell_boxes = self._create_cell_boxes_vectorized(lats_hrrr, lons_hrrr)
        hrrr_gdf = gpd.GeoDataFrame(
            {
                'i': cell_boxes['i'],
                'j': cell_boxes['j'],
                'geometry': cell_boxes['geometry']
            },
            crs=hrrr_crs
        )
        print(f"  Created {len(hrrr_gdf):,} HRRR cells")

        # STEP 3: Classify cells into tiers
        print("\nSTEP 3: Classifying cells...")
        tier_map = np.full(shape, 5, dtype=np.int8)  # Default: Tier 5

        # Coastlines -> Tier 0
        print("  Processing coastlines...")
        self._classify_feature_vectorized(
            tier_map, hrrr_gdf, 'coastline', 0, hrrr_crs,
            buffer_km=config.COASTLINE_BUFFER_KM
        )

        # Lakes -> Tier 0
        print("  Processing lakes...")
        self._classify_lakes_vectorized(tier_map, hrrr_gdf, hrrr_crs)

        # Terrain -> Tiers 1-3
        print("  Processing terrain...")
        self._apply_terrain(tier_map, terrain_analyzer.terrain_var)

        # Urban -> Tier 2
        print("  Processing urban areas...")
        self._classify_feature_vectorized(
            tier_map, hrrr_gdf, 'high_density_urban', 2, hrrr_crs,
            buffer_km=0
        )

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

    def _create_cell_boxes_vectorized(self, lats_hrrr, lons_hrrr):
        """Create box geometries for all HRRR cells (vectorized)"""
        shape = lats_hrrr.shape

        # Get all cell centers
        lats_flat = lats_hrrr.ravel()
        lons_flat = lons_hrrr.ravel()

        # Convert to projected coordinates
        x_flat, y_flat = self.proj(lons_flat, lats_flat)

        # Create boxes (3km x 3km)
        geometries = [
            box(x - 1500, y - 1500, x + 1500, y + 1500)
            for x, y in zip(x_flat, y_flat)
        ]

        # Create i, j indices
        i_indices = np.repeat(np.arange(shape[0]), shape[1])
        j_indices = np.tile(np.arange(shape[1]), shape[0])

        return {
            'i': i_indices,
            'j': j_indices,
            'geometry': geometries
        }

    def _classify_feature_vectorized(self, tier_map, hrrr_gdf, feature_name,
                                     target_tier, crs, buffer_km=0):
        """Classify cells based on feature using vectorized spatial join"""
        feature_gdf = self.loader.data.get(feature_name)
        if feature_gdf is None:
            print(f"    No {feature_name} data")
            return

        # Reproject and buffer
        feature_proj = feature_gdf.to_crs(crs)

        if buffer_km > 0:
            feature_proj = feature_proj.copy()
            feature_proj['geometry'] = feature_proj.geometry.buffer(buffer_km * 1000)

        # Spatial join to find intersections
        joined = gpd.sjoin(hrrr_gdf, feature_proj, how='inner', predicate='intersects')

        # Mark cells in tier map
        for idx in joined.index.unique():
            i = hrrr_gdf.loc[idx, 'i']
            j = hrrr_gdf.loc[idx, 'j']
            tier_map[i, j] = min(tier_map[i, j], target_tier)

        count = len(joined.index.unique())
        print(f"    Tier {target_tier}: {count:,} cells")

    def _classify_lakes_vectorized(self, tier_map, hrrr_gdf, crs):
        """Classify lake boundaries using vectorized operations"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None:
            print("    No lakes data")
            return

        # Reproject, get boundaries, buffer
        lakes_proj = lakes_gdf.to_crs(crs)
        lakes_boundaries = gpd.GeoDataFrame(
            geometry=lakes_proj.boundary,
            crs=crs
        )
        lakes_boundaries['geometry'] = lakes_boundaries.geometry.buffer(
            config.LAKE_BUFFER_KM * 1000
        )

        # Spatial join
        joined = gpd.sjoin(hrrr_gdf, lakes_boundaries, how='inner', predicate='intersects')

        # Mark cells
        for idx in joined.index.unique():
            i = hrrr_gdf.loc[idx, 'i']
            j = hrrr_gdf.loc[idx, 'j']
            tier_map[i, j] = min(tier_map[i, j], 0)

        count = len(joined.index.unique())
        print(f"    Tier 0: {count:,} cells")

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

            # Broadcast to generate all points for this patch
            # Shape: (n_patch_cells, n_points, n_points)
            x_starts = x_centers[:, None, None] - 1500
            y_starts = y_centers[:, None, None] - 1500

            x_offsets = offsets[None, None, :]  # (1, 1, n_points)
            y_offsets = offsets[None, :, None]  # (1, n_points, 1)

            # Broadcast addition
            x_pts = x_starts + x_offsets  # (n_patch_cells, n_points, n_points)
            y_pts = y_starts + y_offsets

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
    generator = RasterGridGeneratorV4()
    lats, lons = generator.generate()
