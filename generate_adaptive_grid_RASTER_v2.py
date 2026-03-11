"""
Raster-based adaptive grid generation (memory-efficient version)
Process each HRRR cell individually, using stride patterns for hierarchical subdivision
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class RasterGridGeneratorV2:
    def __init__(self):
        self.loader = DataLoader()
        self.terrain_analyzer = None  # Will initialize after loading grid

    def generate(self):
        print("=" * 70)
        print(" RASTER-BASED ADAPTIVE GRID (MEMORY-EFFICIENT)")
        print(" Process HRRR cells with stride patterns")
        print("=" * 70)

        start_time = time.time()

        # Load data
        print("\nSTEP 1: Loading data...")
        self.loader.load_all()

        # Get HRRR grid
        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        shape = lats_hrrr.shape

        # Initialize terrain analyzer
        terrain_data = self.loader.hrrr_grid['terrain']
        self.terrain_analyzer = TerrainAnalyzer(terrain_data, lats_hrrr, lons_hrrr)
        self.terrain_analyzer.compute_terrain_variability()

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

        # STEP 2: Classify each HRRR cell into tiers
        print("\nSTEP 2: Classifying HRRR cells into tiers...")
        tier_map = self._classify_cells(lats_hrrr, lons_hrrr, shape)

        # Print tier distribution
        print("\nTier Distribution:")
        for tier in range(6):
            count = np.sum(tier_map == tier)
            pct = 100 * count / tier_map.size
            res = config.TIER_RESOLUTIONS[tier]
            print(f"  Tier {tier} ({res:6.2f}m): {count:7,} cells ({pct:5.2f}%)")

        # STEP 3: Generate points using stride patterns
        print("\nSTEP 3: Generating points with stride patterns...")
        all_lats = []
        all_lons = []

        for tier in range(6):
            tier_lats, tier_lons = self._generate_tier_points(
                tier, tier_map, lats_hrrr, lons_hrrr
            )
            all_lats.append(tier_lats)
            all_lons.append(tier_lons)
            print(f"  Tier {tier}: {len(tier_lats):,} points")

        # Combine all points
        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points generated: {len(lats_all):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # STEP 4: Write output
        print("\nSTEP 4: Writing output...")
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

    def _classify_cells(self, lats_hrrr, lons_hrrr, shape):
        """Classify each HRRR cell - returns tier for each cell (finest tier wins)"""
        # Initialize to Tier 5 (background)
        tier_map = np.full(shape, 5, dtype=np.int8)

        print("  Classifying coastlines (Tier 0)...")
        self._classify_coastlines_raster(tier_map, lats_hrrr, lons_hrrr)

        print("  Classifying lakes (Tier 0)...")
        self._classify_lakes_raster(tier_map, lats_hrrr, lons_hrrr)

        print("  Classifying terrain (Tiers 1-3)...")
        self._classify_terrain_raster(tier_map)

        print("  Classifying urban (Tier 2)...")
        self._classify_urban_raster(tier_map, lats_hrrr, lons_hrrr)

        return tier_map

    def _classify_coastlines_raster(self, tier_map, lats_hrrr, lons_hrrr):
        """Mark cells near coastlines as Tier 0"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None:
            return

        # Reproject and buffer
        coast_proj = coastline_gdf.to_crs(self.hrrr_crs)
        coast_proj['geometry'] = coast_proj.geometry.simplify(100)
        coast_buffered = coast_proj.buffer(config.COASTLINE_BUFFER_KM * 1000)
        coast_union = unary_union(coast_buffered)

        # Check each HRRR cell
        count = 0
        for i in range(tier_map.shape[0]):
            for j in range(tier_map.shape[1]):
                # Create cell box (3km x 3km)
                lat, lon = lats_hrrr[i, j], lons_hrrr[i, j]
                x, y = self.proj(lon, lat)
                cell_box = box(x - 1500, y - 1500, x + 1500, y + 1500)

                # Check if cell intersects buffered coastline
                if cell_box.intersects(coast_union):
                    tier_map[i, j] = 0
                    count += 1

        print(f"    Classified {count:,} cells as Tier 0")

    def _classify_lakes_raster(self, tier_map, lats_hrrr, lons_hrrr):
        """Mark cells near lakes as Tier 0"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None:
            return

        # Reproject, extract boundaries, buffer
        lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
        lakes_boundaries = lakes_proj.boundary
        lakes_buffered = lakes_boundaries.buffer(config.LAKE_BUFFER_KM * 1000)
        lakes_union = unary_union(lakes_buffered)

        count = 0
        for i in range(tier_map.shape[0]):
            for j in range(tier_map.shape[1]):
                lat, lon = lats_hrrr[i, j], lons_hrrr[i, j]
                x, y = self.proj(lon, lat)
                cell_box = box(x - 1500, y - 1500, x + 1500, y + 1500)

                if cell_box.intersects(lakes_union):
                    tier_map[i, j] = min(tier_map[i, j], 0)
                    count += 1

        print(f"    Classified {count:,} cells as Tier 0")

    def _classify_terrain_raster(self, tier_map):
        """Classify based on terrain variability"""
        terrain_var = self.terrain_analyzer.terrain_var

        # Tier 1: Most extreme terrain
        extreme = terrain_var > config.TERRAIN_TIER1_THRESHOLD
        tier_map[extreme] = np.minimum(tier_map[extreme], 1)
        print(f"    Tier 1 (extreme): {extreme.sum():,} cells")

        # Tier 2: Very rugged
        rugged = (terrain_var > config.TERRAIN_TIER2_THRESHOLD) & \
                 (terrain_var <= config.TERRAIN_TIER1_THRESHOLD)
        tier_map[rugged] = np.minimum(tier_map[rugged], 2)
        print(f"    Tier 2 (rugged): {rugged.sum():,} cells")

        # Tier 3: Moderate
        moderate = (terrain_var > config.TERRAIN_TIER3_THRESHOLD) & \
                   (terrain_var <= config.TERRAIN_TIER2_THRESHOLD)
        tier_map[moderate] = np.minimum(tier_map[moderate], 3)
        print(f"    Tier 3 (moderate): {moderate.sum():,} cells")

    def _classify_urban_raster(self, tier_map, lats_hrrr, lons_hrrr):
        """Mark urban cells as Tier 2"""
        urban_gdf = self.loader.data.get('high_density_urban')
        if urban_gdf is None:
            return

        urban_proj = urban_gdf.to_crs(self.hrrr_crs)
        urban_union = unary_union(urban_proj.geometry)

        count = 0
        for i in range(tier_map.shape[0]):
            for j in range(tier_map.shape[1]):
                lat, lon = lats_hrrr[i, j], lons_hrrr[i, j]
                x, y = self.proj(lon, lat)
                cell_box = box(x - 1500, y - 1500, x + 1500, y + 1500)

                if cell_box.intersects(urban_union):
                    tier_map[i, j] = min(tier_map[i, j], 2)
                    count += 1

        print(f"    Classified {count:,} cells as Tier 2")

    def _generate_tier_points(self, tier, tier_map, lats_hrrr, lons_hrrr):
        """Generate points for a specific tier using stride patterns"""
        # Find cells assigned to this tier
        tier_i, tier_j = np.where(tier_map == tier)

        if len(tier_i) == 0:
            return np.array([]), np.array([])

        # Stride = 2^tier (Tier 0: 1, Tier 1: 2, Tier 2: 4, etc.)
        stride = 2 ** tier

        # Base resolution (finest)
        base_res = config.TIER_RESOLUTIONS[0]  # 93.75m

        # Generate sub-grid for each cell
        all_lats = []
        all_lons = []

        for idx in range(len(tier_i)):
            i, j = tier_i[idx], tier_j[idx]
            lat_center, lon_center = lats_hrrr[i, j], lons_hrrr[i, j]
            x_center, y_center = self.proj(lon_center, lat_center)

            # Cell bounds
            x_start = x_center - 1500
            y_start = y_center - 1500

            # Generate strided sub-grid
            # Number of points per side at finest resolution
            n_finest = 32  # 3000m / 93.75m

            # With stride, we get n_finest // stride points
            n_points = n_finest // stride

            # Generate coordinates
            x_offsets = np.arange(n_points) * base_res * stride
            y_offsets = np.arange(n_points) * base_res * stride

            xx, yy = np.meshgrid(x_start + x_offsets, y_start + y_offsets)

            # Convert to lat/lon
            lons_cell, lats_cell = self.proj(xx.ravel(), yy.ravel(), inverse=True)

            all_lats.append(lats_cell)
            all_lons.append(lons_cell)

        return np.concatenate(all_lats), np.concatenate(all_lons)

if __name__ == '__main__':
    generator = RasterGridGeneratorV2()
    lats, lons = generator.generate()
