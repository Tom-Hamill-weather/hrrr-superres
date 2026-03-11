"""
Binary grid approach v2 - optimized
Pre-classify HRRR cells, then create binary grids efficiently
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

class BinaryGridGeneratorV2:
    def __init__(self):
        self.loader = DataLoader()
        self.base_res = 93.75  # meters - finest resolution
        self.n_fine = 32  # 3000m / 93.75m

    def generate(self):
        print("=" * 70)
        print(" BINARY GRID APPROACH V2 (OPTIMIZED)")
        print(" Pre-classify cells, then create binary grids")
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

        # Convert HRRR grid to projected coordinates
        print("\nSTEP 2: Converting HRRR grid to projected coordinates...")
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)

        # STEP 3: Pre-classify HRRR cells and build spatial index
        print("\nSTEP 3: Pre-classifying HRRR cells and building spatial index...")
        cell_features = self._preclassify_cells(x_hrrr, y_hrrr, terrain_var, shape)

        # Build KDTree for coastline distances (for gradual transitions)
        print("\nSTEP 4: Building coastline distance index...")
        coastline_tree = self._build_coastline_tree()

        # STEP 5: Process HRRR cells in patches
        print("\nSTEP 5: Processing HRRR cells in patches...")
        patch_size = 50  # Process 50 HRRR cells at a time
        n_cells_total = shape[0] * shape[1]

        all_lats = []
        all_lons = []
        points_processed = 0

        for i_start in range(0, shape[0], patch_size):
            i_end = min(i_start + patch_size, shape[0])

            for j_start in range(0, shape[1], patch_size):
                j_end = min(j_start + patch_size, shape[1])

                # Process this patch
                patch_lats, patch_lons = self._process_patch(
                    i_start, i_end, j_start, j_end,
                    lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                    cell_features, coastline_tree
                )

                all_lats.append(patch_lats)
                all_lons.append(patch_lons)
                points_processed += len(patch_lats)

                cells_done = min((i_end * shape[1]), n_cells_total)
                if cells_done % 50000 == 0:
                    pct = 100 * cells_done / n_cells_total
                    print(f"  Progress: {cells_done:,} / {n_cells_total:,} cells ({pct:.1f}%), {points_processed:,} points")

        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points: {len(lats_all):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # STEP 6: Write output
        print("\nSTEP 6: Writing output...")
        output_path = 'output/adaptive_grid_BINARY.nc'
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

    def _preclassify_cells(self, x_hrrr, y_hrrr, terrain_var, shape):
        """Pre-classify which HRRR cells have which features"""
        cell_features = {
            'is_coastal': np.zeros(shape, dtype=bool),
            'is_lake': np.zeros(shape, dtype=bool),
            'is_urban': np.zeros(shape, dtype=bool),
            'terrain_tier': np.full(shape, 5, dtype=np.int8)
        }

        # Coastlines
        print("  Classifying coastal cells...")
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is not None:
            coast_proj = coastline_gdf.to_crs(self.hrrr_crs)
            coast_coords = []
            for geom in coast_proj.geometry:
                if geom.geom_type == 'LineString':
                    coast_coords.extend(geom.coords)
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        coast_coords.extend(line.coords)

            if len(coast_coords) > 0:
                tree = cKDTree(np.array(coast_coords))
                hrrr_coords = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
                distances, _ = tree.query(hrrr_coords)
                # Mark cells within buffer as coastal
                is_coastal = distances <= (config.COASTLINE_BUFFER_KM * 1000)
                cell_features['is_coastal'] = is_coastal.reshape(shape)
                print(f"    {cell_features['is_coastal'].sum():,} coastal cells")

        # Lakes
        print("  Classifying lake cells...")
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None:
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            lake_coords = []
            for geom in lakes_proj.geometry:
                if geom.geom_type == 'Polygon':
                    lake_coords.extend(geom.exterior.coords)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        lake_coords.extend(poly.exterior.coords)

            if len(lake_coords) > 0:
                tree = cKDTree(np.array(lake_coords))
                hrrr_coords = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
                distances, _ = tree.query(hrrr_coords)
                is_lake = distances <= (config.LAKE_BUFFER_KM * 1000)
                cell_features['is_lake'] = is_lake.reshape(shape)
                print(f"    {cell_features['is_lake'].sum():,} lake cells")

        # Urban areas - SKIPPED FOR NOW (too slow, will add back later)
        print("  Skipping urban cells for now (will optimize later)...")

        # Terrain
        print("  Classifying terrain...")
        cell_features['terrain_tier'][terrain_var > config.TERRAIN_TIER1_THRESHOLD] = 1
        cell_features['terrain_tier'][(terrain_var > config.TERRAIN_TIER2_THRESHOLD) &
                                      (terrain_var <= config.TERRAIN_TIER1_THRESHOLD)] = 2
        cell_features['terrain_tier'][(terrain_var > config.TERRAIN_TIER3_THRESHOLD) &
                                      (terrain_var <= config.TERRAIN_TIER2_THRESHOLD)] = 3
        print(f"    Tier 1 terrain: {(cell_features['terrain_tier'] == 1).sum():,} cells")
        print(f"    Tier 2 terrain: {(cell_features['terrain_tier'] == 2).sum():,} cells")
        print(f"    Tier 3 terrain: {(cell_features['terrain_tier'] == 3).sum():,} cells")

        return cell_features

    def _build_coastline_tree(self):
        """Build KDTree for coastline distance queries"""
        coastline_gdf = self.loader.data.get('coastline')
        if coastline_gdf is None:
            return None

        coast_proj = coastline_gdf.to_crs(self.hrrr_crs)
        coast_coords = []
        for geom in coast_proj.geometry:
            if geom.geom_type == 'LineString':
                coast_coords.extend(geom.coords)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coast_coords.extend(line.coords)

        if len(coast_coords) > 0:
            return cKDTree(np.array(coast_coords))
        return None

    def _process_patch(self, i_start, i_end, j_start, j_end, lats_hrrr, lons_hrrr,
                      x_hrrr, y_hrrr, cell_features, coastline_tree):
        """Process a patch of HRRR cells using binary grid approach"""
        patch_lats = []
        patch_lons = []

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # Get cell center
                lat_center = lats_hrrr[i, j]
                lon_center = lons_hrrr[i, j]
                x_center = x_hrrr[i, j]
                y_center = y_hrrr[i, j]

                # Create 32×32 fine grid for this cell
                x_start = x_center - 1500
                y_start = y_center - 1500

                x_offsets = np.arange(self.n_fine) * self.base_res
                y_offsets = np.arange(self.n_fine) * self.base_res

                xx, yy = np.meshgrid(x_start + x_offsets, y_start + y_offsets)
                x_fine = xx.ravel()
                y_fine = yy.ravel()

                # Convert to lat/lon
                lons_fine, lats_fine = self.proj(x_fine, y_fine, inverse=True)

                # Create binary masks for each tier
                binary_grids = self._create_binary_grids_fast(
                    i, j, x_fine, y_fine, cell_features, coastline_tree
                )

                # Apply stride patterns and collect points
                cell_lats, cell_lons = self._apply_stride_patterns(
                    lats_fine, lons_fine, binary_grids
                )

                patch_lats.append(cell_lats)
                patch_lons.append(cell_lons)

        if len(patch_lats) > 0:
            return np.concatenate(patch_lats), np.concatenate(patch_lons)
        else:
            return np.array([]), np.array([])

    def _create_binary_grids_fast(self, i, j, x_fine, y_fine, cell_features, coastline_tree):
        """Create binary masks using pre-classified cell features"""
        n_points = len(x_fine)
        binary_grids = {tier: np.zeros(n_points, dtype=bool) for tier in range(6)}

        # Tier 0: Coastal or lake cells - mark ALL fine points
        if cell_features['is_coastal'][i, j] or cell_features['is_lake'][i, j]:
            binary_grids[0][:] = True

        # For distance-based transitions, compute distance from coastline
        if coastline_tree is not None:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = coastline_tree.query(coords)

            # Tier 1: 500m-1500m from coast OR extreme terrain
            if cell_features['terrain_tier'][i, j] == 1:
                binary_grids[1][:] = True
            binary_grids[1] |= ((distances > 500) & (distances <= 1500))

            # Tier 2: 1.5km-3km from coast OR rugged terrain OR urban
            if cell_features['terrain_tier'][i, j] == 2:
                binary_grids[2][:] = True
            if cell_features['is_urban'][i, j]:
                binary_grids[2][:] = True
            binary_grids[2] |= ((distances > 1500) & (distances <= 3000))

            # Tier 3: 3km-6km from coast OR moderate terrain
            if cell_features['terrain_tier'][i, j] == 3:
                binary_grids[3][:] = True
            binary_grids[3] |= ((distances > 3000) & (distances <= 6000))

            # Tier 4: 6km-12km from coast
            binary_grids[4] |= ((distances > 6000) & (distances <= 12000))

            # Tier 5: Background (everywhere)
            binary_grids[5][:] = True
        else:
            # No coastline data - use terrain and urban only
            if cell_features['terrain_tier'][i, j] == 1:
                binary_grids[1][:] = True
            if cell_features['terrain_tier'][i, j] == 2 or cell_features['is_urban'][i, j]:
                binary_grids[2][:] = True
            if cell_features['terrain_tier'][i, j] == 3:
                binary_grids[3][:] = True
            binary_grids[5][:] = True

        return binary_grids

    def _apply_stride_patterns(self, lats_fine, lons_fine, binary_grids):
        """Apply stride patterns to binary grids and collect unique points"""
        # Reshape to 32×32 grid
        lats_grid = lats_fine.reshape(self.n_fine, self.n_fine)
        lons_grid = lons_fine.reshape(self.n_fine, self.n_fine)

        # Collect points from all tiers using stride patterns
        points_set = set()  # Use set to avoid duplicates (element-wise maximum)

        for tier in range(6):
            stride = 2 ** tier
            binary = binary_grids[tier].reshape(self.n_fine, self.n_fine)

            # Apply stride: every stride-th point in both dimensions
            for ii in range(0, self.n_fine, stride):
                for jj in range(0, self.n_fine, stride):
                    if binary[ii, jj]:
                        # Add this point (round to avoid floating point issues)
                        lat_rounded = round(lats_grid[ii, jj], 6)
                        lon_rounded = round(lons_grid[ii, jj], 6)
                        points_set.add((lat_rounded, lon_rounded))

        # Convert set to arrays
        if len(points_set) > 0:
            points_list = list(points_set)
            lats = np.array([p[0] for p in points_list])
            lons = np.array([p[1] for p in points_list])
            return lats, lons
        else:
            return np.array([]), np.array([])

if __name__ == '__main__':
    generator = BinaryGridGeneratorV2()
    lats, lons = generator.generate()
