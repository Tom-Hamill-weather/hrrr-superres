"""
Binary grid approach - correct implementation
For each tier, create binary mask at 93.75m, apply stride patterns, take element-wise maximum
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

class BinaryGridGenerator:
    def __init__(self):
        self.loader = DataLoader()
        self.base_res = 93.75  # meters - finest resolution
        self.n_fine = 32  # 3000m / 93.75m

    def generate(self):
        print("=" * 70)
        print(" BINARY GRID APPROACH")
        print(" Create binary masks at 93.75m, apply stride patterns")
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

        # STEP 2: Prepare feature data (project and build spatial indices)
        print("\nSTEP 2: Preparing feature data...")
        features = self._prepare_features(terrain_var)

        # STEP 3: Process HRRR cells in patches
        print("\nSTEP 3: Processing HRRR cells in patches...")
        patch_size = 100  # Process 100 HRRR cells at a time
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
                    lats_hrrr, lons_hrrr, features
                )

                all_lats.append(patch_lats)
                all_lons.append(patch_lons)
                points_processed += len(patch_lats)

                cells_done = (i_end - 0) * shape[1] + (j_end - 0)
                if cells_done % 10000 == 0:
                    print(f"  Processed {cells_done:,} / {n_cells_total:,} cells, {points_processed:,} points so far")

        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points: {len(lats_all):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # STEP 4: Write output
        print("\nSTEP 4: Writing output...")
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

    def _prepare_features(self, terrain_var):
        """Prepare and index all feature data"""
        features = {}

        # Coastlines - extract coordinates for distance queries
        print("  Preparing coastlines...")
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
                features['coastline_tree'] = cKDTree(np.array(coast_coords))
                features['coastline_buffer'] = config.COASTLINE_BUFFER_KM * 1000
                print(f"    {len(coast_coords):,} coastline points indexed")

        # Lakes - extract boundary coordinates
        print("  Preparing lakes...")
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
                features['lake_tree'] = cKDTree(np.array(lake_coords))
                features['lake_buffer'] = config.LAKE_BUFFER_KM * 1000
                print(f"    {len(lake_coords):,} lake boundary points indexed")

        # Urban areas - create union for containment checks
        print("  Preparing urban areas...")
        urban_gdf = self.loader.data.get('high_density_urban')
        if urban_gdf is not None:
            urban_proj = urban_gdf.to_crs(self.hrrr_crs)
            features['urban_union'] = unary_union(urban_proj.geometry)
            print(f"    Urban union created")

        # Terrain variability
        features['terrain_var'] = terrain_var

        return features

    def _process_patch(self, i_start, i_end, j_start, j_end, lats_hrrr, lons_hrrr, features):
        """Process a patch of HRRR cells using binary grid approach"""
        patch_lats = []
        patch_lons = []

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # Get cell center
                lat_center = lats_hrrr[i, j]
                lon_center = lons_hrrr[i, j]
                x_center, y_center = self.proj(lon_center, lat_center)

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
                binary_grids = self._create_binary_grids(
                    i, j, x_fine, y_fine, lats_fine, lons_fine, features
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

    def _create_binary_grids(self, i, j, x_fine, y_fine, lats_fine, lons_fine, features):
        """Create binary mask for each tier at 93.75m resolution"""
        n_points = len(x_fine)

        # Initialize all tiers to False
        binary_grids = {tier: np.zeros(n_points, dtype=bool) for tier in range(6)}

        # Tier 0: Near coastlines or lakes
        if 'coastline_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['coastline_tree'].query(coords)
            binary_grids[0] |= (distances <= features['coastline_buffer'])

        if 'lake_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['lake_tree'].query(coords)
            binary_grids[0] |= (distances <= features['lake_buffer'])

        # Tier 1: Extreme terrain OR 0.5-1.5km from coastline
        terrain_var = features['terrain_var'][i, j]
        if terrain_var > config.TERRAIN_TIER1_THRESHOLD:
            binary_grids[1][:] = True

        # Distance-based transitions from coastline
        if 'coastline_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['coastline_tree'].query(coords)
            # Tier 1: 0.5-1.5km from coast
            binary_grids[1] |= ((distances > 500) & (distances <= 1500))

        # Tier 2: Rugged terrain OR urban OR 1.5-3km from coastline
        if terrain_var > config.TERRAIN_TIER2_THRESHOLD:
            binary_grids[2][:] = True

        if 'urban_union' in features:
            # Check if points are in urban areas
            for idx in range(n_points):
                pt = Point(x_fine[idx], y_fine[idx])
                if features['urban_union'].contains(pt):
                    binary_grids[2][idx] = True

        if 'coastline_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['coastline_tree'].query(coords)
            # Tier 2: 1.5-3km from coast
            binary_grids[2] |= ((distances > 1500) & (distances <= 3000))

        # Tier 3: Moderate terrain OR 3-6km from coastline
        if terrain_var > config.TERRAIN_TIER3_THRESHOLD:
            binary_grids[3][:] = True

        if 'coastline_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['coastline_tree'].query(coords)
            # Tier 3: 3-6km from coast
            binary_grids[3] |= ((distances > 3000) & (distances <= 6000))

        # Tier 4: 6-12km from coastline
        if 'coastline_tree' in features:
            coords = np.column_stack([x_fine, y_fine])
            distances, _ = features['coastline_tree'].query(coords)
            # Tier 4: 6-12km from coast
            binary_grids[4] |= ((distances > 6000) & (distances <= 12000))

        # Tier 5: Everything else (background)
        binary_grids[5][:] = True

        return binary_grids

    def _apply_stride_patterns(self, lats_fine, lons_fine, binary_grids):
        """Apply stride patterns to binary grids and collect unique points"""
        # Reshape to 32×32 grid
        lats_grid = lats_fine.reshape(self.n_fine, self.n_fine)
        lons_grid = lons_fine.reshape(self.n_fine, self.n_fine)

        # Collect points from all tiers using stride patterns
        points_set = set()  # Use set to avoid duplicates

        for tier in range(6):
            stride = 2 ** tier
            binary = binary_grids[tier].reshape(self.n_fine, self.n_fine)

            # Apply stride: every stride-th point in both dimensions
            for ii in range(0, self.n_fine, stride):
                for jj in range(0, self.n_fine, stride):
                    if binary[ii, jj]:
                        # Add this point
                        points_set.add((lats_grid[ii, jj], lons_grid[ii, jj]))

        # Convert set to arrays
        if len(points_set) > 0:
            points_list = list(points_set)
            lats = np.array([p[0] for p in points_list])
            lons = np.array([p[1] for p in points_list])
            return lats, lons
        else:
            return np.array([]), np.array([])

if __name__ == '__main__':
    generator = BinaryGridGenerator()
    lats, lons = generator.generate()
