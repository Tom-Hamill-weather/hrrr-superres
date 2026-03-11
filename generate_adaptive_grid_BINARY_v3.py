"""
Binary grid approach v3 - TRUE PATCHING
Process patches of HRRR cells, create fine grids at 93.75m, apply stride patterns
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class BinaryGridGeneratorV3:
    def __init__(self):
        self.loader = DataLoader()
        self.base_res = 93.75  # meters - finest resolution
        self.n_fine = 32  # 3000m / 93.75m per HRRR cell

    def generate(self):
        print("=" * 70, flush=True)
        print(" BINARY GRID APPROACH V3 (TRUE PATCHING)", flush=True)
        print(" Process patches of HRRR cells at 93.75m resolution", flush=True)
        print("=" * 70, flush=True)

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

        # STEP 2: Build spatial indices
        print("\nSTEP 2: Building spatial indices...")
        coastline_tree = self._build_coastline_tree()
        lake_tree = self._build_lake_tree()

        # STEP 3: Process in patches
        print("\nSTEP 3: Processing HRRR grid in patches...")
        patch_size = 10  # 10×10 HRRR cells = 320×320 fine points per patch

        all_lats = []
        all_lons = []
        total_points = 0

        n_patches_i = int(np.ceil(shape[0] / patch_size))
        n_patches_j = int(np.ceil(shape[1] / patch_size))
        total_patches = n_patches_i * n_patches_j

        print(f"  Grid size: {shape[0]}×{shape[1]} HRRR cells")
        print(f"  Patch size: {patch_size}×{patch_size} HRRR cells")
        print(f"  Total patches: {total_patches}")

        patch_count = 0

        for i_patch in range(n_patches_i):
            for j_patch in range(n_patches_j):
                i_start = i_patch * patch_size
                i_end = min(i_start + patch_size, shape[0])
                j_start = j_patch * patch_size
                j_end = min(j_start + patch_size, shape[1])

                # Process this patch
                patch_lats, patch_lons = self._process_patch(
                    i_start, i_end, j_start, j_end,
                    lats_hrrr, lons_hrrr, terrain_var,
                    coastline_tree, lake_tree
                )

                all_lats.append(patch_lats)
                all_lons.append(patch_lons)
                total_points += len(patch_lats)

                patch_count += 1
                if patch_count % 100 == 0:
                    pct = 100 * patch_count / total_patches
                    print(f"  Progress: {patch_count}/{total_patches} patches ({pct:.1f}%), {total_points:,} points")

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

    def _build_coastline_tree(self):
        """Build KDTree for coastline"""
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
            print(f"  Coastline: {len(coast_coords):,} points indexed")
            return cKDTree(np.array(coast_coords))
        return None

    def _build_lake_tree(self):
        """Build KDTree for lake boundaries"""
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is None:
            return None

        lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
        lake_coords = []
        for geom in lakes_proj.geometry:
            if geom.geom_type == 'Polygon':
                lake_coords.extend(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    lake_coords.extend(poly.exterior.coords)

        if len(lake_coords) > 0:
            print(f"  Lakes: {len(lake_coords):,} points indexed")
            return cKDTree(np.array(lake_coords))
        return None

    def _process_patch(self, i_start, i_end, j_start, j_end,
                      lats_hrrr, lons_hrrr, terrain_var,
                      coastline_tree, lake_tree):
        """Process a patch of HRRR cells"""
        # Patch dimensions in HRRR cells
        n_i = i_end - i_start
        n_j = j_end - j_start

        # Fine grid dimensions (each HRRR cell → 32×32 fine points)
        n_fine_i = n_i * self.n_fine
        n_fine_j = n_j * self.n_fine

        # Get HRRR cell centers for this patch
        lats_patch = lats_hrrr[i_start:i_end, j_start:j_end]
        lons_patch = lons_hrrr[i_start:i_end, j_start:j_end]
        x_patch, y_patch = self.proj(lons_patch, lats_patch)

        # Create fine grid coordinates
        # For each HRRR cell, create 32×32 sub-grid
        x_fine = np.zeros((n_fine_i, n_fine_j))
        y_fine = np.zeros((n_fine_i, n_fine_j))

        for i in range(n_i):
            for j in range(n_j):
                x_center = x_patch[i, j]
                y_center = y_patch[i, j]

                # Sub-grid for this cell
                x_start = x_center - 1500
                y_start = y_center - 1500

                for ii in range(self.n_fine):
                    for jj in range(self.n_fine):
                        x_fine[i * self.n_fine + ii, j * self.n_fine + jj] = x_start + ii * self.base_res
                        y_fine[i * self.n_fine + ii, j * self.n_fine + jj] = y_start + jj * self.base_res

        # Convert to lat/lon
        lons_fine, lats_fine = self.proj(x_fine, y_fine, inverse=True)

        # Compute distances for entire patch (vectorized)
        coords_fine = np.column_stack([x_fine.ravel(), y_fine.ravel()])

        dist_coast = None
        if coastline_tree is not None:
            dist_coast, _ = coastline_tree.query(coords_fine)
            dist_coast = dist_coast.reshape(n_fine_i, n_fine_j)

        dist_lake = None
        if lake_tree is not None:
            dist_lake, _ = lake_tree.query(coords_fine)
            dist_lake = dist_lake.reshape(n_fine_i, n_fine_j)

        # Create binary grids for each tier
        binary_grids = self._create_binary_grids_patch(
            n_fine_i, n_fine_j, n_i, n_j,
            i_start, j_start, terrain_var,
            dist_coast, dist_lake
        )

        # Apply stride patterns and collect points
        patch_lats, patch_lons = self._apply_strides(
            lats_fine, lons_fine, binary_grids
        )

        return patch_lats, patch_lons

    def _create_binary_grids_patch(self, n_fine_i, n_fine_j, n_i, n_j,
                                    i_start, j_start, terrain_var,
                                    dist_coast, dist_lake):
        """Create binary masks for all tiers for this patch"""
        binary_grids = [np.zeros((n_fine_i, n_fine_j), dtype=bool) for _ in range(6)]

        # Tier 0: Near coastline or lakes
        if dist_coast is not None:
            binary_grids[0] |= (dist_coast <= config.COASTLINE_BUFFER_KM * 1000)
        if dist_lake is not None:
            binary_grids[0] |= (dist_lake <= config.LAKE_BUFFER_KM * 1000)

        # Expand terrain variability to fine grid
        terrain_fine = np.repeat(np.repeat(
            terrain_var[i_start:i_start+n_i, j_start:j_start+n_j],
            self.n_fine, axis=0), self.n_fine, axis=1)

        # Tier 1: Extreme terrain OR 0.5-1.5km from coast
        binary_grids[1] |= (terrain_fine > config.TERRAIN_TIER1_THRESHOLD)
        if dist_coast is not None:
            binary_grids[1] |= ((dist_coast > 500) & (dist_coast <= 1500))

        # Tier 2: Rugged terrain OR 1.5-3km from coast
        binary_grids[2] |= (terrain_fine > config.TERRAIN_TIER2_THRESHOLD)
        if dist_coast is not None:
            binary_grids[2] |= ((dist_coast > 1500) & (dist_coast <= 3000))

        # Tier 3: Moderate terrain OR 3-6km from coast
        binary_grids[3] |= (terrain_fine > config.TERRAIN_TIER3_THRESHOLD)
        if dist_coast is not None:
            binary_grids[3] |= ((dist_coast > 3000) & (dist_coast <= 6000))

        # Tier 4: 6-12km from coast
        if dist_coast is not None:
            binary_grids[4] |= ((dist_coast > 6000) & (dist_coast <= 12000))

        # Tier 5: Background (everywhere)
        binary_grids[5][:, :] = True

        return binary_grids

    def _apply_strides(self, lats_fine, lons_fine, binary_grids):
        """Apply stride decimation to each tier and collect unique points"""
        points_set = set()

        for tier in range(6):
            stride = 2 ** tier
            binary = binary_grids[tier]

            # Decimate: take every stride-th point in both dimensions
            for i in range(0, binary.shape[0], stride):
                for j in range(0, binary.shape[1], stride):
                    if binary[i, j]:
                        lat = round(float(lats_fine[i, j]), 6)
                        lon = round(float(lons_fine[i, j]), 6)
                        points_set.add((lat, lon))

        # Convert to arrays
        if len(points_set) > 0:
            points = list(points_set)
            lats = np.array([p[0] for p in points])
            lons = np.array([p[1] for p in points])
            return lats, lons
        else:
            return np.array([]), np.array([])

if __name__ == '__main__':
    generator = BinaryGridGeneratorV3()
    lats, lons = generator.generate()
