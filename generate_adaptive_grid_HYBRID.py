"""
Hybrid approach: Fast cell classification + multi-tier generation
For each HRRR cell, generate MULTIPLE tiers based on distance and features
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

class HybridGridGenerator:
    def __init__(self):
        self.loader = DataLoader()

    def generate(self):
        print("=" * 70, flush=True)
        print(" HYBRID ADAPTIVE GRID", flush=True)
        print(" Multi-tier generation per cell for gradual transitions", flush=True)
        print("=" * 70, flush=True)

        start_time = time.time()

        # STEP 1: Load data
        print("\nSTEP 1: Loading data...", flush=True)
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

        # Convert to projected coordinates
        print("\nSTEP 2: Converting HRRR grid to projected coordinates...", flush=True)
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)

        # Build spatial indices
        print("\nSTEP 3: Building spatial indices...", flush=True)
        coastline_tree = self._build_coastline_tree()

        # Compute distance from each HRRR cell to coastline
        print("\nSTEP 4: Computing coastline distances...", flush=True)
        if coastline_tree is not None:
            hrrr_coords = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
            dist_to_coast, _ = coastline_tree.query(hrrr_coords)
            dist_to_coast = dist_to_coast.reshape(shape)
            print(f"  Distance range: {dist_to_coast.min():.0f}m to {dist_to_coast.max():.0f}m", flush=True)
        else:
            dist_to_coast = np.full(shape, np.inf)

        # Determine which tiers to generate for each cell
        print("\nSTEP 5: Determining tiers for each cell...", flush=True)
        cell_tiers = self._determine_cell_tiers(dist_to_coast, terrain_var, shape)

        # Generate points
        print("\nSTEP 6: Generating points...", flush=True)
        all_lats = []
        all_lons = []

        n_cells = shape[0] * shape[1]
        cells_processed = 0

        for i in range(shape[0]):
            for j in range(shape[1]):
                tiers = cell_tiers[i, j]

                if len(tiers) > 0:
                    cell_lats, cell_lons = self._generate_cell_points(
                        i, j, lats_hrrr, lons_hrrr, tiers
                    )
                    all_lats.append(cell_lats)
                    all_lons.append(cell_lons)

                cells_processed += 1
                if cells_processed % 100000 == 0:
                    pct = 100 * cells_processed / n_cells
                    total_pts = sum(len(arr) for arr in all_lats)
                    print(f"  Progress: {cells_processed:,}/{n_cells:,} cells ({pct:.1f}%), {total_pts:,} points", flush=True)

        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points: {len(lats_all):,}", flush=True)
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}", flush=True)
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%", flush=True)

        # Write output
        print("\nSTEP 7: Writing output...", flush=True)
        output_path = 'output/adaptive_grid_HYBRID.nc'
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats_all))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))

        lat_var[:] = lats_all
        lon_var[:] = lons_all

        ds.close()

        elapsed = time.time() - start_time
        print(f"\n✓ Complete in {elapsed/60:.1f} minutes", flush=True)
        print(f"  Output: {output_path}", flush=True)

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
            print(f"  Coastline: {len(coast_coords):,} points indexed", flush=True)
            return cKDTree(np.array(coast_coords))
        return None

    def _determine_cell_tiers(self, dist_to_coast, terrain_var, shape):
        """Determine which tiers to generate for each HRRR cell"""
        cell_tiers = np.empty(shape, dtype=object)

        for i in range(shape[0]):
            for j in range(shape[1]):
                tiers = set()

                dist = dist_to_coast[i, j]
                terr = terrain_var[i, j]

                # Distance-based tier assignment (gradual transitions)
                # More conservative to control point budget
                if dist <= 1500:  # 0-1.5km: Tier 0 + background
                    tiers.add(0)
                    tiers.add(5)
                elif dist <= 3000:  # 1.5-3km: Tier 1 + background
                    tiers.add(1)
                    tiers.add(5)
                elif dist <= 6000:  # 3-6km: Tier 2 + background
                    tiers.add(2)
                    tiers.add(5)
                elif dist <= 12000:  # 6-12km: Tier 3 + background
                    tiers.add(3)
                    tiers.add(5)
                else:  # >12km: Tier 5 only
                    tiers.add(5)

                # Terrain-based tiers (add to existing, conservative)
                if terr > config.TERRAIN_TIER1_THRESHOLD:
                    tiers.add(1)
                    tiers.add(5)
                elif terr > config.TERRAIN_TIER2_THRESHOLD:
                    tiers.add(2)
                    tiers.add(5)
                elif terr > config.TERRAIN_TIER3_THRESHOLD:
                    tiers.add(3)
                    tiers.add(5)

                cell_tiers[i, j] = sorted(list(tiers))

        return cell_tiers

    def _generate_cell_points(self, i, j, lats_hrrr, lons_hrrr, tiers):
        """Generate points for all specified tiers in this cell"""
        lat_center = lats_hrrr[i, j]
        lon_center = lons_hrrr[i, j]
        x_center, y_center = self.proj(lon_center, lat_center)

        points_set = set()

        for tier in tiers:
            stride = 2 ** tier
            n_points = 32 // stride

            # Generate sub-grid
            x_start = x_center - 1500
            y_start = y_center - 1500

            x_offsets = np.arange(n_points) * config.TIER_RESOLUTIONS[0] * stride
            y_offsets = np.arange(n_points) * config.TIER_RESOLUTIONS[0] * stride

            xx, yy = np.meshgrid(x_start + x_offsets, y_start + y_offsets)

            # Convert to lat/lon
            lons_tier, lats_tier = self.proj(xx.ravel(), yy.ravel(), inverse=True)

            # Add to set (automatic deduplication)
            for lat, lon in zip(lats_tier, lons_tier):
                points_set.add((round(lat, 6), round(lon, 6)))

        # Convert to arrays
        if len(points_set) > 0:
            points = list(points_set)
            lats = np.array([p[0] for p in points])
            lons = np.array([p[1] for p in points])
            return lats, lons
        else:
            return np.array([]), np.array([])

if __name__ == '__main__':
    generator = HybridGridGenerator()
    lats, lons = generator.generate()
