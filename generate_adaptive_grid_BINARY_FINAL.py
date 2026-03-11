"""
Binary grid approach - FINAL
Per-pixel classification at 93.75m resolution (no blockiness)
Vectorized for performance
Uses actual great-circle distances for HRRR cell spacing

Tier 0 (93.75m): Ski resorts (2km buffer) - HIGHEST RESOLUTION
Tier 1 (187.5m): Coastlines, lakes, extreme terrain
Tier 2 (375m): High-density urban, very rugged terrain
Tier 3 (750m): Suburban areas, roads, parks, forests, rugged terrain
Tier 4 (1.5km): Moderate terrain
Tier 5 (3km): Background
"""
import numpy as np
import geopandas as gpd
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from scipy.spatial import cKDTree
import time
import config
from generate_adaptive_grid import DataLoader, TerrainAnalyzer

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in meters using Haversine formula"""
    R = 6371000  # Earth radius in meters

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c

class BinaryGridGeneratorFinal:
    def __init__(self, test_region=None):
        """
        Initialize grid generator

        Args:
            test_region: Optional region for testing
                'nw_us': Northwest US (lat 42-49, lon -125 to -115)
                None: Full CONUS domain
        """
        self.loader = DataLoader()
        self.base_res = 93.75  # meters (nominal spacing at finest tier)
        self.n_fine = 32  # points per HRRR cell dimension
        self.test_region = test_region

    def generate(self):
        print("=" * 70, flush=True)
        print(" BINARY GRID - FINAL (PER-PIXEL CLASSIFICATION)", flush=True)
        print(" True per-pixel tier assignment at 93.75m", flush=True)
        print("=" * 70, flush=True)

        start_time = time.time()

        # STEP 1: Load data
        print("\nSTEP 1: Loading data...", flush=True)
        self.loader.load_all()

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        terrain = self.loader.hrrr_grid['terrain']
        shape = lats_hrrr.shape

        print(f"  HRRR shape: {shape[0]}×{shape[1]} = {shape[0]*shape[1]:,} cells", flush=True)

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

        # Compute actual cell spacings and orientation vectors (precompute once)
        print("\nSTEP 2b: Precomputing HRRR cell geometry...", flush=True)
        print("  Computing spacing and orientation for each cell...", flush=True)

        # Precompute vectors to neighbors in projected coordinates
        # These encode both distance and direction in one array
        dx_east = np.zeros_like(x_hrrr)
        dy_east = np.zeros_like(x_hrrr)
        dx_north = np.zeros_like(x_hrrr)
        dy_north = np.zeros_like(x_hrrr)

        # East vectors (to j+1 neighbor)
        dx_east[:, :-1] = x_hrrr[:, 1:] - x_hrrr[:, :-1]
        dy_east[:, :-1] = y_hrrr[:, 1:] - y_hrrr[:, :-1]
        dx_east[:, -1] = dx_east[:, -2]  # Replicate last column
        dy_east[:, -1] = dy_east[:, -2]

        # North vectors (to i+1 neighbor)
        dx_north[:-1, :] = x_hrrr[1:, :] - x_hrrr[:-1, :]
        dy_north[:-1, :] = y_hrrr[1:, :] - y_hrrr[:-1, :]
        dx_north[-1, :] = dx_north[-2, :]  # Replicate last row
        dy_north[-1, :] = dy_north[-2, :]

        # Compute actual distances
        dist_east = np.sqrt(dx_east**2 + dy_east**2)
        dist_north = np.sqrt(dx_north**2 + dy_north**2)

        print(f"  East spacing: {dist_east.min():.1f}m to {dist_east.max():.1f}m (mean: {dist_east.mean():.1f}m)", flush=True)
        print(f"  North spacing: {dist_north.min():.1f}m to {dist_north.max():.1f}m (mean: {dist_north.mean():.1f}m)", flush=True)

        # Build coastline buffers
        print("\nSTEP 3: Building coastline buffers...", flush=True)
        coastline_buffers, buffer_bounds = self._build_coastline_buffers()

        # Process in patches
        print("\nSTEP 4: Processing in patches...", flush=True)
        patch_size = 5  # 5×5 HRRR cells = 160×160 fine points per patch

        n_patches_i = int(np.ceil(shape[0] / patch_size))
        n_patches_j = int(np.ceil(shape[1] / patch_size))
        total_patches = n_patches_i * n_patches_j

        print(f"  Patch size: {patch_size}×{patch_size} HRRR cells = {patch_size*32}×{patch_size*32} fine points", flush=True)
        print(f"  Total patches: {total_patches}", flush=True)

        all_lats = []
        all_lons = []
        total_points = 0
        patch_count = 0

        for i_patch in range(n_patches_i):
            for j_patch in range(n_patches_j):
                i_start = i_patch * patch_size
                i_end = min(i_start + patch_size, shape[0])
                j_start = j_patch * patch_size
                j_end = min(j_start + patch_size, shape[1])

                # Check if patch is in test region
                if self.test_region is not None:
                    # Get patch center lat/lon
                    i_center = (i_start + i_end) // 2
                    j_center = (j_start + j_end) // 2
                    patch_lat = lats_hrrr[i_center, j_center]
                    patch_lon = lons_hrrr[i_center, j_center]

                    # Northwest US: lat 42-49, lon -125 to -115
                    if self.test_region == 'nw_us':
                        if not (42 <= patch_lat <= 49 and -125 <= patch_lon <= -115):
                            continue  # Skip this patch

                # Process this patch
                patch_lats, patch_lons = self._process_patch(
                    i_start, i_end, j_start, j_end,
                    x_hrrr, y_hrrr, terrain_var, coastline_buffers, buffer_bounds,
                    dx_east, dy_east, dx_north, dy_north
                )

                all_lats.append(patch_lats)
                all_lons.append(patch_lons)
                total_points += len(patch_lats)

                patch_count += 1
                if patch_count % 500 == 0:
                    pct = 100 * patch_count / total_patches
                    print(f"  Progress: {patch_count}/{total_patches} patches ({pct:.1f}%), {total_points:,} points", flush=True)

        lats_all = np.concatenate(all_lats)
        lons_all = np.concatenate(all_lons)

        print(f"\nTotal points: {len(lats_all):,}", flush=True)
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}", flush=True)
        diff_pct = 100 * (len(lats_all) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%", flush=True)

        # Write output
        print("\nSTEP 5: Writing output...", flush=True)
        if self.test_region:
            output_path = f'output/adaptive_grid_BINARY_{self.test_region}.nc'
        else:
            output_path = 'output/adaptive_grid_BINARY.nc'
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

    def _compute_cell_spacings(self, lats_hrrr, lons_hrrr):
        """Compute actual great-circle spacing for each HRRR cell"""
        ny, nx = lats_hrrr.shape

        # X-direction: spacing to eastern neighbor
        dx_spacing = np.zeros((ny, nx))
        dx_spacing[:, :-1] = haversine_distance(
            lats_hrrr[:, :-1], lons_hrrr[:, :-1],
            lats_hrrr[:, 1:], lons_hrrr[:, 1:]
        )
        # Replicate last column
        dx_spacing[:, -1] = dx_spacing[:, -2]

        # Y-direction: spacing to northern neighbor
        dy_spacing = np.zeros((ny, nx))
        dy_spacing[:-1, :] = haversine_distance(
            lats_hrrr[:-1, :], lons_hrrr[:-1, :],
            lats_hrrr[1:, :], lons_hrrr[1:, :]
        )
        # Replicate last row
        dy_spacing[-1, :] = dy_spacing[-2, :]

        return dx_spacing, dy_spacing

    def _build_coastline_buffers(self):
        """Create buffered geometries for all features at different tier distances

        Starting at 93.75m resolution (tier 0) for ski resorts, with doubled transition distances
        Includes: ski resorts, ocean coastlines, lakes, urban areas, roads
        """
        from shapely.ops import unary_union
        import geopandas as gpd

        all_geometries = {0: [], 1: [], 2: [], 3: []}  # Collect geometries for each tier

        # 0. SKI RESORTS - Tier 0 (93.75m resolution - HIGHEST, 2km buffer around points)
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None and len(ski_gdf) > 0:
            print(f"  Ski resorts: {len(ski_gdf)} locations (Tier 0: 2km buffer, HIGHEST RESOLUTION 93.75m)", flush=True)
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            # Buffer ski resort points by 2km
            ski_buffered = ski_proj.geometry.buffer(2000)
            all_geometries[0].extend(ski_buffered.tolist())

        # 1. OCEAN COASTLINES - Tier 1 (187.5m resolution, 750m buffer)
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant_coastlines = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            print(f"  Ocean: {len(significant_coastlines)} segments (>{50}km), {significant_coastlines['length_km'].sum():.0f}km", flush=True)
            coast_proj = significant_coastlines.to_crs(self.hrrr_crs)
            all_geometries[1].extend(coast_proj.geometry.tolist())

        # 2. LAKES - Tier 1 (187.5m resolution, 750m buffer)
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None and len(lakes_gdf) > 0:
            print(f"  Lakes: {len(lakes_gdf)} features, {lakes_gdf['coastline_length_km'].sum():.0f}km", flush=True)
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            # Use lake boundaries as lines
            all_geometries[1].extend(lakes_proj.geometry.boundary.tolist())

        # 3. HIGH-DENSITY URBAN AREAS - Tier 2 (375m resolution around perimeters)
        urban_high = self.loader.data.get('high_density_urban')
        if urban_high is not None and len(urban_high) > 0:
            urban_proj = urban_high.to_crs(self.hrrr_crs)
            all_geometries[2].extend(urban_proj.geometry.boundary.tolist())
            print(f"  High-density urban: {len(urban_high)} areas (Tier 2: 375m resolution)", flush=True)

        # 4. SUBURBAN AREAS (low-density urban) - Tier 3 (750m resolution around perimeters)
        urban_low = self.loader.data.get('urban')
        if urban_low is not None and len(urban_low) > 0:
            cluster_proj = urban_low.to_crs(self.hrrr_crs)
            all_geometries[3].extend(cluster_proj.geometry.boundary.tolist())
            print(f"  Suburban areas: {len(urban_low)} areas (Tier 3: 750m resolution)", flush=True)

        # 5. PRIMARY ROADS - Tier 3 (750m resolution, 500m buffer on each side)
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None and len(roads_gdf) > 0:
            print(f"  Roads: {len(roads_gdf)} segments (Tier 3: 500m buffer)", flush=True)
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            all_geometries[3].extend(roads_proj.geometry.tolist())

        # 6. NATIONAL PARKS - Tier 3 (750m resolution, full areas)
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None and len(parks_gdf) > 0:
            print(f"  National Parks: {len(parks_gdf)} NPS units (Tier 3: 750m resolution)", flush=True)
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            # Use full park areas (not just boundaries) for higher resolution throughout
            all_geometries[3].extend(parks_proj.geometry.tolist())

        # 7. NATIONAL FORESTS - Tier 3 (750m resolution, full areas)
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None and len(forests_gdf) > 0:
            print(f"  National Forests: {len(forests_gdf)} forest units (Tier 3: 750m resolution)", flush=True)
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            # Use full forest areas (not just boundaries) for higher resolution throughout
            all_geometries[3].extend(forests_proj.geometry.tolist())

        # Create unified buffers for each tier
        print("  Creating and unifying buffers (this may take 2-3 minutes)...", flush=True)
        buffers = {}

        if all_geometries[0]:  # Tier 0: ski resorts only (HIGHEST RESOLUTION - 93.75m)
            geom_gdf = gpd.GeoSeries(all_geometries[0], crs=self.hrrr_crs)
            # Ski resorts already have 2km buffer applied, use directly
            buffers[0] = geom_gdf.union_all()

        if all_geometries[1]:  # Tier 1: coastlines + lakes (187.5m resolution)
            geom_gdf = gpd.GeoSeries(all_geometries[1], crs=self.hrrr_crs)
            # Coastlines and lakes get 750m buffer for tier 1
            buffers[1] = geom_gdf.buffer(750).union_all()
            buffers[2] = geom_gdf.buffer(1500).union_all()
            buffers[3] = geom_gdf.buffer(3000).union_all()
            buffers[4] = geom_gdf.buffer(6000).union_all()
            buffers[5] = geom_gdf.buffer(12000).union_all()

        if all_geometries[2]:  # Tier 2: high-density urban areas only
            geom_gdf = gpd.GeoSeries(all_geometries[2], crs=self.hrrr_crs)
            urban_geom = geom_gdf.union_all()
            # Merge with tier 2 buffer from coastlines
            if 2 in buffers:
                buffers[2] = buffers[2].union(urban_geom)
            else:
                buffers[2] = urban_geom

        if all_geometries[3]:  # Tier 3: suburban + roads + parks + forests
            # NOTE: all_geometries[3] contains:
            #   - Suburban areas (Polygons) - boundaries already extracted
            #   - Roads (LineStrings) - need 500m buffer
            #   - Parks (Polygons) - already full areas, no buffer needed
            #   - Forests (Polygons) - already full areas, no buffer needed
            # So we buffer everything by 500m (safe for all geometry types)
            geom_gdf = gpd.GeoSeries(all_geometries[3], crs=self.hrrr_crs)
            tier3_features = geom_gdf.buffer(500).union_all()
            # Merge with tier 3 distance buffer from coastlines
            if 3 in buffers:
                buffers[3] = buffers[3].union(tier3_features)
            else:
                buffers[3] = tier3_features

        # Simplify buffers for faster intersection (1m tolerance)
        print("  Simplifying buffers...", flush=True)
        buffer_bounds = {}
        for tier in buffers:
            buffers[tier] = buffers[tier].simplify(tolerance=1.0, preserve_topology=True)
            buffer_bounds[tier] = buffers[tier].bounds  # Store bounds for spatial filtering

        print(f"  All feature buffers created for tiers 0-5", flush=True)
        return (buffers, buffer_bounds) if buffers else (None, None)

    def _bounds_intersect(self, bounds1, bounds2):
        """Check if two bounding boxes intersect

        bounds format: (minx, miny, maxx, maxy)
        """
        return not (bounds1[2] < bounds2[0] or  # bounds1 is left of bounds2
                    bounds1[0] > bounds2[2] or  # bounds1 is right of bounds2
                    bounds1[3] < bounds2[1] or  # bounds1 is below bounds2
                    bounds1[1] > bounds2[3])    # bounds1 is above bounds2

    def _process_patch(self, i_start, i_end, j_start, j_end,
                      x_hrrr, y_hrrr, terrain_var, coastline_buffers, buffer_bounds,
                      dx_east, dy_east, dx_north, dy_north):
        """Process a patch using geometric intersection with spatial filtering"""

        # Patch dimensions in HRRR cells
        n_i = i_end - i_start
        n_j = j_end - j_start

        # Extract patch data
        x_patch = x_hrrr[i_start:i_end, j_start:j_end]
        y_patch = y_hrrr[i_start:i_end, j_start:j_end]
        terrain_patch = terrain_var[i_start:i_end, j_start:j_end]

        # Extract precomputed geometry vectors
        dx_east_patch = dx_east[i_start:i_end, j_start:j_end]
        dy_east_patch = dy_east[i_start:i_end, j_start:j_end]
        dx_north_patch = dx_north[i_start:i_end, j_start:j_end]
        dy_north_patch = dy_north[i_start:i_end, j_start:j_end]

        # Repeat values to create fine grid structure
        x_repeated = np.repeat(np.repeat(x_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        y_repeated = np.repeat(np.repeat(y_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        dx_east_repeated = np.repeat(np.repeat(dx_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_east_repeated = np.repeat(np.repeat(dy_east_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dx_north_repeated = np.repeat(np.repeat(dx_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)
        dy_north_repeated = np.repeat(np.repeat(dy_north_patch, self.n_fine, axis=0), self.n_fine, axis=1)

        n_fine_i = n_i * self.n_fine
        n_fine_j = n_j * self.n_fine

        # Create fine grid indices
        i_fine_idx = np.arange(n_fine_i) % self.n_fine
        j_fine_idx = np.arange(n_fine_j) % self.n_fine
        jj_fine_idx, ii_fine_idx = np.meshgrid(j_fine_idx, i_fine_idx)

        # Compute fractional offsets (0 to 1) in east and north directions
        # Using i/n_fine creates n_fine points spanning from 0 to (n_fine-1)/n_fine
        # This leaves a gap of 1/n_fine between adjacent cells
        frac_east = jj_fine_idx / self.n_fine - 0.5  # -0.5 to ~0.47 (for n_fine=32)
        frac_north = ii_fine_idx / self.n_fine - 0.5

        # Apply offsets directly using precomputed vectors
        # These vectors already encode both distance and direction!
        x_fine = x_repeated + frac_east * dx_east_repeated + frac_north * dx_north_repeated
        y_fine = y_repeated + frac_east * dy_east_repeated + frac_north * dy_north_repeated

        # Convert to lat/lon
        lons_fine, lats_fine = self.proj(x_fine, y_fine, inverse=True)

        # Expand terrain to fine grid
        terrain_fine = np.repeat(np.repeat(terrain_patch, self.n_fine, axis=0),
                                 self.n_fine, axis=1)

        # Compute patch bounds for spatial filtering
        patch_bounds = (x_fine.min(), y_fine.min(), x_fine.max(), y_fine.max())

        # Use geometric intersection to determine which points are in coastline buffers
        coast_masks = {}
        if coastline_buffers is not None:
            from shapely.geometry import Point
            from shapely.prepared import prep
            from shapely import vectorized

            # Create Point geometries for fine grid (flatten for efficiency)
            x_flat = x_fine.ravel()
            y_flat = y_fine.ravel()

            # For each tier buffer, check intersection using prepared geometry
            for tier in [0, 1, 2, 3, 4, 5]:  # Tiers 0-5 have feature buffers
                if tier in coastline_buffers:
                    # SPATIAL FILTER: Check if buffer bounds intersect patch bounds
                    if buffer_bounds and tier in buffer_bounds:
                        if not self._bounds_intersect(patch_bounds, buffer_bounds[tier]):
                            # This tier's features don't overlap this patch - skip expensive check
                            continue

                    buffer_geom = coastline_buffers[tier]

                    # Bounding box pre-filter for this specific tier
                    tier_bounds = buffer_geom.bounds
                    in_bounds = (
                        (x_flat >= tier_bounds[0]) & (x_flat <= tier_bounds[2]) &
                        (y_flat >= tier_bounds[1]) & (y_flat <= tier_bounds[3])
                    )

                    # Prepare geometry for faster repeated queries
                    prepared_geom = prep(buffer_geom)

                    # Initialize mask
                    mask_flat = np.zeros(len(x_flat), dtype=bool)

                    # Only check points within bounding box
                    if in_bounds.any():
                        x_check = x_flat[in_bounds]
                        y_check = y_flat[in_bounds]

                        # Use vectorized contains check (much faster!)
                        try:
                            # Try vectorized operation (shapely 2.0+)
                            mask_check = vectorized.contains(buffer_geom, x_check, y_check)
                        except:
                            # Fallback to prepared geometry with list comprehension
                            mask_check = np.array([prepared_geom.contains(Point(x, y))
                                                for x, y in zip(x_check, y_check)])

                        mask_flat[in_bounds] = mask_check

                    coast_masks[tier] = mask_flat.reshape(n_fine_i, n_fine_j)

        # Create binary grids for each tier using geometric intersection
        binary_grids = self._create_binary_grids_intersection(
            n_fine_i, n_fine_j, coast_masks, terrain_fine
        )

        # Apply stride patterns
        patch_lats, patch_lons = self._apply_strides(
            lats_fine, lons_fine, binary_grids
        )

        return patch_lats, patch_lons

    def _create_binary_grids_intersection(self, n_fine_i, n_fine_j, coast_masks, terrain_fine):
        """Create binary masks using UNION of all criteria for each tier

        CRITICAL FIX: Use union approach instead of exclusion
        - Each tier has independent criteria (features + terrain)
        - Then pick finest tier that applies using priority hierarchy
        - This ensures ski resorts, roads, etc. get their resolution regardless of coastline proximity

        Starting at tier 0 (93.75m resolution) for ski resorts with doubled transition distances

        coast_masks: dict with keys 0-5, values are boolean arrays indicating
                     which points intersect with each tier's feature buffers
                     (includes ski resorts, coastlines, lakes, urban, roads merged by tier)
        """
        binary_grids = [np.zeros((n_fine_i, n_fine_j), dtype=bool) for _ in range(6)]

        if coast_masks:
            # STEP 1: Create full criteria maps for each tier (UNION of all features + terrain)
            # Tier 0: 93.75m resolution (HIGHEST RESOLUTION)
            #   - Ski resorts (2km buffer) ← HIGHEST PRIORITY
            tier0_full = np.zeros((n_fine_i, n_fine_j), dtype=bool)
            if 0 in coast_masks:
                tier0_full |= coast_masks[0]

            # Tier 1: 187.5m resolution
            #   - Within 750m of ocean/lakes
            #   - Extreme terrain (>800m std dev)
            tier1_full = np.zeros((n_fine_i, n_fine_j), dtype=bool)
            if 1 in coast_masks:
                tier1_full |= coast_masks[1]
            tier1_full |= (terrain_fine > 800)

            # Tier 2: 375m resolution
            #   - Within 1500m of ocean/lakes
            #   - High-density urban areas
            #   - Very rugged terrain (>600m std dev)
            tier2_full = np.zeros((n_fine_i, n_fine_j), dtype=bool)
            if 2 in coast_masks:
                tier2_full |= coast_masks[2]
            tier2_full |= (terrain_fine > 600)

            # Tier 3: 750m resolution
            #   - Within 3km of ocean/lakes
            #   - Suburban areas (low-density urban)
            #   - Primary roads (500m buffer)
            #   - National Parks (full areas)
            #   - National Forests (full areas)
            #   - Rugged terrain (>300m std dev)
            tier3_full = np.zeros((n_fine_i, n_fine_j), dtype=bool)
            if 3 in coast_masks:
                tier3_full |= coast_masks[3]
            tier3_full |= (terrain_fine > 300)

            # Tier 4: 1.5km resolution
            #   - Within 6km of ocean/lakes
            #   - Moderate terrain (>150m std dev)
            tier4_full = np.zeros((n_fine_i, n_fine_j), dtype=bool)
            if 4 in coast_masks:
                tier4_full |= coast_masks[4]
            tier4_full |= (terrain_fine > 150)

            # Tier 5: 3km resolution (background)
            #   - Within 12km of ocean/lakes
            #   - Everything else
            tier5_full = np.ones((n_fine_i, n_fine_j), dtype=bool)  # Always true

            # STEP 2: Assign tiers using priority (finest resolution wins)
            # This ensures features get their intended resolution, not overridden by coarser tiers
            binary_grids[0] = tier0_full
            binary_grids[1] = tier1_full & ~tier0_full  # Tier 1 only where tier 0 doesn't apply
            binary_grids[2] = tier2_full & ~tier1_full & ~tier0_full
            binary_grids[3] = tier3_full & ~tier2_full & ~tier1_full & ~tier0_full
            binary_grids[4] = tier4_full & ~tier3_full & ~tier2_full & ~tier1_full & ~tier0_full
            binary_grids[5] = tier5_full & ~tier4_full & ~tier3_full & ~tier2_full & ~tier1_full & ~tier0_full

        else:
            # No feature data - terrain only
            tier1_full = (terrain_fine > 800)
            tier2_full = (terrain_fine > 600)
            tier3_full = (terrain_fine > 300)
            tier4_full = (terrain_fine > 150)
            tier5_full = np.ones((n_fine_i, n_fine_j), dtype=bool)

            binary_grids[0] = np.zeros((n_fine_i, n_fine_j), dtype=bool)  # No tier 0 without features
            binary_grids[1] = tier1_full
            binary_grids[2] = tier2_full & ~tier1_full
            binary_grids[3] = tier3_full & ~tier2_full & ~tier1_full
            binary_grids[4] = tier4_full & ~tier3_full & ~tier2_full & ~tier1_full
            binary_grids[5] = tier5_full & ~tier4_full & ~tier3_full & ~tier2_full & ~tier1_full

        return binary_grids

    def _create_binary_grids(self, n_fine_i, n_fine_j, dist_coast, terrain_fine):
        """Create binary masks for each tier based on per-pixel distances

        Uses 375m strip on each side of coastline for finest resolution.
        Rapid transition to coarser resolution.
        Conservative terrain thresholds to control point count.
        """
        binary_grids = [np.zeros((n_fine_i, n_fine_j), dtype=bool) for _ in range(6)]

        if dist_coast is not None:
            # Tier 0: ±375m strip around significant coastlines (ocean + major water bodies)
            # NARROWER band to control point count
            binary_grids[0] = (dist_coast <= 375)

            # Tier 1: 375-750m from coast OR ONLY most extreme terrain
            binary_grids[1] = ((dist_coast > 375) & (dist_coast <= 750))
            binary_grids[1] |= (terrain_fine > 800)  # Raised from 600m - very conservative

            # Tier 2: 750-1500m from coast OR very rugged terrain
            binary_grids[2] = ((dist_coast > 750) & (dist_coast <= 1500))
            binary_grids[2] |= (terrain_fine > 600)  # Raised from 400m

            # Tier 3: 1500-3km from coast OR rugged terrain
            binary_grids[3] = ((dist_coast > 1500) & (dist_coast <= 3000))
            binary_grids[3] |= (terrain_fine > 300)  # Raised from 100m

            # Tier 4: 3-6km from coast OR moderate terrain
            binary_grids[4] = ((dist_coast > 3000) & (dist_coast <= 6000))
            binary_grids[4] |= (terrain_fine > 150)  # Added threshold

            # Tier 5: Everywhere (background)
            binary_grids[5][:, :] = True
        else:
            # No coastline - terrain only (even more conservative)
            binary_grids[1] = (terrain_fine > 800)
            binary_grids[2] = (terrain_fine > 600)
            binary_grids[3] = (terrain_fine > 300)
            binary_grids[4] = (terrain_fine > 150)
            binary_grids[5][:, :] = True

        return binary_grids

    def _apply_strides(self, lats_fine, lons_fine, binary_grids):
        """Apply stride decimation and collect unique points (element-wise maximum)"""
        points_set = set()

        for tier in range(6):
            stride = 2 ** tier
            binary = binary_grids[tier]

            # Apply stride: every stride-th point in both dimensions
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
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate adaptive grid with binary tier classification'
    )
    parser.add_argument(
        '--test-region',
        choices=['nw_us', 'full'],
        default='full',
        help='Region to process: nw_us (Northwest US, lat 42-49, lon -125 to -115) or full (entire CONUS)'
    )

    args = parser.parse_args()

    test_region = args.test_region if args.test_region != 'full' else None

    print(f"\n{'='*70}")
    if test_region:
        print(f" TEST MODE: Processing {test_region.upper()} only")
        print(f" NaN will be saved for rest of domain")
    else:
        print(f" FULL DOMAIN: Processing entire CONUS")
    print(f"{'='*70}\n")

    generator = BinaryGridGeneratorFinal(test_region=test_region)
    lats, lons = generator.generate()
