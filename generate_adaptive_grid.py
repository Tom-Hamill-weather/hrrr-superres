"""
Adaptive Grid Point Generation for XGBoost Weather Downscaling

Generates an irregular grid with ~50M points distributed according to:
- Tier 1 (90m): Recreation areas, coastlines, major lakes
- Tier 2 (270m): Terrain variability, secondary recreation
- Tier 3 (810m): Suburban/exurban, highways
- Tier 4 (3km): Remote/agricultural areas
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
from scipy.ndimage import generic_filter
from scipy.spatial import cKDTree
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import config

class DataLoader:
    """Load and preprocess all geospatial datasets"""

    def __init__(self):
        self.data = {}
        self.hrrr_grid = None

    def load_hrrr_grid(self):
        """Load HRRR grid and terrain"""
        print("\nLoading HRRR grid and terrain...")
        hrrr_dir = os.path.join(config.DATA_DIR, 'hrrr')

        try:
            lats = np.load(os.path.join(hrrr_dir, 'hrrr_lats.npy'))
            lons = np.load(os.path.join(hrrr_dir, 'hrrr_lons.npy'))
            terrain = np.load(os.path.join(hrrr_dir, 'hrrr_terrain.npy'))

            self.hrrr_grid = {
                'lats': lats,
                'lons': lons,
                'terrain': terrain,
                'shape': lats.shape
            }

            print(f"✓ HRRR grid loaded: {lats.shape}")
            print(f"  Lat range: {lats.min():.2f} to {lats.max():.2f}")
            print(f"  Lon range: {lons.min():.2f} to {lons.max():.2f}")
            print(f"  Terrain range: {terrain.min():.1f} to {terrain.max():.1f} m")

        except Exception as e:
            print(f"✗ Error loading HRRR grid: {e}")
            raise

    def load_coastline(self):
        """Load coastline data - GSHHG full resolution from Basemap"""
        print("\nLoading coastline data...")
        try:
            from mpl_toolkits.basemap import Basemap
            from shapely.geometry import LineString

            # Create a Basemap instance covering CONUS with FULL resolution
            # This will load GSHHG full resolution coastline data (~100m accuracy)
            print("  Extracting GSHHG full resolution coastlines from Basemap...")
            print("  (This may take a minute to load the full-resolution database)")

            m = Basemap(
                projection='merc',
                llcrnrlat=20,
                urcrnrlat=55,
                llcrnrlon=-135,
                urcrnrlon=-60,
                resolution='f'  # Full resolution (~100m accuracy)
            )

            # Extract coastline segments and convert to lat/lon
            coastlines = []
            for seg in m.coastsegs:
                if len(seg) >= 2:  # Valid line segment
                    # seg is in map projection coordinates, convert to lat/lon
                    seg_array = np.array(seg)
                    lons, lats = m(seg_array[:, 0], seg_array[:, 1], inverse=True)
                    # Create LineString from lat/lon coordinates
                    coastlines.append(LineString(zip(lons, lats)))

            # Create GeoDataFrame in geographic coordinates (WGS84)
            self.data['ocean_coastline'] = gpd.GeoDataFrame(
                geometry=coastlines,
                crs='EPSG:4326'  # WGS84 lat/lon
            )

            # Calculate coastline length for each segment
            # Reproject to meters (Web Mercator) to calculate length
            self.data['ocean_coastline']['length_km'] = (
                self.data['ocean_coastline'].to_crs('EPSG:3857').geometry.length / 1000
            )

            total_length = self.data['ocean_coastline']['length_km'].sum()
            total_points = sum(len(seg) for seg in m.coastsegs)
            print(f"✓ GSHHG full-resolution coastline loaded")
            print(f"  {len(coastlines)} segments, {total_points:,} points, {total_length:.0f}km total")

        except Exception as e:
            print(f"✗ Error loading coastline: {e}")
            import traceback
            traceback.print_exc()

    def load_lakes(self):
        """Load lake data - filter by CONUS region and coastline length threshold"""
        print("\nLoading lakes data...")
        try:
            lakes_file = os.path.join(config.DATA_DIR, 'natural_earth', 'ne_10m_lakes.shp')
            if os.path.exists(lakes_file):
                lakes_gdf = gpd.read_file(lakes_file)

                # Filter for CONUS region first (approximate bounding box)
                conus_lakes = lakes_gdf.cx[-135:-60, 21:53].copy()

                # Calculate coastline length for each lake
                conus_lakes['coastline_length_km'] = (
                    conus_lakes.to_crs('EPSG:3857').geometry.boundary.length / 1000
                )

                # Filter by coastline length (30km threshold for significant lakes)
                min_coastline_km = 30
                significant_lakes = conus_lakes[conus_lakes['coastline_length_km'] > min_coastline_km].copy()

                # Extract only the lake boundaries (shorelines), not the filled polygons
                # This ensures buffering creates rings around shores, not filled areas
                significant_lakes['geometry'] = significant_lakes.geometry.boundary

                self.data['lakes'] = significant_lakes

                total_length = self.data['lakes']['coastline_length_km'].sum()
                print(f"✓ Lakes loaded: {len(self.data['lakes'])} CONUS features (>{min_coastline_km}km coastline)")
                print(f"  Total lake coastline: {total_length:.0f}km")
            else:
                print(f"⚠ Lakes file not found: {lakes_file}")
        except Exception as e:
            print(f"✗ Error loading lakes: {e}")

    def load_protected_areas(self):
        """Load protected areas (National & State Parks only)"""
        print("\nLoading protected areas (National & State Parks)...")
        padus_dir = os.path.join(config.DATA_DIR, 'padus')

        # Try multiple possible file locations
        possible_files = [
            os.path.join(padus_dir, 'PADUS4_0_Combined.shp'),
            os.path.join(padus_dir, 'PADUS_Combined.shp'),
            os.path.join(padus_dir, 'PAD_US', 'PADUS4_0_Combined.shp'),
            os.path.join(padus_dir, 'PADUS3_0Combined.shp')
        ]

        for padus_file in possible_files:
            if os.path.exists(padus_file):
                try:
                    print(f"  Loading PAD-US from: {padus_file}")
                    # Load and filter for National & State Parks
                    padus_gdf = gpd.read_file(padus_file)

                    # Filter for National Parks
                    national_parks = padus_gdf[
                        padus_gdf['Des_Tp'].isin(['NP', 'NM', 'NRA', 'NSR', 'NS']) &
                        padus_gdf['GAP_Sts'].isin([1, 2])
                    ]

                    # Filter for State Parks
                    state_parks = padus_gdf[
                        (padus_gdf['Des_Tp'].isin(['SP', 'SRA'])) &
                        (padus_gdf['GAP_Sts'].isin([1, 2]))
                    ]

                    # Combine and convert to centroids (for point-based classification)
                    parks_combined = pd.concat([national_parks, state_parks], ignore_index=True)

                    # For large polygons, use centroids to speed up spatial operations
                    parks_combined['geometry'] = parks_combined.geometry.centroid

                    self.data['parks'] = gpd.GeoDataFrame(parks_combined)

                    print(f"✓ Protected areas loaded: {len(self.data['parks'])} parks")
                    print(f"  National: {len(national_parks)}, State: {len(state_parks)}")
                    return

                except Exception as e:
                    print(f"✗ Error loading {padus_file}: {e}")
                    continue

        print(f"⚠ PAD-US data not found. Download from:")
        print("  https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download")
        print(f"  Extract to: {padus_dir}")

    def load_urban_areas(self):
        """Load urban/suburban boundaries, split by urbanized area type and size"""
        print("\nLoading urban areas...")
        try:
            urban_file = os.path.join(config.DATA_DIR, 'urban', 'tl_2020_us_uac10.shp')
            if os.path.exists(urban_file):
                urban_gdf = gpd.read_file(urban_file)

                # Split by Census urban type
                # UATYP10: U = Urbanized Area (≥50k people, major metros)
                #          C = Urban Cluster (2.5k-50k people, towns)
                if 'UATYP10' in urban_gdf.columns:
                    urbanized_areas = urban_gdf[urban_gdf['UATYP10'] == 'U']  # Major metros
                    urban_clusters = urban_gdf[urban_gdf['UATYP10'] == 'C']    # Towns

                    # Further split Urban Clusters by area (proxy for population)
                    # Larger clusters = suburban, smaller = small towns
                    area_threshold = urban_clusters['ALAND10'].quantile(0.6)
                    suburban = urban_clusters[urban_clusters['ALAND10'] > area_threshold]
                    small_towns = urban_clusters[urban_clusters['ALAND10'] <= area_threshold]

                    self.data['high_density_urban'] = urbanized_areas
                    self.data['suburban'] = suburban
                    self.data['small_towns'] = small_towns

                    print(f"✓ Urban areas loaded: {len(urban_gdf)} features")
                    print(f"  Urbanized Areas (≥50k pop): {len(urbanized_areas)} metros")
                    print(f"  Suburban (large clusters): {len(suburban)} areas")
                    print(f"  Small towns (small clusters): {len(small_towns)} towns")
                else:
                    # Fallback if type field not available
                    self.data['urban'] = urban_gdf
                    print(f"✓ Urban areas loaded: {len(urban_gdf)} features")
            else:
                print(f"⚠ Urban areas file not found: {urban_file}")
        except Exception as e:
            print(f"✗ Error loading urban areas: {e}")

    def load_roads(self):
        """Load major highways"""
        print("\nLoading primary roads...")
        try:
            roads_file = os.path.join(config.DATA_DIR, 'roads', 'tl_2023_us_primaryroads.shp')
            if os.path.exists(roads_file):
                self.data['roads'] = gpd.read_file(roads_file)
                print(f"✓ Primary roads loaded: {len(self.data['roads'])} features")
            else:
                print(f"⚠ Roads file not found: {roads_file}")
        except Exception as e:
            print(f"✗ Error loading roads: {e}")

    def load_ski_resorts(self):
        """Load ski resort locations"""
        print("\nLoading ski resorts...")
        try:
            # Priority order: US resorts file, main file, Colorado file
            us_file = os.path.join(config.DATA_DIR, 'ski_resorts', 'us_ski_resorts.geojson')
            ski_file = os.path.join(config.DATA_DIR, 'ski_resorts', 'ski_areas.geojson')
            colorado_file = os.path.join(config.DATA_DIR, 'ski_resorts', 'colorado_ski_resorts.geojson')

            # Try US resorts file first
            if os.path.exists(us_file):
                self.data['ski_resorts'] = gpd.read_file(us_file)
                print(f"✓ Ski resorts loaded: {len(self.data['ski_resorts'])} features (comprehensive US resorts)")
            # Try main file
            elif os.path.exists(ski_file):
                self.data['ski_resorts'] = gpd.read_file(ski_file)
                # Filter for US
                if 'country' in self.data['ski_resorts'].columns:
                    self.data['ski_resorts'] = self.data['ski_resorts'][
                        self.data['ski_resorts']['country'] == 'US'
                    ]

                # If empty or very few resorts, fall back to Colorado file
                if len(self.data['ski_resorts']) < 5 and os.path.exists(colorado_file):
                    print(f"  Main file has only {len(self.data['ski_resorts'])} resorts, loading Colorado resorts...")
                    self.data['ski_resorts'] = gpd.read_file(colorado_file)

                print(f"✓ Ski resorts loaded: {len(self.data['ski_resorts'])} features")
            # Fall back to Colorado file
            elif os.path.exists(colorado_file):
                print(f"  Using Colorado resorts only...")
                self.data['ski_resorts'] = gpd.read_file(colorado_file)
                print(f"✓ Ski resorts loaded: {len(self.data['ski_resorts'])} features (Colorado only)")
            else:
                print(f"⚠ Ski resorts file not found")
        except Exception as e:
            print(f"✗ Error loading ski resorts: {e}")

    def load_golf_courses(self):
        """Load golf course locations"""
        print("\nLoading golf courses...")
        try:
            golf_file = os.path.join(config.DATA_DIR, 'golf', 'us_golf_courses.geojson')

            if os.path.exists(golf_file):
                self.data['golf_courses'] = gpd.read_file(golf_file)
                print(f"✓ Golf courses loaded: {len(self.data['golf_courses'])} features")
            else:
                print(f"⚠ Golf courses file not found: {golf_file}")
                print("  Run download_data.py to fetch from OpenStreetMap")
        except Exception as e:
            print(f"✗ Error loading golf courses: {e}")

    def load_national_parks(self):
        """Load National Park Service boundaries"""
        print("\nLoading national parks...")
        try:
            parks_file = os.path.join(config.DATA_DIR, 'parks', 'nps_boundaries.geojson')

            if os.path.exists(parks_file):
                parks_gdf = gpd.read_file(parks_file)
                # Filter for CONUS
                conus_parks = parks_gdf.cx[-135:-60, 21:53].copy()
                self.data['national_parks'] = conus_parks
                print(f"✓ National Parks loaded: {len(conus_parks)} NPS units")
            else:
                print(f"⚠ National Parks file not found: {parks_file}")
                print("  Run: python -c \"from download_data import download_national_parks; download_national_parks()\"")
        except Exception as e:
            print(f"✗ Error loading national parks: {e}")

    def load_national_forests(self):
        """Load National Forest System boundaries"""
        print("\nLoading national forests...")
        try:
            forests_file = os.path.join(config.DATA_DIR, 'forests', 'S_USA.AdministrativeForest.shp')

            if os.path.exists(forests_file):
                forests_gdf = gpd.read_file(forests_file)
                # Filter for CONUS
                conus_forests = forests_gdf.cx[-135:-60, 21:53].copy()
                self.data['national_forests'] = conus_forests
                print(f"✓ National Forests loaded: {len(conus_forests)} forest units")
            else:
                print(f"⚠ National Forests file not found: {forests_file}")
                print("  Run: python -c \"from download_data import download_national_forests; download_national_forests()\"")
        except Exception as e:
            print(f"✗ Error loading national forests: {e}")

    def load_all(self):
        """Load all datasets"""
        print("="*70)
        print(" LOADING GEOSPATIAL DATASETS")
        print("="*70)

        self.load_hrrr_grid()
        self.load_coastline()
        self.load_lakes()
        self.load_protected_areas()
        self.load_national_parks()
        self.load_national_forests()
        self.load_urban_areas()
        self.load_roads()
        self.load_ski_resorts()
        self.load_golf_courses()

        print("\n" + "="*70)
        print(" DATA LOADING COMPLETE")
        print("="*70)


class TerrainAnalyzer:
    """Analyze terrain variability for Tier 2 classification"""

    def __init__(self, terrain, lats, lons):
        self.terrain = terrain
        self.lats = lats
        self.lons = lons

    def compute_terrain_variability(self):
        """Compute terrain standard deviation in 1km window"""
        print("\n" + "="*70)
        print(" STEP 1/5: COMPUTING TERRAIN VARIABILITY")
        print("="*70)

        # Estimate pixel size in km (approximate)
        # HRRR is ~3km, so window size should be ~1 pixel for 1km window
        window_pixels = max(1, int(config.TERRAIN_WINDOW_SIZE / 3000))

        print(f"\nUsing {window_pixels}x{window_pixels} pixel window (~1km)")
        print(f"Processing {self.terrain.shape[0]} × {self.terrain.shape[1]} grid cells...")
        print("(This may take 2-5 minutes...)")

        start_time = time.time()

        # Compute local standard deviation
        terrain_std = generic_filter(
            self.terrain,
            np.std,
            size=window_pixels,
            mode='constant',
            cval=0
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Terrain variability computed in {elapsed:.1f} seconds")
        print(f"  Std dev range: {terrain_std.min():.1f} to {terrain_std.max():.1f} m")
        print(f"  Cells with std > {config.TERRAIN_TIER1_THRESHOLD}m (extreme): "
              f"{(terrain_std > config.TERRAIN_TIER1_THRESHOLD).sum():,}")
        print(f"  Cells with std > {config.TERRAIN_TIER2_THRESHOLD}m (medium): "
              f"{(terrain_std > config.TERRAIN_TIER2_THRESHOLD).sum():,}")
        print(f"  Cells with std > {config.TERRAIN_TIER3_THRESHOLD}m (lower): "
              f"{(terrain_std > config.TERRAIN_TIER3_THRESHOLD).sum():,}")

        return terrain_std


class TierClassifier:
    """Classify HRRR grid cells into tiers"""

    def __init__(self, data_loader, terrain_variability):
        self.loader = data_loader
        self.terrain_var = terrain_variability
        self.tier_map = None

    def create_tier_map(self):
        """Create tier classification for entire HRRR grid"""
        print("\n" + "="*70)
        print(" STEP 2/5: TIER CLASSIFICATION")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize to Tier 4 (background)
        tier_map = np.full(shape, 4, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)  # Bitfield for criteria met

        # Metadata bits:
        # 0: Urban
        # 1: Suburban
        # 2: Coastline
        # 3: Lake
        # 4: Ski resort
        # 5: Golf course (IMPLEMENTED)
        # 6: National/state park (IMPLEMENTED)
        # 7: Major highway
        # 8: High terrain variability

        print("\n[1/6] Classifying Tier 2: Terrain variability...")
        high_terrain_var = self.terrain_var > config.TERRAIN_STDDEV_THRESHOLD
        tier_map[high_terrain_var] = np.minimum(tier_map[high_terrain_var], 2)
        metadata[high_terrain_var] |= (1 << 8)
        print(f"      ✓ Tier 2 cells from terrain: {high_terrain_var.sum():,}")

        # Classify based on spatial features
        print("\n[2/6] Classifying based on vector data...")

        # Helper function to classify points near features
        def classify_near_features(gdf, tier, buffer_km, metadata_bit, name):
            if gdf is None or len(gdf) == 0:
                print(f"  Skipping {name} (no data)")
                return

            print(f"  Processing {name}...")
            # Create buffer around features
            buffered = gdf.geometry.buffer(buffer_km / 111)  # Rough km to degrees

            # Check which HRRR cells intersect
            count = 0
            for geom in tqdm(buffered, desc=f"  {name}", leave=False):
                # Find grid cells within bounds
                bounds = geom.bounds
                lat_mask = (lats >= bounds[1]) & (lats <= bounds[3])
                lon_mask = (lons >= bounds[0]) & (lons <= bounds[2])
                mask = lat_mask & lon_mask

                # Update tier and metadata
                tier_map[mask] = np.minimum(tier_map[mask], tier)
                metadata[mask] |= (1 << metadata_bit)
                count += mask.sum()

            print(f"    Classified {count:,} cells as Tier {tier}")

        # Tier 1: Recreation areas
        print("\n[3/6] Tier 1 Recreation Areas:")
        classify_near_features(
            self.loader.data.get('coastline'),
            tier=1,
            buffer_km=config.COASTLINE_BUFFER_OFFSHORE_KM,
            metadata_bit=2,
            name="Coastline"
        )

        classify_near_features(
            self.loader.data.get('lakes'),
            tier=1,
            buffer_km=1,
            metadata_bit=3,
            name="Lakes"
        )

        classify_near_features(
            self.loader.data.get('ski_resorts'),
            tier=1,
            buffer_km=2,
            metadata_bit=4,
            name="Ski Resorts"
        )

        # Golf courses (if enabled)
        if config.INCLUDE_GOLF_COURSES:
            classify_near_features(
                self.loader.data.get('golf_courses'),
                tier=1,
                buffer_km=config.GOLF_COURSE_BUFFER_KM,
                metadata_bit=5,
                name="Golf Courses"
            )

        # National & State Parks (if enabled)
        if config.INCLUDE_PARKS:
            classify_near_features(
                self.loader.data.get('parks'),
                tier=1,
                buffer_km=config.PARKS_BUFFER_KM,
                metadata_bit=6,
                name="National & State Parks"
            )

        # Tier 3: Urban/suburban and highways
        print("\n[4/6] Tier 3 Urban/Suburban & Transportation:")
        classify_near_features(
            self.loader.data.get('urban'),
            tier=3,
            buffer_km=0,
            metadata_bit=0,
            name="Urban Areas"
        )

        classify_near_features(
            self.loader.data.get('roads'),
            tier=3,
            buffer_km=0.5,
            metadata_bit=7,
            name="Major Highways"
        )

        self.tier_map = tier_map
        self.metadata_map = metadata

        # Summary statistics
        print("\n" + "="*70)
        print(" TIER CLASSIFICATION SUMMARY")
        print("="*70)
        for tier in [1, 2, 3, 4]:
            count = (tier_map == tier).sum()
            pct = 100 * count / tier_map.size
            print(f"  Tier {tier}: {count:,} cells ({pct:.1f}%)")

        return tier_map, metadata


class AdaptiveGridGenerator:
    """Generate adaptive grid points based on tier classification"""

    def __init__(self, data_loader, tier_map, metadata_map):
        self.loader = data_loader
        self.tier_map = tier_map
        self.metadata_map = metadata_map
        self.points = None

    def generate_points(self):
        """Generate points with tier-appropriate spacing"""
        print("\n" + "="*70)
        print(" STEP 3/5: GENERATING ADAPTIVE GRID POINTS")
        print("="*70)

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        all_points = []
        all_metadata = []

        start_time = time.time()

        for tier_idx, tier in enumerate([1, 2, 3, 4], 1):
            print(f"\n[{tier_idx}/4] Generating Tier {tier} points (resolution: {config.TIER_RESOLUTIONS[tier]}m)...")

            # Get grid cells for this tier
            tier_mask = self.tier_map == tier
            tier_indices = np.argwhere(tier_mask)

            if len(tier_indices) == 0:
                print(f"      No cells in Tier {tier} - skipping")
                continue

            print(f"      {len(tier_indices):,} HRRR cells in Tier {tier}")

            # For each HRRR cell, generate sub-grid points
            resolution_m = config.TIER_RESOLUTIONS[tier]
            hrrr_cell_size_m = 3000  # HRRR is ~3km

            # Number of points per HRRR cell
            if resolution_m >= hrrr_cell_size_m:
                # Tier 4: Just use HRRR cell centers
                points_per_cell = 1
            else:
                # Subdivide HRRR cells
                points_per_cell = int((hrrr_cell_size_m / resolution_m) ** 2)

            expected_points = len(tier_indices) * points_per_cell
            print(f"      {points_per_cell} points per cell = ~{expected_points:,} total points for this tier")

            tier_points = []
            tier_meta = []

            for idx in tqdm(tier_indices, desc=f"      Processing", unit="cells", leave=True):
                i, j = idx
                lat_center = lats_hrrr[i, j]
                lon_center = lons_hrrr[i, j]
                metadata_value = self.metadata_map[i, j]

                if points_per_cell == 1:
                    # Just use center point
                    tier_points.append([lat_center, lon_center, tier])
                    tier_meta.append(metadata_value)
                else:
                    # Generate sub-grid
                    n_per_side = int(np.sqrt(points_per_cell))

                    # Approximate km to degrees
                    lat_spacing = (resolution_m / 1000) / 111
                    lon_spacing = (resolution_m / 1000) / (111 * np.cos(np.radians(lat_center)))

                    # Generate sub-grid centered on HRRR cell
                    half_extent_lat = lat_spacing * n_per_side / 2
                    half_extent_lon = lon_spacing * n_per_side / 2

                    sub_lats = np.linspace(
                        lat_center - half_extent_lat,
                        lat_center + half_extent_lat,
                        n_per_side
                    )
                    sub_lons = np.linspace(
                        lon_center - half_extent_lon,
                        lon_center + half_extent_lon,
                        n_per_side
                    )

                    lat_grid, lon_grid = np.meshgrid(sub_lats, sub_lons)

                    for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel()):
                        tier_points.append([lat, lon, tier])
                        tier_meta.append(metadata_value)

            print(f"      ✓ Generated {len(tier_points):,} points for Tier {tier}")
            all_points.extend(tier_points)
            all_metadata.extend(tier_meta)

        # Convert to arrays
        points_array = np.array(all_points)
        metadata_array = np.array(all_metadata)

        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f" POINT GENERATION COMPLETE")
        print("="*70)
        print(f"  Total points: {len(points_array):,}")
        print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        # Check against target
        if len(points_array) > config.TARGET_TOTAL_POINTS:
            print(f"\n⚠ Warning: Exceeded target of {config.TARGET_TOTAL_POINTS:,} points")
            print(f"  Consider adjusting tier thresholds or resolutions")
        else:
            print(f"\n✓ Within target of {config.TARGET_TOTAL_POINTS:,} points")
            pct_of_target = 100 * len(points_array) / config.TARGET_TOTAL_POINTS
            print(f"  ({pct_of_target:.1f}% of target)")

        self.points = points_array
        self.point_metadata = metadata_array

        return points_array, metadata_array


class OutputWriter:
    """Write output netCDF and visualization"""

    def __init__(self, points, metadata, hrrr_grid):
        self.points = points
        self.metadata = metadata
        self.hrrr_grid = hrrr_grid

    def write_netcdf(self, filename):
        """Write adaptive grid to netCDF file"""
        print("\n" + "="*70)
        print(" STEP 4/5: WRITING OUTPUT NETCDF")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nOutput file: {output_path}")
        print(f"Writing {len(self.points):,} points...")

        start_time = time.time()

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            # Dimensions
            npoints = len(self.points)
            ncfile.createDimension('npoints', npoints)

            # Global attributes
            ncfile.title = 'Adaptive Grid Points for XGBoost Weather Downscaling'
            ncfile.institution = 'TWC/IBM'
            ncfile.source = 'generate_adaptive_grid.py'
            ncfile.history = f'Created {datetime.now().isoformat()}'
            ncfile.conventions = 'CF-1.8'
            ncfile.total_points = npoints
            ncfile.tier1_resolution_m = config.TIER_RESOLUTIONS[1]
            ncfile.tier2_resolution_m = config.TIER_RESOLUTIONS[2]
            ncfile.tier3_resolution_m = config.TIER_RESOLUTIONS[3]
            ncfile.tier4_resolution_m = config.TIER_RESOLUTIONS[4]

            # Variables
            lat_var = ncfile.createVariable('latitude', 'f4', ('npoints',))
            lat_var.units = 'degrees_north'
            lat_var.long_name = 'Latitude'
            lat_var.standard_name = 'latitude'
            lat_var[:] = self.points[:, 0]

            lon_var = ncfile.createVariable('longitude', 'f4', ('npoints',))
            lon_var.units = 'degrees_east'
            lon_var.long_name = 'Longitude'
            lon_var.standard_name = 'longitude'
            lon_var[:] = self.points[:, 1]

            tier_var = ncfile.createVariable('tier', 'i1', ('npoints',))
            tier_var.long_name = 'Grid Tier (1=highest resolution)'
            tier_var.description = 'Tier 1: 90m (recreation), Tier 2: 270m (terrain var), ' \
                                   'Tier 3: 810m (suburban/highway), Tier 4: 3km (background)'
            tier_var[:] = self.points[:, 2].astype(np.int8)

            meta_var = ncfile.createVariable('metadata', 'u2', ('npoints',))
            meta_var.long_name = 'Point Classification Metadata (bitfield)'
            meta_var.description = 'Bit 0: Urban, Bit 1: Suburban, Bit 2: Coastline, ' \
                                   'Bit 3: Lake, Bit 4: Ski Resort, Bit 5: Golf Course, ' \
                                   'Bit 6: Park, Bit 7: Highway, Bit 8: High Terrain Var'
            meta_var[:] = self.metadata

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ NetCDF file written: {output_path}")
        print(f"  Points: {npoints:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Write time: {elapsed:.1f} seconds")
        return output_path

    def create_visualization(self, filename):
        """Create visualization showing point density on HRRR grid"""
        print("\n" + "="*70)
        print(" STEP 5/5: CREATING VISUALIZATION")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nOutput file: {output_path}")

        # Count points per HRRR cell
        lats_hrrr = self.hrrr_grid['lats']
        lons_hrrr = self.hrrr_grid['lons']
        shape = lats_hrrr.shape

        start_time = time.time()
        print(f"\n[1/3] Counting points per HRRR grid cell...")
        print(f"      ({len(self.points):,} points to assign to {shape[0] * shape[1]:,} grid cells)")
        point_count = np.zeros(shape, dtype=np.int32)

        # Build KDTree for HRRR grid centers
        hrrr_points = np.column_stack([lats_hrrr.ravel(), lons_hrrr.ravel()])
        tree = cKDTree(hrrr_points)

        # Find nearest HRRR cell for each adaptive grid point
        adaptive_points = self.points[:, :2]
        _, indices = tree.query(adaptive_points, k=1)

        # Count points
        for idx in tqdm(indices, desc="      Assigning points to cells", unit="points"):
            i = idx // shape[1]
            j = idx % shape[1]
            point_count[i, j] += 1

        print(f"      ✓ Point counting complete")
        print(f"        Max points per cell: {point_count.max():,}")
        print(f"        Mean points per cell: {point_count.mean():.1f}")

        # Create map
        print(f"\n[2/3] Creating map projection and drawing features...")
        fig = plt.figure(figsize=config.PLOT_FIGSIZE)

        # Get map bounds
        lat_min, lat_max = lats_hrrr.min(), lats_hrrr.max()
        lon_min, lon_max = lons_hrrr.min(), lons_hrrr.max()

        # Create Basemap
        m = Basemap(
            projection='lcc',
            lat_0=40,
            lon_0=-96,
            lat_1=33,
            lat_2=45,
            llcrnrlat=lat_min,
            urcrnrlat=lat_max,
            llcrnrlon=lon_min,
            urcrnrlon=lon_max,
            resolution='l'
        )

        # Draw map features
        m.drawcoastlines(linewidth=0.5, color='black')
        m.drawcountries(linewidth=0.5, color='black')
        m.drawstates(linewidth=0.3, color='gray')

        # Plot point density
        x, y = m(lons_hrrr, lats_hrrr)

        # Use log scale for better visualization
        point_count_plot = np.ma.masked_where(point_count == 0, point_count)

        cs = m.pcolormesh(
            x, y, point_count_plot,
            cmap=config.COLORMAP,
            shading='auto',
            norm=matplotlib.colors.LogNorm(vmin=1, vmax=point_count.max())
        )

        # Colorbar
        cbar = plt.colorbar(cs, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Number of Adaptive Grid Points per HRRR Cell (log scale)', fontsize=10)

        # Title
        plt.title(
            f'Adaptive Grid Point Density for XGBoost Weather Downscaling\n'
            f'Total Points: {len(self.points):,} | '
            f'Target: {config.TARGET_TOTAL_POINTS:,}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Save
        print(f"\n[3/3] Saving visualization to file...")
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)

        print(f"\n✓ Visualization saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Time elapsed: {elapsed:.1f} seconds")
        return output_path


def main():
    """Main execution"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION FOR XGBOOST WEATHER DOWNSCALING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target points: {config.TARGET_TOTAL_POINTS:,}")
    print(f"  Tier resolutions: {config.TIER_RESOLUTIONS}")
    print(f"  Terrain variability threshold: {config.TERRAIN_STDDEV_THRESHOLD}m")
    print(f"\nEstimated runtime: 30-60 minutes")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = DataLoader()
    loader.load_all()

    # Analyze terrain
    terrain_analyzer = TerrainAnalyzer(
        loader.hrrr_grid['terrain'],
        loader.hrrr_grid['lats'],
        loader.hrrr_grid['lons']
    )
    terrain_var = terrain_analyzer.compute_terrain_variability()

    # Classify tiers
    classifier = TierClassifier(loader, terrain_var)
    tier_map, metadata_map = classifier.create_tier_map()

    # Generate points
    generator = AdaptiveGridGenerator(loader, tier_map, metadata_map)
    points, metadata = generator.generate_points()

    # Write outputs
    writer = OutputWriter(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_points.nc')
    png_file = writer.create_visualization('adaptive_grid_density.png')

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*70)
    print(" ✓ ADAPTIVE GRID GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  NetCDF: {nc_file}")
    print(f"  Visualization: {png_file}")
    print(f"\nSummary:")
    print(f"  Total points: {len(points):,}")
    print(f"  Total runtime: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
