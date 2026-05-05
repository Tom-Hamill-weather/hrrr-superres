"""
Adaptive Grid Point Generation for XGBoost Weather Downscaling

Generates an irregular grid distributed according to:
- Tier 0 (93.75m):  Ski resorts, golf courses (polygon-interior)
- Tier 1 (187.5m):  Coastlines/water bodies, extreme terrain (σ>600m), Great Lakes shores
- Tier 2 (375m):    Urban areas, rugged terrain (σ 400-600m), coastal strip
- Tier 3 (750m):    Suburban/highways/forests, moderate terrain (σ 100-400m)
- Tier 5 (3000m):   Flat background (HRRR native resolution)

Terrain and coastal classification use the WRF ~800m terrain file
(WRF_CONUS_terrain_info.nc) for sub-HRRR-cell resolution.
Coastal bands are derived from the WRF land_sea_mask via distance transform
(ocean-connected water only; inland lakes excluded).
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.ndimage import uniform_filter, distance_transform_edt, label as ndlabel
from scipy.spatial import cKDTree
import netCDF4 as nc
from datetime import datetime
from tqdm import tqdm
from pyproj import Proj
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
        """Load lake data - separate Great Lakes from inland lakes"""
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

                # Filter by coastline length threshold
                min_coastline_km = config.INLAND_LAKES_MIN_COASTLINE_KM
                significant_lakes = conus_lakes[conus_lakes['coastline_length_km'] > min_coastline_km].copy()

                # Separate Great Lakes from inland lakes
                great_lakes_mask = significant_lakes['name'].str.contains(
                    '|'.join(config.GREAT_LAKES_NAMES), case=False, na=False
                )

                great_lakes = significant_lakes[great_lakes_mask].copy()
                inland_lakes = significant_lakes[~great_lakes_mask].copy()

                # Extract only the lake boundaries (shorelines), not the filled polygons
                # This ensures buffering creates rings around shores, not filled areas
                if len(great_lakes) > 0:
                    great_lakes['geometry'] = great_lakes.geometry.boundary
                    self.data['great_lakes'] = great_lakes
                    print(f"✓ Great Lakes loaded: {len(great_lakes)} features")

                if len(inland_lakes) > 0:
                    inland_lakes['geometry'] = inland_lakes.geometry.boundary
                    self.data['inland_lakes'] = inland_lakes
                    total_length = inland_lakes['coastline_length_km'].sum()
                    print(f"✓ Inland lakes loaded: {len(inland_lakes)} features (>{min_coastline_km}km coastline)")
                    print(f"  Total inland lake coastline: {total_length:.0f}km")
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

                    # Combine all urban categories for tier classification
                    self.data['urban'] = urban_gdf

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
        """Load golf course locations from preprocessed OSM data"""
        print("\nLoading golf courses...")
        try:
            golf_file = os.path.join(config.DATA_DIR, 'golf', 'us_golf_courses_osm.geojson')

            if os.path.exists(golf_file):
                self.data['golf_courses'] = gpd.read_file(golf_file)
                print(f"✓ Golf courses loaded: {len(self.data['golf_courses'])} features")
            else:
                print(f"⚠ Golf courses file not found: {golf_file}")
                print("  Run: python preprocess_golf_courses.py")
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
                self.nps_boundaries = conus_parks  # Store for forest exclusion
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

    def load_trails(self):
        """Load National Recreation Trails and State-designated trails"""
        print("\nLoading trails...")
        try:
            # Load NRT data
            nrt_file = os.path.join(config.DATA_DIR, 'trails', 'national_recreation_trails.shp')
            state_trails_file = os.path.join(config.DATA_DIR, 'trails', 'state_trails.shp')

            nrt_gdf = None
            state_gdf = None

            # CONUS bounds - stricter filtering to exclude Alaska and Hawaii
            # Continental US: roughly 24-49°N, -125 to -67°W
            conus_lat_min, conus_lat_max = 24.0, 49.5
            conus_lon_min, conus_lon_max = -125.0, -67.0

            def is_in_conus(geom):
                """Check if geometry is within CONUS bounds"""
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                # Check if geometry center is in CONUS
                center_lon = (bounds[0] + bounds[2]) / 2
                center_lat = (bounds[1] + bounds[3]) / 2
                return (conus_lon_min <= center_lon <= conus_lon_max and
                        conus_lat_min <= center_lat <= conus_lat_max)

            if os.path.exists(nrt_file):
                nrt_gdf = gpd.read_file(nrt_file)
                n_before = len(nrt_gdf)

                # Filter using explicit bounds checking on geometry
                nrt_gdf = nrt_gdf[nrt_gdf.geometry.apply(is_in_conus)].copy()
                n_after = len(nrt_gdf)
                n_filtered = n_before - n_after
                print(f"✓ National Recreation Trails loaded: {n_after} trails ({n_filtered} filtered out)")
            else:
                print(f"⚠ NRT file not found: {nrt_file}")
                print("  Download from: https://catalog.data.gov/dataset/national-park-service-trails/")

            if os.path.exists(state_trails_file):
                state_gdf = gpd.read_file(state_trails_file)
                n_before = len(state_gdf)

                # Filter using explicit bounds checking
                state_gdf = state_gdf[state_gdf.geometry.apply(is_in_conus)].copy()
                n_after = len(state_gdf)
                n_filtered = n_before - n_after
                print(f"✓ State-designated trails loaded: {n_after} trails ({n_filtered} filtered out)")
            else:
                print(f"⚠ State trails file not found: {state_trails_file}")

            # Combine trails
            if nrt_gdf is not None and state_gdf is not None:
                self.data['trails'] = pd.concat([nrt_gdf, state_gdf], ignore_index=True)
            elif nrt_gdf is not None:
                self.data['trails'] = nrt_gdf
            elif state_gdf is not None:
                self.data['trails'] = state_gdf

            if 'trails' in self.data:
                # Calculate total trail length
                total_km = self.data['trails'].to_crs('EPSG:3857').geometry.length.sum() / 1000
                print(f"  Total trail length: {total_km:,.0f} km")
        except Exception as e:
            print(f"✗ Error loading trails: {e}")

    def load_water_bodies(self):
        """Load manually-defined specific water bodies from config"""
        from shapely.geometry import Polygon

        print("\nLoading specific water bodies...")
        try:
            water_bodies = []
            for key, body in config.SPECIFIC_WATER_BODIES.items():
                coords = body['coords']
                # Create polygon from coordinates (simple bounding box)
                polygon = Polygon(coords)
                water_bodies.append({
                    'geometry': polygon,
                    'name': body['name'],
                    'key': key
                })

            self.data['water_bodies'] = gpd.GeoDataFrame(
                water_bodies,
                crs='EPSG:4326',
                geometry='geometry'
            )
            print(f"✓ Specific water bodies loaded: {len(water_bodies)} regions")
            for body in water_bodies:
                print(f"  - {body['name']}")
        except Exception as e:
            print(f"✗ Error loading water bodies: {e}")

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
        self.load_national_forests()  # Re-enabled for Tier 2 classification
        self.load_trails()
        self.load_water_bodies()  # Manually-defined water bodies
        self.load_urban_areas()
        self.load_roads()
        self.load_ski_resorts()
        self.load_golf_courses()

        print("\n" + "="*70)
        print(" DATA LOADING COMPLETE")
        print("="*70)


class WrfTerrainAnalyzer:
    """Load WRF ~800m terrain data and classify each pixel to a tier.

    Terrain tier is based on local std-dev of elevation over a
    TERRAIN_WINDOW_KM geographic window (uniform_filter — O(n), fast).

    Coastal tier is derived from the WRF land_sea_mask: ocean-connected
    water cells within COASTAL_BAND_KM[i] km of land → Tier i+1.
    Inland lake water cells are excluded by flood-fill from domain edges.
    """

    def __init__(self):
        self.terrain_tier = None   # int8 (ny_wrf, nx_wrf)
        self.coastal_tier = None   # int8 (ny_wrf, nx_wrf)
        self.lats  = None          # float32 (ny_wrf, nx_wrf)
        self.lons  = None          # float32 (ny_wrf, nx_wrf)
        self.lsm   = None          # int32  (ny_wrf, nx_wrf) 1=land 0=water

    def load_and_classify(self):
        print("\n" + "="*70)
        print(" STEP 1/5: WRF TERRAIN & COASTAL CLASSIFICATION")
        print("="*70)

        wrf_file = config.WRF_TERRAIN_FILE
        if not os.path.exists(wrf_file):
            raise FileNotFoundError(f"WRF terrain file not found: {wrf_file}")

        print(f"\nLoading {wrf_file} ...")
        t0 = time.time()
        with nc.Dataset(wrf_file, 'r') as ds:
            terrain = ds.variables['terrain_height'][:].astype(np.float32)
            lsm     = ds.variables['land_sea_mask'][:].astype(np.int8)
            lats    = ds.variables['lats'][:].astype(np.float32)
            lons    = ds.variables['lons'][:].astype(np.float32)

        ny, nx = terrain.shape
        print(f"  Grid: {ny} × {nx} = {ny*nx/1e6:.1f}M pixels")
        print(f"  Lat range: {lats.min():.1f}–{lats.max():.1f}  "
              f"Lon range: {lons.min():.1f}–{lons.max():.1f}")

        # ── Estimate average pixel spacing in metres ─────────────────────────
        # WRF is on a Lambert Conformal projection; spacing varies slightly
        # but is approximately uniform at ~800m.
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        mid_row = ny // 2
        x_row, _ = transformer.transform(lons[mid_row, :], lats[mid_row, :])
        dx_m = float(np.abs(np.diff(x_row)).mean())
        dy_m = float(np.abs(
            np.diff(transformer.transform(lons[:, nx//2], lats[:, nx//2])[1])
        ).mean())
        pixel_m = (dx_m + dy_m) / 2.0
        print(f"  Approx pixel spacing: {pixel_m:.0f} m  (dx={dx_m:.0f} m, dy={dy_m:.0f} m)")

        # ── Terrain std dev ───────────────────────────────────────────────────
        window_cells = max(3, round(config.TERRAIN_WINDOW_KM * 1000.0 / pixel_m))
        print(f"\nTerrain std dev: {window_cells}×{window_cells}-cell window "
              f"(≈{window_cells * pixel_m / 1000:.1f} km)  [uniform_filter, fast]")

        z = terrain.astype(np.float64)
        local_mean    = uniform_filter(z, size=window_cells, mode='reflect')
        local_sq_mean = uniform_filter(z ** 2, size=window_cells, mode='reflect')
        terrain_std   = np.sqrt(np.maximum(0.0, local_sq_mean - local_mean ** 2)).astype(np.float32)

        # Classify — land pixels only; water pixels stay at background Tier 5
        land = (lsm == 1)
        terrain_tier = np.full((ny, nx), 5, dtype=np.int8)
        terrain_tier[land & (terrain_std > config.TERRAIN_TIER3_THRESHOLD)] = 3
        terrain_tier[land & (terrain_std > config.TERRAIN_TIER2_THRESHOLD)] = 2
        terrain_tier[land & (terrain_std > config.TERRAIN_TIER1_THRESHOLD)] = 1

        n1 = int((terrain_tier == 1).sum())
        n2 = int((terrain_tier == 2).sum())
        n3 = int((terrain_tier == 3).sum())
        print(f"  Tier 1 pixels (σ>{config.TERRAIN_TIER1_THRESHOLD}m): {n1:,}")
        print(f"  Tier 2 pixels (σ>{config.TERRAIN_TIER2_THRESHOLD}m): {n2:,}")
        print(f"  Tier 3 pixels (σ>{config.TERRAIN_TIER3_THRESHOLD}m): {n3:,}")

        # ── Coastal tier via WRF land_sea_mask ───────────────────────────────
        print(f"\nCoastal classification (WRF land_sea_mask)...")
        water = (lsm == 0)

        # Identify ocean-connected water via flood-fill from all domain edges.
        # This excludes inland lakes, which are disconnected from the edges.
        labeled_water, n_regions = ndlabel(water)
        # Collect labels that touch any domain edge
        edge_labels = set()
        for edge in [labeled_water[0, :], labeled_water[-1, :],
                     labeled_water[:, 0], labeled_water[:, -1]]:
            edge_labels.update(int(v) for v in edge if v > 0)
        ocean_mask = np.isin(labeled_water, list(edge_labels))
        inland_water_mask = water & ~ocean_mask
        n_ocean = int(ocean_mask.sum())
        n_inland = int(inland_water_mask.sum())
        print(f"  Ocean-connected pixels: {n_ocean:,}")
        print(f"  Inland water (excluded from coastal): {n_inland:,}")

        # Distance from each ocean pixel to nearest land pixel (and vice versa)
        # distance_transform_edt gives distance in pixel units; scale to metres.
        # For ocean pixels: distance to nearest land (→ offshore coastal bands)
        # For land pixels:  distance to nearest ocean (→ onshore coastal bands)
        ocean_border = ocean_mask   # True = ocean, False = land/inland
        dist_ocean_to_land_px = distance_transform_edt(ocean_border)     # distance from each ocean px to nearest non-ocean
        dist_land_to_ocean_px = distance_transform_edt(~ocean_border)    # distance from each non-ocean px to nearest ocean

        dist_ocean_to_land_m = dist_ocean_to_land_px * pixel_m
        dist_land_to_ocean_m = dist_land_to_ocean_px * pixel_m

        # Assign coastal tier
        # Ocean water cells: tier based on distance to nearest land
        # Land cells: tier based on distance to nearest ocean
        coastal_tier = np.full((ny, nx), 99, dtype=np.int8)  # 99 = no coastal assignment

        # distance_transform_edt returns distances in pixel units.  The minimum
        # distance from an ocean pixel to the nearest land pixel is 1.0 px (one
        # grid step away).  With pixel_m ≈ 800 m and config km bands ≤ 0.75 km,
        # a raw km comparison would classify ZERO pixels as Tier 1.
        # Fix: convert config km targets to pixel-count thresholds, guaranteeing
        # that each band covers at least 1 pixel width.
        #   n0 = pixels in Tier-1 band (≥ 1)
        #   n1 = cumulative pixels through Tier-2 band (≥ n0+1)
        #   n2 = cumulative pixels through Tier-3 band (≥ n1+1)
        # Use n + 0.5 as the threshold to capture all pixels at exactly n steps.
        n0 = max(1, round(config.COASTAL_BAND_KM[0] * 1000.0 / pixel_m))
        n1 = max(n0 + 1, round(config.COASTAL_BAND_KM[1] * 1000.0 / pixel_m))
        n2 = max(n1 + 1, round(config.COASTAL_BAND_KM[2] * 1000.0 / pixel_m))
        t0, t1, t2 = n0 + 0.5, n1 + 0.5, n2 + 0.5

        print(f"  Coastal pixel thresholds (pixel_m={pixel_m:.0f}m): "
              f"Tier 1 ≤{n0}px (≈{n0*pixel_m/1000:.1f}km), "
              f"Tier 2 ≤{n1}px (≈{n1*pixel_m/1000:.1f}km), "
              f"Tier 3 ≤{n2}px (≈{n2*pixel_m/1000:.1f}km)")

        # Water side (ocean pixels within N pixel steps of land)
        coastal_tier[ocean_mask & (dist_ocean_to_land_px <= t0)] = 1
        coastal_tier[ocean_mask & (dist_ocean_to_land_px >  t0) &
                                  (dist_ocean_to_land_px <= t1)] = 2
        coastal_tier[ocean_mask & (dist_ocean_to_land_px >  t1) &
                                  (dist_ocean_to_land_px <= t2)] = 3
        coastal_tier[ocean_mask & (dist_ocean_to_land_px >  t2)] = 5   # open ocean → background

        # Land side (land pixels within N pixel steps of ocean)
        # Only assign if it improves on background (i.e. coastal tier < terrain_tier)
        land_t1 = land & (dist_land_to_ocean_px <= t0)
        land_t2 = land & (dist_land_to_ocean_px >  t0) & (dist_land_to_ocean_px <= t1)
        land_t3 = land & (dist_land_to_ocean_px >  t1) & (dist_land_to_ocean_px <= t2)
        coastal_tier[land_t1] = np.minimum(coastal_tier[land_t1], 1)
        coastal_tier[land_t2] = np.minimum(coastal_tier[land_t2], 2)
        coastal_tier[land_t3] = np.minimum(coastal_tier[land_t3], 3)
        # Land beyond coastal bands: mark as no coastal assignment
        coastal_tier[land & (dist_land_to_ocean_px > t2) & (coastal_tier == 99)] = 99

        c1 = int((coastal_tier == 1).sum())
        c2 = int((coastal_tier == 2).sum())
        c3 = int((coastal_tier == 3).sum())
        print(f"  Coastal Tier 1 pixels (≤{n0} steps, ≈{n0*pixel_m/1000:.1f}km): {c1:,}")
        print(f"  Coastal Tier 2 pixels ({n0}–{n1} steps, ≈{n0*pixel_m/1000:.1f}–{n1*pixel_m/1000:.1f}km): {c2:,}")
        print(f"  Coastal Tier 3 pixels ({n1}–{n2} steps, ≈{n1*pixel_m/1000:.1f}–{n2*pixel_m/1000:.1f}km): {c3:,}")

        elapsed = time.time() - t0
        print(f"\n✓ WRF terrain & coastal classification complete in {elapsed:.1f}s")

        self.terrain_tier = terrain_tier
        self.coastal_tier = coastal_tier
        self.lats = lats
        self.lons = lons
        self.lsm  = lsm
        self.pixel_m = pixel_m

        return self


class TierClassifier:
    """Classify HRRR grid cells into feature-based tiers.

    Terrain and coastal tiers are handled separately by WrfTerrainAnalyzer
    and merged per-WRF-pixel in AdaptiveGridGenerator.generate_points().
    This class only applies vector-feature buffers (urban, ski, golf etc.)
    to the HRRR grid.
    """

    def __init__(self, data_loader):
        self.loader = data_loader
        self.tier_map = None
        self.cached_buffers = None

    def load_cached_buffers(self):
        """Load preprocessed tier buffers if available"""
        if not config.USE_CACHED_BUFFERS:
            return False

        cache_file = config.CACHED_BUFFERS_FILE
        if cache_file is None:
            cache_file = os.path.join(config.DATA_DIR, 'preprocessed', 'tier_buffers.gpkg')

        if not os.path.exists(cache_file):
            print(f"\n⚠️  Cached buffers not found: {cache_file}")
            print("   Run: python3 preprocess_feature_buffers.py")
            return False

        try:
            import geopandas as gpd
            print(f"\n✓ Loading cached buffers from: {cache_file}")

            # Verify config fingerprint before trusting the cache
            try:
                fp = gpd.read_file(cache_file, layer='config_fingerprint')
                stored_hash = fp.iloc[0]['hash']
                current_hash = config.cache_config_hash()
                if stored_hash != current_hash:
                    print(f"\n⚠  Cache is STALE — config.py has changed since this cache was built.")
                    print(f"   Re-run:  python3 preprocess_feature_buffers.py")
                    print(f"   Falling back to non-cached feature processing (slower but correct).")
                    return False
                print(f"  ✓ Cache fingerprint matches current config")
            except Exception:
                print(f"\n⚠  Cache has no config fingerprint (old format).")
                print(f"   Re-run:  python3 preprocess_feature_buffers.py")
                print(f"   Falling back to non-cached feature processing.")
                return False

            self.cached_buffers = gpd.read_file(cache_file, layer='tier_buffers')

            # Create projection for cached buffers
            lats_hrrr = self.loader.hrrr_grid['lats']
            lons_hrrr = self.loader.hrrr_grid['lons']
            lat_ref = (lats_hrrr.min() + lats_hrrr.max()) / 2
            lon_ref = (lons_hrrr.min() + lons_hrrr.max()) / 2

            from mpl_toolkits.basemap import Basemap
            proj = Basemap(
                projection='lcc',
                lat_0=lat_ref,
                lon_0=lon_ref,
                lat_1=lat_ref - 5,
                lat_2=lat_ref + 5,
                llcrnrlat=lats_hrrr.min(),
                urcrnrlat=lats_hrrr.max(),
                llcrnrlon=lons_hrrr.min(),
                urcrnrlon=lons_hrrr.max(),
                resolution='l'
            )

            # Convert to lat/lon if needed
            if self.cached_buffers.crs != 'EPSG:4326':
                print(f"  Converting cached buffers to EPSG:4326...")
                self.cached_buffers = self.cached_buffers.to_crs('EPSG:4326')

            print(f"  ✓ Loaded {len(self.cached_buffers)} tier buffers")
            return True

        except Exception as e:
            print(f"\n✗ Error loading cached buffers: {e}")
            return False

    def create_tier_map(self):
        """Create tier classification for entire HRRR grid"""
        print("\n" + "="*70)
        print(" STEP 2/5: TIER CLASSIFICATION")
        print("="*70)

        shape = self.loader.hrrr_grid['shape']
        lats = self.loader.hrrr_grid['lats']
        lons = self.loader.hrrr_grid['lons']

        print(f"\nClassifying {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} HRRR grid cells")

        # Initialize to Tier 5 (background — 3km HRRR native resolution).
        # Features below will override cells to finer tiers.
        # Terrain and coastal classification are handled by WrfTerrainAnalyzer
        # and merged per-WRF-pixel in generate_points().
        tier_map = np.full(shape, 5, dtype=np.int8)
        metadata = np.zeros(shape, dtype=np.uint16)  # Bitfield for criteria met

        # Metadata bits:
        # 0: Urban
        # 1: Suburban
        # 2: Coastline / Specific water bodies
        # 3: Lake
        # 4: Ski resort
        # 5: Golf course
        # 6: National/state park
        # 7: Major highway
        # 8: High terrain variability (set by WRF loop in generate_points)
        # 9: National forest (outside parks)

        # Try to use cached buffers if enabled
        use_cached = self.load_cached_buffers()

        if use_cached and self.cached_buffers is not None:
            print("\n[2/6] Applying cached feature buffers...")
            print("  (Skipping individual feature processing)")

            # Apply cached buffers by tier
            from shapely.geometry import Point
            from shapely.prepared import prep

            for idx, row in self.cached_buffers.iterrows():
                # Cached file uses 0-based tiers matching TIER_RESOLUTIONS directly
                tier = int(row['tier'])
                geom = row['geometry']

                print(f"\n  Processing cached Tier {tier} buffer...")
                prepared_geom = prep(geom)

                # Check each HRRR cell
                count = 0
                bounds = geom.bounds
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        lat, lon = lats[i, j], lons[i, j]

                        # Quick bounds check
                        if not (bounds[1] <= lat <= bounds[3] and bounds[0] <= lon <= bounds[2]):
                            continue

                        # Check if point is in buffer
                        if prepared_geom.contains(Point(lon, lat)):
                            tier_map[i, j] = min(tier_map[i, j], tier)
                            # Set appropriate metadata bit based on tier
                            if tier == 0:
                                metadata[i, j] |= (1 << 4)  # Ski resort bit
                            elif tier == 1:
                                metadata[i, j] |= (1 << 2)  # Coastline/water bit
                            elif tier == 2:
                                metadata[i, j] |= (1 << 0)  # Urban bit
                            elif tier == 3:
                                metadata[i, j] |= (1 << 9)  # Forest/suburban bit
                            count += 1

                print(f"    ✓ Classified {count:,} cells as Tier {tier} or better")

            # Golf courses are excluded from the cache entirely (see
            # preprocess_feature_buffers.py step 6).  No cleanup needed here;
            # polygon-interior golf points are added in generate_points().

        else:
            # Fall back to individual feature processing
            print("\n[2/6] Classifying based on vector data...")
            if use_cached:
                print("  (Cached buffers not available, using full feature processing)")

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

            # Tier 1: Recreation areas (lakes, ski, water bodies)
            # Coastline is handled below via KDTree after both paths complete.
            print("\n[3/6] Tier 1 Recreation Areas:")

            classify_near_features(
                self.loader.data.get('great_lakes'),
                tier=2,
                buffer_km=config.GREAT_LAKES_BUFFER_KM,
                metadata_bit=3,
                name="Great Lakes Shorelines"
            )

            classify_near_features(
                self.loader.data.get('ski_resorts'),
                tier=1,
                buffer_km=2,
                metadata_bit=4,
                name="Ski Resorts"
            )

            # Specific water bodies
            classify_near_features(
                self.loader.data.get('water_bodies'),
                tier=2,
                buffer_km=0,
                metadata_bit=2,
                name="Specific Water Bodies"
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

            # Tier 1: Inland lakes
            print("\n[3.5/6] Tier 1 Inland Lakes:")

            classify_near_features(
                self.loader.data.get('inland_lakes'),
                tier=config.INLAND_LAKES_TIER,
                buffer_km=config.INLAND_LAKES_BUFFER_KM,
                metadata_bit=3,
                name="Inland Lakes"
            )

            # Golf courses: no cell-level tier change needed.
            # Points are added additively by polygon in generate_points(),
            # so existing tier assignments (typically Tier 4 background) are
            # preserved and no coverage holes or gray boxes are created.

            # Tier 3: Urban areas (Urbanized Areas ≥50k pop)
            print("\n[4/6] Tier 3 Urban / Tier 4 Suburban:")
            classify_near_features(
                self.loader.data.get('high_density_urban'),
                tier=3,
                buffer_km=config.URBAN_BUFFER_KM,
                metadata_bit=0,
                name="Urban Areas"
            )

            # Suburban areas (Urban Clusters)
            classify_near_features(
                self.loader.data.get('suburban'),
                tier=config.SUBURBAN_TIER,
                buffer_km=config.SUBURBAN_BUFFER_KM,
                metadata_bit=1,
                name="Suburban Areas"
            )

            classify_near_features(
                self.loader.data.get('roads'),
                tier=config.HIGHWAY_TIER,
                buffer_km=config.HIGHWAY_BUFFER_KM,
                metadata_bit=7,
                name="Major Highways"
            )

            # National forests (excluding national parks) - Tier 3
            if config.INCLUDE_NATIONAL_FORESTS:
                print("  Processing National Forests...")
                forests = self.loader.data.get('national_forests')

                if forests is not None and len(forests) > 0:
                    # Subtract NPS boundaries if configured
                    if config.EXCLUDE_PARKS_FROM_FORESTS and hasattr(self.loader, 'nps_boundaries'):
                        print("    Excluding NPS boundaries from forests...")
                        from shapely.ops import unary_union

                        forests_proj = forests.to_crs('EPSG:5070')
                        parks_proj = self.loader.nps_boundaries.to_crs('EPSG:5070')

                        parks_union = unary_union(parks_proj.geometry)
                        forests_proj['geometry'] = forests_proj.geometry.difference(parks_union)

                        forests_proj = forests_proj[~forests_proj.is_empty]
                        forests = forests_proj.to_crs('EPSG:4326')

                        print(f"    {len(forests)} forest areas after park exclusion")

                    # Classify forest cells
                    count = 0
                    for geom in tqdm(forests.geometry, desc="  National Forests", leave=False):
                        bounds = geom.bounds
                        lat_mask = (lats >= bounds[1]) & (lats <= bounds[3])
                        lon_mask = (lons >= bounds[0]) & (lons <= bounds[2])
                        mask = lat_mask & lon_mask

                        tier_map[mask] = np.minimum(tier_map[mask], config.NATIONAL_FOREST_TIER)
                        metadata[mask] |= (1 << 9)
                        count += mask.sum()

                    print(f"    Classified {count:,} cells as Tier {config.NATIONAL_FOREST_TIER}")
    
        # Coastline bands are handled in generate_points() via polygon-interior
        # generation (same approach as golf courses).  This produces bands that
        # follow the coastline shape continuously rather than snapping to 3 km
        # HRRR cell boundaries.

        self.tier_map = tier_map
        self.metadata_map = metadata

        # Summary statistics
        print("\n" + "="*70)
        print(" TIER CLASSIFICATION SUMMARY")
        print("="*70)
        for tier in [1, 2, 3, 5]:
            count = (tier_map == tier).sum()
            pct = 100 * count / tier_map.size
            print(f"  Tier {tier}: {count:,} cells ({pct:.1f}%)")

        return tier_map, metadata


class AdaptiveGridGenerator:
    """Generate adaptive grid points based on tier classification."""

    def __init__(self, data_loader, tier_map, metadata_map, wrf_analyzer):
        self.loader = data_loader
        self.tier_map = tier_map          # HRRR-resolution feature tiers (5 = background)
        self.metadata_map = metadata_map
        self.wrf = wrf_analyzer           # WrfTerrainAnalyzer (terrain + coastal at WRF resolution)
        self.points = None

    def generate_points(self):
        """Generate points with tier-appropriate spacing"""
        print("\n" + "="*70)
        print(" STEP 3/5: GENERATING ADAPTIVE GRID POINTS")
        print("="*70)

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']

        # Use lists of arrays; vstack at the end (faster than list.extend for large data)
        all_point_arrays = []
        all_meta_arrays  = []

        start_time = time.time()

        # All sub-grid points are placed in HRRR's native LCC projection space so
        # they stay aligned across cell and pixel boundaries.
        hrrr_proj = Proj(proj='lcc', lat_0=38.5, lon_0=-97.5,
                         lat_1=38.5, lat_2=38.5, R=6371229)
        hrrr_crs_str = ('+proj=lcc +lat_0=38.5 +lon_0=-97.5 '
                        '+lat_1=38.5 +lat_2=38.5 +units=m +R=6371229 +no_defs')

        # Globally-aligned golf grid origin anchored at HRRR cell (0,0).
        resolution_golf = config.TIER_RESOLUTIONS[config.GOLF_COURSE_TIER]
        n_golf = round(3000 / resolution_golf)
        x00, y00 = hrrr_proj(lons_hrrr[0, 0], lats_hrrr[0, 0])
        golf_grid_x0 = x00 - (n_golf - 1) / 2.0 * resolution_golf
        golf_grid_y0 = y00 - (n_golf - 1) / 2.0 * resolution_golf

        GOLF_BIT    = 1 << 5
        COAST_BIT   = 1 << 2
        TERRAIN_BIT = 1 << 8

        # ── HRRR LCC grid lookup helpers ──────────────────────────────────────
        _hrrr_flat_x, _hrrr_flat_y = hrrr_proj(lons_hrrr.ravel(), lats_hrrr.ravel())
        _hrrr_gx = _hrrr_flat_x.reshape(lats_hrrr.shape)
        _hrrr_gy = _hrrr_flat_y.reshape(lats_hrrr.shape)
        _hrrr_x0 = _hrrr_gx[0, 0]
        _hrrr_y0 = _hrrr_gy[0, 0]
        _hrrr_dx = (_hrrr_gx[0, -1] - _hrrr_gx[0, 0]) / (_hrrr_gx.shape[1] - 1)
        _hrrr_dy = (_hrrr_gy[-1, 0] - _hrrr_gy[0, 0]) / (_hrrr_gy.shape[0] - 1)
        ny_h, nx_h = self.tier_map.shape

        def _cell_tier_for(px, py):
            """Return feature tier_map value for LCC coordinate arrays px, py."""
            j = np.clip(np.round((px - _hrrr_x0) / _hrrr_dx).astype(int), 0, nx_h - 1)
            i = np.clip(np.round((py - _hrrr_y0) / _hrrr_dy).astype(int), 0, ny_h - 1)
            return self.tier_map[i, j]

        # ── Project WRF pixel centres to HRRR LCC ─────────────────────────────
        print("\n[1/3] WRF-pixel tier mapping...")
        wrf_lats_flat = self.wrf.lats.ravel()
        wrf_lons_flat = self.wrf.lons.ravel()
        wrf_x_flat, wrf_y_flat = hrrr_proj(wrf_lons_flat, wrf_lats_flat)

        # WRF combined tier: best of terrain and coastal at each WRF pixel.
        # coastal_tier == 99 means "no coastal assignment" → treat as Tier 5.
        coast_t = self.wrf.coastal_tier.ravel().astype(np.int8)
        coast_t_clean = np.where(coast_t == 99, np.int8(5), coast_t)
        wrf_terrain_coastal = np.minimum(self.wrf.terrain_tier.ravel(), coast_t_clean)

        # HRRR feature tier at each WRF pixel (nearest-neighbour lookup on 3km grid)
        hrrr_feat_at_wrf = _cell_tier_for(wrf_x_flat, wrf_y_flat)

        # Effective tier: finest of WRF terrain/coastal AND HRRR feature
        effective_tier = np.minimum(wrf_terrain_coastal, hrrr_feat_at_wrf)

        # Per-HRRR-cell minimum effective WRF tier (decides which cells get background points)
        wrf_hrrr_i = np.clip(np.round((wrf_y_flat - _hrrr_y0) / _hrrr_dy).astype(int), 0, ny_h - 1)
        wrf_hrrr_j = np.clip(np.round((wrf_x_flat - _hrrr_x0) / _hrrr_dx).astype(int), 0, nx_h - 1)
        hrrr_flat_idx = wrf_hrrr_i * nx_h + wrf_hrrr_j
        min_wrf_tier_hrrr = np.full(ny_h * nx_h, np.int8(5), dtype=np.int8)
        np.minimum.at(min_wrf_tier_hrrr, hrrr_flat_idx, effective_tier)
        min_wrf_tier_hrrr_2d = min_wrf_tier_hrrr.reshape(ny_h, nx_h)

        # ── WRF-pixel loop: Tiers 1, 2, 3 ────────────────────────────────────
        # Vectorised: generate sub-grid for ALL pixels at tier T simultaneously.
        print("\n[2/3] WRF-pixel sub-grid generation (Tiers 1–3)...")
        pixel_m = float(self.wrf.pixel_m)
        wrf_terr_flat  = self.wrf.terrain_tier.ravel()
        wrf_coast_flat = coast_t_clean

        SKI_BIT = 1 << 4  # matches metadata bit 4 (ski resort)

        for T in [0, 1, 2, 3]:
            resolution_m = float(config.TIER_RESOLUTIONS[T])
            mask = (effective_tier == T)
            n_pix = int(mask.sum())
            if n_pix == 0:
                print(f"  Tier {T}: no WRF pixels — skipping")
                continue

            px = wrf_x_flat[mask]
            py = wrf_y_flat[mask]

            # Metadata: coastal bit if coastal_tier drove the tier; terrain bit if terrain did.
            # Tier 0 pixels originate from HRRR feature buffer (ski resorts); neither coastal
            # nor terrain thresholds reach 0, so set ski bit directly.
            pix_meta = np.zeros(n_pix, dtype=np.uint16)
            if T == 0:
                pix_meta[:] |= SKI_BIT
            else:
                pix_meta[wrf_coast_flat[mask] <= T] |= COAST_BIT
                pix_meta[wrf_terr_flat[mask]  <= T] |= TERRAIN_BIT

            # Sub-grid within each ~pixel_m box, aligned to resolution_m grid
            n_sub = max(1, round(pixel_m / resolution_m))
            if n_sub == 1:
                all_x = px
                all_y = py
                all_meta = pix_meta
            else:
                offsets = (np.arange(n_sub) - (n_sub - 1) / 2.0) * resolution_m
                dx_g, dy_g = np.meshgrid(offsets, offsets)
                dx_f = dx_g.ravel()    # n_sub² offsets
                dy_f = dy_g.ravel()
                # (n_pix, 1) + (1, n_sub²) → (n_pix, n_sub²) → ravel
                all_x = (px[:, None] + dx_f[None, :]).ravel()
                all_y = (py[:, None] + dy_f[None, :]).ravel()
                all_meta = np.repeat(pix_meta, n_sub * n_sub)

            all_lons_t, all_lats_t = hrrr_proj(all_x, all_y, inverse=True)
            n_pts = len(all_lats_t)
            pts_arr = np.empty((n_pts, 3), dtype=np.float64)
            pts_arr[:, 0] = all_lats_t
            pts_arr[:, 1] = all_lons_t
            pts_arr[:, 2] = T
            all_point_arrays.append(pts_arr)
            all_meta_arrays.append(all_meta)
            print(f"  Tier {T} ({resolution_m:.0f}m): {n_pix:,} WRF pixels × {n_sub}²"
                  f" sub-pts = {n_pts:,} points")

        # ── HRRR background: Tier 5 ───────────────────────────────────────────
        # One point at each HRRR cell centre where ALL WRF pixels within are Tier 5.
        print("\n[3/3] HRRR background (Tier 5, 3000m)...")
        bg_mask = (min_wrf_tier_hrrr_2d == 5)
        n_bg = int(bg_mask.sum())
        if n_bg > 0:
            bg_lats = lats_hrrr[bg_mask]
            bg_lons = lons_hrrr[bg_mask]
            pts_bg = np.empty((n_bg, 3), dtype=np.float64)
            pts_bg[:, 0] = bg_lats
            pts_bg[:, 1] = bg_lons
            pts_bg[:, 2] = 5
            all_point_arrays.append(pts_bg)
            all_meta_arrays.append(np.zeros(n_bg, dtype=np.uint16))
            print(f"  {n_bg:,} background cells → {n_bg:,} points")

        # ── Polygon-based golf course generation (Tier 2, 375m) ──────────────
        # Points are placed only within each course's actual polygon boundary,
        # on the globally-aligned HRRR LCC grid.  This replaces the old
        # cell-based approach that filled entire 3km HRRR cells.
        if config.INCLUDE_GOLF_COURSES:
            print(f"\n[Golf] Polygon-based golf course points "
                  f"(Tier {config.GOLF_COURSE_TIER}, {resolution_golf:.0f}m)...")
            golf_courses = self.loader.data.get('golf_courses')
            if golf_courses is not None and len(golf_courses) > 0:
                golf_lcc = golf_courses.to_crs(hrrr_crs_str)
                golf_pts_arrays = []
                golf_meta_arrays = []

                for geom in tqdm(golf_lcc.geometry,
                                 desc="      Golf polygons", unit="course", leave=True):
                    if geom is None or geom.is_empty:
                        continue

                    # Fallback for poorly-defined geometries
                    if geom.geom_type in ('Point', 'MultiPoint'):
                        geom = geom.buffer(150)
                    elif geom.geom_type in ('LineString', 'MultiLineString'):
                        geom = geom.buffer(50)

                    # Drop interior holes; buffer by half grid-spacing so thin
                    # fairways (< 375m wide) still capture at least one row.
                    if geom.geom_type == 'Polygon':
                        if list(geom.interiors):
                            geom = Polygon(geom.exterior)
                    elif geom.geom_type == 'MultiPolygon':
                        geom = geom.__class__([Polygon(p.exterior) for p in geom.geoms])
                    geom = geom.buffer(resolution_golf / 2.0)

                    bounds = geom.bounds
                    k_x0 = math.ceil( (bounds[0] - golf_grid_x0) / resolution_golf)
                    k_x1 = math.floor((bounds[2] - golf_grid_x0) / resolution_golf)
                    k_y0 = math.ceil( (bounds[1] - golf_grid_y0) / resolution_golf)
                    k_y1 = math.floor((bounds[3] - golf_grid_y0) / resolution_golf)

                    if k_x1 < k_x0 or k_y1 < k_y0:
                        continue

                    xs = golf_grid_x0 + np.arange(k_x0, k_x1 + 1) * resolution_golf
                    ys = golf_grid_y0 + np.arange(k_y0, k_y1 + 1) * resolution_golf
                    xx, yy = np.meshgrid(xs, ys)
                    pts_x = xx.ravel()
                    pts_y = yy.ravel()

                    prepared = prep(geom)
                    within = np.array([prepared.contains(Point(x, y))
                                       for x, y in zip(pts_x, pts_y)])

                    kept_x = pts_x[within]
                    kept_y = pts_y[within]
                    if len(kept_x) == 0:
                        continue

                    # Drop points whose HRRR cell already provides equal-or-finer
                    # resolution from the cell-based loop (finest criterion wins).
                    cell_t = _cell_tier_for(kept_x, kept_y)
                    finer  = cell_t > config.GOLF_COURSE_TIER
                    kept_x = kept_x[finer]
                    kept_y = kept_y[finer]
                    if len(kept_x) == 0:
                        continue

                    lons_out, lats_out = hrrr_proj(kept_x, kept_y, inverse=True)
                    n_g = len(kept_x)
                    g_pts = np.empty((n_g, 3), dtype=np.float64)
                    g_pts[:, 0] = lats_out
                    g_pts[:, 1] = lons_out
                    g_pts[:, 2] = config.GOLF_COURSE_TIER
                    golf_pts_arrays.append(g_pts)
                    golf_meta_arrays.append(np.full(n_g, GOLF_BIT, dtype=np.uint16))

                if golf_pts_arrays:
                    golf_combined = np.vstack(golf_pts_arrays)
                    all_point_arrays.append(golf_combined)
                    all_meta_arrays.append(np.concatenate(golf_meta_arrays))
                    golf_count = len(golf_combined)
                else:
                    golf_count = 0
                print(f"      ✓ {golf_count:,} polygon-based golf points "
                      f"(Tier {config.GOLF_COURSE_TIER}, {resolution_golf:.0f}m)")

        # Assemble final arrays
        points_array = np.vstack(all_point_arrays)
        metadata_array = np.concatenate(all_meta_arrays)

        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f" POINT GENERATION COMPLETE")
        print("="*70)
        print(f"  Total points: {len(points_array):,}")
        print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        print(f"\n✓ Total points generated: {len(points_array):,} (no budget cap)")

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
            ncfile.tier0_resolution_m = config.TIER_RESOLUTIONS[0]
            ncfile.tier1_resolution_m = config.TIER_RESOLUTIONS[1]
            ncfile.tier2_resolution_m = config.TIER_RESOLUTIONS[2]
            ncfile.tier3_resolution_m = config.TIER_RESOLUTIONS[3]
            ncfile.tier5_resolution_m = config.TIER_RESOLUTIONS[5]

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
            tier_var.long_name = 'Grid Tier (lower = finer resolution)'
            tier_var.description = ('Tier 0: 93.75m (ski/golf), '
                                    'Tier 1: 187.5m (coastlines/lakes/extreme terrain), '
                                    'Tier 2: 375m (urban/rugged terrain), '
                                    'Tier 3: 750m (suburban/roads/forests/moderate terrain), '
                                    'Tier 5: 3000m (background, HRRR native)')
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
        """Create HRRR-domain point density map using native LCC projection.

        Matches the plotting conventions of visualize_grid.py:
          - HRRR LCC parameters (lat_0=38.5, lon_0=-97.5, lat_1=lat_2=38.5)
          - Map extent set via width/height computed from actual HRRR grid
          - KDTree built in projection space (metres), not lat/lon degrees
          - np.bincount for fast cell assignment
          - Discrete colour bins with ListedColormap
          - Parallels/meridians labelled
        """
        print("\n" + "="*70)
        print(" STEP 5/5: CREATING VISUALIZATION")
        print("="*70)

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        print(f"\nOutput file: {output_path}")

        lats_hrrr = self.hrrr_grid['lats']
        lons_hrrr = self.hrrr_grid['lons']
        shape     = lats_hrrr.shape

        start_time = time.time()

        # ── 1. Build HRRR native LCC projection sized to the actual grid ──
        print("\n[1/4] Building HRRR LCC projection...")
        lat_0, lon_0, lat_1, lat_2 = 38.5, -97.5, 38.5, 38.5

        temp_proj = Basemap(
            projection='lcc', lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            llcrnrlat=21, urcrnrlat=53, llcrnrlon=-135, urcrnrlon=-60,
            resolution=None
        )
        x_hrrr, y_hrrr = temp_proj(lons_hrrr, lats_hrrr)
        width  = x_hrrr.max() - x_hrrr.min()
        height = y_hrrr.max() - y_hrrr.min()

        m = Basemap(
            projection='lcc', lat_0=lat_0, lon_0=lon_0,
            lat_1=lat_1, lat_2=lat_2,
            width=width, height=height, resolution='l', area_thresh=1000
        )

        # ── 2. Count adaptive grid points per HRRR cell in projection space ──
        print("[2/4] Computing point density per HRRR cell...")
        hrrr_pts_proj = np.column_stack([x_hrrr.ravel(), y_hrrr.ravel()])
        tree = cKDTree(hrrr_pts_proj)

        x_adap, y_adap = temp_proj(self.points[:, 1], self.points[:, 0])
        _, indices = tree.query(np.column_stack([x_adap, y_adap]), k=1)

        point_density = np.bincount(
            indices, minlength=hrrr_pts_proj.shape[0]
        ).reshape(shape)

        print(f"    Max points per cell:  {point_density.max():,}")
        print(f"    Mean points per cell: {point_density.mean():.1f}")

        # ── 3. Discrete colour bins (same as visualize_grid.py) ──────────────
        bins   = [0, 5, 10, 50, 100, 200, 400, 800, 1000, 1100]
        labels = ['≤4','5-9','10-49','50-99','100-199',
                  '200-399','400-799','800-999','1000+']
        colors = [
            'White',   '#C4E8FF', '#8FB3FF', '#42F742',
            'Yellow',  'Gold',    'Orange',  '#F6A3AE', 'Orchid',
        ]
        cmap = matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)

        # ── 4. Plot ────────────────────────────────────────────────────────────
        print("[3/4] Drawing map...")
        fig = plt.figure(figsize=(18, 11))
        ax  = fig.add_axes([0.05, 0.05, 0.85, 0.90])

        m.drawcoastlines(linewidth=0.5, color='black', zorder=5)
        m.drawcountries(linewidth=0.5, color='black', zorder=5)
        m.drawstates(linewidth=0.3,   color='gray',  zorder=5)
        m.drawparallels(np.arange(20, 55, 5),   labels=[1,0,0,0],
                        fontsize=13.5, linewidth=0.3, color='gray')
        m.drawmeridians(np.arange(-130, -60, 10), labels=[0,0,0,1],
                        fontsize=13.5, linewidth=0.3, color='gray')

        x_map, y_map = m(lons_hrrr, lats_hrrr)
        density_masked = np.ma.masked_where(point_density == 0, point_density)

        pcm = m.pcolormesh(
            x_map, y_map, density_masked,
            cmap=cmap, norm=norm, shading='auto', zorder=1, rasterized=True
        )

        cax  = fig.add_axes([0.92, 0.15, 0.02, 0.70])
        cbar = plt.colorbar(pcm, cax=cax, spacing='uniform', extend='max')
        tick_pos = [(bins[i] + bins[i+1]) / 2.0 for i in range(len(labels))]
        cbar.set_ticks(tick_pos)
        cbar.set_ticklabels(labels, fontsize=13)
        cbar.set_label('Points per HRRR Cell', fontsize=14)

        total_pts = int(point_density.sum())
        n_cells   = int((point_density > 0).sum())
        ax.set_title(
            f'Adaptive Grid Point Density for XGBoost Weather Downscaling\n'
            f'Total Points: {total_pts:,} | '
            f'HRRR Cells with Points: {n_cells:,} / {point_density.size:,} '
            f'({100*n_cells/point_density.size:.1f}%)',
            fontsize=20, pad=10
        )

        print(f"[4/4] Saving {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"\n✓ Visualization saved: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB  |  Time: {elapsed:.1f}s")
        return output_path


def main():
    """Main execution"""
    overall_start = time.time()

    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION FOR XGBOOST WEATHER DOWNSCALING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target points: no budget cap")
    print(f"  Tier resolutions: {config.TIER_RESOLUTIONS}")
    print(f"  WRF terrain variability thresholds ({config.TERRAIN_WINDOW_KM}km window):")
    print(f"    Tier 1: σ>{config.TERRAIN_TIER1_THRESHOLD}m (extreme)")
    print(f"    Tier 2: σ>{config.TERRAIN_TIER2_THRESHOLD}m (rugged)")
    print(f"    Tier 3: σ>{config.TERRAIN_TIER3_THRESHOLD}m (moderate)")
    print(f"  Coastal bands (WRF land_sea_mask): {config.COASTAL_BAND_KM} km")
    print(f"\nEstimated runtime: 30-60 minutes")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = DataLoader()
    loader.load_all()

    # WRF terrain & coastal classification
    wrf_analyzer = WrfTerrainAnalyzer().load_and_classify()

    # Classify feature-based tiers on HRRR grid
    classifier = TierClassifier(loader)
    tier_map, metadata_map = classifier.create_tier_map()

    # Generate points
    generator = AdaptiveGridGenerator(loader, tier_map, metadata_map, wrf_analyzer)
    points, metadata = generator.generate_points()

    # Write outputs
    writer = OutputWriter(points, metadata, loader.hrrr_grid)
    nc_file = writer.write_netcdf('adaptive_grid_points_intermediate.nc')
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
