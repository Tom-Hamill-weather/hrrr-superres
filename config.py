"""
Configuration settings for adaptive grid generation
"""
import os

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# WRF 1-km terrain file (used for terrain std-dev classification and land/sea mask)
WRF_TERRAIN_FILE = os.path.join(BASE_DIR, 'WRF_CONUS_terrain_info.nc')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# GEN2 TIER SYSTEM (New adaptive grid generation)
# ============================================

# GEN2: Tier resolutions (in meters)
# HIERARCHICAL SUBDIVISION OF HRRR GRID (3km cells)
# Each tier subdivides by powers of 2 for perfect nesting
# HRRR cell = 3000m, subdivisions: /2, /4, /8, /16, /32
TIER_RESOLUTIONS = {
    0: 93.75,   # Tier 0: 93.75m  (3000/32) - 32×32 = 1024 points per HRRR cell - Coastlines, lakes, ski
    1: 187.5,   # Tier 1: 187.5m  (3000/16) - 16×16 = 256 points per HRRR cell - Rugged terrain, major urban
    2: 375,     # Tier 2: 375m    (3000/8)  - 8×8 = 64 points per HRRR cell - Moderate terrain, suburban
    3: 750,     # Tier 3: 750m    (3000/4)  - 4×4 = 16 points per HRRR cell - Slightly rugged terrain
    4: 1500,    # Tier 4: 1500m   (3000/2)  - 2×2 = 4 points per HRRR cell - Transition tier
    5: 3000     # Tier 5: 3000m   (3000/1)  - 1×1 = 1 point per HRRR cell - Background (HRRR native)
}

# GEN2: Terrain variability thresholds (meters std dev)
# Applied at WRF ~800m pixel level (not HRRR 3km cell level).
# Std dev is computed over TERRAIN_WINDOW_KM geographic window on the WRF terrain grid.
# Only land pixels receive terrain-based tier assignment; water pixels use coastal tiers.
#   σ > TERRAIN_TIER1_THRESHOLD → Tier 1 (187.5m)
#   σ > TERRAIN_TIER2_THRESHOLD → Tier 2 (375m)
#   σ > TERRAIN_TIER3_THRESHOLD → Tier 3 (750m)
#   σ ≤ TERRAIN_TIER3_THRESHOLD → Tier 5 background (3km)
TERRAIN_TIER1_THRESHOLD = 200     # meters std dev → Tier 1 (187.5m) — most extreme terrain
TERRAIN_TIER2_THRESHOLD = 100     # meters std dev → Tier 2 (375m)   — very rugged terrain
TERRAIN_TIER3_THRESHOLD = 40      # meters std dev → Tier 3 (750m)   — moderate terrain
TERRAIN_WINDOW_KM = 10.0          # Geographic window for std dev (10km)

# GEN2: Coastal band distances (km from nearest land/water boundary)
# Applied using the WRF land_sea_mask + distance transform.
# Ocean-connected water cells AND nearby land cells receive these coastal tier assignments.
# The tier is a lower bound — terrain/feature assignments are kept if they are finer.
#   0 – COASTAL_BAND_KM[0]: Tier 1 (187.5m)
#   COASTAL_BAND_KM[0] – COASTAL_BAND_KM[1]: Tier 2 (375m)
#   COASTAL_BAND_KM[1] – COASTAL_BAND_KM[2]: Tier 3 (750m)
#   Beyond COASTAL_BAND_KM[2]: no coastal assignment (background / terrain drives)
COASTAL_BAND_KM = [0.75, 1.5, 2.25]   # Tier 1 / Tier 2 / Tier 3 outer edges

# GEN2: Tier constraint enforcement parameters
# Ensures no adjacent cells differ by more than MAX_TIER_JUMP tiers
MAX_TIER_JUMP = 1              # Maximum tier difference between adjacent cells
CONSTRAINT_MAX_ITERATIONS = 10  # Maximum smoothing iterations
CONSTRAINT_CONVERGENCE_TOLERANCE = 0.01  # Convergence threshold for early stopping

# GEN2: Transition algorithm selection
# 'hybrid' (recommended): Distance blending + constraint enforcement
# 'distance_only': Distance-based assignment only
# 'constraint_only': Core classification + constraint enforcement
TRANSITION_ALGORITHM = 'hybrid'

# GEN2: Distance weighting function for blending
# 'inverse': weight = 1 / (1 + distance)
# 'gaussian': weight = exp(-(distance^2) / (2 * sigma^2))
DISTANCE_WEIGHT_FUNCTION = 'inverse'
DISTANCE_WEIGHT_SIGMA = 3.0  # Sigma for Gaussian weighting (in HRRR cells)

# No target point total — point count is determined entirely by tier assignments
TARGET_TOTAL_POINTS = None

# CONUS domain (will be defined by HRRR grid)
# Standard HRRR CONUS: ~1799 x 1059 grid points at 3km

# AWS HRRR data URLs
HRRR_AWS_BASE = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
# HRRR grid files are available from NOAA repositories

# ============================================
# GEN2: FEATURE BUFFERS AND CLASSIFICATION
# ============================================

# Tier 0 features (93.75m resolution, 1024 pts/cell - VERY EXPENSIVE!)
LAKE_BUFFER_KM = 1.0               # ±1.0km buffer around large lake shorelines
LAKE_MIN_AREA_KM2 = 2000           # Minimum lake size (2000km² - ONLY largest Great Lakes)
INCLUDE_LAKES_IN_TIER0 = True      # Include large lake shorelines in Tier 0

# Lake strategy: Two-tier approach
# Great Lakes (5 largest): Treat like coastlines - high res along shores only
GREAT_LAKES_NAMES = ['Superior', 'Michigan', 'Huron', 'Erie', 'Ontario']  # The five Great Lakes
GREAT_LAKES_BUFFER_KM = 0.5        # Shoreline buffer for Great Lakes (Tier 1, like coastlines)

# Inland lakes: Intermediate resolution, not abandoned
INLAND_LAKES_MIN_COASTLINE_KM = 30  # Minimum coastline length for inland lakes
INLAND_LAKES_TIER = 2              # Tier 2 (375m resolution) for inland lakes
INLAND_LAKES_BUFFER_KM = 0.5       # Smaller buffer for inland lakes
SKI_RESORT_BUFFER_KM = 4.0         # Ski resorts in Tier 0 with 4km buffer

# Tier 1 features (187.5m resolution, 256 pts/cell - EXPENSIVE!)
# Use sparingly - only for critical high-resolution needs
PARKS_BUFFER_KM = 0.0              # No parks (too many points)
INCLUDE_PARKS = False              # Disabled
SKI_RESORT_BUFFER_KM_TIER1 = 0.0   # Ski resorts moved to Tier 0
# Major urban areas: Tier 2 (375m resolution)

# Specific water bodies for Tier 1 (187.5m resolution)
SPECIFIC_WATER_BODIES = {
    'puget_sound': {
        'coords': [(-122.9, 47.1), (-122.2, 47.1), (-122.2, 48.4), (-122.9, 48.4)],
        'name': 'Puget Sound'
    },
    'san_francisco_bay': {
        'coords': [(-122.6, 37.4), (-121.8, 37.4), (-121.8, 38.1), (-122.6, 38.1)],
        'name': 'San Francisco Bay'
    },
    'san_diego_harbor': {
        'coords': [(-117.3, 32.6), (-117.0, 32.6), (-117.0, 32.8), (-117.3, 32.8)],
        'name': 'San Diego Harbor'
    },
    'long_island_sound': {
        'coords': [(-73.8, 40.8), (-72.0, 40.8), (-72.0, 41.3), (-73.8, 41.3)],
        'name': 'Long Island Sound'
    },
    'narragansett_bay': {
        'coords': [(-71.5, 41.3), (-71.2, 41.3), (-71.2, 41.7), (-71.5, 41.7)],
        'name': 'Narragansett Bay'
    }
}

# Tier 2 features (375m resolution, 64 pts/cell - MODERATE COST)
HIGHWAY_BUFFER_KM = 0.5            # 500m buffer around major highways
INCLUDE_HIGHWAYS = True            # Enable/disable highway classification (can be slow - disable for testing)
# Urban area buffers: Census UA polygons are fragmented (trace individual parcels).
# A 1.5 km buffer (half HRRR cell width) ensures every HRRR cell partially
# overlapping a UA polygon gets classified — without this, most cell centers
# land in the gaps between polygon fragments and are missed entirely.
URBAN_BUFFER_KM = 1.5              # Buffer applied to Urbanized Areas (≥50k pop) → Tier 2
SUBURBAN_BUFFER_KM = 1.5          # Buffer applied to Urban Clusters (suburban) → Tier 3

# Golf course configuration (Tier 0: 93.75m resolution, polygon interior only)
INCLUDE_GOLF_COURSES = True        # Enable golf courses
GOLF_COURSE_BUFFER_KM = 2.5        # buffer used only for post-cache cleanup of old cache files
GOLF_COURSE_TIER = 0               # Tier 0 (93.75m, 1024 pts/cell)

# Suburban areas tier (Urban Clusters — smaller population centers)
SUBURBAN_TIER = 2                  # Tier 2 (375m resolution)

# Major roads tier (TIGER Primary Roads)
HIGHWAY_TIER = 3                   # Tier 3 (750m resolution)

# National forest configuration
INCLUDE_NATIONAL_FORESTS = True    # Enable national forests
NATIONAL_FOREST_TIER = 3           # Tier 3 (750m resolution)
EXCLUDE_PARKS_FROM_FORESTS = True  # Subtract NPS boundaries from forests

# ============================================
# PREPROCESSING / CACHING (legacy generate_adaptive_grid.py)
# ============================================
# Use cached feature buffers (run preprocess_feature_buffers.py first)
USE_CACHED_BUFFERS = True           # Enable to use preprocessed buffers (saves 5-10 min per run)
CACHED_BUFFERS_FILE = None          # Auto-detect: data/preprocessed/tier_buffers.gpkg

# ============================================
# HI-RES BINARY GRID (generate_hires_points.py)
# ============================================
# New binary-mask patch-based point generator.
# Prerequisites: python3 preprocess_hires_features.py
HIRES_PATCH_SIZE  = 64             # Patch size in HRRR cells (64 × 32 = 2048 hi-res pixels)
HIRES_OUTPUT_FILE = 'hires_points.nc'   # Output filename in OUTPUT_DIR

# ============================================
# FAST TESTING MODE
# ============================================
# For faster testing, disable slow data loading
FAST_TEST_MODE = False              # Test with all features enabled

# Data source URLs and settings
DATA_SOURCES = {
    'hrrr_grid': {
        'url': 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20150101/conus/hrrr.t00z.wrfsubhf00.grib2',
        'description': 'HRRR grid definition and terrain'
    },
    'protected_areas': {
        'url': 'https://gapanalysis.usgs.gov/apps/gapanalysis/downloads/PAD-US/',
        'description': 'PAD-US protected areas database'
    },
    'urban_areas': {
        'url': 'https://www2.census.gov/geo/tiger/TIGER2020/UAC/',
        'description': 'Census Urban Areas 2020'
    },
    'primary_roads': {
        'url': 'https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/',
        'description': 'TIGER Primary Roads'
    },
    'coastline': {
        'source': 'natural_earth',
        'description': 'Natural Earth coastline data'
    },
    'lakes': {
        'source': 'natural_earth',
        'description': 'Natural Earth lakes data'
    },
    'ski_resorts': {
        'url': 'https://raw.githubusercontent.com/openskistats/openskistats.github.io/master/data/ski_areas.geojson',
        'description': 'OpenSkiStats ski resort locations'
    }
}

# Visualization settings
PLOT_DPI = 150
PLOT_FIGSIZE = (16, 10)
COLORMAP = 'viridis'

# ============================================
# CACHE FINGERPRINTING
# ============================================
# Every config value that affects the geometry written by preprocess_feature_buffers.py
# is collected here.  Changing any of these requires re-running the preprocessor.
# Excluded intentionally:
#   - TERRAIN_* thresholds and COASTAL_BAND_KM: applied at runtime from WRF data, not cached
#   - TIER_RESOLUTIONS: affects point density only, not cached buffer geometries
CACHE_RELEVANT_CONFIG = {
    'SKI_RESORT_BUFFER_KM':          SKI_RESORT_BUFFER_KM,
    'GREAT_LAKES_BUFFER_KM':         GREAT_LAKES_BUFFER_KM,
    'INLAND_LAKES_BUFFER_KM':        INLAND_LAKES_BUFFER_KM,
    'INLAND_LAKES_TIER':             INLAND_LAKES_TIER,
    'INLAND_LAKES_MIN_COASTLINE_KM': INLAND_LAKES_MIN_COASTLINE_KM,
    'LAKE_MIN_AREA_KM2':             LAKE_MIN_AREA_KM2,
    'INCLUDE_LAKES_IN_TIER0':        INCLUDE_LAKES_IN_TIER0,
    'URBAN_BUFFER_KM':               URBAN_BUFFER_KM,
    'SUBURBAN_BUFFER_KM':           SUBURBAN_BUFFER_KM,
    'SUBURBAN_TIER':                 SUBURBAN_TIER,
    'HIGHWAY_BUFFER_KM':             HIGHWAY_BUFFER_KM,
    'HIGHWAY_TIER':                  HIGHWAY_TIER,
    'INCLUDE_HIGHWAYS':              INCLUDE_HIGHWAYS,
    'NATIONAL_FOREST_TIER':          NATIONAL_FOREST_TIER,
    'INCLUDE_NATIONAL_FORESTS':      INCLUDE_NATIONAL_FORESTS,
    'EXCLUDE_PARKS_FROM_FORESTS':    EXCLUDE_PARKS_FROM_FORESTS,
    'INCLUDE_PARKS':                 INCLUDE_PARKS,
    'PARKS_BUFFER_KM':               PARKS_BUFFER_KM,
}


def cache_config_hash():
    """Return an MD5 hex digest of all cache-relevant config values.

    Used by preprocess_feature_buffers.py (to stamp the GPKG) and by
    generate_adaptive_grid.py (to verify the cache is still current).
    """
    import hashlib
    import json
    serialized = json.dumps(CACHE_RELEVANT_CONFIG, sort_keys=True,
                            default=str, allow_nan=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def coast_config_hash():
    """Retained for backward compatibility; no longer used (coastal bands are
    computed at runtime from the WRF land_sea_mask, not cached as ring polygons).
    """
    import hashlib
    import json
    serialized = json.dumps({'COASTAL_BAND_KM': COASTAL_BAND_KM},
                            sort_keys=True, default=str, allow_nan=True)
    return hashlib.md5(serialized.encode()).hexdigest()
