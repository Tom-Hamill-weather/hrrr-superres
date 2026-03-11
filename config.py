"""
Configuration settings for adaptive grid generation
"""
import os

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

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
# Re-enabled with narrow coastal buffer, budget now available for terrain
# Still conservative to balance with rapid coastal transitions
TERRAIN_TIER1_THRESHOLD = 600     # meters std dev → Most extreme terrain only (Tier 1: 187.5m, 256 pts/cell)
TERRAIN_TIER2_THRESHOLD = 400     # meters std dev → Very rugged (Tier 2: 375m, 64 pts/cell)
TERRAIN_TIER3_THRESHOLD = 100     # meters std dev → Moderate terrain (Tier 3: 750m, 16 pts/cell)
# Below TIER3 threshold → Tier 5 background (3km, 1 pt/cell)
TERRAIN_WINDOW_SIZE = 10000    # meters (10km window - needs to be > HRRR cell size)

# GEN2: Transition algorithm parameters
# Controls smooth gradual transitions between tier regions
TRANSITION_DISTANCE_KM = 3.0  # Distance over which transitions occur (matches rapid coast degradation)

# GEN2: Transition tier thresholds (km from nearest core region)
# RAPID TRANSITIONS from coastline to control point budget
# Degradation schedule: 93.75m (coast) → 187.5m (0.5km) → 375m (1km) → 750m (2km)
TRANSITION_THRESHOLDS = {
    1: [0.0, 0.5],      # 0-0.5km from coast: Tier 1 (187.5m - immediate step down)
    2: [0.5, 1.0],      # 0.5-1km: Tier 2 (375m)
    3: [1.0, 2.0],      # 1-2km: Tier 3 (750m - reaches 0.75km at 2km as requested)
    4: [2.0, 3.0],      # 2-3km: Tier 4 (1500m)
    5: [3.0, float('inf')]  # Beyond 3km: Tier 5 (let terrain/urban dominate)
}

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

# GEN2: Target total points (reduced from 50M to aim for ~20M)
TARGET_TOTAL_POINTS = 20_000_000

# CONUS domain (will be defined by HRRR grid)
# Standard HRRR CONUS: ~1799 x 1059 grid points at 3km

# AWS HRRR data URLs
HRRR_AWS_BASE = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
# HRRR grid files are available from NOAA repositories

# ============================================
# GEN2: FEATURE BUFFERS AND CLASSIFICATION
# ============================================

# Tier 0 features (93.75m resolution, 1024 pts/cell - VERY EXPENSIVE!)
# HARD CONSTRAINT: Coastlines and major lakes MUST have finest resolution
# Buffer must catch ALL cells that touch coastline (HRRR cells are 3km wide)
# Using intersects predicate, so buffer = safety margin
COASTLINE_BUFFER_KM = 1.5          # ±1.5km buffer to capture more coastal cells
COASTLINE_BUFFER_ONSHORE_KM = 1.0  # 1.0 km onshore component
COASTLINE_BUFFER_OFFSHORE_KM = 1.0 # 1.0 km offshore component
LAKE_BUFFER_KM = 1.0               # ±1.0km buffer around large lake shorelines
LAKE_MIN_AREA_KM2 = 2000           # Minimum lake size (2000km² - ONLY largest Great Lakes)
INCLUDE_LAKES_IN_TIER0 = True      # Include large lake shorelines in Tier 0
SKI_RESORT_BUFFER_KM = 0.0         # No ski resorts in Tier 0 (moved to Tier 1)

# Tier 1 features (187.5m resolution, 256 pts/cell - EXPENSIVE!)
# Use sparingly - only for critical high-resolution needs
GOLF_COURSE_BUFFER_KM = 0.0        # No golf courses (too many points)
INCLUDE_GOLF_COURSES = False       # Disabled
PARKS_BUFFER_KM = 0.0              # No parks (too many points)
INCLUDE_PARKS = False              # Disabled
SKI_RESORT_BUFFER_KM_TIER1 = 2.0   # Ski resorts moved to Tier 1 with narrow buffer
# Major urban areas: DISABLED for Tier 1 (moved to Tier 2)

# Tier 2 features (500m resolution)
HIGHWAY_BUFFER_KM = 0.5            # 500m buffer around major highways
INCLUDE_HIGHWAYS = False           # Enable/disable highway classification (can be slow - disable for testing)
# Small urban clusters (2.5k-50k population): No buffer, use polygon boundaries directly
# Suburban areas: Use Census-designated suburban area polygons

# ============================================
# FAST TESTING MODE
# ============================================
# For faster testing, disable slow data loading
FAST_TEST_MODE = True              # Skip golf, parks, highways for fast testing

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
