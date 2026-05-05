"""
Configuration snapshot: adaptive_grid_points_23M.nc
Created: 2026-04-15 07:34
Total points: 23,402,599

Tier breakdown:
  Tier 0:  1,843,516 pts  ( 7.9%)  93.75m  — ski resorts, golf courses
  Tier 1:  7,244,721 pts  (31.0%)  187.5m  — coastline 0–0.5 km
  Tier 2:  3,917,909 pts  (16.7%)  375m    — coastline 0.5–1 km, urban ≥50k, extreme terrain
  Tier 3:  3,874,737 pts  (16.6%)  750m    — coastline 1–2 km, rugged terrain, national forests
  Tier 4:  6,521,716 pts  (27.9%)  1500m   — background, suburbs, roads, moderate terrain

To restore: copy these values back into config.py, revert the hardcoded tier
assignments in generate_adaptive_grid.py (extreme→2, rugged→3, moderate→4),
and revert preprocess_feature_buffers.py (suburbs/roads/forests → all_geometries[4]).
Then re-run: python3 preprocess_feature_buffers.py
"""

# ── Tier resolutions (unchanged across configs) ─────────────────────────────
TIER_RESOLUTIONS = {
    0: 93.75,
    1: 187.5,
    2: 375,
    3: 750,
    4: 1500,
    5: 3000,
}

# ── Coastline transition bands ───────────────────────────────────────────────
TRANSITION_THRESHOLDS = {
    1: [0.0, 0.5],       # 0–0.5 km  → Tier 1 (187.5m)
    2: [0.5, 1.0],       # 0.5–1 km  → Tier 2 (375m)
    3: [1.0, 2.0],       # 1–2 km    → Tier 3 (750m)
    4: [2.0, 3.0],       # 2–3 km    → Tier 4 (1500m)
    5: [3.0, float('inf')],
}

# ── Terrain thresholds (σ of HRRR terrain elevation in 10 km window) ────────
# generate_adaptive_grid.py assigns tiers with hardcoded numbers:
#   extreme_terrain → min(tier_map, 2)   extreme terrain → Tier 2 (375m)
#   rugged_terrain  → min(tier_map, 3)   rugged terrain  → Tier 3 (750m)
#   moderate_terrain → min(tier_map, 4)  moderate terrain → Tier 4 (1500m, no-op)
TERRAIN_TIER1_THRESHOLD = 600    # σ > 600m → Tier 2 (extreme)
TERRAIN_TIER2_THRESHOLD = 400    # σ > 400m → Tier 3 (rugged)
TERRAIN_TIER3_THRESHOLD = 100    # σ > 100m → Tier 4 (moderate, same as background)

# ── Recreation features ──────────────────────────────────────────────────────
GOLF_COURSE_TIER = 0             # polygon-interior at 93.75m
SKI_RESORT_BUFFER_KM = 4.0       # 4 km buffer → Tier 0

# ── Lakes ────────────────────────────────────────────────────────────────────
GREAT_LAKES_BUFFER_KM = 0.5      # shoreline only → Tier 2 (via cached buffer Tier 2 bucket)
INLAND_LAKES_TIER = 2
INLAND_LAKES_BUFFER_KM = 0.5
INLAND_LAKES_MIN_COASTLINE_KM = 30

# ── Urban / suburban ─────────────────────────────────────────────────────────
URBAN_BUFFER_KM = 1.5            # Urbanized Areas ≥50k → Tier 3 (cached buffer)
SUBURBAN_BUFFER_KM = 1.5         # Urban Clusters → Tier 4 (hardcoded in code/preprocessor)
# Note: no SUBURBAN_TIER variable existed; tier was hardcoded as 4

# ── Roads ────────────────────────────────────────────────────────────────────
INCLUDE_HIGHWAYS = True
HIGHWAY_BUFFER_KM = 0.5          # TIGER Primary Roads → Tier 4 (hardcoded in code/preprocessor)
# Note: no HIGHWAY_TIER variable existed; tier was hardcoded as 4

# ── National forests ─────────────────────────────────────────────────────────
INCLUDE_NATIONAL_FORESTS = True
NATIONAL_FOREST_TIER = 4         # S_USA.AdministrativeForest minus NPS boundaries
EXCLUDE_PARKS_FROM_FORESTS = True
