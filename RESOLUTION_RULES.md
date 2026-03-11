# Adaptive Grid Resolution Rules

## Overview

The adaptive grid uses a 6-tier resolution system based on geographic features and terrain variability. Each tier has a different spatial resolution (grid spacing) determined by applying stride patterns to the finest 93.75m base grid.

**Base Resolution**: 93.75m (HRRR 3km grid divided by 32)

---

## Tier Definitions

### Tier 0: 93.75m resolution (stride=1, keep all points)
**Features**:
- **Ski resorts**: 2km buffer around ski resort boundaries
- **Hiking trails**: 1km buffer around National Recreation Trails and State-designated trails

**Purpose**: Highest resolution for outdoor recreation areas where detailed weather conditions are critical for safety and planning.

**Typical locations**: Major ski resorts, popular hiking trails (Pacific Crest Trail, Appalachian Trail, etc.)

---

### Tier 1: 187.5m resolution (stride=2, keep every 2nd point in each dimension)
**Features**:
- **Coastlines**: 750m buffer from ocean shorelines
- **Lake shores**: 750m buffer from major lake boundaries (>30km coastline)
- **Extreme terrain**: Terrain standard deviation >800m in 3×3 HRRR cell window

**Purpose**: High resolution for land-sea interfaces and extremely rugged mountain areas where sharp gradients exist.

**Typical locations**: Pacific coast, Great Lakes, Rocky Mountains peaks

---

### Tier 2: 375m resolution (stride=4, keep every 4th point)
**Features**:
- **Coastlines**: 1500m buffer from ocean shorelines
- **Lake shores**: 1500m buffer from major lakes
- **High-density urban**: Cities and metropolitan areas (≥50k population)
- **Suburban**: Large urban clusters (2.5k-50k population, upper 40%)
- **Very rugged terrain**: Terrain standard deviation >600m

**Purpose**: Enhanced resolution for population centers and moderately rough terrain.

**Typical locations**: Los Angeles, Chicago, Seattle metro areas; Sierra Nevada, Cascades

---

### Tier 3: 750m resolution (stride=8, keep every 8th point)
**Features**:
- **Coastlines**: 3km buffer from ocean shorelines
- **Lake shores**: 3km buffer from major lakes
- **Small towns**: Small urban clusters (2.5k-50k population, lower 60%)
- **Major roads**: 1km buffer around primary highways
- **National parks**: Park boundaries
- **Rugged terrain**: Terrain standard deviation >300m

**Purpose**: Moderate resolution for smaller communities, transportation corridors, protected areas, and varied terrain.

**Typical locations**: Small cities, interstate highways, Yellowstone/Yosemite, Appalachian Mountains

**Note**: National Forests were previously included at this tier but have been **removed** and replaced by the trail-based approach at Tier 0.

---

### Tier 4: 1.5km resolution (stride=16, keep every 16th point)
**Features**:
- **Coastlines**: 6km buffer from ocean shorelines
- **Lake shores**: 6km buffer from major lakes
- **Moderate terrain**: Terrain standard deviation >150m

**Purpose**: Baseline resolution for areas with some geographic variability.

**Typical locations**: Plains near coasts, rolling hills

---

### Tier 5: 3km resolution (stride=32, regular grid)
**Features**:
- **Background grid**: All remaining areas

**Purpose**: Matches HRRR native resolution for flat, featureless areas.

**Typical locations**: Great Plains, agricultural regions, deserts

---

## Tier Priority and Overlap

When a grid point qualifies for multiple tiers, the **highest resolution tier wins**. Tiers are processed in order from 0 (finest) to 5 (coarsest).

**Example**: A point near the coast AND in high-density urban:
- Qualifies for Tier 1 (coastline 750m)
- Qualifies for Tier 2 (urban)
- **Assigned Tier 1** (higher priority)

All qualifying features are tracked in the metadata flags, so you can see which features influenced each point's resolution.

---

## Metadata Flags

Each grid point has metadata tracking which feature(s) caused its resolution:

| Bit | Feature | Description |
|-----|---------|-------------|
| 0 | Coastline | Near ocean shoreline |
| 1 | Lake | Near lake boundary |
| 2 | Ski resort | Within ski resort buffer |
| 3 | High-density urban | In major metropolitan area |
| 4 | Suburban | In suburban area |
| 5 | Small town | In small town |
| 6 | Major road | Near primary highway |
| 7 | National park | In national park |
| 8 | National forest | **Deprecated** (no longer used) |
| 9 | Extreme terrain | Very rugged topography (σ>800m) |
| 10 | High terrain | Rugged topography (σ>600m) |
| 11 | Moderate terrain | Varied topography (σ>300m or >150m) |
| 12 | Background | Background grid point |
| 13 | **Hiking trail** | **New**: Near recreation trail |

Multiple flags can be set for a single point if it qualifies under multiple criteria.

---

## Key Design Decisions

### Trail-Based Recreation Coverage (New)
**Replaced**: National Forest blanket coverage at Tier 3 (750m)
**With**: Hiking trails at Tier 0 (93.75m) with 1km buffer

**Rationale**:
- Targets highest resolution where people actually recreate (on trails)
- More computationally efficient (~300x faster)
- Better coverage of outdoor recreation areas
- Trail data updated more frequently than forest boundaries

**Trails included**:
- National Recreation Trails (NRT): ~100,000 km
- State-designated trails: ~150,000 km
- Total: ~250,000 km of trails

**Computation approach**: Trail→Grid method iterates through trail points and marks nearby grid cells, rather than checking each grid point against trail geometries. This is much faster for line features.

### Coastline Buffer Degradation
Coastlines have 5 buffer distances (750m, 1.5km, 3km, 6km, 12km) creating a gradual resolution transition from shore to inland. This captures land-sea interaction effects that decay with distance.

### Terrain Variability
Terrain standard deviation computed using a 3×3 HRRR cell window (~9km × 9km). This captures local terrain roughness that affects weather patterns:
- Extreme (>800m): Mountain peaks, deep valleys
- High (>600m): Mountain ranges
- Moderate (>300m): Hills, foothills
- Lower (>150m): Rolling terrain

### Urban Hierarchy
Urban areas split by population density:
- High-density (≥50k): Major metros → Tier 2 (375m)
- Suburban (large clusters): Suburban areas → Tier 2 (375m)
- Small towns (small clusters): Small communities → Tier 3 (750m)

This reflects the importance of detailed forecasts for larger population centers.

---

## Expected Point Counts (CONUS)

Based on recent full-domain run:

| Tier | Resolution | Points | Percentage |
|------|-----------|--------|-----------|
| 0 | 93.75m | ~125k | 1.0% |
| 1 | 187.5m | ~21.7M | 17.1% |
| 2 | 375m | ~61.1M | 48.2% |
| 3 | 750m | ~193.2M | (pre-stride) |
| 4 | 1.5km | ~180.1M | (pre-stride) |
| 5 | 3km | ~1.6M | 12.7% |

**Total**: ~12.7M points after stride application (target: 20M)

**Note**: Tier 3 and 4 contain many points before stride is applied, but only a fraction survive the stride pattern. The total is below target due to forest removal; adding trails is expected to increase Tier 0 significantly.

---

## Computational Efficiency

**Stage 2 processing** (index generation):
- Coastlines/lakes: Patch-based STRtree intersection (~80 min)
- Terrain: Direct HRRR grid processing (~10 sec)
- Urban/roads/parks: Patch-based polygon containment (~80 min)
- **Trails: Trail→Grid rasterization (~5-10 min)**

Total Stage 2: ~170 min for CONUS

**Memory**: 5.7 GB for Stage 2 checkpoint (749M indices before stride)

---

## Output Format

NetCDF file with variables:
- `latitude`: Point latitudes (degrees North)
- `longitude`: Point longitudes (degrees East)
- `tier`: Resolution tier (0-5)
- `metadata`: Bit flags indicating features (16-bit integer)

Use `inspect_metadata.py` to decode metadata and analyze grid composition.

---

## References

- Implementation: `generate_adaptive_grid_SPARSE_v2.py`
- Data loader: `generate_adaptive_grid.py`
- Configuration: `config.py`
- Trail implementation notes: `NOTES_trail_based_resolution.md`
