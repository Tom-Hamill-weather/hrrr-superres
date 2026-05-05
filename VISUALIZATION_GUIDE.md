# Visualization Guide for Adaptive Grid Points

**Date:** 2026-03-18
**For use with:** `output/adaptive_grid_points.nc` (16M points)

---

## Quick Start

### 1. **Overall Grid Visualization** (Already Created!)

The grid generation automatically created:
```
output/adaptive_grid_density.png (701 KB)
```

This shows the full CONUS grid with color-coded point density.

**Alternative version with discrete colors:**
```bash
python visualize_grid.py
```
Creates: `output/adaptive_grid_density_discrete.png (951 KB)`

**Differences:**
- **adaptive_grid_density.png** - Continuous color scale, created automatically
- **adaptive_grid_density_discrete.png** - Discrete color bins, Lambert Conformal projection, geographic features

---

## Detailed Regional Visualization

### 2. **Zoom into Specific Regions** - `visualize_point_grid_binary.py`

**Best for:** Examining point distribution in specific areas of interest

**Usage:**
```bash
python visualize_point_grid_binary.py LAT LON --input-file adaptive_grid_points.nc

# Examples:
python visualize_point_grid_binary.py 33.4 -118.4 --input-file adaptive_grid_points.nc   # Catalina Island
python visualize_point_grid_binary.py 40.7 -74.0 --input-file adaptive_grid_points.nc    # New York City
python visualize_point_grid_binary.py 47.6 -122.3 --input-file adaptive_grid_points.nc   # Seattle (Puget Sound)
python visualize_point_grid_binary.py 37.8 -122.4 --input-file adaptive_grid_points.nc   # San Francisco Bay
python visualize_point_grid_binary.py 44.0 -103.5 --input-file adaptive_grid_points.nc   # Black Hills (terrain)
python visualize_point_grid_binary.py 39.0 -106.0 --input-file adaptive_grid_points.nc   # Colorado Rockies
```

**Options:**
```bash
--box-sizes 0.4 0.2 0.05    # Box sizes in degrees (creates multiple zoom levels)
--input-file FILENAME       # Input NetCDF file (default: adaptive_grid_SPARSE_trails.nc)
```

**What it shows:**
- **Point positions** - Exact lat/lon of each grid point
- **Local spacing** - Points colored by nearest-neighbor distance:
  - 🔴 Red: ~94m spacing (Tier 0 - not used)
  - 🟠 Orange: ~188m spacing (Tier 1)
  - ⚪ Gray: ~375m spacing (Tier 2)
  - 🟢 Green: ~750m spacing (Tier 3)
  - 🔵 Blue: ~1.5km spacing (Tier 4)
  - 🟣 Purple: ~3km spacing (Tier 5)
- **Geographic context** - Coastlines, state/county boundaries
- **Multiple zoom levels** - Creates visualizations at different scales

**Output:** `output/point_grid_binary_LAT_LON_BOXDEGdeg.png`

---

## Recommended Test Locations

### High-Resolution Areas (Tier 1 - 187.5m)
Test these to verify high-resolution coverage:

1. **Catalina Island, CA:** `33.4, -118.4`
   - Coastline buffer (0.5km onshore)
   - Should show dense orange points

2. **Puget Sound, WA:** `47.6, -122.3`
   - Specific water body (Tier 1)
   - Complex coastline

3. **San Francisco Bay, CA:** `37.8, -122.4`
   - Specific water body (Tier 1)
   - Great Lakes shoreline (if near Great Lakes instead)

4. **Lake Superior shore:** `47.5, -87.5`
   - Great Lakes shoreline buffer (0.5km)
   - Should show orange points near shore, degrading inland

### Mid-Resolution Areas (Tier 2 - 375m)
Test these to verify golf course and inland lake coverage:

5. **Pebble Beach, CA:** `36.6, -121.9`
   - Golf courses with 1.5km buffer
   - Should show gray points around courses

6. **Lake Tahoe, CA/NV:** `39.0, -120.0`
   - Large inland lake (Tier 2, 0.5km buffer)
   - Should show gray points around shoreline

7. **Finger Lakes, NY:** `42.5, -76.9`
   - Multiple inland lakes
   - Should show gray points near lakes

### Terrain-Based Resolution (Tier 2-3)

8. **Rocky Mountain NP, CO:** `40.3, -105.7`
   - Extreme terrain (Tier 1) in peaks
   - Rugged terrain (Tier 2) in foothills
   - Should show orange/gray points

9. **Black Hills, SD:** `44.0, -103.5`
   - Moderate terrain variability
   - Should show green points (Tier 3)

### Forest/Urban Areas (Tier 3 - 750m)

10. **Olympic National Forest, WA:** `47.8, -123.8`
    - National forest at Tier 3
    - Should show green points

11. **Denver Metro, CO:** `39.7, -105.0`
    - Urban area at Tier 3
    - Should show green points

### Background Areas (Tier 4 - 1.5km)

12. **Kansas plains:** `38.5, -98.5`
    - Flat, rural area
    - Should show blue points (coarse spacing)

---

## Other Visualization Scripts

### ✅ `visualize_grid.py` - Full CONUS Visualization

**Status:** ✅ Updated and working with `adaptive_grid_points.nc`

**Usage:**
```bash
python visualize_grid.py
```

**What it does:**
- Full CONUS visualization with point density heatmap
- Uses HRRR native Lambert Conformal projection
- Discrete color bins for cleaner visualization
- Shows coastlines, state boundaries, geographic features
- Outputs: `output/adaptive_grid_density_discrete.png`

**Color scale:**
- ⚪ **White:** <4 points per HRRR cell (background areas)
- 🟡 **Yellow:** 4-9 points (Tier 4)
- 🟢 **Green:** 10-49 points (Tier 3)
- 🔵 **Blue-teal:** 50-99 points (Tier 2)
- 🔵 **Dark blue:** 100-199 points (Tier 2)
- 🔴 **Red:** 200+ points (Tier 1, high-density areas)

**Output details:**
- File size: ~0.9 MB
- Resolution: High-res PNG
- Includes statistics: max/mean/median points per cell
- Median point density: 4.0 points/cell

### 🚫 Other Scripts - Not Updated

- `visualize_point_grid.py` - Older version, looks for SPARSE files
- `visualize_grid_fixed.py` - Specific to older grid format
- `visualize_grid_local.py` - Local domain version
- `visualize_catalina.py` - Specific to Catalina Island demo
- `visualize_osm_trails*.py` - For trail visualization (feature data, not grid)

---

## Tips for Effective Visualization

### 1. Start Wide, Then Zoom
```bash
# First look at a large area
python visualize_point_grid_binary.py 40.0 -105.0 --box-sizes 1.0 --input-file adaptive_grid_points.nc

# Then zoom into interesting features
python visualize_point_grid_binary.py 40.0 -105.0 --box-sizes 0.2 0.05 --input-file adaptive_grid_points.nc
```

### 2. Test Your Configuration Changes

If you modified feature buffers or tier assignments, visualize:
- Before/after comparison locations
- Boundary regions between tiers
- Areas where features overlap

### 3. Verify Point Budget Allocation

Check that high-resolution areas (Tier 1-2) are where you expect:
- Coastlines should have orange points
- Great Lakes shores should have orange points
- Inland lakes should have gray points
- Golf courses should have gray buffer zones
- National forests should have green points

### 4. Resolution Validation

The color coding shows **actual spacing** (nearest neighbor distance), which should match tier expectations:
- Orange (~188m) = Tier 1 cells (256 points @ 187.5m resolution)
- Gray (~375m) = Tier 2 cells (64 points @ 375m resolution)
- Green (~750m) = Tier 3 cells (16 points @ 750m resolution)
- Blue (~1.5km) = Tier 4 cells (4 points @ 1.5km resolution)

---

## Creating Custom Visualizations

### Using the NetCDF File Directly

The output file `adaptive_grid_points.nc` contains:

**Variables:**
- `latitude(npoints)` - Point latitudes
- `longitude(npoints)` - Point longitudes
- `tier(npoints)` - Tier classification (1-4)
- `metadata(npoints)` - Bitfield with feature classifications

**Example Python code:**
```python
import netCDF4 as nc
import matplotlib.pyplot as plt

# Load data
with nc.Dataset('output/adaptive_grid_points.nc', 'r') as ds:
    lats = ds.variables['latitude'][:]
    lons = ds.variables['longitude'][:]
    tiers = ds.variables['tier'][:]
    metadata = ds.variables['metadata'][:]

# Filter by tier
tier1_mask = (tiers == 1)
tier1_lats = lats[tier1_mask]
tier1_lons = lons[tier1_mask]

# Filter by metadata (e.g., coastline points)
# Bit 2: Coastline
coastline_mask = (metadata & (1 << 2)) > 0
coast_lats = lats[coastline_mask]
coast_lons = lons[coastline_mask]

# Create your custom visualization
plt.figure(figsize=(12, 8))
plt.scatter(lons, lats, c=tiers, s=1, cmap='RdYlGn_r')
plt.colorbar(label='Tier')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Adaptive Grid - All Points')
plt.savefig('my_custom_viz.png', dpi=150)
```

---

## Summary

**Primary tool:** `visualize_point_grid_binary.py`

**Quick command:**
```bash
python visualize_point_grid_binary.py LAT LON --input-file adaptive_grid_points.nc
```

**Already created:**
- `output/adaptive_grid_density.png` - Full CONUS overview

**Test locations ready:** 12 recommended locations above covering all tier types and features.
