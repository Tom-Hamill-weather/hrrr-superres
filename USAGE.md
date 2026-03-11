# Usage Guide: Adaptive Grid Generation

## Overview

This application generates an adaptive grid of ~50M points for training XGBoost weather downscaling models across CONUS. The grid has high resolution (90m) in recreation areas and variable terrain, and coarser resolution (up to 3km) in remote/agricultural areas.

## Installation

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note on Basemap:** Basemap can be tricky to install. If you encounter issues:

```bash
# Alternative 1: Use conda
conda install -c conda-forge basemap basemap-data-hires

# Alternative 2: Use cartopy instead
# Edit generate_adaptive_grid.py to use cartopy instead of basemap
```

**Note on pygrib:** pygrib requires GRIB API libraries:

```bash
# On macOS
brew install eccodes

# On Linux (Ubuntu/Debian)
sudo apt-get install libeccodes-dev

# Then install pygrib
pip install pygrib
```

### 2. Verify Installation

```bash
python -c "import numpy, geopandas, netCDF4, matplotlib; print('All core packages imported successfully')"
```

## Two-Stage Workflow

### Stage 1: Data Acquisition

Download all necessary geospatial datasets:

```bash
python download_data.py
```

This will:
- Download HRRR grid and terrain from AWS (~500MB)
- Download Natural Earth coastline and lakes data
- Download Census urban areas
- Download TIGER primary roads
- Download ski resort locations

**Large datasets requiring manual download:**

1. **PAD-US (Protected Areas)**: Very large dataset (~5GB)
   - Download from: https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download
   - Extract to: `data/padus/`

2. **Golf Courses** (optional): Can extract from OpenStreetMap
   - Use overpass API or download OSM extracts
   - Place in: `data/golf/`

**Expected data directory structure after download:**

```
data/
├── hrrr/
│   ├── hrrr_lats.npy
│   ├── hrrr_lons.npy
│   └── hrrr_terrain.npy
├── natural_earth/
│   ├── ne_10m_coastline.shp
│   └── ne_10m_lakes.shp
├── urban/
│   └── tl_2020_us_uac10.shp
├── roads/
│   └── tl_2023_us_primaryroads.shp
├── ski_resorts/
│   └── ski_areas.geojson
├── padus/          # Manual download
└── golf/           # Optional
```

### Stage 2: Generate Adaptive Grid

Once data is downloaded, generate the adaptive grid:

```bash
python generate_adaptive_grid.py
```

This will:
1. Load all geospatial datasets
2. Compute terrain variability from HRRR terrain
3. Classify each HRRR grid cell into tiers (1-4)
4. Generate sub-grid points according to tier resolutions
5. Write output netCDF file with point locations and metadata
6. Create visualization showing point density

**Expected outputs:**

```
output/
├── adaptive_grid_points.nc     # NetCDF with lat/lon and metadata
└── adaptive_grid_density.png   # Visualization of point distribution
```

## Configuration

Edit `config.py` to adjust parameters:

### Grid Resolutions

```python
TIER_RESOLUTIONS = {
    1: 90,      # Tier 1 resolution (meters)
    2: 270,     # Tier 2 resolution (meters)
    3: 810,     # Tier 3 resolution (meters)
    4: 3000     # Tier 4 resolution (meters)
}
```

### Terrain Variability Threshold

```python
TERRAIN_STDDEV_THRESHOLD = 50  # meters
TERRAIN_WINDOW_SIZE = 1000     # meters (1km window)
```

### Target Point Count

```python
TARGET_TOTAL_POINTS = 50_000_000  # 50M points
```

### Recreation Area Buffers

```python
COASTLINE_BUFFER_ONSHORE_KM = 1    # 1 km onshore
COASTLINE_BUFFER_OFFSHORE_KM = 3   # 3 km offshore
LAKE_MIN_AREA_KM2 = 5              # Minimum lake size
```

## Output Format

### NetCDF File Structure

The output `adaptive_grid_points.nc` contains:

**Dimensions:**
- `npoints`: Number of adaptive grid points (~50M)

**Variables:**
- `latitude(npoints)`: Latitude of each point (degrees_north)
- `longitude(npoints)`: Longitude of each point (degrees_east)
- `tier(npoints)`: Grid tier (1-4, int8)
- `metadata(npoints)`: Classification metadata (uint16 bitfield)

**Metadata Bitfield:**
- Bit 0: Urban area
- Bit 1: Suburban area
- Bit 2: Coastline proximity
- Bit 3: Lake proximity
- Bit 4: Ski resort proximity
- Bit 5: Golf course proximity
- Bit 6: National/state park
- Bit 7: Major highway proximity
- Bit 8: High terrain variability

**Example: Reading the output in Python**

```python
import netCDF4 as nc
import numpy as np

# Open file
ds = nc.Dataset('output/adaptive_grid_points.nc', 'r')

# Read coordinates
lats = ds.variables['latitude'][:]
lons = ds.variables['longitude'][:]
tiers = ds.variables['tier'][:]
metadata = ds.variables['metadata'][:]

# Filter for Tier 1 points
tier1_mask = (tiers == 1)
tier1_lats = lats[tier1_mask]
tier1_lons = lons[tier1_mask]

# Check which points are near coastlines (bit 2)
coastal_mask = (metadata & (1 << 2)) > 0
coastal_lats = lats[coastal_mask]
coastal_lons = lons[coastal_mask]

# Summary
print(f"Total points: {len(lats):,}")
print(f"Tier 1 points: {tier1_mask.sum():,}")
print(f"Coastal points: {coastal_mask.sum():,}")

ds.close()
```

## Tier Definitions

### Tier 1 (90m resolution)
**High-priority recreation areas:**
- Coastline: 1km onshore to 3km offshore
- Significant inland lakes (>5 km²)
- Ski resorts and surrounding areas
- Golf courses
- National and state parks

### Tier 2 (270m resolution)
**Secondary recreation and terrain variability:**
- Areas with elevation std dev > 50m in 1km window
- Secondary recreation areas
- Complex terrain regions

### Tier 3 (810m resolution)
**Suburban/exurban and transportation:**
- Suburban areas
- Exurban development
- Major highways and interstate corridors
- Rural areas with some activity

### Tier 4 (3km resolution)
**Background coverage:**
- Remote areas
- Agricultural regions
- Low-priority rural areas

## Troubleshooting

### HRRR Download Issues

If HRRR data download fails:
1. Check AWS HRRR bucket availability
2. Try a different date in `download_data.py`
3. Manually download HRRR file from: https://noaa-hrrr-bdp-pds.s3.amazonaws.com

### Memory Issues

If you run out of memory:
1. Process data in chunks
2. Reduce target point count in `config.py`
3. Use a machine with more RAM (recommended: 32GB+)

### Missing Data Files

If certain datasets fail to download:
- Check URLs in `config.py` for updates
- Manually download from source websites
- The application will skip missing datasets and warn you

### Basemap Installation

If basemap fails to install:
1. Use conda instead of pip
2. Or modify `generate_adaptive_grid.py` to use cartopy
3. Or use simplified matplotlib plotting without basemap

## Performance Notes

- **Data download**: 10-30 minutes (depending on connection)
- **Grid generation**: 30-60 minutes (depending on CPU)
- **Memory usage**: Peak ~16-32 GB RAM
- **Output file size**: ~2-3 GB for 50M points

## Next Steps

After generating the adaptive grid:

1. **Quality Check**: Examine the visualization to ensure proper point distribution
2. **Integration**: Use the netCDF file in your XGBoost training pipeline
3. **Validation**: Test model performance with adaptive vs. regular grids
4. **Iteration**: Adjust tier thresholds and resolutions based on results

## Support

For issues or questions:
- Check the main README.md for algorithm details
- Review config.py for parameter documentation
- Inspect intermediate outputs in the data/ directory
