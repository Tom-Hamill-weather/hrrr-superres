# Quick Start Guide

## Complete Workflow for Adaptive Grid Generation

This guide walks you through the complete process from installation to generating the adaptive grid.

---

## Step 1: Installation (5 minutes)

### Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Troubleshooting:**

If `pygrib` fails to install:
```bash
# macOS
brew install eccodes
pip install pygrib

# Linux (Ubuntu/Debian)
sudo apt-get install libeccodes-dev
pip install pygrib
```

If `basemap` fails to install:
```bash
conda install -c conda-forge basemap basemap-data-hires
```

### Verify Installation

```bash
python test_installation.py
```

This will check all dependencies and show what data is missing.

---

## Step 2: Data Acquisition (15-30 minutes)

### Download Required Datasets

```bash
python download_data.py
```

This will automatically download:
- ✓ HRRR grid and terrain (~500 MB)
- ✓ Natural Earth coastlines and lakes
- ✓ Census urban areas
- ✓ Primary roads
- ✓ Ski resort locations

### Manual Downloads (Optional but Recommended)

**PAD-US Protected Areas** (~5 GB):
1. Visit: https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download
2. Download the latest PAD-US geodatabase
3. Extract to: `data/padus/`

**Golf Courses** (Optional):
- Can be extracted from OpenStreetMap or omitted

### Verify Data

```bash
python test_installation.py
```

Should now show all required datasets as available.

---

## Step 3: Generate Adaptive Grid (30-60 minutes)

### Run Grid Generation

```bash
python generate_adaptive_grid.py
```

**What this does:**
1. Loads all geospatial datasets
2. Computes terrain variability from HRRR terrain
3. Classifies HRRR grid cells into 4 tiers based on:
   - Recreation areas (coastlines, lakes, ski resorts)
   - Terrain complexity
   - Urban/suburban areas
   - Highway corridors
4. Generates sub-grid points at tier-appropriate resolutions:
   - Tier 1: 90m
   - Tier 2: 270m
   - Tier 3: 810m
   - Tier 4: 3km
5. Writes output netCDF file
6. Creates visualization map

**Expected runtime:** 30-60 minutes (depending on CPU)

**Expected output:**
- `output/adaptive_grid_points.nc` (~2-3 GB)
- `output/adaptive_grid_density.png`

---

## Step 4: Inspect Results (2 minutes)

### Run Output Inspector

```bash
python inspect_output.py
```

**This produces:**
- Detailed statistics about the generated grid
- Tier distribution charts
- Sample CSV of 10,000 random points for manual inspection

**Output files:**
- `output/tier_distribution.png`
- `output/sample_points.csv`

---

## Complete Command Sequence

For a fresh installation, run these commands in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation
python test_installation.py

# 3. Download data
python download_data.py

# 4. Generate adaptive grid
python generate_adaptive_grid.py

# 5. Inspect results
python inspect_output.py
```

---

## Expected Outputs

After completing all steps, you should have:

```
output/
├── adaptive_grid_points.nc        # Main output: ~50M points with metadata
├── adaptive_grid_density.png      # Map showing point density
├── tier_distribution.png          # Charts showing tier statistics
└── sample_points.csv              # Sample of 10K points for inspection
```

---

## Using the Output in Your XGBoost Pipeline

### Reading the Grid in Python

```python
import netCDF4 as nc
import numpy as np

# Load adaptive grid
ds = nc.Dataset('output/adaptive_grid_points.nc', 'r')

lats = ds.variables['latitude'][:]     # (npoints,)
lons = ds.variables['longitude'][:]    # (npoints,)
tiers = ds.variables['tier'][:]        # (npoints,) - values 1-4
metadata = ds.variables['metadata'][:] # (npoints,) - bitfield

ds.close()

# Use these coordinates for your XGBoost training/inference
# Each (lat, lon) pair is a location where you need to:
# 1. Extract HRRR features
# 2. Extract terrain features
# 3. Run XGBoost inference
```

### Extracting Features at Grid Points

For each point, you'll need to:
1. Find nearest HRRR grid cell
2. Interpolate HRRR variables to point location
3. Extract local terrain features (elevation, gradient)
4. Extract land-use features
5. Run XGBoost model to predict temperature/wind

---

## Customization

### Adjust Grid Resolution

Edit `config.py`:

```python
# Change target point count
TARGET_TOTAL_POINTS = 30_000_000  # Reduce to 30M for faster testing

# Change tier resolutions
TIER_RESOLUTIONS = {
    1: 120,    # Coarsen Tier 1 from 90m to 120m
    2: 360,    # Coarsen Tier 2
    3: 1080,   # Coarsen Tier 3
    4: 3000    # Keep Tier 4 at HRRR resolution
}
```

### Adjust Terrain Threshold

```python
# Make terrain classification more/less aggressive
TERRAIN_STDDEV_THRESHOLD = 75  # Increase from 50m to be more selective
```

### Adjust Coastline Buffer

```python
# Wider/narrower coastal zone
COASTLINE_BUFFER_OFFSHORE_KM = 5  # Increase from 3km
```

Then re-run:
```bash
python generate_adaptive_grid.py
```

---

## Troubleshooting

### "Out of Memory" Errors

**Solution:** Reduce target point count
```python
# In config.py
TARGET_TOTAL_POINTS = 30_000_000  # Reduce from 50M
```

### HRRR Download Fails

**Solution:** The AWS bucket URL may have changed. Check:
- https://registry.opendata.aws/noaa-hrrr-pds/

Update the URL in `download_data.py` and re-run.

### Missing Basemap

**Solution:** Install via conda or use cartopy alternative
```bash
conda install -c conda-forge basemap
```

### Generation Takes Too Long

**Solution:** Test on smaller region first
- Modify `generate_adaptive_grid.py` to subset the HRRR grid
- Or reduce target points temporarily

---

## Next Steps

1. **Validate Output**: Examine visualizations and statistics
2. **Integrate with XGBoost**: Update your training pipeline to use these points
3. **Test Forecast Quality**: Compare adaptive grid vs. regular grid performance
4. **Iterate**: Adjust tier criteria based on forecast skill scores
5. **Deploy**: Use adaptive grid for operational CONUS forecasting

---

## Support

For detailed documentation:
- **Algorithm details**: See `README.md`
- **Usage instructions**: See `USAGE.md`
- **Configuration**: See `config.py` with inline comments

For issues:
- Check that all data files downloaded successfully
- Verify system has sufficient RAM (16-32 GB recommended)
- Test with smaller point count first (10M) before scaling to 50M
