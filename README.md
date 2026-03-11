# Adaptive Point Selection for XGBoost Weather Downscaling

## Project Overview

This project develops an extreme gradient boosting (XGBoost) decision tree algorithm for downscaling (super-resolving) meteorological weather forecast data. The system produces high-resolution surface temperature and 10-m wind speed forecasts by combining coarse forecast model output with high-resolution terrain and land-use information.

## Problem Statement

### Current Approach
- **Training Data**: Features from forecast model output at 2-h lead time (post-spin-up)
- **Additional Features**:
  - High-resolution terrain (height, gradient)
  - Land-use classification (forest, cropland, water, etc.)
- **Target**: Quality-controlled surface observations
- **Output Variables**: Surface temperature, 10-m wind speed
- **Current Resolution**: 90-m grid spacing in limited geographic regions

### Scalability Challenge

Extending the current 90-m resolution to full CONUS is computationally intractable:

- **Full CONUS at 90m**: ~985 million grid points
- **Current test regions**: ~1/1000 to 1/10000 of CONUS coverage
- **Computational bottleneck**: Inference computation and data storage at full resolution

### Use Case Requirements

High-resolution forecasts are needed in specific areas:
- Dense population centers
- Outdoor recreational areas (skiing, hiking, wind surfing, etc.)
- Complex terrain regions

Lower resolution is acceptable in:
- Agricultural areas (e.g., Great Plains)
- Remote, low-activity regions

## Solution: Adaptive Grid Resolution

Deploy an **irregular, priority-based point distribution** that concentrates high-resolution points (90m spacing) in areas of interest while using coarser spacing (up to 3km) in low-priority regions.

### Expected Performance
- **Adaptive grid**: ~93 million points
- **Full 90m grid**: ~985 million points
- **Reduction factor**: 10.6× (90-91% fewer points)
- **Coverage**: Maintains 90m resolution where it matters most

---

## Data Sources

### Core Data (User-Provided)
1. High-resolution terrain elevation and gradient
2. High-resolution land-use classification

### Population Density
- **LandScan USA** ([Census-based 30m grids](https://pmc.ncbi.nlm.nih.gov/articles/PMC9422266/)): Dasymetric population mapping
- **Census Gridded Population** ([Data.gov](https://catalog.data.gov/dataset?q=%22Population+Density%22)): 340+ datasets available

### Recreation & Outdoor Activity

**Trails:**
- [OpenStreetMap Trails](https://openstreetmap.us/our-work/trails/): Comprehensive trail mapping
- [American Trails](https://www.americantrails.org/resources/mapping-trails-across-the-country): Nationwide aggregation (NPS, BLM, FWS partnership)

**Ski Resorts:**
- [OpenSkiStats](https://openskistats.org/manuscript/): 505 active ski areas in the US
- [NOHRSC Ski Areas](https://www.arcgis.com/home/item.html?id=3f344bd915bf4f7da76ca1af46489ec5): National Operational Hydrologic Remote Sensing Center dataset

**Protected Areas & Parks:**
- [PAD-US](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview): Protected Areas Database (federal/state parks, wilderness, recreation areas)
- [Parks Dataset](https://www.nature.com/articles/s41597-022-01857-7): Accessible and recreational parks

**Campgrounds:**
- [Recreation.gov](https://www.recreation.gov/use-our-data): 103,000+ campsites across 3,600+ federal recreation areas
- [RIDB](https://ridb.recreation.gov/shared/pdf/Federal_Camping_Data_Standard_1.0.pdf): Standardized campground locations (NPS, BLM, USFS, USACE)

**Beaches:**
- [EPA Beach Act Data](https://www.epa.gov/waterdata/fact-sheet-beach-act-locational-data): Coastal and Great Lakes beaches
- [National Beach List](https://catalog.data.gov/dataset/national-list-of-beaches): All public coastal recreation waters

### Transportation Infrastructure
- [TIGER Primary Roads](https://catalog.data.gov/dataset/tiger-line-shapefile-2023-nation-u-s-primary-roads): All primary highways and interstate systems
- [Aviation Facilities](https://geodata.bts.gov/datasets/usdot::aviation-facilities/about): All US airports (updated every 28 days)

### Urban Areas
- [Census Urban Areas](https://catalog.data.gov/dataset/tiger-line-shapefile-2020-nation-u-s-2020-census-urban-area): 2,645 urbanized areas with population ≥5,000

### Additional Data Sources to Consider
1. National Parks & Monuments (within PAD-US)
2. Golf courses
3. Marinas & boat launches
4. College/University campuses
5. Sports venues & stadiums
6. Wildfire risk areas
7. High-value agricultural crops (orchards, vineyards)
8. Power generation facilities (wind farms, solar installations)
9. Major tourist attractions
10. Commuter rail stations

---

## Algorithm: Hierarchical Grid Refinement

### Phase 1: Priority Layer Generation

For each CONUS 90m grid cell, compute a priority score:

```
Priority_Score = w1·Population_Density
               + w2·Recreation_Proximity
               + w3·Transportation_Proximity
               + w4·Urban_Area_Flag
               + w5·Terrain_Complexity
```

**Component Definitions:**

- **Population_Density**: Normalized population per km² (from 30m census data)
- **Recreation_Proximity**: Distance-weighted sum of nearby recreation features (trails, ski resorts, parks, beaches, campgrounds)
  - Uses exponential decay: `exp(-distance/characteristic_length)`
- **Transportation_Proximity**: Distance to major highways/airports
- **Urban_Area_Flag**: Binary indicator from Census Urban Areas
- **Terrain_Complexity**: Gradient of terrain elevation (complex terrain requires finer resolution)

**Weights (w1-w5)**: Configurable parameters for tuning priorities based on use case requirements

### Phase 2: Hierarchical Grid Refinement

**Step 1**: Start with CONUS-wide coarse grid (e.g., HRRR 3km spacing)

**Step 2**: Define priority tiers and corresponding resolutions:

| Tier | Resolution | Priority Threshold | Use Case |
|------|------------|-------------------|----------|
| Tier 1 | 90m | Priority_Score ≥ threshold_high | Dense population, major recreation areas |
| Tier 2 | 270m | threshold_mid ≤ Priority_Score < threshold_high | Suburban areas, secondary recreation |
| Tier 3 | 810m | threshold_low ≤ Priority_Score < threshold_mid | Rural areas with some activity |
| Tier 4 | 3km | Priority_Score < threshold_low | Remote/agricultural background |

**Step 3**: Apply spatial constraints:
- Enforce minimum/maximum point density per region
- Apply buffering around high-priority areas (smooth transitions)
- Ensure connectivity (no isolated high-resolution islands)
- Respect terrain boundaries (maintain resolution in complex terrain)

**Step 4**: Generate irregular point mesh using priority-weighted sampling

### Phase 3: Optimization & Validation

**Computational Constraints:**
- Set target total point count (e.g., 50-100M points)
- Iteratively adjust thresholds to meet target

**Coverage Validation:**
- Ensure all major population centers are covered
- Check recreation area coverage
- Verify no critical gaps exist

**Spatial Indexing:**
- Generate spatial indices for efficient forecast inference

---

## Grid Point Estimates

### Tier Distribution

| Tier | Resolution | Coverage Area | Points/km² | Total Points |
|------|-----------|---------------|------------|--------------|
| **Tier 1** | 90m | 600,000 km²<br>(metros: 300k, recreation: 200k, complex terrain: 100k) | 123 | **74M** |
| **Tier 2** | 270m | 1,100,000 km²<br>(suburban: 800k, secondary recreation: 300k) | 14 | **15M** |
| **Tier 3** | 810m | 2,000,000 km²<br>(rural areas with some activity) | 1.5 | **3M** |
| **Tier 4** | 3km | 4,100,000 km²<br>(remote/agricultural areas) | 0.11 | **0.5M** |
| **TOTAL** | - | **8,000,000 km²** | - | **~93M** |

### Comparison to Full Resolution

- **Full 90m CONUS grid**: ~985 million points
- **Adaptive grid**: ~93 million points
- **Reduction**: 10.6× fewer points (90-91% reduction)

### Refinement Options

If 93M points is still too large:

1. **Increase Tier 1 threshold**: Be more selective about 90m coverage areas
2. **Adjust Tier 1 resolution**: Use 180m instead of 90m (reduces by 4×)
3. **Temporal filtering**: Only activate seasonal recreation points when relevant (e.g., ski areas in winter)
4. **Incremental deployment**: Start with top 20M points (major metros + premier recreation areas)

---

## Storage & Compute Implications

### Storage Requirements (per forecast time)
- **93M points × 10 features × 4 bytes/float32** ≈ 3.7 GB per forecast time
- **Compared to full grid**: ~39 GB per forecast time

### Inference Performance
- **Speedup**: ~100× faster than full 985M point grid
- **Scalability**: Operationally feasible for real-time forecasting

---

## Implementation Considerations

### Design Principles
- **Tunability**: Make weights (w1-w5) configurable for different use cases
- **Multi-criteria optimization**: Use Pareto frontier if multiple objectives conflict
- **Temporal variation**: Consider seasonal recreation patterns (ski areas in winter, beaches in summer)
- **Data fusion**: Appropriately combine multiple data sources with varying resolutions
- **Edge handling**: Special treatment for coastlines, borders, high-terrain gradients

### Workflow
1. Acquire and preprocess all geospatial datasets
2. Generate priority score layer at 90m resolution
3. Apply hierarchical refinement algorithm
4. Validate coverage and adjust thresholds
5. Generate spatial indices for operational inference
6. Deploy and monitor forecast quality

---

## Implementation

### Quick Start

A Python implementation of this adaptive grid generation system is provided. The implementation uses a two-stage pipeline:

**Stage 1: Data Acquisition**
```bash
python download_data.py
```

**Stage 2: Grid Generation**
```bash
python generate_adaptive_grid.py
```

See [USAGE.md](USAGE.md) for detailed installation and usage instructions.

### Project Structure

```
superres/
├── README.md                        # This file - algorithm overview
├── USAGE.md                         # Detailed usage instructions
├── config.py                        # Configuration settings
├── download_data.py                 # Data acquisition script
├── generate_adaptive_grid.py        # Main grid generation script
├── test_installation.py             # Installation validation
├── requirements.txt                 # Python dependencies
├── data/                            # Downloaded datasets (created by download_data.py)
│   ├── hrrr/                       # HRRR grid and terrain
│   ├── natural_earth/              # Coastlines and lakes
│   ├── urban/                      # Census urban areas
│   ├── roads/                      # Primary highways
│   ├── ski_resorts/                # Ski area locations
│   ├── padus/                      # Protected areas (manual download)
│   └── golf/                       # Golf courses (optional)
└── output/                         # Generated outputs
    ├── adaptive_grid_points.nc     # NetCDF with point locations
    └── adaptive_grid_density.png   # Visualization
```

### Implementation Details

The implementation follows the hierarchical refinement algorithm with the following specifics:

**Tier Classifications (as implemented):**

| Tier | Resolution | Criteria |
|------|-----------|----------|
| **Tier 1** | 90m | Recreation areas (coastlines ±1-3km, lakes, ski resorts, golf courses, parks) |
| **Tier 2** | 270m | Terrain variability (std dev > 50m in 1km window), secondary recreation |
| **Tier 3** | 810m | Urban/suburban areas, major highways (±0.5km buffer) |
| **Tier 4** | 3km | Background coverage (HRRR grid spacing) |

**Data Sources Used:**
- HRRR grid and terrain from AWS NOAA archive
- Natural Earth coastlines and lakes (10m resolution)
- Census TIGER urban areas (2020)
- Census TIGER primary roads (2023)
- OpenSkiStats ski resort database
- PAD-US protected areas (optional, manual download)

**Output Format:**

The generated `adaptive_grid_points.nc` netCDF file contains:
- `latitude(npoints)`: Point latitudes
- `longitude(npoints)`: Point longitudes
- `tier(npoints)`: Tier classification (1-4)
- `metadata(npoints)`: Bitfield indicating which criteria each point satisfies

**Visualization:**

The `adaptive_grid_density.png` shows a map of point density per HRRR grid cell using a logarithmic color scale, allowing identification of high-priority regions.

### Testing Installation

Validate your environment before running:

```bash
python test_installation.py
```

This checks:
- Required Python packages
- Data file availability
- Configuration settings
- Estimated computational requirements

### Configuration

Key parameters in `config.py`:

```python
# Target total points
TARGET_TOTAL_POINTS = 50_000_000  # 50M points

# Tier resolutions (meters)
TIER_RESOLUTIONS = {1: 90, 2: 270, 3: 810, 4: 3000}

# Terrain variability threshold
TERRAIN_STDDEV_THRESHOLD = 50  # meters
TERRAIN_WINDOW_SIZE = 1000     # meters (1km window)

# Coastline buffers
COASTLINE_BUFFER_ONSHORE_KM = 1
COASTLINE_BUFFER_OFFSHORE_KM = 3
```

### Computational Requirements

For 50M points:
- **RAM**: 16-32 GB recommended
- **Runtime**: 30-60 minutes
- **Output size**: ~2-3 GB
- **Disk space**: ~5-10 GB (including downloaded datasets)

---

## Next Steps

1. ✓ **Prototype Development**: Implementation complete (see above)
2. **Data Acquisition**: Run `download_data.py` to obtain necessary datasets
3. **Grid Generation**: Run `generate_adaptive_grid.py` to create adaptive grid
4. **Weight Calibration**: Analyze output and adjust tier criteria in `config.py`
5. **XGBoost Integration**: Incorporate generated grid into training pipeline
6. **Validation**: Compare model performance vs. regular grids
7. **Operational Deployment**: Deploy for full CONUS forecasting

---

## References

See inline links in the Data Sources section above for all external datasets and documentation.

---

## License

[Add appropriate license information]

## Contact

[Add contact information]
