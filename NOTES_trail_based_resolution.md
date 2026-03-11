# Trail-Based High Resolution Approach

## Overview
Replace National Forest blanket coverage with targeted high resolution along hiking trails where outdoor recreation actually occurs.

## Design Decision
- **Resolution**: Tier 0 (93.75m, stride=1, keep all points)
- **Trails to include**:
  - National Recreation Trails (NRTs) - ~100,000 km
  - State-designated trails - ~150,000 km
  - **Total**: ~250,000 km
- **Buffer**: 1km around trails (configurable)

## Why This Approach is Better

**Computational Efficiency:**
- Current (forests): 749M grid points checked against 109 polygon boundaries
- Trail→Grid: ~2.5M trail points × circle rasterization
- **Speedup**: ~300x fewer operations, each operation 100x faster
- **Estimated processing time**: 5-10 minutes (vs current forest processing in patch loop)

**Coverage Comparison:**
| Approach | Coverage Area | Index Count | Targeting |
|----------|---------------|-------------|-----------|
| National Forests (current) | 737,000 km² | 99M indices | Blanket forest coverage |
| NRT + State Trails | 500,000 km² | 80-100M indices | Where people actually hike |

**Advantages:**
- Higher resolution where it matters (trails, not entire forests)
- Faster processing (trail→grid vs grid→polygon)
- More maintainable (trail data updated more frequently than forest boundaries)

## Data Sources

**National Recreation Trails:**
- Source: National Park Service GIS data
- Available at: https://hub.arcgis.com/datasets/fedmaps::national-park-service-trails/about
- Also: https://catalog.data.gov/dataset/national-park-service-trails/
- Format: Shapefile, GeoJSON, Feature Service
- ~1,200 designated trails
- Coverage: National Parks + USFS + BLM + state/local

**State-Designated Trails:**
- Source: Individual state GIS portals or OpenStreetMap with state trail designation tags
- May need to compile from multiple sources
- Alternative: Use OSM query for trails tagged with state designation

**Alternative/Supplement:**
- OpenStreetMap: `route=hiking` or `highway=path/footway` with quality filters
- Filter by: trail designation, sac_scale, usage frequency

## Implementation Algorithm

### Efficient Trail→Grid Approach

**Key insight**: Instead of checking each grid point against trail geometries, iterate through trail points and mark nearby grid cells.

```python
def add_trails_to_indices(trail_geometries, buffer_m, index_sets, proj, n_fine):
    """
    Add trail-buffered indices using efficient Trail→Grid approach

    Args:
        trail_geometries: List of LineString/MultiLineString geometries
        buffer_m: Buffer distance in meters (e.g., 1000 for 1km)
        index_sets: Dictionary to add indices to
        proj: Basemap projection object (HRRR)
        n_fine: Fine grid subdivision factor (32 for 93.75m resolution)
    """
    base_resolution = 93.75  # meters per fine grid cell
    buffer_cells = int(buffer_m / base_resolution) + 1

    # Sample points along each trail
    sample_spacing = 100  # meters between samples along trail

    for trail_geom in trail_geometries:
        # Sample points every 100m along trail
        trail_points = sample_along_line(trail_geom, spacing=sample_spacing)

        # Project trail points to HRRR coordinate system
        x_pts, y_pts = proj(trail_points.lon, trail_points.lat)

        # For each trail point, find nearby grid indices
        for x, y in zip(x_pts, y_pts):
            # Convert projected coordinates to fine grid indices
            i_fine_center = int(y / base_resolution)
            j_fine_center = int(x / base_resolution)

            # Add all grid cells within buffer distance
            # Use circle rasterization (much faster than actual distance checks)
            for di in range(-buffer_cells, buffer_cells + 1):
                for dj in range(-buffer_cells, buffer_cells + 1):
                    # Check if within circular buffer
                    dist_cells_sq = di*di + dj*dj
                    if dist_cells_sq <= buffer_cells * buffer_cells:
                        i_fine = i_fine_center + di
                        j_fine = j_fine_center + dj
                        # Add to appropriate index set (e.g., 'trails' for Tier 0)
                        index_sets['trails'].add((i_fine, j_fine))

def sample_along_line(linestring, spacing=100):
    """
    Sample points at regular intervals along a LineString

    Args:
        linestring: Shapely LineString or MultiLineString
        spacing: Distance in meters between sample points

    Returns:
        Array of (lon, lat) points
    """
    # If MultiLineString, process each component
    if linestring.geom_type == 'MultiLineString':
        points = []
        for line in linestring.geoms:
            points.extend(sample_along_line(line, spacing))
        return points

    # For LineString: sample at regular distances
    length = linestring.length  # Note: this is in degrees, need to convert
    # For more accurate sampling, project to meters first or use geodesic

    points = []
    num_samples = int(length / (spacing / 111000)) + 1  # rough degrees conversion
    for i in range(num_samples):
        fraction = i / max(num_samples - 1, 1)
        point = linestring.interpolate(fraction, normalized=True)
        points.append((point.x, point.y))  # lon, lat

    return np.array(points)
```

### Integration into Current System

**Option 1: Replace forests entirely**
```python
# In _preproject_features():
# REMOVE forest processing
# ADD:
trails_gdf = self.loader.data.get('trails')  # NRT + State trails
if trails_gdf is not None:
    add_trails_to_indices(trails_gdf.geometry, buffer_m=1000,
                         index_sets=index_sets, proj=self.proj, n_fine=self.n_fine)
```

**Option 2: Keep both (trails + forests)**
- Keep forests at Tier 3 (750m)
- Add trails at Tier 0 (93.75m)
- Trails override forests where they overlap (higher priority)

**Recommended: Option 1** (trails only) for computational efficiency and targeted coverage.

## Tier Assignment Update

Current tier 0: Ski resorts only
Proposed tier 0: Ski resorts + Trails (1km buffer)

```python
# Tier 0: Ski resorts + Trails (stride=1, 93.75m resolution)
tier0_indices = sorted_sets['ski_resorts'] | sorted_sets['trails']
```

## Data Loading

Add to DataLoader class (generate_adaptive_grid.py):

```python
def load_trails(self):
    """Load National Recreation Trails and State-designated trails"""
    print("\nLoading trails...")

    # Load NRT data
    nrt_file = os.path.join(config.DATA_DIR, 'trails', 'national_recreation_trails.shp')
    if os.path.exists(nrt_file):
        nrt_gdf = gpd.read_file(nrt_file)
        print(f"✓ National Recreation Trails loaded: {len(nrt_gdf)} trails")
    else:
        print(f"⚠ NRT file not found: {nrt_file}")
        nrt_gdf = None

    # Load state trails data
    state_trails_file = os.path.join(config.DATA_DIR, 'trails', 'state_trails.shp')
    if os.path.exists(state_trails_file):
        state_gdf = gpd.read_file(state_trails_file)
        print(f"✓ State-designated trails loaded: {len(state_gdf)} trails")
    else:
        print(f"⚠ State trails file not found: {state_trails_file}")
        state_gdf = None

    # Combine
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
```

## Metadata Flag

Add new metadata bit flag:

```python
METADATA_TRAIL = 1 << 13  # New flag for hiking trails
```

Update flag list in all relevant files:
- generate_adaptive_grid_SPARSE_v2.py
- inspect_metadata.py

## Performance Expectations

**Processing time breakdown:**
- Sample trail points (2.5M points): ~10 seconds
- Project to grid coordinates: ~5 seconds
- Rasterize circles and add to index sets: ~5 minutes
- **Total: ~5-10 minutes** (vs forests processed during 80-min patch loop)

**Memory usage:**
- Trail geometries: ~50-100 MB
- Sampled points: ~20 MB (2.5M × 8 bytes)
- Result index set: ~800 MB (100M indices × 8 bytes)
- **Total additional**: ~1 GB (well within current memory budget)

## Next Steps (When Ready to Implement)

1. Download trail data:
   - NRT from https://catalog.data.gov/dataset/national-park-service-trails/
   - State trails from state GIS portals or compile from OSM

2. Create data/trails/ directory and add shapefiles

3. Implement `load_trails()` in generate_adaptive_grid.py

4. Implement `add_trails_to_indices()` helper function

5. Integrate into patch processing or as separate pre-processing step

6. Update tier assignment logic to include 'trails' in Tier 0

7. Add METADATA_TRAIL flag and update inspection tools

8. Test with Western Washington region first

9. Run full CONUS with trails replacing forests

10. Compare: coverage, point count, processing time, memory usage

## Configuration Options

Make these configurable:

```python
TRAIL_BUFFER_M = 1000        # Buffer distance around trails
TRAIL_SAMPLE_SPACING_M = 100 # Spacing between sampled points along trail
TRAIL_TIER = 0               # Resolution tier for trails (0 = 93.75m)
INCLUDE_NRT = True           # Include National Recreation Trails
INCLUDE_STATE_TRAILS = True  # Include state-designated trails
```

## Questions to Resolve Before Implementation

1. Should trails be Tier 0 (93.75m) or Tier 1 (187.5m)?
   - Leaning toward Tier 0 for best coverage where people hike

2. Keep ski resorts at Tier 0 also?
   - Yes, both trails and ski resorts at Tier 0

3. What to do with National Forests?
   - Recommend: Remove entirely, or move to lower tier (Tier 4/5)

4. Buffer distance: 1km sufficient?
   - 1km = ~10-15 minute walk from trail
   - Could make configurable: 500m, 1km, 2km options

5. State trails data compilation strategy?
   - Option A: Manual download from each state GIS portal
   - Option B: OSM extract with state trail tags
   - Option C: Hybrid: official data where available, OSM supplement

## References

- [National Park Service Trails GIS Data](https://hub.arcgis.com/datasets/fedmaps::national-park-service-trails/about)
- [Data.gov - NPS Trails](https://catalog.data.gov/dataset/national-park-service-trails/)
- [NPS Open Data Portal](https://public-nps.opendata.arcgis.com/)
- [Partnership for National Trails System GIS Network](https://pnts.org/new/national-trails-system-gis-network/)
