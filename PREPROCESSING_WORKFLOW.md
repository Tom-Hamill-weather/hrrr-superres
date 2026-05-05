# Preprocessing Workflow for Adaptive Grid Generation

**Date:** 2026-03-18
**Purpose:** Cache expensive feature processing to speed up grid generation iterations

---

## Overview

The preprocessing system allows you to pre-compute expensive buffer and geometry operations once, then reuse them across multiple grid generation runs. This saves **5-10 minutes per run** when iterating on grid configurations.

---

## Two Preprocessing Approaches

### 1. **Feature Buffer Preprocessing** (RECOMMENDED - Updated for current config)

**Script:** `preprocess_feature_buffers.py`

**Status:** ✅ **UPDATED FOR CURRENT CONFIGURATION**

**What it does:**
- Pre-computes unified buffer geometries for all features by tier
- Saves to: `data/preprocessed/tier_buffers.gpkg`
- Integrates with: Main grid generation pipeline (needs integration)

**Current configuration (2026-03-18):**
- **Tier 1** (187.5m resolution):
  - Coastlines: 0.5km onshore buffer
  - Great Lakes (5): 0.5km shoreline buffer
  - Ski resorts: 2.0km buffer
  - Specific water bodies: exact boundaries

- **Tier 2** (375m resolution):
  - Inland lakes (216): 0.5km buffer
  - Golf courses (15,063): 1.5km buffer

- **Tier 3** (750m resolution):
  - Urban areas: polygon boundaries
  - Major highways: 0.5km buffer (if enabled)
  - National forests: polygon boundaries (NPS subtracted)

**Usage:**
```bash
# Run once after changing feature configuration
python3 preprocess_feature_buffers.py

# Output: data/preprocessed/tier_buffers.gpkg
```

**Integration status:** ⚠️ Not yet integrated into `generate_adaptive_grid.py`
**TODO:** Modify `generate_adaptive_grid.py` to optionally load from cached buffers

---

### 2. **Stage-Based Mask Generation** (EXPERIMENTAL - Not updated)

**Scripts:**
- `stage1_generate_masks.py` - Generate boolean masks
- `stage2_generate_grid.py` - Load masks and generate grid

**Status:** ⚠️ **NOT UPDATED FOR CURRENT CONFIGURATION**

**What it does:**
- **Stage 1:** Generates individual boolean masks at 93.75m resolution
  - Saves to: `output/masks/*.npz` (compressed numpy arrays)
  - Each feature type gets its own mask file
- **Stage 2:** Loads masks and applies tier logic to generate final grid
  - Allows fast iteration on tier assignments

**Why it's not updated:**
- More complex than buffer preprocessing
- Not integrated with main `generate_adaptive_grid.py`
- Appears to be an experimental alternative approach
- Main script already processes features efficiently

**If you want to use this approach:**
- Would need significant updates to match new configuration
- Requires separating Great Lakes/inland lakes
- Requires adding golf course masks
- Requires updating tier assignments

---

## Recommended Workflow

### Current Workflow (No Preprocessing)

```bash
# Every run processes all features from scratch
python3 generate_adaptive_grid.py
```

**Time:** ~15-20 minutes per run
**Pro:** Simple, always uses latest config
**Con:** Slow when iterating on configurations

---

### With Preprocessing (Recommended)

#### Step 1: Initial setup (once)
```bash
# Download and prepare all feature data
python3 preprocess_golf_courses.py  # Get OSM golf course data
# ... other data download scripts
```

#### Step 2: Run preprocessing (after feature config changes)
```bash
# Pre-compute feature buffers
python3 preprocess_feature_buffers.py

# Output: data/preprocessed/tier_buffers.gpkg
# Time: ~5-10 minutes
```

#### Step 3: Generate grid (fast iterations)
```bash
# TODO: Update generate_adaptive_grid.py to support --use-cached flag
# python3 generate_adaptive_grid.py --use-cached

# For now, run normally:
python3 generate_adaptive_grid.py
```

**Time:** Currently same as no preprocessing (integration pending)
**Future time:** ~5-10 minutes per run (after integration)

---

## When to Re-run Preprocessing

Re-run `preprocess_feature_buffers.py` when you change:

✅ **Yes, re-run:**
- Buffer distances (coastline, lakes, golf courses, roads)
- Which features are included (enable/disable golf courses, forests, etc.)
- Feature data files (download new OSM data, updated shapefiles)
- Tier assignments for features

❌ **No, don't re-run:**
- Terrain thresholds (processed separately)
- Grid resolution/subdivision settings
- Point budget targets
- Output file names/formats

---

## Integration TODO

To complete the preprocessing integration:

1. **Modify `generate_adaptive_grid.py`:**
   - Add `--use-cached` command-line flag
   - Add method to load from `data/preprocessed/tier_buffers.gpkg`
   - Skip feature buffering when cached buffers exist
   - Fall back to full processing if cache is missing/stale

2. **Add cache validation:**
   - Check if config has changed since cache was created
   - Store config hash in cached file metadata
   - Warn if cache is out of date

3. **Document in main README:**
   - Add preprocessing to workflow documentation
   - Explain when to use cached vs fresh processing

---

## File Locations

```
superres/
├── preprocess_feature_buffers.py       # ✅ UPDATED - Feature buffer preprocessing
├── stage1_generate_masks.py            # ⚠️  OLD - Boolean mask generation
├── stage2_generate_grid.py             # ⚠️  OLD - Grid from masks
├── generate_adaptive_grid.py           # Main grid generation script
├── data/
│   └── preprocessed/
│       └── tier_buffers.gpkg          # Cached feature buffers (created by preprocessing)
└── output/
    └── masks/                         # Boolean masks (created by stage1, if used)
        ├── terrain_*.npz
        ├── coastline_*.npz
        └── ...
```

---

## Summary

**Current Status (2026-03-18):**
- ✅ `preprocess_feature_buffers.py` - UPDATED and ready to use
- ⚠️  Integration into main script - PENDING
- ⚠️  Stage-based approach - NOT UPDATED (experimental alternative)

**Next Steps:**
1. Run `preprocess_feature_buffers.py` to create cached buffers
2. Integrate cache loading into `generate_adaptive_grid.py`
3. Test performance improvements
4. Document final workflow
