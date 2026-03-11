"""
Sparse Index Approach - Efficient Grid Generation with Metadata
================================================================

IMPROVEMENTS IN V2:
- Fixed HRRR projection parameters to match actual HRRR grid
- Added metadata tracking for each point (why it has this resolution)
- Added western Washington test region

Algorithm:
1. Process features individually (no massive unions)
2. Store sparse (i,j) index sets where mask = 1
3. Use bounding boxes to skip irrelevant features
4. Merge index sets at the end with tier priority
5. Track metadata: which feature caused this resolution
6. Apply stride patterns and convert to lat/lon only at final step

Memory efficient: Only stores indices of masked points (~5-10% of domain)
"""

import numpy as np
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, box
from shapely.prepared import prep
from shapely import vectorized
try:
    from shapely import contains_xy
    HAS_CONTAINS_XY = True
except ImportError:
    HAS_CONTAINS_XY = False
from shapely.strtree import STRtree
import time
import os
import pickle
from generate_adaptive_grid import DataLoader, TerrainAnalyzer
import config

# Metadata bit flags for tracking why each point has its resolution
METADATA_COASTLINE = 1 << 0    # Near coastline
METADATA_LAKE = 1 << 1          # Near lake
METADATA_SKI_RESORT = 1 << 2    # At ski resort
METADATA_URBAN_HIGH = 1 << 3    # High-density urban (≥50k pop)
METADATA_URBAN_SUBURBAN = 1 << 4 # Suburban (large clusters)
METADATA_SMALL_TOWN = 1 << 5    # Small town (small clusters)
METADATA_ROAD = 1 << 6          # Near major road
METADATA_PARK = 1 << 7          # In national park
METADATA_FOREST = 1 << 8        # In national forest
METADATA_TERRAIN_EXTREME = 1 << 9  # Extreme terrain variability
METADATA_TERRAIN_HIGH = 1 << 10     # High terrain variability
METADATA_TERRAIN_MODERATE = 1 << 11 # Moderate terrain variability
METADATA_BACKGROUND = 1 << 12      # Background grid point

class SparseGridGenerator:
    def __init__(self, test_region=None):
        self.loader = DataLoader()
        self.n_fine = 32  # points per HRRR cell dimension
        self.test_region = test_region

    def generate(self, start_stage=1):
        """
        Generate adaptive grid with stage-by-stage checkpointing.

        Stages:
          1: Data preparation (load, project, buffer geometries)
          2: Index generation (terrain + patch processing)
          3: Tier assignment (apply tier logic with batching)
          4: Output generation (convert to lat/lon and write NetCDF)
        """
        print("="*70)
        print(" SPARSE INDEX APPROACH V2 - EFFICIENT GRID GENERATION")
        print(" WITH METADATA TRACKING AND CORRECT HRRR PROJECTION")
        print(f" Starting from stage {start_stage}")
        print("="*70)

        start_time = time.time()

        # STAGE 1: Data Preparation
        if start_stage <= 1:
            stage_data = self._run_stage1_data_prep()
        else:
            stage_data = self._load_stage1_checkpoint()

        # STAGE 2: Index Generation
        if start_stage <= 2:
            index_sets = self._run_stage2_indices(stage_data)
        else:
            index_sets, stage_data = self._load_stage2_checkpoint()

        # STAGE 3: Tier Assignment
        if start_stage <= 3:
            index_tier_metadata = self._run_stage3_tiers(index_sets, stage_data)
        else:
            index_tier_metadata, stage_data = self._load_stage3_checkpoint()

        # STAGE 4: Output Generation
        lats_out, lons_out, tiers_out, metadata_out, output_path = self._run_stage4_output(
            index_tier_metadata, stage_data
        )

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ Complete in {elapsed/60:.1f} minutes")
        print(f"  Output: {output_path}")
        print(f"{'='*70}")

        return lats_out, lons_out, tiers_out, metadata_out

    def _run_stage1_data_prep(self):
        """Stage 1: Load data, create projection, precompute geometry"""
        print("\n" + "="*70)
        print(" STAGE 1: DATA PREPARATION")
        print("="*70)

        # Load data
        print("\nLoading data...")
        self.loader.load_all()

        lats_hrrr = self.loader.hrrr_grid['lats']
        lons_hrrr = self.loader.hrrr_grid['lons']
        terrain = self.loader.hrrr_grid['terrain']
        shape = lats_hrrr.shape

        # Apply test region filter if specified
        if self.test_region:
            lats_hrrr, lons_hrrr, terrain, shape = self._apply_test_region(
                lats_hrrr, lons_hrrr, terrain
            )

        print(f"HRRR shape: {shape[0]}×{shape[1]} = {shape[0]*shape[1]:,} cells")

        n_fine_i = shape[0] * self.n_fine
        n_fine_j = shape[1] * self.n_fine
        print(f"Fine grid: {n_fine_i}×{n_fine_j} = {n_fine_i*n_fine_j:,} points")

        # Create projection with CORRECT HRRR parameters
        print("\nUsing HRRR native projection parameters:")
        print("  lat_0 = 38.5°N")
        print("  lon_0 = 97.5°W")
        print("  lat_1 = lat_2 = 38.5° (tangent latitude)")

        self.proj = Basemap(
            projection='lcc',
            lat_0=38.5,      # HRRR central latitude
            lon_0=-97.5,     # HRRR central longitude
            lat_1=38.5,      # HRRR first standard parallel
            lat_2=38.5,      # HRRR second standard parallel
            llcrnrlat=lats_hrrr.min(),
            urcrnrlat=lats_hrrr.max(),
            llcrnrlon=lons_hrrr.min(),
            urcrnrlon=lons_hrrr.max(),
            resolution=None
        )
        self.hrrr_crs = self.proj.proj4string

        # Convert to projected coordinates
        print("\nConverting to projected coordinates...")
        x_hrrr, y_hrrr = self.proj(lons_hrrr, lats_hrrr)

        # Precompute cell geometry
        print("Precomputing cell geometry...")
        dx_east, dy_east, dx_north, dy_north = self._compute_cell_vectors(x_hrrr, y_hrrr)

        # Save stage 1 checkpoint
        self._save_stage1_checkpoint(lats_hrrr, lons_hrrr, terrain, shape,
                                     x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                                     n_fine_i, n_fine_j)

        return {
            'lats_hrrr': lats_hrrr,
            'lons_hrrr': lons_hrrr,
            'terrain': terrain,
            'shape': shape,
            'x_hrrr': x_hrrr,
            'y_hrrr': y_hrrr,
            'dx_east': dx_east,
            'dy_east': dy_east,
            'dx_north': dx_north,
            'dy_north': dy_north,
            'n_fine_i': n_fine_i,
            'n_fine_j': n_fine_j,
        }

    def _save_stage1_checkpoint(self, lats_hrrr, lons_hrrr, terrain, shape,
                                 x_hrrr, y_hrrr, dx_east, dy_east, dx_north, dy_north,
                                 n_fine_i, n_fine_j):
        """Save stage 1 checkpoint"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage1')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n✓ Saving Stage 1 checkpoint...")
        np.save(os.path.join(checkpoint_dir, 'lats_hrrr.npy'), lats_hrrr)
        np.save(os.path.join(checkpoint_dir, 'lons_hrrr.npy'), lons_hrrr)
        np.save(os.path.join(checkpoint_dir, 'terrain.npy'), terrain)
        np.save(os.path.join(checkpoint_dir, 'x_hrrr.npy'), x_hrrr)
        np.save(os.path.join(checkpoint_dir, 'y_hrrr.npy'), y_hrrr)
        np.save(os.path.join(checkpoint_dir, 'dx_east.npy'), dx_east)
        np.save(os.path.join(checkpoint_dir, 'dy_east.npy'), dy_east)
        np.save(os.path.join(checkpoint_dir, 'dx_north.npy'), dx_north)
        np.save(os.path.join(checkpoint_dir, 'dy_north.npy'), dy_north)

        with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
            f.write(f"{shape[0]}\n{shape[1]}\n{n_fine_i}\n{n_fine_j}\n")

        print(f"  → {checkpoint_dir}")

    def _load_stage1_checkpoint(self):
        """Load stage 1 checkpoint"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage1')
        print(f"\n{'='*70}")
        print(" LOADING STAGE 1 CHECKPOINT")
        print(f"{'='*70}")
        print(f"Loading from: {checkpoint_dir}")

        with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'r') as f:
            shape = (int(f.readline().strip()), int(f.readline().strip()))
            n_fine_i = int(f.readline().strip())
            n_fine_j = int(f.readline().strip())

        # Recreate projection (can't easily pickle Basemap)
        lats_hrrr = np.load(os.path.join(checkpoint_dir, 'lats_hrrr.npy'))
        lons_hrrr = np.load(os.path.join(checkpoint_dir, 'lons_hrrr.npy'))

        self.proj = Basemap(
            projection='lcc',
            lat_0=38.5, lon_0=-97.5, lat_1=38.5, lat_2=38.5,
            llcrnrlat=lats_hrrr.min(), urcrnrlat=lats_hrrr.max(),
            llcrnrlon=lons_hrrr.min(), urcrnrlon=lons_hrrr.max(),
            resolution=None
        )
        self.hrrr_crs = self.proj.proj4string

        return {
            'lats_hrrr': lats_hrrr,
            'lons_hrrr': lons_hrrr,
            'terrain': np.load(os.path.join(checkpoint_dir, 'terrain.npy')),
            'shape': shape,
            'x_hrrr': np.load(os.path.join(checkpoint_dir, 'x_hrrr.npy')),
            'y_hrrr': np.load(os.path.join(checkpoint_dir, 'y_hrrr.npy')),
            'dx_east': np.load(os.path.join(checkpoint_dir, 'dx_east.npy')),
            'dy_east': np.load(os.path.join(checkpoint_dir, 'dy_east.npy')),
            'dx_north': np.load(os.path.join(checkpoint_dir, 'dx_north.npy')),
            'dy_north': np.load(os.path.join(checkpoint_dir, 'dy_north.npy')),
            'n_fine_i': n_fine_i,
            'n_fine_j': n_fine_j,
        }

    def _run_stage2_indices(self, stage_data):
        """Stage 2: Generate index sets from terrain and features"""
        print("\n" + "="*70)
        print(" STAGE 2: INDEX GENERATION")
        print("="*70)

        # Initialize sparse index sets for each mask
        print("\nInitializing sparse index sets...")
        index_sets = self._initialize_index_sets()

        # Process terrain (fast - already on HRRR grid)
        print("\nProcessing terrain thresholds...")
        self._process_terrain(stage_data['terrain'], index_sets,
                             stage_data['n_fine_i'], stage_data['n_fine_j'])

        # Process features in patches (avoids creating full 2B point grid)
        print("\nProcessing features in patches...")
        self._process_features_patched(
            stage_data['lats_hrrr'], stage_data['lons_hrrr'],
            stage_data['x_hrrr'], stage_data['y_hrrr'],
            stage_data['dx_east'], stage_data['dy_east'],
            stage_data['dx_north'], stage_data['dy_north'],
            stage_data['shape'], index_sets
        )

        # Save stage 2 checkpoint
        self._save_stage2_checkpoint(index_sets, stage_data['n_fine_i'], stage_data['n_fine_j'])

        return index_sets

    def _save_stage2_checkpoint(self, index_sets, n_fine_i, n_fine_j):
        """Save stage 2 checkpoint (index sets)"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage2')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n✓ Saving Stage 2 checkpoint...")
        with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'w') as f:
            f.write(f"{n_fine_i}\n{n_fine_j}\n")

        # Save each index set as separate numpy file (memory-efficient)
        total_size = 0
        for key, index_set in index_sets.items():
            if len(index_set) > 0:
                # Pre-allocate array and fill it to avoid intermediate list
                arr = np.empty((len(index_set), 2), dtype=np.int32)
                for idx, (i, j) in enumerate(index_set):
                    arr[idx, 0] = i
                    arr[idx, 1] = j
                npy_file = os.path.join(checkpoint_dir, f'{key}.npy')
                np.save(npy_file, arr)
                total_size += os.path.getsize(npy_file)
                if len(index_set) > 1_000_000:
                    print(f"  {key}: {len(index_set):,} indices")

        checkpoint_size_mb = total_size / (1024**2)
        print(f"  → {checkpoint_dir} ({checkpoint_size_mb:.1f} MB)")

    def _load_stage2_checkpoint(self):
        """Load stage 2 checkpoint"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage2')
        print(f"\n{'='*70}")
        print(" LOADING STAGE 2 CHECKPOINT")
        print(f"{'='*70}")
        print(f"Loading from: {checkpoint_dir}")

        with open(os.path.join(checkpoint_dir, 'metadata.txt'), 'r') as f:
            n_fine_i = int(f.readline().strip())
            n_fine_j = int(f.readline().strip())

        # Load index sets
        index_sets = self._initialize_index_sets()
        for key in index_sets.keys():
            npy_file = os.path.join(checkpoint_dir, f'{key}.npy')
            if os.path.exists(npy_file):
                arr = np.load(npy_file)
                index_sets[key] = set(tuple(row) for row in arr)

        total_indices = sum(len(s) for s in index_sets.values())
        print(f"  Restored {total_indices:,} indices from {len(index_sets)} index sets")

        # Also need stage 1 data for later stages
        stage_data = self._load_stage1_checkpoint()

        return index_sets, stage_data

    def _run_stage3_tiers(self, index_sets, stage_data):
        """Stage 3: Apply tier logic and stride patterns"""
        print("\n" + "="*70)
        print(" STAGE 3: TIER ASSIGNMENT")
        print("="*70)

        index_tier_metadata = self._apply_tier_logic_batched(
            index_sets, stage_data['n_fine_i'], stage_data['n_fine_j']
        )

        # Save stage 3 checkpoint
        self._save_stage3_checkpoint(index_tier_metadata)

        return index_tier_metadata

    def _save_stage3_checkpoint(self, index_tier_metadata):
        """Save stage 3 checkpoint (tier assignments)"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage3')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n✓ Saving Stage 3 checkpoint...")

        # Convert dict to arrays for efficient storage
        indices = np.array(list(index_tier_metadata.keys()), dtype=np.int32)
        tiers_metadata = np.array([index_tier_metadata[tuple(idx)] for idx in indices], dtype=np.int16)

        np.save(os.path.join(checkpoint_dir, 'indices.npy'), indices)
        np.save(os.path.join(checkpoint_dir, 'tiers_metadata.npy'), tiers_metadata)

        size_mb = (indices.nbytes + tiers_metadata.nbytes) / (1024**2)
        print(f"  → {checkpoint_dir} ({size_mb:.1f} MB, {len(indices):,} points)")

    def _load_stage3_checkpoint(self):
        """Load stage 3 checkpoint"""
        checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoint_stage3')
        print(f"\n{'='*70}")
        print(" LOADING STAGE 3 CHECKPOINT")
        print(f"{'='*70}")
        print(f"Loading from: {checkpoint_dir}")

        indices = np.load(os.path.join(checkpoint_dir, 'indices.npy'))
        tiers_metadata = np.load(os.path.join(checkpoint_dir, 'tiers_metadata.npy'))

        # Reconstruct dictionary
        index_tier_metadata = {
            tuple(idx): tuple(tm) for idx, tm in zip(indices, tiers_metadata)
        }

        print(f"  Restored {len(index_tier_metadata):,} tier assignments")

        # Also need stage 1 data for output generation
        stage_data = self._load_stage1_checkpoint()

        return index_tier_metadata, stage_data

    def _run_stage4_output(self, index_tier_metadata, stage_data):
        """Stage 4: Convert to lat/lon and write output"""
        print("\n" + "="*70)
        print(" STAGE 4: OUTPUT GENERATION")
        print("="*70)

        # Convert indices to lat/lon
        print("\nConverting indices to lat/lon...")
        lats_out, lons_out, tiers_out, metadata_out = self._indices_to_latlon(
            index_tier_metadata,
            stage_data['lats_hrrr'], stage_data['lons_hrrr'],
            stage_data['x_hrrr'], stage_data['y_hrrr'],
            stage_data['dx_east'], stage_data['dy_east'],
            stage_data['dx_north'], stage_data['dy_north']
        )

        print(f"\nTotal points: {len(lats_out):,}")
        print(f"Target: {config.TARGET_TOTAL_POINTS:,}")
        diff_pct = 100 * (len(lats_out) - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
        print(f"Difference: {diff_pct:+.1f}%")

        # Write output
        print("\nWriting output...")
        if self.test_region:
            output_path = f'output/adaptive_grid_SPARSE_{self.test_region}.nc'
        else:
            output_path = 'output/adaptive_grid_SPARSE.nc'
        self._write_output(lats_out, lons_out, tiers_out, metadata_out, output_path)

        return lats_out, lons_out, tiers_out, metadata_out, output_path

    def _apply_test_region(self, lats_hrrr, lons_hrrr, terrain):
        """Apply test region filter"""
        if self.test_region == 'west_wa':
            # Western Washington state: roughly 45°N-49°N, 124°W-120°W
            print("Applying Western Washington test region filter...")
            mask = ((lats_hrrr >= 45.0) & (lats_hrrr <= 49.5) &
                   (lons_hrrr >= -124.8) & (lons_hrrr <= -119.5))

            # Find bounding box in array indices
            rows, cols = np.where(mask)
            i_min, i_max = rows.min(), rows.max() + 1
            j_min, j_max = cols.min(), cols.max() + 1

            lats_sub = lats_hrrr[i_min:i_max, j_min:j_max]
            lons_sub = lons_hrrr[i_min:i_max, j_min:j_max]
            terrain_sub = terrain[i_min:i_max, j_min:j_max]

            print(f"  Subset shape: {lats_sub.shape}")
            print(f"  Lat range: {lats_sub.min():.2f} to {lats_sub.max():.2f}")
            print(f"  Lon range: {lons_sub.min():.2f} to {lons_sub.max():.2f}")

            return lats_sub, lons_sub, terrain_sub, lats_sub.shape
        elif self.test_region == 'nw_us':
            # Northwest US (existing region)
            print("Applying Northwest US test region filter...")
            mask = ((lats_hrrr >= 42.0) & (lats_hrrr <= 49.0) &
                   (lons_hrrr >= -125.0) & (lons_hrrr <= -115.0))

            rows, cols = np.where(mask)
            i_min, i_max = rows.min(), rows.max() + 1
            j_min, j_max = cols.min(), cols.max() + 1

            lats_sub = lats_hrrr[i_min:i_max, j_min:j_max]
            lons_sub = lons_hrrr[i_min:i_max, j_min:j_max]
            terrain_sub = terrain[i_min:i_max, j_min:j_max]

            print(f"  Subset shape: {lats_sub.shape}")
            print(f"  Lat range: {lats_sub.min():.2f} to {lats_sub.max():.2f}")
            print(f"  Lon range: {lons_sub.min():.2f} to {lons_sub.max():.2f}")

            return lats_sub, lons_sub, terrain_sub, lats_sub.shape
        else:
            return lats_hrrr, lons_hrrr, terrain, lats_hrrr.shape

    def _compute_cell_vectors(self, x_hrrr, y_hrrr):
        """Compute orientation vectors for HRRR cells"""
        dx_east = np.zeros_like(x_hrrr)
        dy_east = np.zeros_like(x_hrrr)
        dx_north = np.zeros_like(x_hrrr)
        dy_north = np.zeros_like(x_hrrr)

        dx_east[:, :-1] = x_hrrr[:, 1:] - x_hrrr[:, :-1]
        dy_east[:, :-1] = y_hrrr[:, 1:] - y_hrrr[:, :-1]
        dx_east[:, -1] = dx_east[:, -2]
        dy_east[:, -1] = dy_east[:, -2]

        dx_north[:-1, :] = x_hrrr[1:, :] - x_hrrr[:-1, :]
        dy_north[:-1, :] = y_hrrr[1:, :] - y_hrrr[:-1, :]
        dx_north[-1, :] = dx_north[-2, :]
        dy_north[-1, :] = dy_north[-2, :]

        return dx_east, dy_east, dx_north, dy_north

    def _initialize_index_sets(self):
        """Initialize dictionary of sets for sparse indices"""
        return {
            # Coastlines at multiple buffer distances
            'coastline_750m': set(),
            'coastline_1500m': set(),
            'coastline_3000m': set(),
            'coastline_6000m': set(),
            'coastline_12000m': set(),

            # Lakes at multiple buffer distances
            'lakes_750m': set(),
            'lakes_1500m': set(),
            'lakes_3000m': set(),
            'lakes_6000m': set(),

            # Terrain variability thresholds
            'terrain_gt800': set(),
            'terrain_gt600': set(),
            'terrain_gt300': set(),
            'terrain_gt150': set(),

            # Urban areas
            'urban_high': set(),
            'urban_suburban': set(),
            'small_towns': set(),

            # Recreation and transportation
            'ski_resorts': set(),
            'roads': set(),

            # Protected areas
            'parks': set(),
            'forests': set(),
        }

    def _process_terrain(self, terrain, index_sets, n_fine_i, n_fine_j):
        """Process terrain variability thresholds"""
        print("\n" + "="*70)
        print(" STEP 1/5: COMPUTING TERRAIN VARIABILITY")
        print("="*70)
        print("\nUsing 3x3 pixel window (~1km)")
        print(f"Processing {terrain.shape[0]} × {terrain.shape[1]} grid cells...")
        print("(This may take 2-5 minutes...)")

        start = time.time()
        from scipy.ndimage import generic_filter
        terrain_std = generic_filter(terrain, np.std, size=3, mode='constant', cval=0)
        elapsed = time.time() - start

        print(f"\n✓ Terrain variability computed in {elapsed:.1f} seconds")
        print(f"  Std dev range: {terrain_std.min():.1f} to {terrain_std.max():.1f} m")
        print(f"  Cells with std > 600m (extreme): {(terrain_std > 600).sum():,}")
        print(f"  Cells with std > 400m (medium): {(terrain_std > 400).sum():,}")
        print(f"  Cells with std > 100m (lower): {(terrain_std > 100).sum():,}")

        # Expand to fine grid and add to sets
        for threshold, key in [(800, 'terrain_gt800'), (600, 'terrain_gt600'),
                              (300, 'terrain_gt300'), (150, 'terrain_gt150')]:
            hrrr_cells = np.where(terrain_std > threshold)
            for i_hrrr, j_hrrr in zip(hrrr_cells[0], hrrr_cells[1]):
                i_start = i_hrrr * self.n_fine
                j_start = j_hrrr * self.n_fine
                for di in range(self.n_fine):
                    for dj in range(self.n_fine):
                        index_sets[key].add((i_start + di, j_start + dj))
            print(f"  {key}: {len(index_sets[key]):,} indices")

    # The rest of the methods follow similar pattern from the original
    # I'll add them in chunks...

    def _process_features_patched(self, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                                   dx_east, dy_east, dx_north, dy_north,
                                   shape, index_sets):
        """Process geographic features using patch-based approach with pre-buffering"""
        patch_size = 10  # HRRR cells per patch
        fine_patch_size = patch_size * self.n_fine  # 320 fine points per patch

        n_patches_i = (shape[0] + patch_size - 1) // patch_size
        n_patches_j = (shape[1] + patch_size - 1) // patch_size
        total_patches = n_patches_i * n_patches_j

        print(f"  Patch size: {patch_size}×{patch_size} HRRR cells = {fine_patch_size}×{fine_patch_size} fine points")
        print(f"  Total patches: {total_patches:,}")

        # Pre-compute buffered geometries
        self._preproject_features()

        # Process patches
        print("  Processing patches with pre-buffered geometries...")
        start_total_indices = sum(len(s) for s in index_sets.values())

        patch_count = 0
        for pi in range(n_patches_i):
            for pj in range(n_patches_j):
                # Process this patch
                self._process_single_patch(pi, pj, patch_size, shape,
                                          lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                                          dx_east, dy_east, dx_north, dy_north,
                                          index_sets)

                patch_count += 1
                if patch_count % 500 == 0 or patch_count == total_patches:
                    total_indices = sum(len(s) for s in index_sets.values())
                    pct = 100 * patch_count / total_patches
                    print(f"  Progress: {patch_count}/{total_patches} patches ({pct:.1f}%), {total_indices:,} total indices")

        final_total = sum(len(s) for s in index_sets.values())
        print(f"  Complete: {total_patches}/{total_patches} patches, {final_total:,} total indices")

    def _preproject_features(self):
        """Pre-project features and PRE-COMPUTE all buffered geometries"""
        print("  Pre-computing buffered geometries (this will take a few minutes but saves hours later)...")

        self.buffered_features = {}
        self.spatial_indices = {}

        # Coastlines - buffer at multiple distances
        coastline_gdf = self.loader.data.get('ocean_coastline')
        if coastline_gdf is not None:
            significant = coastline_gdf[coastline_gdf['length_km'] > 50].copy()
            coast_proj = significant.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(coast_proj)} coastline segments...")

            for dist, key in [(750, 'coastline_750m'), (1500, 'coastline_1500m'),
                             (3000, 'coastline_3000m'), (6000, 'coastline_6000m'),
                             (12000, 'coastline_12000m')]:
                buffered = [geom.buffer(dist) for geom in coast_proj.geometry]
                self.buffered_features[key] = buffered
                self.spatial_indices[key] = STRtree(buffered)

            del coast_proj, significant

        # Lakes - buffer at multiple distances
        lakes_gdf = self.loader.data.get('lakes')
        if lakes_gdf is not None:
            lakes_proj = lakes_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(lakes_proj)} lake boundaries...")

            for dist, key in [(750, 'lakes_750m'), (1500, 'lakes_1500m'),
                             (3000, 'lakes_3000m'), (6000, 'lakes_6000m')]:
                buffered = [geom.buffer(dist) for geom in lakes_proj.geometry]
                self.buffered_features[key] = buffered
                self.spatial_indices[key] = STRtree(buffered)

            del lakes_proj

        # Ski resorts - buffer at 2km
        ski_gdf = self.loader.data.get('ski_resorts')
        if ski_gdf is not None:
            ski_proj = ski_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(ski_proj)} ski resorts...")
            buffered = [geom.buffer(2000) for geom in ski_proj.geometry]
            self.buffered_features['ski_resorts'] = buffered
            self.spatial_indices['ski_resorts'] = STRtree(buffered)
            del ski_proj

        # Urban areas - HIGH DENSITY (>=50k population) → Tier 2
        high_density_gdf = self.loader.data.get('high_density_urban')
        if high_density_gdf is not None:
            high_density_proj = high_density_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(high_density_proj)} high-density urban boundaries...")
            boundaries = list(high_density_proj.geometry)
            self.buffered_features['urban_high'] = boundaries
            self.spatial_indices['urban_high'] = STRtree(boundaries)
            del high_density_proj

        # Urban areas - SUBURBAN (larger clusters) → Tier 2
        suburban_gdf = self.loader.data.get('suburban')
        if suburban_gdf is not None:
            suburban_proj = suburban_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(suburban_proj)} suburban boundaries...")
            boundaries = list(suburban_proj.geometry)
            self.buffered_features['urban_suburban'] = boundaries
            self.spatial_indices['urban_suburban'] = STRtree(boundaries)
            del suburban_proj

        # Urban areas - SMALL TOWNS (smaller clusters) → Tier 3
        small_towns_gdf = self.loader.data.get('small_towns')
        if small_towns_gdf is not None:
            small_towns_proj = small_towns_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(small_towns_proj)} small town boundaries...")
            boundaries = list(small_towns_proj.geometry)
            self.buffered_features['small_towns'] = boundaries
            self.spatial_indices['small_towns'] = STRtree(boundaries)
            del small_towns_proj

        # Roads - buffer at 1km
        roads_gdf = self.loader.data.get('roads')
        if roads_gdf is not None:
            roads_proj = roads_gdf.to_crs(self.hrrr_crs)
            print(f"    Buffering {len(roads_proj)} road segments...")
            buffered = [geom.buffer(1000) for geom in roads_proj.geometry]
            self.buffered_features['roads'] = buffered
            self.spatial_indices['roads'] = STRtree(buffered)
            del roads_proj

        # Parks
        parks_gdf = self.loader.data.get('national_parks')
        if parks_gdf is not None:
            parks_proj = parks_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(parks_proj)} national parks...")
            boundaries = list(parks_proj.geometry)
            self.buffered_features['parks'] = boundaries
            self.spatial_indices['parks'] = STRtree(boundaries)
            del parks_proj

        # Forests
        forests_gdf = self.loader.data.get('national_forests')
        if forests_gdf is not None:
            forests_proj = forests_gdf.to_crs(self.hrrr_crs)
            print(f"    Processing {len(forests_proj)} national forests...")
            boundaries = list(forests_proj.geometry)
            self.buffered_features['forests'] = boundaries
            self.spatial_indices['forests'] = STRtree(boundaries)
            del forests_proj

        print("  ✓ All geometries buffered and indexed")

    def _process_single_patch(self, pi, pj, patch_size, shape,
                              lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                              dx_east, dy_east, dx_north, dy_north,
                              index_sets):
        """Process a single patch of the grid"""
        # Compute patch bounds in HRRR coordinates
        i_start = pi * patch_size
        i_end = min((pi + 1) * patch_size, shape[0])
        j_start = pj * patch_size
        j_end = min((pj + 1) * patch_size, shape[1])

        # Get center of patch for bounding box
        i_center = (i_start + i_end) // 2
        j_center = (j_start + j_end) // 2

        # Create bounding box in projection coordinates
        margin = patch_size * 3000 * 2  # Extra margin for buffers
        x_center = x_hrrr[i_center, j_center]
        y_center = y_hrrr[i_center, j_center]
        patch_bounds = box(x_center - margin, y_center - margin,
                          x_center + margin, y_center + margin)

        # Generate fine grid for this patch
        fine_i_start = i_start * self.n_fine
        fine_j_start = j_start * self.n_fine
        patch_shape = ((i_end - i_start) * self.n_fine, (j_end - j_start) * self.n_fine)

        x_patch = np.zeros(patch_shape)
        y_patch = np.zeros(patch_shape)

        for i_hrrr in range(i_start, i_end):
            for j_hrrr in range(j_start, j_end):
                for i_local in range(self.n_fine):
                    for j_local in range(self.n_fine):
                        frac_east = (j_local / self.n_fine) - 0.5
                        frac_north = (i_local / self.n_fine) - 0.5

                        x = x_hrrr[i_hrrr, j_hrrr] + frac_east * dx_east[i_hrrr, j_hrrr] + frac_north * dx_north[i_hrrr, j_hrrr]
                        y = y_hrrr[i_hrrr, j_hrrr] + frac_east * dy_east[i_hrrr, j_hrrr] + frac_north * dy_north[i_hrrr, j_hrrr]

                        i_patch = (i_hrrr - i_start) * self.n_fine + i_local
                        j_patch = (j_hrrr - j_start) * self.n_fine + j_local
                        x_patch[i_patch, j_patch] = x
                        y_patch[i_patch, j_patch] = y

        # Process each feature type
        for key in self.buffered_features.keys():
            if key not in self.spatial_indices:
                continue

            # Query spatial index for nearby features
            tree = self.spatial_indices[key]
            nearby_indices = tree.query(patch_bounds)

            if len(nearby_indices) == 0:
                continue

            # Process each nearby feature
            features = self.buffered_features[key]
            for idx in nearby_indices:
                buffered_geom = features[idx]
                self._process_buffered_feature(
                    buffered_geom, key, x_patch, y_patch,
                    patch_bounds, index_sets, fine_i_start, fine_j_start
                )

    def _process_buffered_feature(self, buffered_geom, mask_key, x_patch, y_patch,
                                   patch_bounds, index_sets, i_offset, j_offset):
        """Process a single PRE-BUFFERED feature using hierarchical multigrid approach"""
        # Hierarchical approach: check coarse grid first
        coarse_stride = 5
        x_coarse = x_patch[::coarse_stride, ::coarse_stride]
        y_coarse = y_patch[::coarse_stride, ::coarse_stride]

        if HAS_CONTAINS_XY:
            coarse_inside = contains_xy(buffered_geom, x_coarse.ravel(), y_coarse.ravel())
        else:
            coarse_inside = vectorized.contains(buffered_geom, x_coarse.ravel(), y_coarse.ravel())

        if not coarse_inside.any():
            return

        # Expand coarse hits to fine grid
        coarse_inside_2d = coarse_inside.reshape(x_coarse.shape)
        fine_mask = np.zeros(x_patch.shape, dtype=bool)

        for ci in range(coarse_inside_2d.shape[0]):
            for cj in range(coarse_inside_2d.shape[1]):
                if coarse_inside_2d[ci, cj]:
                    i_start = max(0, ci * coarse_stride - coarse_stride)
                    i_end = min(x_patch.shape[0], (ci + 1) * coarse_stride + coarse_stride)
                    j_start = max(0, cj * coarse_stride - coarse_stride)
                    j_end = min(x_patch.shape[1], (cj + 1) * coarse_stride + coarse_stride)
                    fine_mask[i_start:i_end, j_start:j_end] = True

        # Check fine grid points
        x_fine = x_patch[fine_mask]
        y_fine = y_patch[fine_mask]

        if len(x_fine) == 0:
            return

        if HAS_CONTAINS_XY:
            fine_inside = contains_xy(buffered_geom, x_fine, y_fine)
        else:
            fine_inside = vectorized.contains(buffered_geom, x_fine, y_fine)

        # Add to index set
        fine_indices = np.where(fine_mask)
        for idx in np.where(fine_inside)[0]:
            i_local = fine_indices[0][idx]
            j_local = fine_indices[1][idx]
            i_global = i_offset + i_local
            j_global = j_offset + j_local
            index_sets[mask_key].add((i_global, j_global))

    def _apply_tier_logic(self, index_sets, n_fine_i, n_fine_j):
        """Apply tier priority and stride patterns, returning dict with tier and metadata"""

        # Tier 0: Ski resorts (stride=1, 93.75m)
        tier0_indices = index_sets['ski_resorts']
        tier0_dict = {idx: (0, METADATA_SKI_RESORT) for idx in tier0_indices}

        # Tier 1: Coastlines/lakes 750m OR extreme terrain (stride=2, 187.5m)
        tier1_full = (index_sets['coastline_750m'] | index_sets['lakes_750m'] |
                     index_sets['terrain_gt800'])
        tier1_indices = tier1_full - tier0_indices

        # Build metadata for tier 1
        tier1_dict = {}
        for idx in tier1_indices:
            metadata = 0
            if idx in index_sets['coastline_750m']:
                metadata |= METADATA_COASTLINE
            if idx in index_sets['lakes_750m']:
                metadata |= METADATA_LAKE
            if idx in index_sets['terrain_gt800']:
                metadata |= METADATA_TERRAIN_EXTREME
            tier1_dict[idx] = (1, metadata)

        # Tier 2: Coastlines/lakes 1500m OR high-density urban OR suburban OR very rugged (stride=4, 375m)
        tier2_full = (index_sets['coastline_1500m'] | index_sets['lakes_1500m'] |
                     index_sets['urban_high'] | index_sets['urban_suburban'] | index_sets['terrain_gt600'])
        tier2_indices = tier2_full - tier1_full - tier0_indices

        tier2_dict = {}
        for idx in tier2_indices:
            metadata = 0
            if idx in index_sets['coastline_1500m']:
                metadata |= METADATA_COASTLINE
            if idx in index_sets['lakes_1500m']:
                metadata |= METADATA_LAKE
            if idx in index_sets['urban_high']:
                metadata |= METADATA_URBAN_HIGH
            if idx in index_sets['urban_suburban']:
                metadata |= METADATA_URBAN_SUBURBAN
            if idx in index_sets['terrain_gt600']:
                metadata |= METADATA_TERRAIN_HIGH
            tier2_dict[idx] = (2, metadata)

        # Tier 3: Coastlines/lakes 3km OR small towns OR roads OR parks OR forests OR rugged (stride=8, 750m)
        tier3_full = (index_sets['coastline_3000m'] | index_sets['lakes_3000m'] |
                     index_sets['small_towns'] | index_sets['roads'] |
                     index_sets['parks'] | index_sets['forests'] | index_sets['terrain_gt300'])
        tier3_indices = tier3_full - tier2_full - tier1_full - tier0_indices

        tier3_dict = {}
        for idx in tier3_indices:
            metadata = 0
            if idx in index_sets['coastline_3000m']:
                metadata |= METADATA_COASTLINE
            if idx in index_sets['lakes_3000m']:
                metadata |= METADATA_LAKE
            if idx in index_sets['small_towns']:
                metadata |= METADATA_SMALL_TOWN
            if idx in index_sets['roads']:
                metadata |= METADATA_ROAD
            if idx in index_sets['parks']:
                metadata |= METADATA_PARK
            if idx in index_sets['forests']:
                metadata |= METADATA_FOREST
            if idx in index_sets['terrain_gt300']:
                metadata |= METADATA_TERRAIN_MODERATE
            tier3_dict[idx] = (3, metadata)

        # Tier 4: Coastlines/lakes 6km OR moderate terrain (stride=16, 1.5km)
        tier4_full = (index_sets['coastline_6000m'] | index_sets['lakes_6000m'] |
                     index_sets['terrain_gt150'])
        tier4_indices = tier4_full - tier3_full - tier2_full - tier1_full - tier0_indices

        tier4_dict = {}
        for idx in tier4_indices:
            metadata = 0
            if idx in index_sets['coastline_6000m']:
                metadata |= METADATA_COASTLINE
            if idx in index_sets['lakes_6000m']:
                metadata |= METADATA_LAKE
            if idx in index_sets['terrain_gt150']:
                metadata |= METADATA_TERRAIN_MODERATE
            tier4_dict[idx] = (4, metadata)

        print(f"  Tier 0 (93.75m): {len(tier0_indices):,} indices")
        print(f"  Tier 1 (187.5m): {len(tier1_indices):,} indices")
        print(f"  Tier 2 (375m): {len(tier2_indices):,} indices")
        print(f"  Tier 3 (750m): {len(tier3_indices):,} indices")
        print(f"  Tier 4 (1.5km): {len(tier4_indices):,} indices")

        # Apply stride patterns
        print("  Applying stride decimation...")
        final_dict = {}

        # Tier 0: stride=1 (keep all)
        print(f"    Processing Tier 0 (stride=1)...")
        final_dict.update(tier0_dict)

        # Tier 1: stride=2
        print(f"    Processing Tier 1 (stride=2)...")
        for idx, (tier, metadata) in tier1_dict.items():
            i, j = idx
            if (i & 1) == 0 and (j & 1) == 0:
                final_dict[idx] = (tier, metadata)

        # Tier 2: stride=4
        print(f"    Processing Tier 2 (stride=4)...")
        for idx, (tier, metadata) in tier2_dict.items():
            i, j = idx
            if (i & 3) == 0 and (j & 3) == 0:
                final_dict[idx] = (tier, metadata)

        # Tier 3: stride=8
        print(f"    Processing Tier 3 (stride=8)...")
        for idx, (tier, metadata) in tier3_dict.items():
            i, j = idx
            if (i & 7) == 0 and (j & 7) == 0:
                final_dict[idx] = (tier, metadata)

        # Tier 4: stride=16
        print(f"    Processing Tier 4 (stride=16)...")
        for idx, (tier, metadata) in tier4_dict.items():
            i, j = idx
            if (i & 15) == 0 and (j & 15) == 0:
                final_dict[idx] = (tier, metadata)

        # Tier 5: Background grid (stride=32, 3km)
        print("  Generating Tier 5 background grid...")
        tier5_count = 0
        for i in range(0, n_fine_i, 32):
            for j in range(0, n_fine_j, 32):
                if (i, j) not in final_dict:
                    final_dict[(i, j)] = (5, METADATA_BACKGROUND)
                    tier5_count += 1

        print(f"  Tier 5 (3km): {tier5_count:,} background points")
        print(f"  Final indices after stride: {len(final_dict):,}")

        return final_dict

    def _apply_tier_logic_batched(self, index_sets, n_fine_i, n_fine_j):
        """Apply tier logic with batched processing to reduce memory usage"""
        print("  Converting index sets to sorted arrays for efficient processing...")

        # Convert sets to sorted numpy arrays for memory-efficient lookups
        sorted_sets = {}
        for key, index_set in index_sets.items():
            if len(index_set) > 0:
                # Convert (i,j) tuples to int64 encoding: i * 2^32 + j
                # Vectorized approach: much faster than list comprehension
                arr = np.array(list(index_set), dtype=np.int32)  # Shape: (N, 2)
                encoded = arr[:, 0].astype(np.int64) * (2**32) + arr[:, 1].astype(np.int64)
                sorted_sets[key] = np.sort(encoded)
            else:
                sorted_sets[key] = np.array([], dtype=np.int64)
            # Free original set memory
            index_sets[key] = None

        print(f"  Converted {len(sorted_sets)} index sets to sorted arrays")

        # Process tiers with priority (high to low)
        final_results = []  # List of (i, j, tier, metadata) tuples

        # Tier 0: Ski resorts (stride=1, keep all)
        tier0_indices = sorted_sets['ski_resorts']
        print(f"  Tier 0 (93.75m): {len(tier0_indices):,} indices")
        for encoded_idx in tier0_indices:
            i = int(encoded_idx // (2**32))
            j = int(encoded_idx % (2**32))
            final_results.append((i, j, 0, METADATA_SKI_RESORT))

        # Track assigned indices to avoid duplicates
        assigned = set(tier0_indices)

        # Tier 1: Coastlines/lakes 750m OR extreme terrain (stride=2)
        print(f"  Tier 1 (187.5m): Processing...")
        tier1_combined = np.unique(np.concatenate([
            sorted_sets['coastline_750m'],
            sorted_sets['lakes_750m'],
            sorted_sets['terrain_gt800']
        ]))
        tier1_new = tier1_combined[~np.isin(tier1_combined, list(assigned))]
        print(f"    {len(tier1_new):,} new indices in Tier 1")

        # Process Tier 1 in batches
        batch_size = 10_000_000
        for batch_start in range(0, len(tier1_new), batch_size):
            batch_end = min(batch_start + batch_size, len(tier1_new))
            batch = tier1_new[batch_start:batch_end]

            for encoded_idx in batch:
                i = int(encoded_idx // (2**32))
                j = int(encoded_idx % (2**32))

                # Apply stride=2
                if (i & 1) == 0 and (j & 1) == 0:
                    metadata = 0
                    if np.searchsorted(sorted_sets['coastline_750m'], encoded_idx) < len(sorted_sets['coastline_750m']) and \
                       sorted_sets['coastline_750m'][np.searchsorted(sorted_sets['coastline_750m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_COASTLINE
                    if np.searchsorted(sorted_sets['lakes_750m'], encoded_idx) < len(sorted_sets['lakes_750m']) and \
                       sorted_sets['lakes_750m'][np.searchsorted(sorted_sets['lakes_750m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_LAKE
                    if np.searchsorted(sorted_sets['terrain_gt800'], encoded_idx) < len(sorted_sets['terrain_gt800']) and \
                       sorted_sets['terrain_gt800'][np.searchsorted(sorted_sets['terrain_gt800'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_TERRAIN_EXTREME

                    final_results.append((i, j, 1, metadata))
                    assigned.add(encoded_idx)

        # Tier 2: Coastlines/lakes 1500m OR high-density urban OR suburban OR very rugged (stride=4)
        print(f"  Tier 2 (375m): Processing...")
        tier2_combined = np.unique(np.concatenate([
            sorted_sets['coastline_1500m'],
            sorted_sets['lakes_1500m'],
            sorted_sets['urban_high'],
            sorted_sets['urban_suburban'],
            sorted_sets['terrain_gt600']
        ]))
        tier2_new = tier2_combined[~np.isin(tier2_combined, list(assigned))]
        print(f"    {len(tier2_new):,} new indices in Tier 2")

        for batch_start in range(0, len(tier2_new), batch_size):
            batch_end = min(batch_start + batch_size, len(tier2_new))
            batch = tier2_new[batch_start:batch_end]

            for encoded_idx in batch:
                i = int(encoded_idx // (2**32))
                j = int(encoded_idx % (2**32))

                # Apply stride=4
                if (i & 3) == 0 and (j & 3) == 0:
                    metadata = 0
                    if np.searchsorted(sorted_sets['coastline_1500m'], encoded_idx) < len(sorted_sets['coastline_1500m']) and \
                       sorted_sets['coastline_1500m'][np.searchsorted(sorted_sets['coastline_1500m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_COASTLINE
                    if np.searchsorted(sorted_sets['lakes_1500m'], encoded_idx) < len(sorted_sets['lakes_1500m']) and \
                       sorted_sets['lakes_1500m'][np.searchsorted(sorted_sets['lakes_1500m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_LAKE
                    if np.searchsorted(sorted_sets['urban_high'], encoded_idx) < len(sorted_sets['urban_high']) and \
                       sorted_sets['urban_high'][np.searchsorted(sorted_sets['urban_high'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_URBAN_HIGH
                    if np.searchsorted(sorted_sets['urban_suburban'], encoded_idx) < len(sorted_sets['urban_suburban']) and \
                       sorted_sets['urban_suburban'][np.searchsorted(sorted_sets['urban_suburban'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_URBAN_SUBURBAN
                    if np.searchsorted(sorted_sets['terrain_gt600'], encoded_idx) < len(sorted_sets['terrain_gt600']) and \
                       sorted_sets['terrain_gt600'][np.searchsorted(sorted_sets['terrain_gt600'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_TERRAIN_HIGH

                    final_results.append((i, j, 2, metadata))
                    assigned.add(encoded_idx)

        # Tier 3: Coastlines/lakes 3km OR small towns OR roads OR parks OR forests OR rugged (stride=8)
        print(f"  Tier 3 (750m): Processing...")
        tier3_combined = np.unique(np.concatenate([
            sorted_sets['coastline_3000m'],
            sorted_sets['lakes_3000m'],
            sorted_sets['small_towns'],
            sorted_sets['roads'],
            sorted_sets['parks'],
            sorted_sets['forests'],
            sorted_sets['terrain_gt300']
        ]))
        tier3_new = tier3_combined[~np.isin(tier3_combined, list(assigned))]
        print(f"    {len(tier3_new):,} new indices in Tier 3")

        for batch_start in range(0, len(tier3_new), batch_size):
            batch_end = min(batch_start + batch_size, len(tier3_new))
            batch = tier3_new[batch_start:batch_end]

            for encoded_idx in batch:
                i = int(encoded_idx // (2**32))
                j = int(encoded_idx % (2**32))

                # Apply stride=8
                if (i & 7) == 0 and (j & 7) == 0:
                    metadata = 0
                    if np.searchsorted(sorted_sets['coastline_3000m'], encoded_idx) < len(sorted_sets['coastline_3000m']) and \
                       sorted_sets['coastline_3000m'][np.searchsorted(sorted_sets['coastline_3000m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_COASTLINE
                    if np.searchsorted(sorted_sets['lakes_3000m'], encoded_idx) < len(sorted_sets['lakes_3000m']) and \
                       sorted_sets['lakes_3000m'][np.searchsorted(sorted_sets['lakes_3000m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_LAKE
                    if np.searchsorted(sorted_sets['small_towns'], encoded_idx) < len(sorted_sets['small_towns']) and \
                       sorted_sets['small_towns'][np.searchsorted(sorted_sets['small_towns'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_SMALL_TOWN
                    if np.searchsorted(sorted_sets['roads'], encoded_idx) < len(sorted_sets['roads']) and \
                       sorted_sets['roads'][np.searchsorted(sorted_sets['roads'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_ROAD
                    if np.searchsorted(sorted_sets['parks'], encoded_idx) < len(sorted_sets['parks']) and \
                       sorted_sets['parks'][np.searchsorted(sorted_sets['parks'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_PARK
                    if np.searchsorted(sorted_sets['forests'], encoded_idx) < len(sorted_sets['forests']) and \
                       sorted_sets['forests'][np.searchsorted(sorted_sets['forests'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_FOREST
                    if np.searchsorted(sorted_sets['terrain_gt300'], encoded_idx) < len(sorted_sets['terrain_gt300']) and \
                       sorted_sets['terrain_gt300'][np.searchsorted(sorted_sets['terrain_gt300'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_TERRAIN_MODERATE

                    final_results.append((i, j, 3, metadata))
                    assigned.add(encoded_idx)

        # Tier 4: Coastlines/lakes 6km OR moderate terrain (stride=16)
        print(f"  Tier 4 (1.5km): Processing...")
        tier4_combined = np.unique(np.concatenate([
            sorted_sets['coastline_6000m'],
            sorted_sets['lakes_6000m'],
            sorted_sets['terrain_gt150']
        ]))
        tier4_new = tier4_combined[~np.isin(tier4_combined, list(assigned))]
        print(f"    {len(tier4_new):,} new indices in Tier 4")

        for batch_start in range(0, len(tier4_new), batch_size):
            batch_end = min(batch_start + batch_size, len(tier4_new))
            batch = tier4_new[batch_start:batch_end]

            for encoded_idx in batch:
                i = int(encoded_idx // (2**32))
                j = int(encoded_idx % (2**32))

                # Apply stride=16
                if (i & 15) == 0 and (j & 15) == 0:
                    metadata = 0
                    if np.searchsorted(sorted_sets['coastline_6000m'], encoded_idx) < len(sorted_sets['coastline_6000m']) and \
                       sorted_sets['coastline_6000m'][np.searchsorted(sorted_sets['coastline_6000m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_COASTLINE
                    if np.searchsorted(sorted_sets['lakes_6000m'], encoded_idx) < len(sorted_sets['lakes_6000m']) and \
                       sorted_sets['lakes_6000m'][np.searchsorted(sorted_sets['lakes_6000m'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_LAKE
                    if np.searchsorted(sorted_sets['terrain_gt150'], encoded_idx) < len(sorted_sets['terrain_gt150']) and \
                       sorted_sets['terrain_gt150'][np.searchsorted(sorted_sets['terrain_gt150'], encoded_idx)] == encoded_idx:
                        metadata |= METADATA_TERRAIN_MODERATE

                    final_results.append((i, j, 4, metadata))
                    assigned.add(encoded_idx)

        # Tier 5: Background grid (stride=32)
        print("  Tier 5 (3km): Generating background grid...")
        tier5_count = 0
        for i in range(0, n_fine_i, 32):
            for j in range(0, n_fine_j, 32):
                encoded_idx = i * (2**32) + j
                if encoded_idx not in assigned:
                    final_results.append((i, j, 5, METADATA_BACKGROUND))
                    tier5_count += 1

        print(f"  Tier 5 (3km): {tier5_count:,} background points")
        print(f"  Final indices after stride: {len(final_results):,}")

        # Convert to dictionary format expected by downstream code
        final_dict = {(i, j): (tier, metadata) for i, j, tier, metadata in final_results}

        return final_dict

    def _indices_to_latlon(self, index_tier_metadata, lats_hrrr, lons_hrrr, x_hrrr, y_hrrr,
                           dx_east, dy_east, dx_north, dy_north):
        """Convert (i,j) indices to lat/lon coordinates by reconstructing from HRRR grid"""
        lats = []
        lons = []
        tiers = []
        metadata = []

        for (i, j), (tier, meta) in index_tier_metadata.items():
            # Find parent HRRR cell
            i_hrrr = i // self.n_fine
            j_hrrr = j // self.n_fine

            # Find position within cell
            i_local = i % self.n_fine
            j_local = j % self.n_fine

            # Get fractions within cell (-0.5 to +0.5)
            frac_east = (j_local / self.n_fine) - 0.5
            frac_north = (i_local / self.n_fine) - 0.5

            # Reconstruct projected coordinates
            x = x_hrrr[i_hrrr, j_hrrr] + frac_east * dx_east[i_hrrr, j_hrrr] + frac_north * dx_north[i_hrrr, j_hrrr]
            y = y_hrrr[i_hrrr, j_hrrr] + frac_east * dy_east[i_hrrr, j_hrrr] + frac_north * dy_north[i_hrrr, j_hrrr]

            # Convert to lat/lon
            lon, lat = self.proj(x, y, inverse=True)

            lats.append(lat)
            lons.append(lon)
            tiers.append(tier)
            metadata.append(meta)

        return np.array(lats), np.array(lons), np.array(tiers, dtype=np.int8), np.array(metadata, dtype=np.int32)

    def _write_output(self, lats, lons, tiers, metadata, output_path):
        """Write output NetCDF file with tier and metadata"""
        ds = nc.Dataset(output_path, 'w')
        ds.createDimension('points', len(lats))

        lat_var = ds.createVariable('latitude', 'f4', ('points',))
        lon_var = ds.createVariable('longitude', 'f4', ('points',))
        tier_var = ds.createVariable('tier', 'i1', ('points',))
        meta_var = ds.createVariable('metadata', 'i4', ('points',))

        lat_var.units = 'degrees_north'
        lat_var.long_name = 'Latitude'
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'Longitude'
        tier_var.long_name = 'Resolution tier (0=93.75m, 1=187.5m, 2=375m, 3=750m, 4=1.5km, 5=3km)'
        meta_var.long_name = 'Metadata bit flags indicating why point has this resolution'

        # Add metadata bit flag descriptions
        meta_var.bit_0 = 'Coastline'
        meta_var.bit_1 = 'Lake'
        meta_var.bit_2 = 'Ski resort'
        meta_var.bit_3 = 'High-density urban (≥50k pop)'
        meta_var.bit_4 = 'Suburban (large clusters)'
        meta_var.bit_5 = 'Small town (small clusters)'
        meta_var.bit_6 = 'Major road'
        meta_var.bit_7 = 'National park'
        meta_var.bit_8 = 'National forest'
        meta_var.bit_9 = 'Extreme terrain variability'
        meta_var.bit_10 = 'High terrain variability'
        meta_var.bit_11 = 'Moderate terrain variability'
        meta_var.bit_12 = 'Background grid point'

        lat_var[:] = lats
        lon_var[:] = lons
        tier_var[:] = tiers
        meta_var[:] = metadata

        # Add global attributes
        ds.projection = 'Lambert Conformal Conic'
        ds.lat_0 = '38.5'
        ds.lon_0 = '-97.5'
        ds.lat_1 = '38.5'
        ds.lat_2 = '38.5'
        ds.description = 'Adaptive grid with correct HRRR projection and metadata tracking'

        ds.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate adaptive grid with sparse index approach V2')
    parser.add_argument('--test-region', choices=['west_wa', 'nw_us', 'full'], default='west_wa',
                       help='Region to process (default: west_wa for testing)')
    parser.add_argument('--start-from-stage', type=int, choices=[1, 2, 3, 4], default=1,
                       help='Stage to resume from: 1=data prep, 2=indices, 3=tiers, 4=output (default: 1)')

    args = parser.parse_args()

    test_region = args.test_region if args.test_region != 'full' else None

    generator = SparseGridGenerator(test_region=test_region)
    lats, lons, tiers, metadata = generator.generate(start_stage=args.start_from_stage)
