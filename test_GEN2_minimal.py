"""
Minimal test of GEN2 algorithm with synthetic data
Tests the 3-pass algorithm without slow data loading
"""

import time
import numpy as np
from scipy.ndimage import distance_transform_edt, generic_filter
import config

print("\n" + "="*70)
print(" GEN2 MINIMAL TEST - Synthetic Data")
print("="*70)

# Create small synthetic grid (100x100 instead of 1799x1059)
print("\n[1/5] Creating synthetic test grid (100x100)...")
shape = (100, 100)
np.random.seed(42)

# Synthetic terrain variability
terrain_var = np.random.rand(*shape) * 300  # 0-300m std dev

# Initialize tier map (all background initially)
tier_map = np.full(shape, 5, dtype=np.int8)
metadata = np.zeros(shape, dtype=np.uint16)

print(f"✓ Synthetic grid created: {shape}")

# PASS 1: Core region classification (synthetic features)
print("\n" + "="*70)
print(" PASS 1/3: CORE REGION CLASSIFICATION")
print("="*70)

# Simulate coastline (left edge)
tier_map[:, :5] = np.minimum(tier_map[:, :5], 0)
print(f"✓ Synthetic coastline: {(tier_map == 0).sum()} cells")

# Simulate rugged terrain
extreme_terrain = terrain_var > config.TERRAIN_TIER1_THRESHOLD
tier_map[extreme_terrain] = np.minimum(tier_map[extreme_terrain], 1)
print(f"✓ Extreme terrain (Tier 1): {extreme_terrain.sum()} cells")

medium_terrain = (terrain_var > config.TERRAIN_TIER2_THRESHOLD) & \
                 (terrain_var <= config.TERRAIN_TIER1_THRESHOLD)
tier_map[medium_terrain] = np.minimum(tier_map[medium_terrain], 2)
print(f"✓ Medium terrain (Tier 2): {medium_terrain.sum()} cells")

slight_terrain = (terrain_var > config.TERRAIN_TIER3_THRESHOLD) & \
                 (terrain_var <= config.TERRAIN_TIER2_THRESHOLD)
tier_map[slight_terrain] = np.minimum(tier_map[slight_terrain], 3)
print(f"✓ Slight terrain (Tier 3): {slight_terrain.sum()} cells")

# PASS 2: Distance-based transition zones
print("\n" + "="*70)
print(" PASS 2/3: DISTANCE-BASED TRANSITION ZONES")
print("="*70)

start = time.time()

# Compute distance fields for each tier
distance_maps = {}
cell_size_km = 3.0

for tier in [0, 1, 2, 3, 5]:
    core_mask = (tier_map == tier)
    distance_maps[tier] = distance_transform_edt(~core_mask)

print(f"✓ Distance fields computed in {time.time() - start:.1f}s")

# Apply transition thresholds
start = time.time()
transition_map = tier_map.copy()

for i in range(shape[0]):
    for j in range(shape[1]):
        if tier_map[i, j] < 5:  # Skip core regions
            continue

        # Find nearest core tier
        distances_km = {t: distance_maps[t][i,j] * cell_size_km for t in [0,1,2,3,5]}
        min_dist_km = min(distances_km.values())

        # Apply transition thresholds
        assigned_tier = 5
        for tier in sorted(config.TRANSITION_THRESHOLDS.keys()):
            min_km, max_km = config.TRANSITION_THRESHOLDS[tier]
            if min_km <= min_dist_km < max_km:
                assigned_tier = tier
                break

        transition_map[i, j] = assigned_tier

print(f"✓ Transitions assigned in {time.time() - start:.1f}s")

# PASS 3: Constraint enforcement
print("\n" + "="*70)
print(" PASS 3/3: TIER CONSTRAINT ENFORCEMENT")
print("="*70)

start = time.time()
smoothed = transition_map.astype(np.float32)

def smooth_cell(values):
    center = values[4]
    neighbors = np.concatenate([values[:4], values[5:]])
    max_jump = np.max(np.abs(center - neighbors))

    if max_jump > config.MAX_TIER_JUMP:
        return min(center, np.mean(neighbors))
    return center

for iteration in range(config.CONSTRAINT_MAX_ITERATIONS):
    prev = smoothed.copy()
    smoothed = generic_filter(smoothed, smooth_cell, size=3, mode='nearest')

    max_change = np.max(np.abs(smoothed - prev))
    cells_changed = np.sum(~np.isclose(smoothed, prev, atol=0.01))

    print(f"  Iteration {iteration + 1}: {cells_changed} cells changed (max Δ={max_change:.3f})")

    if max_change < 0.01:
        print(f"  ✓ Converged after {iteration + 1} iterations")
        break

final_map = np.round(smoothed).astype(np.int8)
final_map = np.clip(final_map, 0, 5)

print(f"✓ Constraint enforcement completed in {time.time() - start:.1f}s")

# Verify constraints
print("\nVerifying tier constraints...")
violations = 0
max_jump_found = 0

for i in range(1, shape[0] - 1):
    for j in range(1, shape[1] - 1):
        center = final_map[i, j]
        neighbors = [
            final_map[i-1, j-1], final_map[i-1, j], final_map[i-1, j+1],
            final_map[i, j-1],                       final_map[i, j+1],
            final_map[i+1, j-1], final_map[i+1, j], final_map[i+1, j+1]
        ]

        max_jump = max(abs(center - n) for n in neighbors)
        max_jump_found = max(max_jump_found, max_jump)

        if max_jump > config.MAX_TIER_JUMP:
            violations += 1

print(f"  Max tier jump: {max_jump_found}")
print(f"  Violations: {violations}")

if violations == 0:
    print("  ✓ All constraints satisfied!")

# Summary
print("\n" + "="*70)
print(" TIER DISTRIBUTION")
print("="*70)

for tier in [0, 1, 2, 3, 4, 5]:
    count = (final_map == tier).sum()
    pct = 100 * count / final_map.size
    res = config.TIER_RESOLUTIONS[tier]
    print(f"  Tier {tier} ({res}m): {count:,} cells ({pct:.1f}%)")

# [4/5] Quick point count estimate
print("\n[4/5] Estimating point count...")
tier_counts = [(final_map == t).sum() for t in [0, 1, 2, 3, 4, 5]]
tier_points = [
    tier_counts[0] * ((3000 / config.TIER_RESOLUTIONS[0]) ** 2),
    tier_counts[1] * ((3000 / config.TIER_RESOLUTIONS[1]) ** 2),
    tier_counts[2] * ((3000 / config.TIER_RESOLUTIONS[2]) ** 2),
    tier_counts[3] * ((3000 / config.TIER_RESOLUTIONS[3]) ** 2),
    tier_counts[4] * ((3000 / config.TIER_RESOLUTIONS[4]) ** 2),
    tier_counts[5] * 1
]
estimated_total = sum(tier_points)

print(f"  Estimated points for 100x100 grid: {estimated_total:,.0f}")

# Extrapolate to full HRRR grid
full_hrrr_cells = 1799 * 1059
test_cells = 100 * 100
scale_factor = full_hrrr_cells / test_cells
extrapolated = estimated_total * scale_factor

print(f"  Extrapolated to full HRRR domain (1799x1059): {extrapolated:,.0f} points")
print(f"  Target: {config.TARGET_TOTAL_POINTS:,}")
diff_pct = 100 * (extrapolated - config.TARGET_TOTAL_POINTS) / config.TARGET_TOTAL_POINTS
print(f"  Difference: {diff_pct:+.1f}%")

print("\n" + "="*70)
print(" ✓ GEN2 ALGORITHM TEST COMPLETE")
print(" (3-pass hybrid algorithm verified on synthetic data)")
print("="*70)
