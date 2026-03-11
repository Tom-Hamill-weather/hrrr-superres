"""
Diagnostic script to analyze sub-grid generation and identify gap patterns
"""
import numpy as np

def analyze_subgrid_generation(resolution_m=300, hrrr_cell_size_m=3000):
    """
    Analyze how points are distributed within HRRR cells
    """
    print(f"\n{'='*70}")
    print(f" SUBGRID GENERATION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  HRRR cell size: {hrrr_cell_size_m}m = {hrrr_cell_size_m/1000}km")
    print(f"  Target resolution: {resolution_m}m = {resolution_m/1000}km")

    # Current implementation
    points_per_cell = int((hrrr_cell_size_m / resolution_m) ** 2)
    n_per_side = int(np.sqrt(points_per_cell))

    print(f"  Points per cell: {points_per_cell} ({n_per_side} × {n_per_side})")

    # CURRENT ALGORITHM (from generate_adaptive_grid_GEN2.py)
    half_extent = (n_per_side - 1) / 2
    offsets = np.linspace(-half_extent, half_extent, n_per_side)

    print(f"\n--- CURRENT ALGORITHM ---")
    print(f"  half_extent = (n_per_side - 1) / 2 = ({n_per_side} - 1) / 2 = {half_extent}")
    print(f"  Offsets (in units of spacing): {offsets}")

    # Convert to actual distances (meters from cell center)
    offset_meters = offsets * resolution_m
    print(f"\n  Actual offsets (meters from cell center):")
    print(f"    First point: {offset_meters[0]:.1f}m")
    print(f"    Last point:  {offset_meters[-1]:.1f}m")
    print(f"    Range:       {offset_meters[-1] - offset_meters[0]:.1f}m")

    # Cell boundaries
    cell_half_size = hrrr_cell_size_m / 2
    print(f"\n  HRRR cell boundaries: ±{cell_half_size:.1f}m from center")

    # Gap at edges
    edge_gap = cell_half_size - offset_meters[-1]
    print(f"  Gap at each edge: {edge_gap:.1f}m")
    print(f"  Coverage: {(offset_meters[-1] - offset_meters[0]) / hrrr_cell_size_m * 100:.1f}%")

    # Gap between adjacent cells
    print(f"\n  Adjacent cells:")
    print(f"    Cell 1 rightmost point: {offset_meters[-1]:.1f}m from center")
    print(f"    Cell 2 leftmost point:  {hrrr_cell_size_m + offset_meters[0]:.1f}m from Cell 1 center")
    print(f"                          = {offset_meters[0]:.1f}m from Cell 2 center")
    inter_cell_gap = (hrrr_cell_size_m + offset_meters[0]) - offset_meters[-1]
    print(f"    Gap between cells: {inter_cell_gap:.1f}m")
    print(f"    (This is {inter_cell_gap/resolution_m:.2f} × point spacing)")

    # PROPOSED FIX
    print(f"\n--- PROPOSED FIX ---")
    # Option 1: Extend to cell edges
    print(f"\nOption 1: Extend points to cell boundaries")
    print(f"  half_extent = n_per_side / 2 = {n_per_side / 2}")
    offsets_fix1 = np.linspace(-n_per_side / 2.0, n_per_side / 2.0, n_per_side)
    offset_meters_fix1 = offsets_fix1 * resolution_m
    print(f"  Offsets: {offsets_fix1}")
    print(f"  Range: {offset_meters_fix1[0]:.1f}m to {offset_meters_fix1[-1]:.1f}m")
    print(f"  Coverage: {(offset_meters_fix1[-1] - offset_meters_fix1[0]) / hrrr_cell_size_m * 100:.1f}%")
    edge_gap_fix1 = cell_half_size - abs(offset_meters_fix1[-1])
    print(f"  Gap at each edge: {edge_gap_fix1:.1f}m")
    inter_cell_gap_fix1 = (hrrr_cell_size_m + offset_meters_fix1[0]) - offset_meters_fix1[-1]
    print(f"  Gap between cells: {inter_cell_gap_fix1:.1f}m")

    # Option 2: Adjust offsets to span cell
    print(f"\nOption 2: Adjust offsets to span full cell (points at ±(cell_size/2 - spacing/2))")
    max_offset_units = (hrrr_cell_size_m / 2.0 - resolution_m / 2.0) / resolution_m
    print(f"  max_offset = (cell_size/2 - spacing/2) / spacing")
    print(f"             = ({hrrr_cell_size_m/2:.1f} - {resolution_m/2:.1f}) / {resolution_m}")
    print(f"             = {max_offset_units:.3f}")
    offsets_fix2 = np.linspace(-max_offset_units, max_offset_units, n_per_side)
    offset_meters_fix2 = offsets_fix2 * resolution_m
    print(f"  Range: {offset_meters_fix2[0]:.1f}m to {offset_meters_fix2[-1]:.1f}m")
    print(f"  Coverage: {(offset_meters_fix2[-1] - offset_meters_fix2[0]) / hrrr_cell_size_m * 100:.1f}%")
    edge_gap_fix2 = cell_half_size - abs(offset_meters_fix2[-1])
    print(f"  Gap at each edge: {edge_gap_fix2:.1f}m (= spacing/2)")
    inter_cell_gap_fix2 = (hrrr_cell_size_m + offset_meters_fix2[0]) - offset_meters_fix2[-1]
    print(f"  Gap between cells: {inter_cell_gap_fix2:.1f}m (= spacing)")

    # Visualize point positions for one cell
    print(f"\n--- POINT POSITIONS (ONE CELL) ---")
    print(f"\nCurrent algorithm - points along one axis:")
    for i, (offset_units, offset_m) in enumerate(zip(offsets[:5], offset_meters[:5])):
        print(f"  Point {i}: {offset_m:+7.1f}m (offset = {offset_units:+5.2f} × spacing)")
    print(f"  ...")
    for i, (offset_units, offset_m) in enumerate(zip(offsets[-3:], offset_meters[-3:]), start=n_per_side-3):
        print(f"  Point {i}: {offset_m:+7.1f}m (offset = {offset_units:+5.2f} × spacing)")

    print(f"\nProposed Fix (Option 2) - points along one axis:")
    for i, (offset_units, offset_m) in enumerate(zip(offsets_fix2[:5], offset_meters_fix2[:5])):
        print(f"  Point {i}: {offset_m:+7.1f}m (offset = {offset_units:+5.2f} × spacing)")
    print(f"  ...")
    for i, (offset_units, offset_m) in enumerate(zip(offsets_fix2[-3:], offset_meters_fix2[-3:]), start=n_per_side-3):
        print(f"  Point {i}: {offset_m:+7.1f}m (offset = {offset_units:+5.2f} × spacing)")

    print(f"\n{'='*70}")
    print(f" RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\nThe current algorithm leaves {inter_cell_gap:.1f}m gaps between adjacent HRRR cells.")
    print(f"This is visible as systematic gaps in the visualization.")
    print(f"\nRecommended fix: Use Option 2 (adjust offsets to span full cell)")
    print(f"  - Points will extend to within spacing/2 of cell edges")
    print(f"  - Gap between cells = exactly one point spacing")
    print(f"  - This eliminates visible systematic gaps while avoiding duplicate points")
    print(f"\nCode change in generate_adaptive_grid_GEN2.py:")
    print(f"  REPLACE:")
    print(f"    half_extent = (n_per_side - 1) / 2")
    print(f"  WITH:")
    print(f"    half_extent = (hrrr_cell_size_m / 2.0 - resolution_m / 2.0) / resolution_m")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Analyze Tier 0 (300m resolution)
    analyze_subgrid_generation(resolution_m=300, hrrr_cell_size_m=3000)

    # Also analyze Tier 1 (300m) - should be same
    print("\n" + "="*70)
    print(" TIER 1 ANALYSIS (same resolution)")
    print("="*70)
    analyze_subgrid_generation(resolution_m=300, hrrr_cell_size_m=3000)

    # Analyze Tier 2 (500m)
    print("\n" + "="*70)
    print(" TIER 2 ANALYSIS")
    print("="*70)
    analyze_subgrid_generation(resolution_m=500, hrrr_cell_size_m=3000)
