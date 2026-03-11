"""
Test DataLoader timing to identify bottlenecks
"""
import time
import sys
from generate_adaptive_grid import DataLoader

print("="*70)
print(" DATA LOADER TIMING TEST")
print("="*70)

loader = DataLoader()

# Time each load step individually
steps = [
    ('HRRR Grid', loader.load_hrrr_grid),
    ('Coastline', loader.load_coastline),
    ('Lakes', loader.load_lakes),
    ('Protected Areas', loader.load_protected_areas),
    ('Urban Areas', loader.load_urban_areas),
    ('Roads', loader.load_roads),
    ('Ski Resorts', loader.load_ski_resorts),
    ('Golf Courses', loader.load_golf_courses),
]

total_start = time.time()
timings = []

for name, func in steps:
    print(f"\n{'='*70}")
    print(f" Testing: {name}")
    print('='*70)

    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        timings.append((name, elapsed, 'SUCCESS'))
        print(f"\n✓ {name}: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        timings.append((name, elapsed, f'FAILED: {e}'))
        print(f"\n✗ {name}: {elapsed:.2f}s - ERROR: {e}")

    # Show progress
    total_elapsed = time.time() - total_start
    print(f"   [Cumulative: {total_elapsed:.2f}s]")
    sys.stdout.flush()

total_time = time.time() - total_start

print("\n" + "="*70)
print(" TIMING SUMMARY")
print("="*70)

for name, elapsed, status in timings:
    pct = 100 * elapsed / total_time
    print(f"  {name:20s}: {elapsed:6.2f}s ({pct:5.1f}%) - {status}")

print(f"\n  TOTAL: {total_time:.2f}s")
print("="*70)
