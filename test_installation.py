"""
Test script to validate installation and data availability
"""
import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("="*70)
    print(" TESTING PACKAGE IMPORTS")
    print("="*70)

    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'geopandas': 'GeoPandas',
        'shapely': 'Shapely',
        'pygrib': 'pygrib (GRIB2 support)',
        'netCDF4': 'netCDF4',
        'requests': 'Requests',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
    }

    failed = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(module)

    # Test basemap separately (optional)
    try:
        from mpl_toolkits.basemap import Basemap
        print("✓ Basemap")
    except ImportError:
        print("⚠ Basemap (optional, but recommended)")
        print("  Install with: conda install -c conda-forge basemap")

    if failed:
        print(f"\n✗ {len(failed)} required packages failed to import")
        print("  Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages imported successfully")
        return True

def test_data_availability():
    """Check which data files are available"""
    print("\n" + "="*70)
    print(" CHECKING DATA AVAILABILITY")
    print("="*70)

    import config

    data_checks = {
        'HRRR Grid': [
            os.path.join(config.DATA_DIR, 'hrrr', 'hrrr_lats.npy'),
            os.path.join(config.DATA_DIR, 'hrrr', 'hrrr_lons.npy'),
        ],
        'HRRR Terrain': [
            os.path.join(config.DATA_DIR, 'hrrr', 'hrrr_terrain.npy'),
        ],
        'Coastline': [
            os.path.join(config.DATA_DIR, 'natural_earth', 'ne_10m_coastline.shp'),
        ],
        'Lakes': [
            os.path.join(config.DATA_DIR, 'natural_earth', 'ne_10m_lakes.shp'),
        ],
        'Urban Areas': [
            os.path.join(config.DATA_DIR, 'urban', 'tl_2020_us_uac10.shp'),
        ],
        'Primary Roads': [
            os.path.join(config.DATA_DIR, 'roads', 'tl_2023_us_primaryroads.shp'),
        ],
        'Ski Resorts': [
            os.path.join(config.DATA_DIR, 'ski_resorts', 'ski_areas.geojson'),
        ],
    }

    missing = []
    for name, files in data_checks.items():
        all_exist = all(os.path.exists(f) for f in files)
        if all_exist:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            missing.append(name)

    # Optional datasets
    print("\nOptional datasets:")
    padus_dir = os.path.join(config.DATA_DIR, 'padus')
    if os.path.exists(padus_dir) and len(os.listdir(padus_dir)) > 0:
        print("✓ PAD-US (Protected Areas)")
    else:
        print("⚠ PAD-US (Protected Areas) - requires manual download")

    if missing:
        print(f"\n⚠ {len(missing)} required datasets missing")
        print("  Run: python download_data.py")
        return False
    else:
        print("\n✓ All required datasets available")
        return True

def test_config():
    """Test configuration settings"""
    print("\n" + "="*70)
    print(" CONFIGURATION")
    print("="*70)

    import config

    print(f"Target points: {config.TARGET_TOTAL_POINTS:,}")
    print(f"Tier resolutions: {config.TIER_RESOLUTIONS}")
    print(f"Terrain threshold: {config.TERRAIN_STDDEV_THRESHOLD}m")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")

    return True

def estimate_resources():
    """Estimate computational requirements"""
    print("\n" + "="*70)
    print(" RESOURCE ESTIMATES")
    print("="*70)

    import config

    # Rough estimates
    target_points = config.TARGET_TOTAL_POINTS
    gb_per_million_points = 0.05  # ~50 MB per million points

    estimated_ram = target_points / 1e6 * gb_per_million_points * 3  # 3x for working memory
    estimated_output_size = target_points / 1e6 * gb_per_million_points

    print(f"For {target_points:,} points:")
    print(f"  Estimated RAM needed: {estimated_ram:.1f} GB")
    print(f"  Estimated output size: {estimated_output_size:.1f} GB")
    print(f"  Estimated runtime: 30-60 minutes")

    # Check available RAM
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"\nSystem RAM: {available_ram_gb:.1f} GB")
        if available_ram_gb < estimated_ram:
            print(f"⚠ Warning: May need more RAM (have {available_ram_gb:.1f} GB, need ~{estimated_ram:.1f} GB)")
        else:
            print(f"✓ Sufficient RAM available")
    except ImportError:
        print("\n(Install psutil to check system RAM: pip install psutil)")

    return True

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" ADAPTIVE GRID GENERATION - INSTALLATION TEST")
    print("="*70)

    results = {
        'Package Imports': test_imports(),
        'Configuration': test_config(),
        'Data Availability': test_data_availability(),
        'Resources': estimate_resources(),
    }

    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Ready to generate adaptive grid")
        print("  Run: python generate_adaptive_grid.py")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED - Please address issues above")
        print("="*70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
