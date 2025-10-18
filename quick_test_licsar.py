#!/usr/bin/env python3
"""
Quick test script to verify LiCSAR functionality in MintPy
"""

import sys
import os
import numpy as np

# Add MintPy to path
sys.path.insert(0, '/raid-gpu2/maref/Software/Code-dev/MintPy/src')

def test_imports():
    """Test that all LiCSAR modules can be imported."""
    print("Testing imports...")
    
    try:
        from mintpy.utils import utils0 as ut
        print("✓ mintpy.utils.utils0 imported")
        
        from mintpy import prep_licsar
        print("✓ mintpy.prep_licsar imported")
        
        from mintpy import create_licsar_geometry
        print("✓ mintpy.create_licsar_geometry imported")
        
        from mintpy import load_data
        print("✓ mintpy.load_data imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_enu_functions():
    """Test the ENU-based geometry functions."""
    print("\nTesting ENU geometry functions...")
    
    try:
        from mintpy.utils import utils0 as ut
        
        # Create test data
        print("Creating test E,N,U data...")
        shape = (100, 100)
        e_data = np.random.uniform(-0.8, 0.8, shape).astype(np.float32)
        n_data = np.random.uniform(-0.6, 0.6, shape).astype(np.float32)
        u_data = np.random.uniform(0.3, 0.7, shape).astype(np.float32)
        
        # Normalize to unit vectors
        magnitude = np.sqrt(e_data**2 + n_data**2 + u_data**2)
        e_data /= magnitude
        n_data /= magnitude
        u_data /= magnitude
        
        print(f"Test data shape: {shape}")
        print(f"E range: {np.min(e_data):.3f} to {np.max(e_data):.3f}")
        print(f"N range: {np.min(n_data):.3f} to {np.max(n_data):.3f}")
        print(f"U range: {np.min(u_data):.3f} to {np.max(u_data):.3f}")
        
        # Test incidence angle calculation
        inc_angle = ut.incidence_angle_from_enu(e_data, n_data, u_data)
        if inc_angle is not None:
            print(f"✓ Incidence angle calculated: {np.min(inc_angle):.1f}° to {np.max(inc_angle):.1f}°")
        else:
            print("✗ Incidence angle calculation failed")
            return False
        
        # Test azimuth angle calculation
        az_angle = ut.azimuth_angle_from_enu(e_data, n_data, u_data)
        if az_angle is not None:
            print(f"✓ Azimuth angle calculated: {np.min(az_angle):.1f}° to {np.max(az_angle):.1f}°")
        else:
            print("✗ Azimuth angle calculation failed")
            return False
        
        # Test slant range calculation
        temp_atr = {
            'HEIGHT': '693000',  # Sentinel-1 altitude
            'EARTH_RADIUS': str(ut.EARTH_RADIUS),
        }
        slant_range = ut.incidence_angle2slant_range_distance(temp_atr, inc_angle)
        print(f"✓ Slant range calculated: {np.min(slant_range):.0f}m to {np.max(slant_range):.0f}m")
        
        return True
        
    except Exception as e:
        print(f"✗ ENU function test failed: {e}")
        return False

def test_geometry_creation_interface():
    """Test the geometry creation interface (without actual files)."""
    print("\nTesting geometry creation interface...")
    
    try:
        from mintpy.create_licsar_geometry import create_parser, cmd_line_parse
        
        # Test parser creation
        parser = create_parser()
        print("✓ Argument parser created")
        
        # Test with mock arguments
        test_args = ['--geom-dir', '/fake/path', '--output', 'test.h5']
        inps = cmd_line_parse(test_args)
        print(f"✓ Command line parsing works: geom_dir={inps.geom_dir}, output={inps.output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geometry creation interface test failed: {e}")
        return False

def test_load_data_integration():
    """Test load_data integration (check for processor support)."""
    print("\nTesting load_data integration...")
    
    try:
        from mintpy.load_data import PROCESSOR_LIST
        
        if 'licsar' in PROCESSOR_LIST:
            print("✓ LiCSAR processor is in PROCESSOR_LIST")
        else:
            print("✗ LiCSAR processor not found in PROCESSOR_LIST")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Load data integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MINTPY LICSAR FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("ENU Functions Test", test_enu_functions()))
    results.append(("Geometry Creation Interface", test_geometry_creation_interface()))
    results.append(("Load Data Integration", test_load_data_integration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - LiCSAR support is working!")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())