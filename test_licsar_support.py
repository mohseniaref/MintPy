#!/usr/bin/env python3
"""
Test script for LiCSAR data support in MintPy.
This script tests:
1. prep_licsar.py functionality
2. create_licsar_geometry.py functionality 
3. load_data.py with LiCSAR data

Usage:
    python test_licsar_support.py --data-dir /path/to/licsar/data
    
Example directory structure for LiCSAR data:
    /data/
    ├── 20190601_20190613/
    │   ├── 20190601_20190613.geo.cc.tif
    │   ├── 20190601_20190613.geo.unw.tif
    │   ├── metadata.txt
    │   └── baselines.txt
    ├── 20190601_20190625/
    │   ├── 20190601_20190625.geo.cc.tif
    │   ├── 20190601_20190625.geo.unw.tif
    │   └── metadata.txt
    └── geo/
        ├── track.geo.E.tif
        ├── track.geo.N.tif
        ├── track.geo.U.tif
        └── track.geo.hgt.tif
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import h5py
import glob
import shutil
from pathlib import Path

# Import MintPy modules
try:
    from mintpy.utils import readfile, utils0 as ut
    from mintpy.objects.stackDict import geometryDict
    from mintpy import load_data
except ImportError:
    raise ImportError('Please make sure MintPy is in your PYTHONPATH')


def create_parser():
    parser = argparse.ArgumentParser(description='Test LiCSAR support in MintPy')
    parser.add_argument('--data-dir', '-d', dest='data_dir', type=str, required=True,
                      help='Directory containing LiCSAR data for testing')
    parser.add_argument('--work-dir', '-w', dest='work_dir', type=str, default='./licsar_test',
                      help='Working directory for test outputs')
    parser.add_argument('--mintpy-dir', '-m', dest='mintpy_dir', type=str, 
                      help='MintPy repository directory. If not provided, will use the current PYTHONPATH')
    
    return parser


def run_command(cmd, print_msg=True):
    """Run command line and return exit code."""
    if print_msg:
        print(f'Running command: {cmd}')
    exit_code = subprocess.call(cmd, shell=True)
    if print_msg:
        if exit_code == 0:
            print('Command completed successfully')
        else:
            print(f'Command failed with exit code {exit_code}')
    return exit_code


def test_prep_licsar(data_dir, work_dir):
    """Test prep_licsar.py functionality."""
    print('\n' + '='*50)
    print('TESTING: prep_licsar.py')
    print('='*50)
    
    # Find test files
    unw_files = glob.glob(os.path.join(data_dir, '**/*unw*.tif'), recursive=True)
    coherence_files = glob.glob(os.path.join(data_dir, '**/*cc*.tif'), recursive=True)
    dem_files = glob.glob(os.path.join(data_dir, '**/*geo.hgt*.tif'), recursive=True)
    e_files = glob.glob(os.path.join(data_dir, '**/*geo.E*.tif'), recursive=True)
    n_files = glob.glob(os.path.join(data_dir, '**/*geo.N*.tif'), recursive=True)
    u_files = glob.glob(os.path.join(data_dir, '**/*geo.U*.tif'), recursive=True)
    
    # Check if we have required files
    print(f'Found {len(unw_files)} unwrapped interferogram files')
    print(f'Found {len(coherence_files)} coherence files')
    print(f'Found {len(dem_files)} DEM files')
    print(f'Found {len(e_files)} E component files')
    print(f'Found {len(n_files)} N component files')
    print(f'Found {len(u_files)} U component files')
    
    if not all([unw_files, coherence_files, dem_files, e_files, n_files, u_files]):
        print('WARNING: Not all required file types were found for a complete test')
    
    # Test prep_licsar with each file type
    test_files = []
    if unw_files:
        test_files.append(unw_files[0])
    if coherence_files:
        test_files.append(coherence_files[0])
    if dem_files:
        test_files.append(dem_files[0])
    if e_files:
        test_files.append(e_files[0])
    if n_files:
        test_files.append(n_files[0])
    if u_files:
        test_files.append(u_files[0])
    
    success = True
    for test_file in test_files:
        cmd = f'prep_licsar.py "{test_file}"'
        exit_code = run_command(cmd)
        if exit_code != 0:
            success = False
        
        # Check if .rsc file was created
        rsc_file = test_file + '.rsc'
        if os.path.isfile(rsc_file):
            print(f'✓ Created {rsc_file}')
            
            # Read metadata to verify
            meta = readfile.read_roipac_rsc(rsc_file)
            print(f'  File dimensions: {meta["WIDTH"]}x{meta["LENGTH"]}')
            if 'X_FIRST' in meta:
                print(f'  Geocoded with bounds: lon={meta["X_FIRST"]}, lat={meta["Y_FIRST"]}')
        else:
            print(f'✗ Failed to create {rsc_file}')
            success = False
    
    return success


def test_create_licsar_geometry(data_dir, work_dir):
    """Test create_licsar_geometry.py functionality."""
    print('\n' + '='*50)
    print('TESTING: create_licsar_geometry.py')
    print('='*50)
    
    # Find a directory with geometry files
    geom_dirs = []
    
    # Look for directories containing E,N,U files
    e_files = glob.glob(os.path.join(data_dir, '**/*geo.E*.tif'), recursive=True)
    for e_file in e_files:
        geom_dir = os.path.dirname(e_file)
        # Check if it has N and U files too
        n_file = glob.glob(os.path.join(geom_dir, '*geo.N*.tif'))
        u_file = glob.glob(os.path.join(geom_dir, '*geo.U*.tif'))
        if n_file and u_file:
            geom_dirs.append(geom_dir)
    
    if not geom_dirs:
        print('ERROR: No suitable geometry directory found with E,N,U files')
        return False
    
    # Use the first directory with complete E,N,U files
    geom_dir = geom_dirs[0]
    print(f'Using geometry directory: {geom_dir}')
    
    # Create output directory in work_dir
    geom_out_dir = os.path.join(work_dir, 'geometry')
    os.makedirs(geom_out_dir, exist_ok=True)
    
    # Run create_licsar_geometry.py
    geom_file = os.path.join(geom_out_dir, 'geometryGeo.h5')
    cmd = f'create_licsar_geometry.py --geom-dir "{geom_dir}" --output "{geom_file}"'
    exit_code = run_command(cmd)
    
    if exit_code != 0:
        print('ERROR: create_licsar_geometry.py failed')
        return False
    
    # Check if geometry file was created
    if not os.path.isfile(geom_file):
        print(f'ERROR: Geometry file was not created at {geom_file}')
        return False
    
    print(f'✓ Created geometry file: {geom_file}')
    
    # Verify geometry file contents
    required_datasets = ['height', 'incidenceAngle', 'azimuthAngle', 
                         'slantRangeDistance', 'latitude', 'longitude', 'shadowMask']
    
    try:
        with h5py.File(geom_file, 'r') as f:
            datasets = list(f.keys())
            print(f'Geometry file contains {len(datasets)} datasets: {", ".join(datasets)}')
            
            missing = [ds for ds in required_datasets if ds not in datasets]
            if missing:
                print(f'WARNING: Missing required datasets: {", ".join(missing)}')
                return False
            
            # Check dataset shapes
            shape_info = []
            for ds in datasets:
                if ds in f:
                    shape = f[ds].shape
                    shape_info.append(f'{ds}: {shape}')
            
            print('Dataset shapes:')
            for info in shape_info:
                print(f'  {info}')
    
    except Exception as e:
        print(f'ERROR reading geometry file: {str(e)}')
        return False
    
    return True


def test_load_data(data_dir, work_dir):
    """Test load_data.py with LiCSAR data."""
    print('\n' + '='*50)
    print('TESTING: load_data.py with LiCSAR data')
    print('='*50)
    
    # Create template file
    template_file = os.path.join(work_dir, 'smallbaselineApp_licsar.txt')
    
    # Get directories containing unwrapped interferograms
    unw_files = glob.glob(os.path.join(data_dir, '**/*unw*.tif'), recursive=True)
    if not unw_files:
        print('ERROR: No unwrapped interferogram files found')
        return False
    
    # Get interferogram directory
    ifg_dir = os.path.dirname(unw_files[0])
    print(f'Using interferogram directory: {ifg_dir}')
    
    # Find geometry files directory
    geom_dir = None
    e_files = glob.glob(os.path.join(data_dir, '**/*geo.E*.tif'), recursive=True)
    if e_files:
        geom_dir = os.path.dirname(e_files[0])
        print(f'Using geometry directory: {geom_dir}')
    else:
        print('WARNING: No E,N,U files found for geometry')
    
    # Create template file
    with open(template_file, 'w') as f:
        f.write('# MintPy template file for LiCSAR data\n')
        f.write('mintpy.load.processor      = licsar\n')
        f.write(f'mintpy.load.unwFile       = {os.path.join(ifg_dir, "*.geo.unw*.tif")}\n')
        f.write(f'mintpy.load.corFile       = {os.path.join(ifg_dir, "*.geo.cc*.tif")}\n')
        
        if geom_dir:
            f.write(f'mintpy.load.demFile        = {os.path.join(geom_dir, "*.geo.hgt*.tif")}\n')
            f.write(f'mintpy.load.bEastFile      = {os.path.join(geom_dir, "*.geo.E*.tif")}\n')
            f.write(f'mintpy.load.bNorthFile     = {os.path.join(geom_dir, "*.geo.N*.tif")}\n')
            f.write(f'mintpy.load.bUpFile        = {os.path.join(geom_dir, "*.geo.U*.tif")}\n')
        
        f.write('mintpy.subset.lalo        = None\n')
        f.write('mintpy.reference.lalo     = auto\n')
        f.write('mintpy.troposphericDelay.method = no\n')
        f.write('mintpy.network.coherenceBased   = yes\n')
        f.write('mintpy.topographicResidual.pixelwiseGeometry = yes\n')
        f.write('mintpy.geocode            = yes\n')
    
    print(f'Created template file: {template_file}')
    
    # Run load_data.py
    cmd = f'load_data.py {template_file}'
    exit_code = run_command(cmd)
    
    if exit_code != 0:
        print('ERROR: load_data.py failed')
        return False
    
    # Check output files
    ifgram_stack_file = os.path.join(work_dir, 'inputs/ifgramStack.h5')
    geom_file = os.path.join(work_dir, 'inputs/geometryGeo.h5')
    
    if not os.path.isfile(ifgram_stack_file):
        print(f'ERROR: ifgramStack.h5 not created at {ifgram_stack_file}')
        return False
    
    if not os.path.isfile(geom_file):
        print(f'ERROR: geometryGeo.h5 not created at {geom_file}')
        return False
    
    print(f'✓ Created ifgramStack.h5: {ifgram_stack_file}')
    print(f'✓ Created geometryGeo.h5: {geom_file}')
    
    # Verify ifgram stack contents
    try:
        with h5py.File(ifgram_stack_file, 'r') as f:
            datasets = list(f.keys())
            print(f'ifgramStack contains datasets: {", ".join(datasets)}')
            
            if 'unwrapPhase' not in datasets or 'coherence' not in datasets:
                print('WARNING: Missing required datasets in ifgramStack')
                return False
            
            num_ifgrams = f['unwrapPhase'].shape[0]
            print(f'Found {num_ifgrams} interferograms')
            
            # Print date12 list
            if 'date12' in f.keys():
                dates = [date.decode('utf-8') for date in f['date12'][:]]
                print(f'Date pairs: {dates[:5]}{"..." if len(dates) > 5 else ""}')
    
    except Exception as e:
        print(f'ERROR reading ifgramStack file: {str(e)}')
        return False
    
    return True


def main():
    """Main function to run all tests."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create work directory
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    
    # Set MintPy path if provided
    if args.mintpy_dir:
        mintpy_dir = os.path.abspath(args.mintpy_dir)
        sys.path.insert(0, mintpy_dir)
        os.environ['PYTHONPATH'] = f"{mintpy_dir}:{os.environ.get('PYTHONPATH', '')}"
        print(f'Using MintPy from: {mintpy_dir}')
    
    # Print test environment info
    print('\nTEST ENVIRONMENT:')
    print(f'Data directory: {args.data_dir}')
    print(f'Work directory: {work_dir}')
    print(f'Python executable: {sys.executable}')
    print(f'PYTHONPATH: {os.environ.get("PYTHONPATH", "Not set")}')
    
    # Run tests
    print('\nRunning LiCSAR support tests...\n')
    
    # Test prep_licsar.py
    prep_success = test_prep_licsar(args.data_dir, work_dir)
    
    # Test create_licsar_geometry.py
    geom_success = test_create_licsar_geometry(args.data_dir, work_dir)
    
    # Test load_data.py
    load_success = test_load_data(args.data_dir, work_dir)
    
    # Print summary
    print('\n' + '='*50)
    print('TEST SUMMARY:')
    print('='*50)
    print(f'prep_licsar.py:           {"✓ PASSED" if prep_success else "✗ FAILED"}')
    print(f'create_licsar_geometry.py: {"✓ PASSED" if geom_success else "✗ FAILED"}')
    print(f'load_data.py with LiCSAR: {"✓ PASSED" if load_success else "✗ FAILED"}')
    
    overall_success = prep_success and geom_success and load_success
    print(f'\nOverall test result: {"✓ PASSED" if overall_success else "✗ FAILED"}')
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())