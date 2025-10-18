#!/usr/bin/env python3
"""
Simple LiCSAR geometry file creator that directly uses GDAL and writes to HDF5.
This avoids the complexity of the geometryDict object for initial testing.
"""

import os
import sys
import h5py
import numpy as np
from osgeo import gdal

def create_simple_licsar_geometry(geom_dir, out_file):
    """Create a simple LiCSAR geometry file with all required datasets."""
    
    print(f'Creating LiCSAR geometry file from: {geom_dir}')
    
    # Find files
    import glob
    hgt_files = glob.glob(os.path.join(geom_dir, '*.geo.hgt.tif'))
    e_files = glob.glob(os.path.join(geom_dir, '*.geo.E.tif'))
    n_files = glob.glob(os.path.join(geom_dir, '*.geo.N.tif'))
    u_files = glob.glob(os.path.join(geom_dir, '*.geo.U.tif'))
    
    if not all([hgt_files, e_files, n_files, u_files]):
        print('ERROR: Missing required geometry files')
        return None
    
    hgt_file = hgt_files[0]
    e_file = e_files[0]
    n_file = n_files[0]
    u_file = u_files[0]
    
    print(f'Height file: {hgt_file}')
    print(f'E file: {e_file}')
    print(f'N file: {n_file}')
    print(f'U file: {u_file}')
    
    # Read data
    print('Reading geometry data...')
    ds_hgt = gdal.Open(hgt_file, gdal.GA_ReadOnly)
    ds_e = gdal.Open(e_file, gdal.GA_ReadOnly)
    ds_n = gdal.Open(n_file, gdal.GA_ReadOnly)
    ds_u = gdal.Open(u_file, gdal.GA_ReadOnly)
    
    height = ds_hgt.GetRasterBand(1).ReadAsArray()
    e_data = ds_e.GetRasterBand(1).ReadAsArray()
    n_data = ds_n.GetRasterBand(1).ReadAsArray()
    u_data = ds_u.GetRasterBand(1).ReadAsArray()
    
    # Get geotransform
    gt = ds_hgt.GetGeoTransform()
    width = ds_hgt.RasterXSize
    length = ds_hgt.RasterYSize
    
    print(f'Data dimensions: {length} x {width}')
    
    # Calculate incidence and azimuth angles
    print('Calculating incidence and azimuth angles...')
    
    # Incidence angle: θ = arccos(|U|)
    inc_angle = np.rad2deg(np.arccos(np.abs(u_data)))
    
    # Azimuth angle: α = atan2(E, N)
    az_angle = np.rad2deg(np.arctan2(e_data, n_data))
    az_angle = np.where(az_angle < 0, az_angle + 360, az_angle)
    
    print(f'Incidence angle range: {np.nanmin(inc_angle):.2f}° - {np.nanmax(inc_angle):.2f}°')
    print(f'Azimuth angle range: {np.nanmin(az_angle):.2f}° - {np.nanmax(az_angle):.2f}°')
    
    # Calculate slant range
    sat_height = 693000  # meters, Sentinel-1 approximate altitude
    slant_range = sat_height / np.cos(np.deg2rad(inc_angle))
    print(f'Slant range: {np.nanmin(slant_range):.0f}m - {np.nanmax(slant_range):.0f}m')
    
    # Calculate lat/lon
    print('Calculating latitude and longitude...')
    x = np.arange(width) * gt[1] + gt[0] + gt[1] / 2
    y = np.arange(length) * gt[5] + gt[3] + gt[5] / 2
    lon, lat = np.meshgrid(x, y)
    
    # Shadow mask (simplified)
    shadow_mask = np.zeros((length, width), dtype=np.uint8)
    
    # Write to HDF5
    print(f'Writing to HDF5 file: {out_file}')
    with h5py.File(out_file, 'w') as f:
        # Required datasets
        f.create_dataset('height', data=height, compression='gzip')
        f.create_dataset('incidenceAngle', data=inc_angle, compression='gzip')
        f.create_dataset('azimuthAngle', data=az_angle, compression='gzip')
        f.create_dataset('slantRangeDistance', data=slant_range, compression='gzip')
        f.create_dataset('latitude', data=lat, compression='gzip')
        f.create_dataset('longitude', data=lon, compression='gzip')
        f.create_dataset('shadowMask', data=shadow_mask, compression='gzip')
        
        # Additional datasets
        f.create_dataset('basisEast', data=e_data, compression='gzip')
        f.create_dataset('basisNorth', data=n_data, compression='gzip')
        f.create_dataset('basisUp', data=u_data, compression='gzip')
        
        # Add attributes
        f.attrs['FILE_TYPE'] = 'geometry'
        f.attrs['PROCESSOR'] = 'licsar'
        f.attrs['PLATFORM'] = 'Sen'
        f.attrs['WIDTH'] = width
        f.attrs['LENGTH'] = length
        f.attrs['X_FIRST'] = gt[0]
        f.attrs['Y_FIRST'] = gt[3]
        f.attrs['X_STEP'] = gt[1]
        f.attrs['Y_STEP'] = gt[5]
        f.attrs['X_UNIT'] = 'degrees'
        f.attrs['Y_UNIT'] = 'degrees'
        f.attrs['WAVELENGTH'] = 0.05546576  # Sentinel-1 C-band
    
    # Close datasets
    ds_hgt = None
    ds_e = None
    ds_n = None
    ds_u = None
    
    print(f'✓ Successfully created geometry file: {out_file}')
    print(f'  Datasets: height, incidenceAngle, azimuthAngle, slantRangeDistance,')
    print(f'            latitude, longitude, shadowMask, basisEast, basisNorth, basisUp')
    
    return out_file


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python simple_licsar_geometry.py <geom_dir> <output_file>')
        sys.exit(1)
    
    geom_dir = sys.argv[1]
    out_file = sys.argv[2]
    
    create_simple_licsar_geometry(geom_dir, out_file)
