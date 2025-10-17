#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Your Name, Oct 2025                              #
############################################################


import os
import sys
import time
import h5py
import numpy as np
import warnings

from mintpy.constants import SPEED_OF_LIGHT
from mintpy.objects import sensor
from mintpy.objects.stackDict import geometryDict
from mintpy.utils import readfile, utils0 as ut, utils1, writefile, arg_utils


###############################################################
def create_parser():
    """Create command line parser."""
    parser = arg_utils.create_argument_parser(
        name='create_licsar_geometry',
        synopsis='Create a geometry HDF5 file from LiCSAR E,N,U component files',
    )

    # input
    parser.add_argument('--geom-dir', '-g', dest='geom_dir', type=str, required=True,
                        help='Directory containing the LiCSAR geometry files (*.geo.{hgt,E,N,U}.tif)')
    parser.add_argument('--output', '-o', dest='output_file', type=str, default='geometryGeo.h5',
                        help='Output file name (default: %(default)s)')
    parser.add_argument('--update', dest='update_mode', action='store_true',
                        help='Enable update mode, and skip datasets existed in the output file.')

    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


#########################################################################
def create_licsar_geometry_file(geom_dir, out_file='geometryGeo.h5', update_mode=False, print_msg=True):
    """Create a comprehensive geometry file from LiCSAR products using MintPy's built-in functions.
    
    Parameters:
        geom_dir    - str, directory containing LiCSAR geometry files (.geo.E.tif, .geo.N.tif, etc.)
        out_file    - str, output geometry file name
        update_mode - bool, if True, skip writing existing datasets in the output file
        print_msg   - bool, print processing messages if True
    Returns:
        out_file    - str, absolute path of output geometry file
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    
    vprint(f'Creating geometry file from LiCSAR data in: {geom_dir}')
    
    # Find LiCSAR geometry files
    pattern = os.path.join(geom_dir, '*.geo.*.tif')
    geo_files = utils1.get_file_list(pattern, abspath=True)
    
    if not geo_files:
        raise FileNotFoundError(f'No LiCSAR geometry files found in {geom_dir}')
    
    # Categorize files
    height_file = None
    e_file = None
    n_file = None
    u_file = None
    
    for geo_file in geo_files:
        basename = os.path.basename(geo_file).lower()
        if '.geo.hgt.' in basename or '.geo.dem.' in basename:
            height_file = geo_file
        elif '.geo.e.' in basename:
            e_file = geo_file
        elif '.geo.n.' in basename:
            n_file = geo_file
        elif '.geo.u.' in basename:
            u_file = geo_file
    
    # Check required files
    if not height_file:
        vprint("WARNING: No height/DEM file found. Elevation-dependent incidence angle won't be available.")
    
    # Read metadata
    if height_file:
        atr_dict = readfile.read_attribute(height_file)
    elif e_file:
        atr_dict = readfile.read_attribute(e_file)
    else:
        raise FileNotFoundError('At least one of height or E component file is required')
    
    # Initialize dataset dictionary for geometryDict
    ds_dict = {}
    
    # Add DEM/height
    if height_file:
        ds_dict['height'] = height_file
        vprint(f'height: {height_file}')
    
    # Read E, N, U components for LOS calculation
    e_data = None
    n_data = None
    u_data = None
    
    if all([e_file, n_file, u_file]):
        vprint('Reading E, N, U components for pixel-wise geometry calculations...')
        try:
            # Read using GDAL directly to avoid XML parsing issues
            from osgeo import gdal
            ds_e = gdal.Open(e_file, gdal.GA_ReadOnly)
            ds_n = gdal.Open(n_file, gdal.GA_ReadOnly)
            ds_u = gdal.Open(u_file, gdal.GA_ReadOnly)
            
            e_data = ds_e.GetRasterBand(1).ReadAsArray()
            n_data = ds_n.GetRasterBand(1).ReadAsArray()
            u_data = ds_u.GetRasterBand(1).ReadAsArray()
            
            ds_e = None
            ds_n = None
            ds_u = None
            
        except Exception as e:
            vprint(f'Error reading E,N,U components: {str(e)}')
            vprint('Trying alternative readfile method...')
            e_data, e_meta = readfile.read(e_file)
            n_data, n_meta = readfile.read(n_file)
            u_data, u_meta = readfile.read(u_file)
        
        # Save basis vector components as well (useful for advanced analysis)
        ds_dict['basisEast'] = e_file
        ds_dict['basisNorth'] = n_file
        ds_dict['basisUp'] = u_file
        vprint(f'E component: {e_file}')
        vprint(f'N component: {n_file}')
        vprint(f'U component: {u_file}')
    else:
        vprint('WARNING: E, N, U component files not all available')
        vprint('Calculation of pixel-wise incidence/azimuth angles and slant range will be limited')
        if any([e_file, n_file, u_file]):
            missing = []
            if not e_file:
                missing.append('E')
            if not n_file:
                missing.append('N')
            if not u_file:
                missing.append('U')
            vprint(f'Missing components: {", ".join(missing)}')
    
    # Check if output file already exists in update mode
    if update_mode and os.path.isfile(out_file):
        vprint(f'Update mode is enabled, skip datasets already in {out_file}')
        ds_dict_out = {}
        for key in ds_dict.keys():
            if key not in readfile.get_dataset_list(out_file):
                ds_dict_out[key] = ds_dict[key]
        ds_dict = ds_dict_out
    
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        vprint(f'Created directory: {out_dir}')
    
    # Create geometryDict object
    geom_obj = geometryDict(
        processor='licsar',
        datasetDict=ds_dict,
        extraMetadata=atr_dict,
    )
    
    # Write to HDF5 file (this automatically sets the output file to absolute path)
    out_file = geom_obj.write2hdf5(outputFile=out_file, access_mode='a', compression=None)
    
    # If E, N, U components are available, calculate pixel-wise incidence/azimuth angles
    if all([e_data is not None, n_data is not None, u_data is not None]):
        vprint('Calculating pixel-wise geometry from E, N, U components...')
        
        # Calculate incidence angle using MintPy's utility function
        inc_angle = ut.incidence_angle_from_enu(e_data, n_data, u_data)
        az_angle = ut.azimuth_angle_from_enu(e_data, n_data, u_data)
        
        if inc_angle is not None and az_angle is not None:
            vprint(f'Incidence angle range: {np.nanmin(inc_angle):.2f}° - {np.nanmax(inc_angle):.2f}°')
            vprint(f'Azimuth angle range: {np.nanmin(az_angle):.2f}° - {np.nanmax(az_angle):.2f}°')
            
            # Calculate slant range using MintPy function
            temp_atr = atr_dict.copy()
            temp_atr.update({
                'HEIGHT': '693000',  # Sentinel-1 approximate altitude in meters
                'EARTH_RADIUS': str(ut.EARTH_RADIUS),  # Use MintPy's standard Earth radius
            })
            slant_range = ut.incidence_angle2slant_range_distance(temp_atr, inc_angle)
            vprint(f'Slant range: {np.nanmin(slant_range):.0f}m - {np.nanmax(slant_range):.0f}m')
            
            # Calculate lat/lon if not already in geometry file
            length, width = inc_angle.shape
            try:
                lat, lon = ut.get_lat_lon(atr_dict, dimension=2)
            except Exception as e:
                vprint(f'WARNING: Could not extract lat/lon from metadata: {str(e)}')
                lat, lon = None, None
            
            # Create shadow mask (simplified - areas with no valid incidence angle)
            shadow_mask = np.zeros((length, width), dtype=np.bool_)
            if inc_angle is not None:
                shadow_mask = np.isnan(inc_angle)
            
            # Add calculated datasets to HDF5 file
            with h5py.File(out_file, 'a') as f:
                # incidence angle (required)
                if 'incidenceAngle' not in f.keys() or not update_mode:
                    vprint('Adding incidenceAngle dataset')
                    writefile.write_hdf5_block(
                        out_file,
                        data=inc_angle,
                        datasetName='incidenceAngle',
                        block=[0, 0, length, width],
                    )
                
                # azimuth angle (required)
                if 'azimuthAngle' not in f.keys() or not update_mode:
                    vprint('Adding azimuthAngle dataset')
                    writefile.write_hdf5_block(
                        out_file,
                        data=az_angle,
                        datasetName='azimuthAngle',
                        block=[0, 0, length, width],
                    )
                
                # slant range distance (required)
                if 'slantRangeDistance' not in f.keys() or not update_mode:
                    vprint('Adding slantRangeDistance dataset')
                    writefile.write_hdf5_block(
                        out_file,
                        data=slant_range,
                        datasetName='slantRangeDistance',
                        block=[0, 0, length, width],
                    )
                
                # shadow mask (required)
                if 'shadowMask' not in f.keys() or not update_mode:
                    vprint('Adding shadowMask dataset')
                    writefile.write_hdf5_block(
                        out_file,
                        data=shadow_mask,
                        datasetName='shadowMask',
                        block=[0, 0, length, width],
                    )
                
                # latitude and longitude (required for geo files)
                if lat is not None and lon is not None:
                    if 'latitude' not in f.keys() or not update_mode:
                        vprint('Adding latitude dataset')
                        writefile.write_hdf5_block(
                            out_file,
                            data=lat,
                            datasetName='latitude',
                            block=[0, 0, length, width],
                        )
                    
                    if 'longitude' not in f.keys() or not update_mode:
                        vprint('Adding longitude dataset')
                        writefile.write_hdf5_block(
                            out_file,
                            data=lon,
                            datasetName='longitude',
                            block=[0, 0, length, width],
                        )
    
    # Final check of required datasets
    with h5py.File(out_file, 'r') as f:
        available_datasets = list(f.keys())
    
    required_datasets = ['height', 'incidenceAngle', 'azimuthAngle', 
                         'slantRangeDistance', 'latitude', 'longitude', 'shadowMask']
    
    missing_datasets = [ds for ds in required_datasets if ds not in available_datasets]
    if missing_datasets:
        vprint(f"WARNING: The following required datasets are missing: {', '.join(missing_datasets)}")
    else:
        vprint(f"SUCCESS: All required datasets are present in {out_file}")
    
    vprint(f'Geometry file created: {out_file}')
    return out_file


#########################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    
    # Create geometry file
    create_licsar_geometry_file(
        geom_dir=inps.geom_dir,
        out_file=inps.output_file,
        update_mode=inps.update_mode,
    )
    
    return


#########################################################################
if __name__ == '__main__':
    main(sys.argv[1:])