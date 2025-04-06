############################################################
# Program is part of MintPy                                #
# Adapted for LiCSAR products                              #
# Author: Your Name, Apr 2025                              #
############################################################

import datetime as dt
import os

from mintpy.constants import SPEED_OF_LIGHT
from mintpy.objects import sensor
from mintpy.utils import readfile, utils1 as ut, writefile


#########################################################################
def add_licsar_metadata(fname, meta, is_ifg=True):
    '''Read/extract metadata from LiCSAR metadata and baseline files and add to metadata dictionary.

    LiCSAR metadata is provided in two files:
    1. metadata.txt: Contains general metadata about the product.
    2. baselines.txt: Contains baseline information for interferograms.

    Parameters:
        fname  - str, path to the LiCSAR data file, e.g. *unw_phase_clip*.tif, *dem_clip*.tif
        meta   - dict, existing metadata
        is_ifg - bool, is the data file interferogram (unw/corr) or geometry (dem/angles)
    Returns:
        meta   - dict, updated metadata
    '''

    # Read metadata.txt
    meta_file = os.path.join(os.path.dirname(fname), 'metadata.txt')
    licsar_meta = {}
    with open(meta_file) as f:
        for line in f:
            key, value = line.strip().split('=')
            licsar_meta[key.strip()] = value.strip()

    # Add universal metadata
    meta['PROCESSOR'] = 'LiCSAR'
    meta['MASTER_DATE'] = licsar_meta.get('master', 'Unknown')
    meta['CENTER_TIME'] = licsar_meta.get('center_time', 'Unknown')
    meta['HEADING'] = float(licsar_meta.get('heading', '0')) % 360. - 360.  # ensure negative value
    meta['AVG_INCIDENCE_ANGLE'] = licsar_meta.get('avg_incidence_angle', 'Unknown')
    meta['AZIMUTH_RESOLUTION'] = licsar_meta.get('azimuth_resolution', 'Unknown')
    meta['RANGE_RESOLUTION'] = licsar_meta.get('range_resolution', 'Unknown')
    meta['CENTRE_RANGE'] = licsar_meta.get('centre_range_m', 'Unknown')
    meta['AVG_HEIGHT'] = licsar_meta.get('avg_height', 'Unknown')
    meta['APPLIED_DEM'] = licsar_meta.get('applied_DEM', 'Unknown')

    # Add orbit direction and corner coordinates
    meta['ORBIT_DIRECTION'] = 'ASCENDING' if abs(meta['HEADING']) < 90 else 'DESCENDING'
    N = float(meta['Y_FIRST'])
    W = float(meta['X_FIRST'])
    S = N + float(meta['Y_STEP']) * int(meta['LENGTH'])
    E = W + float(meta['X_STEP']) * int(meta['WIDTH'])

    if meta['ORBIT_DIRECTION'] == 'ASCENDING':
        meta['LAT_REF1'] = str(S)
        meta['LAT_REF2'] = str(S)
        meta['LAT_REF3'] = str(N)
        meta['LAT_REF4'] = str(N)
        meta['LON_REF1'] = str(W)
        meta['LON_REF2'] = str(E)
        meta['LON_REF3'] = str(W)
        meta['LON_REF4'] = str(E)
    else:
        meta['LAT_REF1'] = str(N)
        meta['LAT_REF2'] = str(N)
        meta['LAT_REF3'] = str(S)
        meta['LAT_REF4'] = str(S)
        meta['LON_REF1'] = str(E)
        meta['LON_REF2'] = str(W)
        meta['LON_REF3'] = str(E)
        meta['LON_REF4'] = str(W)

    # Hard-coded metadata for Sentinel-1
    meta['PLATFORM'] = 'Sentinel-1'
    meta['ANTENNA_SIDE'] = -1
    meta['WAVELENGTH'] = SPEED_OF_LIGHT / sensor.SEN['carrier_frequency']
    meta['RANGE_PIXEL_SIZE'] = sensor.SEN['range_pixel_size']
    meta['AZIMUTH_PIXEL_SIZE'] = sensor.SEN['azimuth_pixel_size']

    # Read baselines.txt for interferogram-specific metadata
    if is_ifg:
        baseline_file = os.path.join(os.path.dirname(fname), 'baselines.txt')
        master_date = licsar_meta.get('master', 'Unknown')
        slave_date = os.path.basename(fname).split('_')[1]  # Extract slave date from filename
        with open(baseline_file) as f:
            for line in f:
                fields = line.strip().split()
                if fields[0] == master_date and fields[1] == slave_date:
                    meta['P_BASELINE_TOP_HDR'] = fields[2]
                    meta['P_BASELINE_BOTTOM_HDR'] = fields[3]
                    break

        meta['DATE12'] = f"{master_date[2:]}-{slave_date[2:]}"

    return meta


#########################################################################
def prep_licsar(inps):
    """Prepare LiCSAR metadata files"""

    inps.file = ut.get_file_list(inps.file, abspath=True)

    # For each filename, generate metadata rsc file
    for fname in inps.file:
        is_ifg = any([x in fname for x in ['unw_phase', 'corr']])
        meta = readfile.read_gdal_vrt(fname)
        meta = add_licsar_metadata(fname, meta, is_ifg=is_ifg)

        # Write
        rsc_file = fname + '.rsc'
        writefile.write_roipac_rsc(meta, out_file=rsc_file)

    return
