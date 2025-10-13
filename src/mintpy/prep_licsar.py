############################################################
# Program is part of MintPy                                #
# Adapted for LiCSAR products                              #
# Author: Your Name, Apr 2025                              #
############################################################

import datetime as dt
import os
import re
import warnings

from mintpy.constants import SPEED_OF_LIGHT
from mintpy.objects import sensor
from mintpy.utils import readfile, utils1 as ut, writefile


def _safe_float(value):
    """Convert *value* to float when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(date_str):
    """Return ``datetime.date`` from a LiCSAR date string."""

    if not date_str:
        return None

    digits = ''.join(ch for ch in str(date_str) if ch.isdigit())
    for length, fmt in ((8, '%Y%m%d'), (6, '%y%m%d')):
        if len(digits) >= length:
            try:
                return dt.datetime.strptime(digits[:length], fmt).date()
            except ValueError:
                continue
    return None


def _extract_slave_date_from_fname(fname, master_date):
    """Infer the slave date from the file name when it is not present in metadata."""

    name = os.path.basename(fname)
    for match in re.findall(r'\d{6,8}', name):
        candidate = _parse_date(match)
        if candidate and candidate != master_date:
            return candidate
    return None


def _read_key_value_file(meta_file, separators=('=', ':')):
    """Return a dictionary with lower-case keys from ``meta_file``."""

    if not os.path.isfile(meta_file):
        raise FileNotFoundError(f'LiCSAR metadata file not found: {meta_file}')

    info = {}
    with open(meta_file) as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith('#'):
                continue

            for sep in separators:
                if sep in text:
                    key, value = text.split(sep, 1)
                    info[key.strip().lower()] = value.strip()
                    break
    return info


def _get_first(info, *keys):
    """Return the first available value from ``info`` using ``keys``."""

    for key in keys:
        if key and key.lower() in info:
            return info[key.lower()]
    return None


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
    licsar_meta = _read_key_value_file(meta_file)

    # Add universal metadata
    meta['PROCESSOR'] = 'LiCSAR'

    master_date = _parse_date(_get_first(licsar_meta, 'master', 'masterdate'))
    slave_date = _parse_date(_get_first(licsar_meta, 'slave', 'slavedate'))
    if slave_date is None:
        slave_date = _extract_slave_date_from_fname(fname, master_date)

    if master_date:
        meta['MASTER_DATE'] = master_date.strftime('%Y%m%d')
    if slave_date:
        meta['SLAVE_DATE'] = slave_date.strftime('%Y%m%d')

    center_time = _get_first(licsar_meta, 'center_time', 'centre_time')
    if center_time:
        meta['CENTER_TIME'] = center_time

    heading = _safe_float(_get_first(licsar_meta, 'heading', 'track_heading'))
    if heading is not None:
        heading = heading % 360.0
        if heading > 180.0:
            heading -= 360.0
        meta['HEADING'] = heading

    value_map = {
        'AVG_INCIDENCE_ANGLE': ['avg_incidence_angle', 'incidence_angle'],
        'AZIMUTH_RESOLUTION': ['azimuth_resolution', 'azimuth_pixel_spacing'],
        'RANGE_RESOLUTION': ['range_resolution', 'range_pixel_spacing'],
        'CENTRE_RANGE': ['centre_range_m', 'center_range', 'center_range_m'],
        'AVG_HEIGHT': ['avg_height', 'average_height'],
        'APPLIED_DEM': ['applied_dem', 'applied_dem_file', 'applied_DEM'],
        'ALOOKS': ['azimuth_looks', 'looks_azimuth'],
        'RLOOKS': ['range_looks', 'looks_range'],
    }
    for meta_key, candidates in value_map.items():
        value = _get_first(licsar_meta, *candidates)
        if value is not None:
            meta[meta_key] = value

    # Add orbit direction and corner coordinates
    orbit_direction = _get_first(licsar_meta, 'orbit_direction', 'pass')
    if orbit_direction:
        meta['ORBIT_DIRECTION'] = orbit_direction.upper()
    elif heading is not None:
        meta['ORBIT_DIRECTION'] = 'ASCENDING' if abs(heading) < 90 else 'DESCENDING'

    try:
        N = float(meta['Y_FIRST'])
        W = float(meta['X_FIRST'])
        S = N + float(meta['Y_STEP']) * int(meta['LENGTH'])
        E = W + float(meta['X_STEP']) * int(meta['WIDTH'])
    except (KeyError, TypeError, ValueError):
        N = S = E = W = None

    if meta.get('ORBIT_DIRECTION') and None not in (N, S, E, W):
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
    meta['PLATFORM'] = 'Sen'
    meta['ANTENNA_SIDE'] = -1
    meta['WAVELENGTH'] = SPEED_OF_LIGHT / sensor.SEN['carrier_frequency']
    if 'RLOOKS' in meta:
        looks = _safe_float(meta['RLOOKS'])
        meta['RANGE_PIXEL_SIZE'] = sensor.SEN['range_pixel_size'] * looks if looks else sensor.SEN['range_pixel_size']
    else:
        meta['RANGE_PIXEL_SIZE'] = sensor.SEN['range_pixel_size']
    if 'ALOOKS' in meta:
        looks = _safe_float(meta['ALOOKS'])
        meta['AZIMUTH_PIXEL_SIZE'] = sensor.SEN['azimuth_pixel_size'] * looks if looks else sensor.SEN['azimuth_pixel_size']
    else:
        meta['AZIMUTH_PIXEL_SIZE'] = sensor.SEN['azimuth_pixel_size']

    # Read baselines.txt for interferogram-specific metadata
    if is_ifg:
        baseline_file = os.path.join(os.path.dirname(fname), 'baselines.txt')
        if not os.path.isfile(baseline_file):
            raise FileNotFoundError(f'LiCSAR baseline file not found: {baseline_file}')

        pair_found = False
        with open(baseline_file) as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith('#'):
                    continue

                fields = text.split()
                if len(fields) < 4:
                    continue

                base_master = _parse_date(fields[0])
                base_slave = _parse_date(fields[1])
                if master_date and base_master and master_date != base_master:
                    continue
                if slave_date and base_slave and slave_date != base_slave:
                    continue

                meta['P_BASELINE_TOP_HDR'] = fields[2]
                meta['P_BASELINE_BOTTOM_HDR'] = fields[3]
                pair_found = True
                break

        if not pair_found:
            warnings.warn(
                f'Baseline info for {os.path.basename(fname)} was not found in {baseline_file}.',
                category=UserWarning,
            )

        if master_date and slave_date:
            meta['DATE12'] = f"{master_date.strftime('%y%m%d')}-{slave_date.strftime('%y%m%d')}"

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
