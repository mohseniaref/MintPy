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


def _extract_dates_from_fname(fname):
    """Extract both master and slave dates from interferogram filename.
    
    LiCSAR interferogram filenames follow the pattern: YYYYMMDD_YYYYMMDD.geo.unw.tif
    where the first date is the master and the second is the slave.
    """
    name = os.path.basename(fname)
    dates = re.findall(r'\d{8}', name)
    if len(dates) >= 2:
        master = _parse_date(dates[0])
        slave = _parse_date(dates[1])
        return master, slave
    return None, None


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

    # Read metadata.txt - first try same directory, then parent directory
    meta_file = os.path.join(os.path.dirname(fname), 'metadata.txt')
    if not os.path.isfile(meta_file):
        # Try parent directory (for LiCSAR structure where metadata is in GEOC/)
        meta_file = os.path.join(os.path.dirname(os.path.dirname(fname)), 'metadata.txt')
    licsar_meta = _read_key_value_file(meta_file)

    # Add universal metadata
    meta['PROCESSOR'] = 'licsar'

    # For interferograms, extract dates from filename first (YYYYMMDD_YYYYMMDD pattern)
    # For geometry files, try to get from metadata
    fname_master, fname_slave = _extract_dates_from_fname(fname)
    
    if fname_master and fname_slave:
        # Interferogram with dates in filename
        master_date = fname_master
        slave_date = fname_slave
    else:
        # Geometry file or other product - use metadata
        master_date = _parse_date(_get_first(licsar_meta, 'master', 'masterdate'))
        slave_date = _parse_date(_get_first(licsar_meta, 'slave', 'slavedate'))

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
    
    # Calculate ALOOKS and RLOOKS from ground pixel spacing
    # LOOKS = (desired ground pixel size) / (native single-look pixel spacing)
    # For Sentinel-1:
    #   - range pixel spacing (slant) = 2.3 m
    #   - range pixel spacing (ground) = slant / sin(incidence)
    #   - azimuth pixel spacing = 13.9 m
    import math
    
    if 'RLOOKS' not in meta:
        if 'X_STEP' in meta:
            x_step_deg = abs(_safe_float(meta['X_STEP']))
            if x_step_deg:
                # Convert X_STEP from degrees to meters
                lat = _safe_float(meta.get('Y_FIRST', 0))
                x_step_m = x_step_deg * 111320 * math.cos(math.radians(lat)) if lat else x_step_deg * 111320
                
                # Get ground range pixel spacing
                # slant range spacing = 2.3 m, convert to ground range using incidence angle
                inc_angle = _safe_float(meta.get('AVG_INCIDENCE_ANGLE', 37.0))  # default 37° if not available
                range_spacing_ground = sensor.SEN['range_pixel_size'] / math.sin(math.radians(inc_angle))
                
                # Calculate looks
                meta['RLOOKS'] = str(int(round(x_step_m / range_spacing_ground)))
            else:
                meta['RLOOKS'] = '1'
        else:
            meta['RLOOKS'] = '1'
    
    if 'ALOOKS' not in meta:
        if 'Y_STEP' in meta:
            y_step_deg = abs(_safe_float(meta['Y_STEP']))
            if y_step_deg:
                # Convert Y_STEP from degrees to meters (constant 111320 m/degree)
                y_step_m = y_step_deg * 111320
                
                # Azimuth pixel spacing from sensor definition
                az_spacing = sensor.SEN['azimuth_pixel_size']
                
                # Calculate looks
                meta['ALOOKS'] = str(int(round(y_step_m / az_spacing)))
            else:
                meta['ALOOKS'] = '1'
        else:
            meta['ALOOKS'] = '1'
    
    # Ensure pixel sizes are set (use values from .rsc if available, otherwise calculate from looks)
    if 'RANGE_PIXEL_SIZE' not in meta:
        if 'RLOOKS' in meta:
            looks = _safe_float(meta['RLOOKS'])
            meta['RANGE_PIXEL_SIZE'] = sensor.SEN['range_pixel_size'] * looks if looks else sensor.SEN['range_pixel_size']
        else:
            meta['RANGE_PIXEL_SIZE'] = sensor.SEN['range_pixel_size']
    
    if 'AZIMUTH_PIXEL_SIZE' not in meta:
        if 'ALOOKS' in meta:
            looks = _safe_float(meta['ALOOKS'])
            meta['AZIMUTH_PIXEL_SIZE'] = sensor.SEN['azimuth_pixel_size'] * looks if looks else sensor.SEN['azimuth_pixel_size']
        else:
            meta['AZIMUTH_PIXEL_SIZE'] = sensor.SEN['azimuth_pixel_size']

    # Read baselines file for interferogram-specific metadata
    # LiCSAR baselines file format: frame_master_date acquisition_date perp_baseline temporal_baseline
    # For an interferogram between dates A and B, we need baselines for both dates relative to frame master
    if is_ifg:
        baseline_file = os.path.join(os.path.dirname(fname), 'baselines.txt')
        if not os.path.isfile(baseline_file):
            # Try parent directory (for LiCSAR structure where baselines is in GEOC/)
            baseline_file = os.path.join(os.path.dirname(os.path.dirname(fname)), 'baselines.txt')
        if not os.path.isfile(baseline_file):
            baseline_file = os.path.join(os.path.dirname(fname), 'baselines')
        if not os.path.isfile(baseline_file):
            baseline_file = os.path.join(os.path.dirname(os.path.dirname(fname)), 'baselines')

        if os.path.isfile(baseline_file):
            # Read all baselines into a dictionary: {acquisition_date: (perp_baseline, temporal_baseline)}
            baselines_dict = {}
            frame_master = None
            with open(baseline_file) as f:
                for line in f:
                    text = line.strip()
                    if not text or text.startswith('#'):
                        continue

                    fields = text.split()
                    if len(fields) < 4:
                        continue

                    if frame_master is None:
                        frame_master = _parse_date(fields[0])
                    
                    acq_date = _parse_date(fields[1])
                    if acq_date:
                        try:
                            perp_baseline = float(fields[2])
                            temp_baseline = float(fields[3])
                            baselines_dict[acq_date] = (perp_baseline, temp_baseline)
                        except ValueError:
                            continue

            # For interferogram, compute baseline between master and slave dates
            if master_date and slave_date and master_date in baselines_dict and slave_date in baselines_dict:
                perp_master, _ = baselines_dict[master_date]
                perp_slave, _ = baselines_dict[slave_date]
                
                # Perpendicular baseline is the difference
                perp_baseline = perp_slave - perp_master
                meta['P_BASELINE_TOP_HDR'] = str(perp_baseline)
                meta['P_BASELINE_BOTTOM_HDR'] = str(perp_baseline)
                
                # Temporal baseline in days
                if master_date and slave_date:
                    temp_baseline = (slave_date - master_date).days
                    meta['T_BASELINE'] = str(temp_baseline)
            else:
                # Baseline not found for this pair
                if master_date and slave_date:
                    warnings.warn(
                        f'Baseline info for {master_date.strftime("%Y%m%d")}_{slave_date.strftime("%Y%m%d")} '
                        f'not found in {baseline_file}.',
                        category=UserWarning,
                    )
        else:
            warnings.warn(
                f'LiCSAR baseline file not found: {baseline_file}',
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
        # Detect if file is interferogram or coherence
        # LiCSAR: .geo.unw.tif or .geo.cc.tif
        # ISCE: unw_phase, corr
        is_ifg = any([x in fname for x in ['unw_phase', 'corr', '.unw.', '.cc.']])
        meta = readfile.read_gdal_vrt(fname)
        meta = add_licsar_metadata(fname, meta, is_ifg=is_ifg)
        
        # LiCSAR coherence files store values as uint8 (0-255) for space efficiency
        # Add metadata to indicate this needs normalization to 0-1 range
        if any([x in fname for x in ['.cc.', '.cor', 'coherence', 'corr']]):
            meta['DATA_TYPE'] = 'coherence'
            # Check if data type is uint8 (byte) which indicates 0-255 scaling
            try:
                from osgeo import gdal
                ds = gdal.Open(fname, gdal.GA_ReadOnly)
                if ds:
                    bnd = ds.GetRasterBand(1)
                    gdal_dtype = bnd.DataType
                    # gdal.GDT_Byte = 1 (uint8, 0-255)
                    if gdal_dtype == 1:  # GDT_Byte
                        meta['COHERENCE_SCALE_FACTOR'] = '255.0'
                        print(f'  detected uint8 coherence file (0-255 range), will normalize to 0-1 during loading')
                    ds = None
            except:
                pass

        # Write
        rsc_file = fname + '.rsc'
        writefile.write_roipac_rsc(meta, out_file=rsc_file)

    return
