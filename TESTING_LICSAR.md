# Testing LiCSAR Support in MintPy

This document provides instructions for testing the LiCSAR product support in MintPy.

## Setup and Requirements

1. Make sure MintPy is installed and in your PYTHONPATH
2. You need LiCSAR test data with the following structure:
   - Interferogram files (*.geo.unw.tif, *.geo.cc.tif)
   - Metadata files (metadata.txt, baselines.txt)
   - Geometry files (*.geo.E.tif, *.geo.N.tif, *.geo.U.tif, *.geo.hgt.tif)

## Running the Tests

### Method 1: Using the test script

```bash
# Run the test script with your LiCSAR data directory
python test_licsar_support.py --data-dir /path/to/licsar/data
```

### Method 2: Testing individual components

#### 1. Test `prep_licsar.py`

```bash
# Process metadata for interferogram files
prep_licsar.py /path/to/licsar/data/*geo.unw.tif

# Process metadata for geometry files
prep_licsar.py /path/to/licsar/data/*geo.{E,N,U,hgt}.tif
```

#### 2. Test geometry file creation

```bash
# Create enhanced geometry file with pixel-wise calculations
create_licsar_geometry.py --geom-dir /path/to/licsar/geometry/files --output geometryGeo.h5
```

#### 3. Test loading LiCSAR data

Create a template file `smallbaselineApp_licsar.txt`:

```
mintpy.load.processor      = licsar
mintpy.load.unwFile       = /path/to/licsar/data/*.geo.unw*.tif
mintpy.load.corFile       = /path/to/licsar/data/*.geo.cc*.tif
mintpy.load.demFile       = /path/to/licsar/data/*.geo.hgt*.tif
mintpy.load.bEastFile     = /path/to/licsar/data/*.geo.E*.tif
mintpy.load.bNorthFile    = /path/to/licsar/data/*.geo.N*.tif
mintpy.load.bUpFile       = /path/to/licsar/data/*.geo.U*.tif
```

Then run:

```bash
load_data.py smallbaselineApp_licsar.txt
```

## Expected Results

1. `prep_licsar.py` should create .rsc metadata files for each input file
2. `create_licsar_geometry.py` should create a complete geometryGeo.h5 file with all 7 required datasets
3. `load_data.py` should create ifgramStack.h5 and geometryGeo.h5 in the ./inputs directory

## Troubleshooting

1. **Missing components**: Make sure your LiCSAR data includes E, N, U component files 
2. **Missing .rsc files**: Verify that `prep_licsar.py` ran successfully
3. **Missing datasets**: Check if the geometry file contains all required datasets