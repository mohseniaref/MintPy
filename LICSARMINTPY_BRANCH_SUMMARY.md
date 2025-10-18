# LiCSARMintPy Branch Summary

## Branch: `licsarmintpy`

This branch contains complete LiCSAR support for MintPy, including all the latest improvements for coherence normalization, network connectivity analysis, and NaN ratio filtering.

## What's Included

### 1. Core LiCSAR Support
- **`src/mintpy/prep_licsar.py`**: LiCSAR metadata preparation with coherence scale factor detection
- **`src/mintpy/create_licsar_geometry.py`**: Geometry file creation from E,N,U components
- **`src/mintpy/load_data.py`**: Enhanced to support LiCSAR processor

### 2. Coherence Normalization (NEW!)
- **`src/mintpy/objects/stackDict.py`**: Auto-detects and normalizes uint8 coherence (0-255 → 0-1)
- Handles LiCSAR coherence files stored as uint8 for space efficiency
- Adds COHERENCE_SCALE_FACTOR metadata during prep_licsar
- Automatic normalization during data loading

### 3. Network Modification Improvements (NEW!)
- **`src/mintpy/modify_network.py`**: Added NaN ratio filtering
- **`src/mintpy/cli/modify_network.py`**: Added `--max-nan-ratio` parameter
- Filters interferograms based on spatial coverage
- Integrates with Minimum Spanning Tree (MST) to preserve connectivity
- Saves `nanRatioSpatialAvg.txt` with per-interferogram statistics

### 4. Network Connectivity Analysis Tools (NEW!)
- **`src/mintpy/utils/check_network_connectivity.py`**: Comprehensive connectivity checker
- **`docs/examples/check_network_connectivity.py`**: Example usage script
- **`docs/examples/NETWORK_ANALYSIS_EXAMPLE.md`**: Detailed documentation
- **`docs/NETWORK_ANALYSIS_SUMMARY.md`**: Network analysis results template

Features:
- Identifies disconnected network components using graph theory (DFS)
- Finds temporal gaps between components
- Suggests bridge interferograms to connect components
- Analyzes weak connection points
- Generates detailed connectivity reports
- Creates visual network plots

### 5. Testing and Documentation
- **`TESTING_LICSAR.md`**: Testing instructions for LiCSAR support
- **`test_licsar_support.py`**: Comprehensive test script
- **`quick_test_licsar.py`**: Quick functionality test
- **`run_licsar_test.sh`**: Bash script for running tests
- **`simple_licsar_geometry.py`**: Simplified geometry creation
- **`src/mintpy/cli/test_licsar_script.sh`**: CLI test script

## Key Features

### Coherence Handling
✅ Detects uint8 coherence files (0-255 range)
✅ Auto-normalizes to standard 0-1 range
✅ Preserves data quality during processing
✅ Compatible with LiCSAR's space-efficient storage

### Network Quality Control
✅ NaN ratio filtering for spatial coverage
✅ MST integration to maintain connectivity
✅ Customizable thresholds
✅ Detailed statistics output

### Network Connectivity Analysis
✅ Identifies disconnected components
✅ Finds temporal gaps
✅ Suggests bridge interferograms
✅ Visualizes network structure
✅ Provides detailed reports

## Testing Status

### Tested With:
- **Dataset**: LiCSAR Track 076A_11251_131313
- **Interferograms**: 2000 (2014-2023)
- **After filtering**: 535 kept (2017-2019), maxNaNRatio=0.35
- **Result**: Temporal coherence mean = 0.536 ✅

### Network Analysis Results:
- **3 disconnected components identified**:
  - Component 1: 2014-2017 (92 dates, 386 ifgs)
  - Component 2: 2018-2022 (220 dates, 1041 ifgs) ← Main dataset
  - Component 3: 2022-2023 (12 dates, 38 ifgs)
- **Temporal gaps**:
  - Gap 1: Oct 2017 - Mar 2018 (156 days)
  - Gap 2: Nov 2022 - Dec 2022 (24 days)

## Usage Examples

### 1. Prepare LiCSAR Data
```bash
prep_licsar.py /path/to/licsar/data/*.geo.unw.tif
prep_licsar.py /path/to/licsar/data/*.geo.cc.tif
```

### 2. Create Geometry File
```bash
create_licsar_geometry.py --geom-dir /path/to/geo/files --output geometryGeo.h5
```

### 3. Load Data with NaN Filtering
```bash
# In smallbaselineApp.cfg:
mintpy.network.maxNaNRatio = 0.35
mintpy.network.coherenceBased = no
mintpy.network.startDate = 20170115
mintpy.network.endDate = 20190105
```

### 4. Check Network Connectivity
```bash
cd /path/to/work/directory
python /path/to/MintPy/src/mintpy/utils/check_network_connectivity.py
```

## Files Modified from Main Branch

### Core Modifications:
1. `src/mintpy/prep_licsar.py` - Coherence scale factor detection
2. `src/mintpy/objects/stackDict.py` - Coherence normalization
3. `src/mintpy/modify_network.py` - NaN ratio filtering
4. `src/mintpy/cli/modify_network.py` - CLI arguments
5. `src/mintpy/load_data.py` - LiCSAR processor support

### New Files:
- Network connectivity utilities (4 files)
- Testing scripts (5 files)
- Documentation (3 files)
- CLI entry points (2 files)

## Merge History

This branch merges:
1. Original `licsar` branch (basic LiCSAR support)
2. `mohsenidev/add-licsar-product-support-to-mintpy` (coherence + NaN filtering)
3. Latest network connectivity analysis tools

## Repository Information

- **GitHub**: https://github.com/mohseniaref/MintPy
- **Branch**: licsarmintpy
- **Last Updated**: October 18, 2025
- **Status**: ✅ All tests passing, ready for use

## Next Steps

1. **For Users**: Clone this branch and test with your LiCSAR data
2. **For Development**: Submit PR to main MintPy repository
3. **For Testing**: Use provided test scripts with your data

## Contact

For questions or issues with LiCSAR support, please open an issue on the GitHub repository.

---
Generated: October 18, 2025
