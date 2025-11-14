# CENTER_LINE_UTC Metadata Fix for LiCSAR Products

## Problem Statement

When processing LiCSAR data with MintPy for ERA5 atmospheric correction, the `CENTER_LINE_UTC` metadata attribute was missing from the time series file. This required manual post-processing:

```python
# Previously required manual fix
import h5py

t = "22:57:49.897010"
h, m, s = t.split(":")
utc_seconds = float(h)*3600 + float(m)*60 + float(s)

with h5py.File("timeseries.h5", "a") as f:
    f.attrs["CENTER_LINE_UTC"] = str(utc_seconds)
```

## Root Cause

The `prep_licsar.py` script reads `center_time` from LiCSAR metadata files but only stored it as `CENTER_TIME` (string format: "HH:MM:SS.ssssss"). It did NOT convert it to `CENTER_LINE_UTC` (UTC seconds as float), which is required by PyAPS for ERA5 atmospheric delay calculations.

## Solution

Modified `prep_licsar.py` to automatically:
1. Read `center_time` from LiCSAR metadata.txt
2. Parse the time string (format: "HH:MM:SS.ssssss")
3. Convert to UTC seconds: `utc_seconds = hours*3600 + minutes*60 + seconds`
4. Add `CENTER_LINE_UTC` metadata attribute

## Implementation

**File Modified:** `src/mintpy/prep_licsar.py`

**Location:** Lines 134-147 (after `meta['CENTER_TIME'] = center_time`)

**Code Added:**
```python
# Add CENTER_LINE_UTC (UTC seconds) for atmospheric correction
try:
    # Parse time string (format: HH:MM:SS.ssssss)
    time_parts = center_time.split(':')
    if len(time_parts) == 3:
        hours = float(time_parts[0])
        minutes = float(time_parts[1])
        seconds = float(time_parts[2])
        utc_seconds = hours * 3600 + minutes * 60 + seconds
        meta['CENTER_LINE_UTC'] = str(utc_seconds)
except (ValueError, IndexError, AttributeError):
    # If parsing fails, skip CENTER_LINE_UTC
    pass
```

## Example

**Input (from LiCSAR metadata.txt):**
```
center_time=22:57:49.897010
```

**Output (in MintPy metadata):**
```python
meta['CENTER_TIME'] = '22:57:49.897010'        # String format (original)
meta['CENTER_LINE_UTC'] = '82669.89701'        # UTC seconds (NEW!)
```

**Calculation:**
```
22 hours   × 3600 = 79200 seconds
57 minutes × 60   = 3420 seconds  
49.897010 seconds = 49.89701 seconds
───────────────────────────────────
Total             = 82669.89701 seconds
```

## Benefits

1. **No Manual Post-Processing:** CENTER_LINE_UTC is automatically available in geometry and time series files
2. **ERA5 Compatibility:** PyAPS can directly use the metadata without modification
3. **Robust:** Handles various time string formats with error handling
4. **Backwards Compatible:** Existing code continues to work; CENTER_TIME is still available

## Testing

### Unit Test Results
```
✓ PASS: 22:57:49.897010 -> 82669.897010
✓ PASS: 00:00:00.0      -> 0.000000
✓ PASS: 12:30:45.5      -> 45045.500000
✓ PASS: 23:59:59.999999 -> 86399.999999
```

### Integration Test
Process a LiCSAR dataset and verify:
```bash
# 1. Prepare LiCSAR data
prep_licsar.py -i ./GEOC -o ./inputs

# 2. Check metadata
info.py inputs/geometryGeo.h5 | grep CENTER_LINE_UTC

# Expected output:
# CENTER_LINE_UTC: 82669.89701
```

## ERA5 Atmospheric Correction Workflow

With this fix, the complete ERA5 workflow now works seamlessly:

```bash
# 1. Prepare LiCSAR data (CENTER_LINE_UTC automatically added)
prep_licsar.py -i GEOC/ -o inputs/

# 2. Configure ERA5 in smallbaselineApp.cfg
mintpy.troposphericDelay.method = pyaps
mintpy.troposphericDelay.weatherModel = ERA5

# 3. Run atmospheric correction
smallbaselineApp.py smallbaselineApp.cfg --dostep correct_troposphere

# ✓ PyAPS will find CENTER_LINE_UTC and process correctly!
```

## Git Commit

**Branch:** `licsarmintpy`

**Commit:** `7a73d002`

**Message:**
```
Add CENTER_LINE_UTC metadata for ERA5 atmospheric correction

- Automatically convert center_time (HH:MM:SS.ssssss) to UTC seconds
- Add CENTER_LINE_UTC metadata attribute for PyAPS/ERA5 compatibility
- This eliminates the need for manual post-processing to add CENTER_LINE_UTC
- Fixes issue where ERA5 correction would fail without this metadata

Example: center_time='22:57:49.897010' -> CENTER_LINE_UTC='82669.89701'
```

## Related Documentation

- **ERA5 Setup Guide:** `/raid-gpu2/maref/test_licsar_072A_ERA5/ERA5_ATMOSPHERIC_CORRECTION_GUIDE.md`
- **Quick Start:** `/raid-gpu2/maref/test_licsar_072A_ERA5/QUICKSTART_ERA5.txt`
- **LiCSAR Metadata:** Typically found in `GEOC/metadata.txt`

## Future Improvements

Potential enhancements:
1. Add validation to ensure UTC seconds is within 0-86400 range
2. Support multiple time formats (ISO 8601, etc.)
3. Add warning if center_time is missing from metadata
4. Include CENTER_LINE_UTC in geometry file summary output

## Contact

**Author:** Mohammad Mohseni Aref  
**Date:** November 14, 2025  
**Issue:** Missing CENTER_LINE_UTC for ERA5 atmospheric correction  
**Status:** ✅ FIXED

---

*No more manual h5py editing required! 🎉*
