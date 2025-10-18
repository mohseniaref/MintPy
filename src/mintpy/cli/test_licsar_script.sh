#!/bin/bash

# LiCSAR MintPy Processing Test Script
# This script tests the prep_licsar.py functionality

echo "=== LiCSAR MintPy Processing Test ==="

# Set paths
MINTPY_DIR="/raid-gpu2/maref/Software/Code-dev/MintPy"
LICSAR_DATA="/raid2-manaslu/maref/Extra/072A_05090_131313/GEOC"
TEST_DIR="/raid2-manaslu/maref/test_licsar_processing"

echo "MintPy Directory: $MINTPY_DIR"
echo "LiCSAR Data: $LICSAR_DATA"
echo "Test Directory: $TEST_DIR"

# Create test directory
mkdir -p $TEST_DIR
cd $MINTPY_DIR

# Set Python path
export PYTHONPATH=$MINTPY_DIR/src:$PYTHONPATH

echo -e "\n=== Step 1: Prepare a few interferograms ==="
# Process a few interferograms for testing
python src/mintpy/cli/prep_licsar.py $LICSAR_DATA/201902*/20*.geo.unw.tif

echo -e "\n=== Step 2: Prepare coherence files ==="
# Process corresponding coherence files  
python src/mintpy/cli/prep_licsar.py $LICSAR_DATA/201902*/20*.geo.cc.tif

echo -e "\n=== Step 3: Check generated files ==="
# Check what was generated
echo "Generated RSC files:"
find $LICSAR_DATA -name "*.rsc" -newer $TEST_DIR | head -10

echo -e "\n=== Step 4: Show sample metadata ==="
# Show sample metadata
if [ -f "$LICSAR_DATA/20190204_20190306/20190204_20190306.geo.unw.tif.rsc" ]; then
    echo "Sample RSC content:"
    head -20 "$LICSAR_DATA/20190204_20190306/20190204_20190306.geo.unw.tif.rsc"
fi

echo -e "\n=== LiCSAR MintPy Test Complete ==="
echo "The LiCSAR data has been prepared for MintPy processing!"
echo "You can now use these .tif and .rsc files with MintPy load_data.py"

# Example load_data command (commented out)
echo -e "\nExample usage with load_data.py:"
echo "load_data.py -t LiCSAR $LICSAR_DATA/*/20*.geo.unw.tif"