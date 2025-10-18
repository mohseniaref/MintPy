#!/bin/bash
# Run the LiCSAR support test script

# Check if a data directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/licsar/data"
    echo "No data directory provided. Using default /raid2-manaslu/maref/inputs/LiCSAR_data if it exists."
    DATA_DIR="/raid2-manaslu/maref/inputs/LiCSAR_data"
else
    DATA_DIR="$1"
fi

# Check if the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

# Create a test work directory
WORK_DIR="./licsar_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORK_DIR"

# Run the test script
python test_licsar_support.py --data-dir "$DATA_DIR" --work-dir "$WORK_DIR"

# Check the exit code
if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
    echo "Test outputs available in: $WORK_DIR"
else
    echo "Test failed! Check the output above for errors."
fi