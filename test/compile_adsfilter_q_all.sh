#!/bin/bash
set -euo pipefail

# Create bin directory if it doesn't exist
mkdir -p ./bin

# Build
nvcc -O3 -std=c++17 \
    -lstdc++fs \
    ./adsfilter_q_all.cu \
    -o ./bin/adsfilter_q_all

echo "Compilation completed. Executable: ./bin/adsfilter_q_all"

# Check if arguments are provided, if not use default paths
if [ $# -eq 0 ]; then
    echo "Running test with default paths..."
    ./bin/adsfilter_q_all ../dataset/adsfilter/filter.coo ../dataset/adsfilter
else
    echo "Running test with provided arguments..."
    ./bin/adsfilter_q_all "$@"
fi
