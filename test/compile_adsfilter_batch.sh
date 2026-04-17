#!/bin/bash
set -euo pipefail

# Create bin directory if it doesn't exist
mkdir -p ./bin

# Build
nvcc -O3 -std=c++17 \
    -lstdc++fs \
    ./adsfilter_batch.cu \
    -o ./bin/adsfilter_batch

echo "Compilation completed. Executable: ./bin/adsfilter_batch"

# Run test if arguments are provided, otherwise show usage
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage: ./compile_adsfilter_batch.sh [filter_file] [query_dir] [batch_size]"
    echo "  filter_file: Path to filter.coo file (default: ../dataset/adsfilter/filter.coo)"
    echo "  query_dir:   Directory containing query files (default: ../dataset/adsfilter)"
    echo "  batch_size:  Number of queries per batch (default: 4)"
    echo ""
    echo "Running with default parameters..."
    ./bin/adsfilter_batch ../dataset/adsfilter/filter.coo ../dataset/adsfilter 4
else
    echo "Running test..."
    ./bin/adsfilter_batch "$@"
fi
