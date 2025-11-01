#!/bin/bash
set -euo pipefail

# Create bin directory if it doesn't exist
mkdir -p ./bin

# Build
nvcc -O3 -std=c++17 \
    -I../src/ivf \
    -I../simple_ivf/cutlass/include \
    -lstdc++fs \
    ./tencent_batch.cu \
    -o ./bin/tencent_batch

echo "Compilation completed. Executable: ./bin/tencent_batch"

# Run test if arguments are provided, otherwise show usage
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage: ./compile_tencent_batch.sh [filter_file] [query_dir] [batch_size]"
    echo "  filter_file: Path to filter.coo file (default: ../result/output/filter.coo)"
    echo "  query_dir:   Directory containing query files (default: ../result/output)"
    echo "  batch_size:  Number of queries per batch (default: 4)"
    echo ""
    echo "Running with default parameters..."
    ./bin/tencent_batch ../dataset/tencent/filter.coo ../dataset/tencent 4
else
    echo "Running test..."
    ./bin/tencent_batch "$@"
fi
