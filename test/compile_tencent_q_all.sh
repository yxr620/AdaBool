#!/bin/bash
set -euo pipefail

# Create bin directory if it doesn't exist
mkdir -p ./bin

# Build
nvcc -O3 -std=c++17 \
    -I../src/ivf \
    -I../simple_ivf/cutlass/include \
    -lstdc++fs \
    ./tencent_q_all.cu \
    -o ./bin/tencent_q_all

echo "Compilation completed. Executable: ./bin/tencent_q_all"

# Check if arguments are provided, if not use default paths
if [ $# -eq 0 ]; then
    echo "Running test with default paths..."
    ./bin/tencent_q_all ../dataset/tencent/filter.coo ../dataset/tencent
else
    echo "Running test with provided arguments..."
    ./bin/tencent_q_all "$@"
fi
