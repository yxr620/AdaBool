#!/bin/bash
set -euo pipefail

mkdir -p ./bin
# Build
nvcc -O3 -std=c++17 \
    -I../src/ivf \
    -I../simple_ivf/cutlass/include \
    -lstdc++fs \
    ./MV_benchmark.cu \
    -o ./bin/MV_benchmark

echo "Compilation completed. Executable: ./bin/MV_benchmark"

echo "Running benchmark..."
./bin/MV_benchmark "$@"
