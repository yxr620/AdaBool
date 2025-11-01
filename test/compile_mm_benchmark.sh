#!/bin/bash
set -euo pipefail

mkdir -p ./bin
# Build
nvcc -O3 -std=c++17 \
    -I../src/ivf \
    -I../simple_ivf/cutlass/include \
    -lstdc++fs \
    ./MM_benchmark.cu \
    -o ./bin/MM_benchmark

echo "Compilation completed. Executable: ./bin/MM_benchmark"

echo "Running benchmark..."
./bin/MM_benchmark "$@"
