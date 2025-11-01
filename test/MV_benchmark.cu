#include <iostream>
#include <vector>
#include <map>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <random>
#include <algorithm>
#include <chrono>
#include <set>

#include <cuda_runtime.h>

#include "../src/AdaBool/data_utility.hpp"
#include "../src/AdaBool/inverted_list.h"
#include "../src/AdaBool/len_buffer.h"
#include "../src/AdaBool/load_balance_transformer.cu.h"
#include "../src/AdaBool/key_transformer.cu.h"

constexpr uint32_t kThreadNum = 64;
constexpr uint32_t kKeyStep = 256;

// Dense and sparse union implementations reused from combined_correctness_check.cu
cudaError_t ListUnion_dense(const InvertedListBitmapWorkerImpl<kKeyStep, kThreadNum> &worker, uint32_t *d_res, cudaStream_t stream = 0)
{
    // process LIST_ARRAY1
    auto &meta1 = worker.metas->at(LIST_ARRAY1);
    auto &headers1 = worker.headers->at(LIST_ARRAY1);
    auto &query_bitmap = worker.query_bitmap.d;
    RunKeyBPTransformKernel1<kThreadNum, kKeyStep>(stream, meta1, query_bitmap, headers1.d, d_res);
    // process LIST_ARRAY4
    auto &meta2 = worker.metas->at(LIST_ARRAY4);
    auto &headers2 = worker.headers->at(LIST_ARRAY4);
    auto &arrays2 = worker.arrays->at(LIST_ARRAY4);
    RunKeyBPTransformKernel<kThreadNum, kKeyStep, LIST_ARRAY4>(stream, meta2, query_bitmap, headers2.d, arrays2.d, d_res);
    // process LIST_ARRAY8
    auto &meta3 = worker.metas->at(LIST_ARRAY8);
    auto &headers3 = worker.headers->at(LIST_ARRAY8);
    auto &arrays3 = worker.arrays->at(LIST_ARRAY8);
    RunKeyBPTransformKernel<kThreadNum, kKeyStep, LIST_ARRAY8>(stream, meta3, query_bitmap, headers3.d, arrays3.d, d_res);
    // process LIST_ARRAY16
    auto &meta4 = worker.metas->at(LIST_ARRAY16);
    auto &headers4 = worker.headers->at(LIST_ARRAY16);
    auto &arrays4 = worker.arrays->at(LIST_ARRAY16);
    RunKeyBPTransformKernel<kThreadNum, kKeyStep, LIST_ARRAY16>(stream, meta4, query_bitmap, headers4.d, arrays4.d, d_res);
    // process LIST_ARRAY32
    auto &meta5 = worker.metas->at(LIST_ARRAY32);
    auto &headers5 = worker.headers->at(LIST_ARRAY32);
    auto &arrays5 = worker.arrays->at(LIST_ARRAY32);
    RunKeyBPTransformKernel<kThreadNum, kKeyStep, LIST_ARRAY32>(stream, meta5, query_bitmap, headers5.d, arrays5.d, d_res);

    // process LIST_BITSET
    auto &meta6 = worker.metas->at(LIST_BITSET);
    auto &headers6 = worker.headers->at(LIST_BITSET);
    auto &arrays6 = worker.arrays->at(LIST_BITSET);
    RunKeyBPTransformKernelBitset<kThreadNum, kKeyStep, LIST_BITSET>(stream, meta6, query_bitmap, headers6.d, arrays6.d, d_res);

    return cudaSuccess;
}

cudaError_t ListUnion_sparse(const InvertedListWorker &worker, uint32_t *d_res, cudaStream_t stream = 0)
{
    // process LIST_ARRAY1
    auto indices_beg1 = worker.tasks[LIST_ARRAY1].indices_beg.d;
    auto value_header1 = worker.tasks[LIST_ARRAY1].d_value_headers;
    auto value_arrays1 = worker.tasks[LIST_ARRAY1].d_value_arrays;
    worker.tasks[LIST_ARRAY1].worker.RunListUnion1(
        stream, d_res, indices_beg1, value_header1, value_arrays1);
    // process LIST_ARRAY4
    auto indices_beg2 = worker.tasks[LIST_ARRAY4].indices_beg.d;
    auto value_header2 = worker.tasks[LIST_ARRAY4].d_value_headers;
    auto value_arrays2 = worker.tasks[LIST_ARRAY4].d_value_arrays;
    worker.tasks[LIST_ARRAY4].worker.RunListUnion<LIST_ARRAY4>(
        stream, d_res, indices_beg2, value_header2, value_arrays2);
    // process LIST_ARRAY8
    auto indices_beg3 = worker.tasks[LIST_ARRAY8].indices_beg.d;
    auto value_header3 = worker.tasks[LIST_ARRAY8].d_value_headers;
    auto value_arrays3 = worker.tasks[LIST_ARRAY8].d_value_arrays;
    worker.tasks[LIST_ARRAY8].worker.RunListUnion<LIST_ARRAY8>(
        stream, d_res, indices_beg3, value_header3, value_arrays3);
    // process LIST_ARRAY16
    auto indices_beg4 = worker.tasks[LIST_ARRAY16].indices_beg.d;
    auto value_header4 = worker.tasks[LIST_ARRAY16].d_value_headers;
    auto value_arrays4 = worker.tasks[LIST_ARRAY16].d_value_arrays;
    worker.tasks[LIST_ARRAY16].worker.RunListUnion<LIST_ARRAY16>(
        stream, d_res, indices_beg4, value_header4, value_arrays4);
    // process LIST_ARRAY32
    auto indices_beg5 = worker.tasks[LIST_ARRAY32].indices_beg.d;
    auto value_header5 = worker.tasks[LIST_ARRAY32].d_value_headers;
    auto value_arrays5 = worker.tasks[LIST_ARRAY32].d_value_arrays;
    worker.tasks[LIST_ARRAY32].worker.RunListUnion<LIST_ARRAY32>(
        stream, d_res, indices_beg5, value_header5, value_arrays5);

    // process LIST_BITSET
    auto indices_beg = worker.tasks[LIST_BITSET].indices_beg.d;
    auto value_header = worker.tasks[LIST_BITSET].d_value_headers;
    auto value_arrays = worker.tasks[LIST_BITSET].d_value_arrays;
    worker.tasks[LIST_BITSET].worker.RunListUnionBitset(
        stream, d_res, indices_beg, value_header, value_arrays);

    return cudaSuccess;
}

static double DurationMs(std::chrono::high_resolution_clock::time_point a,
                         std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Collect set bit indices from a bitmap
static void CollectIdsFromBitmap(const uint8_t* bitmap, size_t bitmap_bytes, std::vector<uint32_t>& out) {
    out.clear();
    if (!bitmap) return;
    for (size_t byte_idx = 0; byte_idx < bitmap_bytes; ++byte_idx) {
        uint8_t byte = bitmap[byte_idx];
        while (byte) {
            unsigned lsb_index = __builtin_ctz(static_cast<unsigned>(byte));
            out.push_back(static_cast<uint32_t>(byte_idx * 8 + lsb_index));
            byte &= (byte - 1);
        }
    }
}

// CPU-based ground truth computation for correctness verification
static std::vector<uint32_t> ComputeGroundTruth(
    const std::multimap<uint32_t, uint32_t>& kvs,
    const std::vector<uint32_t>& query_list,
    uint32_t ncols) {
    
    std::vector<bool> result(ncols, false);
    std::set<uint32_t> query_set(query_list.begin(), query_list.end());
    
    // For each (row, col) pair in the matrix
    for (const auto& kv : kvs) {
        uint32_t row = kv.first;
        uint32_t col = kv.second;
        // If this row is in the query, mark the column
        if (query_set.count(row) > 0) {
            result[col] = true;
        }
    }
    
    // Collect all set columns
    std::vector<uint32_t> gt;
    for (uint32_t i = 0; i < ncols; ++i) {
        if (result[i]) {
            gt.push_back(i);
        }
    }
    return gt;
}

int main(int argc, char** argv) {
    // Defaults inspired by large_* tests
    uint64_t rows = 1e4;      // number of rows
    uint64_t cols = 1e4;      // number of columns
    double density = 0.001;    // matrix density (nnz / (rows * cols))
    double alpha = 1.0;       // Zipfian distribution parameter
    double query_ratio = 1e-3;       // fraction of rows in query
    int warmups = 2;
    int repeats = 10;

    if (argc >= 2) rows = std::stoull(argv[1]);
    if (argc >= 3) cols = std::stoull(argv[2]);
    if (argc >= 4) density = std::stod(argv[3]);
    if (argc >= 5) query_ratio = std::stod(argv[4]);
    if (argc >= 6) warmups = std::stoi(argv[5]);
    if (argc >= 7) repeats = std::stoi(argv[6]);
    if (argc >= 8) alpha = std::stod(argv[7]);

    std::cout<<"argc: "<<argc<<std::endl;
    std::cout<<"rows: "<<rows<<", cols: "<<cols<<", density: "<<density<<", alpha: "<<alpha
             <<", query_ratio: "<<query_ratio<<", warmups: "<<warmups<<", repeats: "<<repeats<<std::endl;

    std::mt19937_64 rng{233};

    // Generate dataset
    auto t0 = std::chrono::high_resolution_clock::now();
    auto [max_key, kvs] = GenLargeDataset_Zipfian_rows(&rng, rows, cols, density, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();

    const uint32_t nrowsA = static_cast<uint32_t>(max_key + 1);
    const uint32_t ncolsA = static_cast<uint32_t>(cols);
    const size_t nnzA = kvs.size();

    // Build query
    std::vector<uint8_t> query_bitmap((nrowsA + 7) / 8, 0);
    std::vector<uint32_t> query_list;
    {
        std::uniform_int_distribution<uint32_t> dist(0, nrowsA - 1);
        const size_t target = std::max<size_t>(1, static_cast<size_t>(nrowsA * query_ratio));
        std::set<uint32_t> chosen;
        while (chosen.size() < target) {
            uint32_t r = dist(rng);
            if (chosen.insert(r).second) {
                query_list.push_back(r);
                query_bitmap[r / 8] |= (1u << (r % 8));
            }
        }
        std::sort(query_list.begin(), query_list.end());
    }

    // Dataset summary
    std::cout << "Dataset: nnz=" << nnzA
              << ", rows=" << nrowsA
              << ", cols=" << ncolsA
              << ", max_key=" << max_key
              << ", gen_time_ms=" << DurationMs(t0, t1)
              << ", density=" << (static_cast<double>(nnzA) / nrowsA / ncolsA)
              << "\n";
    std::cout << "Query: size=" << query_list.size()
              << " (ratio=" << query_ratio << ")\n";
    std::cout << "Bench: warmups=" << warmups << ", repeats=" << repeats << "\n";

    // Compute ground truth on CPU
    std::cout << "\nComputing ground truth on CPU...\n";
    auto t_gt_start = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> gt = ComputeGroundTruth(kvs, query_list, ncolsA);
    auto t_gt_end = std::chrono::high_resolution_clock::now();
    std::cout << "Ground truth computed: " << gt.size() << " result indices, time=" 
              << DurationMs(t_gt_start, t_gt_end) << " ms\n";

    // ========== Custom Methods Setup ==========
    // Build inverted index for custom methods
    ivf_list list;
    list.BuildFrom(kvs);

    // Prepare workers
    auto dense_worker_mem = list.Prepare(query_bitmap.size(), query_bitmap.data());
    auto sparse_worker_mem = list.Prepare(query_list);

    const size_t res_bytes = (cols + 7) / 8;
    LenBuffer<OUTPUT, uint8_t> res_dense, res_sparse;
    res_dense.Alloc(res_bytes);
    res_sparse.Alloc(res_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing helpers
    auto bench = [&](auto fn) {
        // warmups
        for (int i = 0; i < warmups; ++i) fn();
        std::vector<double> times;
        times.reserve(repeats);
        for (int i = 0; i < repeats; ++i) {
            auto t_beg = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(stream);
            auto t_end = std::chrono::high_resolution_clock::now();
            times.push_back(DurationMs(t_beg, t_end));
        }
        double sum = 0.0, mn = 1e99, mx = -1e99;
        for (double v : times) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
        return std::tuple<double,double,double>(sum / times.size(), mn, mx);
    };

    // Run DenseCore benchmark (with prepare)
    auto dense_call_no_init = [&](){
        auto dense_worker = list.Prepare(query_bitmap.size(), query_bitmap.data());
        ListUnion_dense(*dense_worker,
            reinterpret_cast<uint32_t*>(res_dense.d), stream);
    };
    cudaMemsetAsync(res_dense.d, 0, res_dense.len, stream);
    auto [avg_dense_core, min_dense_core, max_dense_core] = bench(dense_call_no_init);

    // Run Dense benchmark (with prepare and memset)
    auto dense_call = [&]() {
        auto dense_worker = list.Prepare(query_bitmap.size(), query_bitmap.data());
        cudaMemsetAsync(res_dense.d, 0, res_dense.len * sizeof(uint8_t), stream);
        ListUnion_dense(*dense_worker, reinterpret_cast<uint32_t*>(res_dense.d), stream);
    };
    auto [avg_dense, min_dense, max_dense] = bench(dense_call);

    // Run Sparse benchmark (with prepare and memset)
    auto sparse_call = [&]() {
        auto sparse_worker = list.Prepare(query_list);
        cudaMemsetAsync(res_sparse.d, 0, res_sparse.len * sizeof(uint8_t), stream);
        ListUnion_sparse(*sparse_worker, reinterpret_cast<uint32_t*>(res_sparse.d), stream);
    };
    auto [avg_sparse, min_sparse, max_sparse] = bench(sparse_call);

    // Report
    std::cout << "\n========== Performance Results ==========\n";
    std::cout << "Timing (ms):\n";
    std::cout << "  DenseCore avg=" << avg_dense_core << ", min=" << min_dense_core << ", max=" << max_dense_core << "\n";
    std::cout << "  Dense     avg=" << avg_dense << ", min=" << min_dense << ", max=" << max_dense << "\n";
    std::cout << "  Sparse    avg=" << avg_sparse << ", min=" << min_sparse << ", max=" << max_sparse << "\n";

    // Correctness check against CPU ground truth
    // Recompute once to produce fresh outputs (not timed)
    
    // Use the workers created for memory measurement
    cudaMemsetAsync(res_dense.d, 0, res_dense.len * sizeof(uint8_t), stream);
    ListUnion_dense(*dense_worker_mem, reinterpret_cast<uint32_t*>(res_dense.d), stream);
    cudaStreamSynchronize(stream);
    
    cudaMemsetAsync(res_sparse.d, 0, res_sparse.len * sizeof(uint8_t), stream);
    ListUnion_sparse(*sparse_worker_mem, reinterpret_cast<uint32_t*>(res_sparse.d), stream);
    cudaStreamSynchronize(stream);

    // Extract Dense/Sparse results
    std::vector<uint8_t> host_dense(res_dense.len);
    std::vector<uint8_t> host_sparse(res_sparse.len);
    cudaMemcpy(host_dense.data(), res_dense.d, res_dense.len * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_sparse.data(), res_sparse.d, res_sparse.len * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    std::vector<uint32_t> ids_dense, ids_sparse;
    CollectIdsFromBitmap(host_dense.data(), host_dense.size(), ids_dense);
    CollectIdsFromBitmap(host_sparse.data(), host_sparse.size(), ids_sparse);
    std::sort(ids_dense.begin(), ids_dense.end());
    std::sort(ids_sparse.begin(), ids_sparse.end());

    // Compare
    auto cmp_sets = [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
        if (a.size() != b.size()) return false;
        return std::equal(a.begin(), a.end(), b.begin());
    };

    bool ok_dense = cmp_sets(gt, ids_dense);
    bool ok_sparse = cmp_sets(gt, ids_sparse);
    
    std::cout << "\n========== Correctness vs CPU Ground Truth ==========\n";
    std::cout << "  Dense    : " << (ok_dense ? "OK" : "FAIL") << " (gt=" << gt.size() << ", dense=" << ids_dense.size() << ")\n";
    std::cout << "  Sparse   : " << (ok_sparse ? "OK" : "FAIL") << " (gt=" << gt.size() << ", sparse=" << ids_sparse.size() << ")\n";

    // Cleanup
    cudaStreamDestroy(stream);

    return 0;
}
