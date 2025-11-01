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
#include "../src/AdaBool/key_batch_transformer.cu.h"
#include "../src/AdaBool/inverted_list.cu.h"

constexpr uint32_t kThreadNum = 64;
constexpr uint32_t kKeyStep = 256;

// Dense batch list union wrapper (writes batch of bitmaps)
static void BatchListUnion(
    InvertedListBatchWorkerImpl<kKeyStep, kThreadNum> &worker,
    cudaStream_t stream,
    uint32_t *d_res,
    uint32_t res_len_u32) {
    worker.RunArrays(stream, d_res, res_len_u32);
}

// Sparse batch list union wrapper (mirrors batch_inverted_list_test.cu)
static cudaError_t SparseBatchListUnion(
    const InvertedListWorker &worker,
    uint32_t *d_res,
    const uint32_t* d_qidx,
    uint32_t res_len_per_batch_bytes,
    cudaStream_t stream = 0)
{
    uint32_t res_len_per_batch_u32 = res_len_per_batch_bytes / 4;

    // process LIST_ARRAY1
    RunSparseBatchListUnion1(
        worker.tasks[LIST_ARRAY1].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_ARRAY1].indices_beg.d, worker.tasks[LIST_ARRAY1].d_value_headers, worker.tasks[LIST_ARRAY1].d_value_arrays);

    // process LIST_ARRAY4
    RunSparseBatchListUnion<LIST_ARRAY4>(
        worker.tasks[LIST_ARRAY4].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_ARRAY4].indices_beg.d, worker.tasks[LIST_ARRAY4].d_value_headers, worker.tasks[LIST_ARRAY4].d_value_arrays);

    // process LIST_ARRAY8
    RunSparseBatchListUnion<LIST_ARRAY8>(
        worker.tasks[LIST_ARRAY8].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_ARRAY8].indices_beg.d, worker.tasks[LIST_ARRAY8].d_value_headers, worker.tasks[LIST_ARRAY8].d_value_arrays);

    // process LIST_ARRAY16
    RunSparseBatchListUnion<LIST_ARRAY16>(
        worker.tasks[LIST_ARRAY16].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_ARRAY16].indices_beg.d, worker.tasks[LIST_ARRAY16].d_value_headers, worker.tasks[LIST_ARRAY16].d_value_arrays);

    // process LIST_ARRAY32
    RunSparseBatchListUnion<LIST_ARRAY32>(
        worker.tasks[LIST_ARRAY32].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_ARRAY32].indices_beg.d, worker.tasks[LIST_ARRAY32].d_value_headers, worker.tasks[LIST_ARRAY32].d_value_arrays);

    // process LIST_BITSET
    RunSparseBatchListUnionBitset(
        worker.tasks[LIST_BITSET].worker, stream, d_res, d_qidx, res_len_per_batch_u32,
        worker.tasks[LIST_BITSET].indices_beg.d, worker.tasks[LIST_BITSET].d_value_headers, worker.tasks[LIST_BITSET].d_value_arrays);

    return cudaSuccess;
}

static inline uint32_t align_up_u32(uint32_t x, uint32_t a) { return (x + a - 1) & ~(a - 1); }
static inline uint64_t align_up_u64(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

// Collect set bit indices from a bitmap (limit to ncols bits)
static void CollectIdsFromBitmap(const uint8_t* bitmap, size_t bitmap_bytes, uint32_t ncols, std::vector<uint32_t>& out) {
    out.clear();
    if (!bitmap) return;
    const size_t total_bits = std::min<size_t>(bitmap_bytes * 8, ncols);
    for (size_t byte_idx = 0; byte_idx < (total_bits + 7) / 8; ++byte_idx) {
        uint8_t byte = bitmap[byte_idx];
        while (byte) {
            unsigned lsb_index = __builtin_ctz(static_cast<unsigned>(byte));
            uint32_t bit_idx = static_cast<uint32_t>(byte_idx * 8 + lsb_index);
            if (bit_idx < ncols) out.push_back(bit_idx);
            byte &= (byte - 1);
        }
    }
}

template <class T>
static double DurationMs(T a, T b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// CPU-based sparse matrix-vector multiply for correctness checking
// Computes result = query_rows x matrix (boolean OR-AND)
static void CPUSparseMatrixVectorMultiply(
    const std::multimap<uint32_t, uint32_t>& matrix_kvs,
    const std::vector<uint32_t>& query_rows,
    uint32_t ncols,
    std::set<uint32_t>& result) {
    result.clear();
    std::set<uint32_t> query_set(query_rows.begin(), query_rows.end());
    for (const auto& kv : matrix_kvs) {
        if (query_set.count(kv.first)) {
            result.insert(kv.second);
        }
    }
}

int main(int argc, char** argv) {
    // Defaults inspired by MV_benchmark, with batch size 4
    uint64_t rows = 1e4;      // number of rows
    uint64_t cols = 1e4;       // number of columns
    double density = 0.001;          // matrix density
    double alpha = 1.0;              // Zipfian distribution parameter
    double query_ratio = 0.001;      // fraction of rows in each query
    int warmups = 2;
    int repeats = 10;
    const uint32_t kBatchSize = 4;

    if (argc >= 2) rows = std::stoull(argv[1]);
    if (argc >= 3) cols = std::stoull(argv[2]);
    if (argc >= 4) density = std::stod(argv[3]);
    if (argc >= 5) query_ratio = std::stod(argv[4]);
    if (argc >= 6) warmups = std::stoi(argv[5]);
    if (argc >= 7) repeats = std::stoi(argv[6]);
    if (argc >= 8) alpha = std::stod(argv[7]);

    std::cout << "argc: " << argc << std::endl;
    std::cout << "rows: " << rows << ", cols: " << cols << ", density: " << density << ", alpha: " << alpha
              << ", query_ratio: " << query_ratio << ", warmups: " << warmups << ", repeats: " << repeats
              << ", batch_size: " << kBatchSize << std::endl;

    std::mt19937_64 rng{233};

    // Generate dataset
    auto t0 = std::chrono::high_resolution_clock::now();
    auto [max_key, kvs] = GenLargeDataset_Zipfian_rows(&rng, rows, cols, density, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();

    const uint32_t nrowsA = static_cast<uint32_t>(max_key + 1);
    const uint32_t ncolsA = static_cast<uint32_t>(cols);
    const size_t nnzA = kvs.size();
    const uint32_t padded_cols_128 = static_cast<uint32_t>(align_up_u64(ncolsA, 128));

    // Build batch queries: bitmaps and key lists
    const uint32_t query_bytes = (nrowsA + 7) / 8;
    const uint32_t query_bytes_aligned = align_up_u32(query_bytes, 16u); // 128-bit alignment
    std::vector<uint8_t> query_bitmap(kBatchSize * query_bytes_aligned, 0);
    std::vector<std::vector<uint32_t>> query_lists(kBatchSize);

    auto dist_row = std::uniform_int_distribution<uint32_t>(0, nrowsA ? (nrowsA - 1) : 0);
    size_t per_query = std::max<size_t>(1, static_cast<size_t>(nrowsA * query_ratio));
    for (uint32_t b = 0; b < kBatchSize; ++b) {
        std::set<uint32_t> uniq;
        while (uniq.size() < per_query) {
            uniq.insert(dist_row(rng));
        }
        query_lists[b].assign(uniq.begin(), uniq.end());
        // mark bits into aligned bitmap slice
        uint8_t* qbm = query_bitmap.data() + b * query_bytes_aligned;
        for (auto r : query_lists[b]) {
            qbm[r >> 3] |= static_cast<uint8_t>(1u << (r & 7));
        }
    }

    // Dataset summary
    std::cout << "Dataset: nnz=" << nnzA
              << ", rows=" << nrowsA
              << ", cols=" << ncolsA
              << ", max_key=" << max_key
              << ", gen_time_ms=" << DurationMs(t0, t1)
              << ", density=" << (static_cast<double>(nnzA) / nrowsA / ncolsA)
              << "\n";
    std::cout << "Batch size: " << kBatchSize << ", per-query rows ~" << per_query
              << " (ratio=" << query_ratio << ")\n";
    std::cout << "Bench: warmups=" << warmups << ", repeats=" << repeats << "\n";

    // ========== Custom Methods Setup ==========
    // Build IVF list and workers
    ivf_list list;
    list.BuildFrom(kvs);

    auto dense_worker = list.PrepareBatch(kBatchSize, query_bytes_aligned, query_bitmap.data());

    // Prepare sparse worker inputs
    std::vector<uint32_t> flat_keys; flat_keys.reserve(1024);
    std::vector<uint32_t> flat_qidx; flat_qidx.reserve(1024);
    for (uint32_t b = 0; b < kBatchSize; ++b) {
        for (auto k : query_lists[b]) { flat_keys.push_back(k); flat_qidx.push_back(b); }
    }
    auto sparse_worker = list.Prepare(flat_keys);

    // Result buffers (aligned to 16 bytes stride)
    const uint32_t res_bytes = (ncolsA + 7) / 8;
    const uint32_t res_bytes_aligned = align_up_u32(res_bytes, 16u);
    LenBuffer<OUTPUT, uint8_t> res_dense, res_sparse;
    res_dense.Alloc(kBatchSize * res_bytes_aligned);
    res_sparse.Alloc(kBatchSize * res_bytes_aligned);
    LenBuffer<INPUT, uint32_t> qidx_map;
    qidx_map.AllocFrom(flat_qidx);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto bench = [&](auto fn) {
        // warmup
        for (int i = 0; i < warmups; ++i) fn();
        double sum = 0.0, mn = 1e100, mx = -1.0;
        for (int i = 0; i < repeats; ++i) {
            auto t2 = std::chrono::high_resolution_clock::now();
            fn();
            auto t3 = std::chrono::high_resolution_clock::now();
            double dt = DurationMs(t2, t3);
            sum += dt; mn = std::min(mn, dt); mx = std::max(mx, dt);
        }
        return std::tuple<double,double,double>{sum / repeats, mn, mx};
    };

    // Dense benchmark
    auto dense_call = [&]() {
        LENBUF_CUDA_CHECK(cudaMemsetAsync(res_dense.d, 0, res_dense.len, stream));
        BatchListUnion(*dense_worker, stream, reinterpret_cast<uint32_t*>(res_dense.d), res_bytes_aligned / 4);
        LENBUF_CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    auto [avg_dense, min_dense, max_dense] = bench(dense_call);

    // Sparse benchmark
    auto sparse_call = [&]() {
        LENBUF_CUDA_CHECK(cudaMemsetAsync(res_sparse.d, 0, res_sparse.len, stream));
        SparseBatchListUnion(*sparse_worker, reinterpret_cast<uint32_t*>(res_sparse.d), qidx_map.d, res_bytes_aligned, stream);
        LENBUF_CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    auto [avg_sparse, min_sparse, max_sparse] = bench(sparse_call);

    // Report
    std::cout << "\n========== Performance Results ==========\n";
    std::cout << "Timing (ms):\n";
    std::cout << "  Dense     avg=" << avg_dense << ", min=" << min_dense << ", max=" << max_dense << "\n";
    std::cout << "  Sparse    avg=" << avg_sparse << ", min=" << min_sparse << ", max=" << max_sparse << "\n";

    // Correctness: compare dense and sparse against CPU ground truth for all queries
    // First, run one more time to get fresh outputs
    dense_call();
    sparse_call();

    // Copy back
    LENBUF_CUDA_CHECK(cudaMemcpy(res_dense.h, res_dense.d, res_dense.len, cudaMemcpyDeviceToHost));
    LENBUF_CUDA_CHECK(cudaMemcpy(res_sparse.h, res_sparse.d, res_sparse.len, cudaMemcpyDeviceToHost));

    std::cout << "\n========== Correctness vs CPU Ground Truth ==========\n";
    bool all_ok = true;
    for (uint32_t b = 0; b < kBatchSize; ++b) {
        // Compute CPU ground truth
        std::set<uint32_t> gt;
        CPUSparseMatrixVectorMultiply(kvs, query_lists[b], ncolsA, gt);

        // Dense ids
        const uint8_t* dense_slice = res_dense.h + b * res_bytes_aligned;
        std::vector<uint32_t> dense_ids; CollectIdsFromBitmap(dense_slice, res_bytes_aligned, ncolsA, dense_ids);
        std::set<uint32_t> dense_set(dense_ids.begin(), dense_ids.end());

        // Sparse ids
        const uint8_t* sparse_slice = res_sparse.h + b * res_bytes_aligned;
        std::vector<uint32_t> sparse_ids; CollectIdsFromBitmap(sparse_slice, res_bytes_aligned, ncolsA, sparse_ids);
        std::set<uint32_t> sparse_set(sparse_ids.begin(), sparse_ids.end());

        bool ok_dense = (dense_set == gt);
        bool ok_sparse = (sparse_set == gt);
        std::cout << "Batch " << b << ": gt=" << gt.size()
                  << ", dense=" << dense_set.size() << (ok_dense ? " (OK)" : " (MISMATCH)")
                  << ", sparse=" << sparse_set.size() << (ok_sparse ? " (OK)" : " (MISMATCH)")
                  << "\n";
        all_ok &= ok_dense && ok_sparse;
        if (!ok_dense || !ok_sparse) {
            // Print small diffs to help debug
            std::vector<uint32_t> diff1, diff2;
            std::set_difference(dense_set.begin(), dense_set.end(), gt.begin(), gt.end(), std::back_inserter(diff1));
            std::set_difference(gt.begin(), gt.end(), dense_set.begin(), dense_set.end(), std::back_inserter(diff2));
            if (!diff1.empty()) { std::cout << "  Dense extra: "; for (size_t i=0;i<std::min<size_t>(10,diff1.size());++i) std::cout << diff1[i] << ' '; std::cout << "\n"; }
            if (!diff2.empty()) { std::cout << "  Dense missing: "; for (size_t i=0;i<std::min<size_t>(10,diff2.size());++i) std::cout << diff2[i] << ' '; std::cout << "\n"; }
            diff1.clear(); diff2.clear();
            std::set_difference(sparse_set.begin(), sparse_set.end(), gt.begin(), gt.end(), std::back_inserter(diff1));
            std::set_difference(gt.begin(), gt.end(), sparse_set.begin(), sparse_set.end(), std::back_inserter(diff2));
            if (!diff1.empty()) { std::cout << "  Sparse extra: "; for (size_t i=0;i<std::min<size_t>(10,diff1.size());++i) std::cout << diff1[i] << ' '; std::cout << "\n"; }
            if (!diff2.empty()) { std::cout << "  Sparse missing: "; for (size_t i=0;i<std::min<size_t>(10,diff2.size());++i) std::cout << diff2[i] << ' '; std::cout << "\n"; }
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);

    if (all_ok) {
        std::cout << "\n✅ MM_benchmark correctness check passed for all batches." << std::endl;
        return 0;
    }
    std::cout << "\n❌ MM_benchmark correctness check failed." << std::endl;
    return 1;
}
