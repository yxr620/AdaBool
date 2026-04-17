#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <cmath>

#include <cuda_runtime.h>
#include "../src/AdaBool/inverted_list.h"
#include "../src/AdaBool/len_buffer.h"
#include "../src/AdaBool/load_balance_transformer.cu.h"
#include "../src/AdaBool/key_transformer.cu.h"
#include "../src/AdaBool/key_batch_transformer.cu.h"
#include "../src/AdaBool/inverted_list.cu.h"

constexpr uint32_t kThreadNum = 64;
constexpr uint32_t kKeyStep = 256;

// Dense batch union implementation
template<uint32_t BatchSize>
void BatchListUnion_dense(
    InvertedListBatchWorkerImpl<kKeyStep, kThreadNum> &worker,
    cudaStream_t stream,
    uint32_t *d_res,
    uint32_t res_len_u32)
{
    worker.RunArrays(stream, d_res, res_len_u32);
}

// Sparse batch union implementation
cudaError_t BatchListUnion_sparse(
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

// Collect set bit indices from a bitmap
static void CollectIdsFromBitmap(const uint8_t* bitmap, size_t bitmap_bytes, uint32_t ncols, std::vector<uint32_t>& out) {
    out.clear();
    if (!bitmap) return;
    const size_t total_bits = std::min<size_t>(bitmap_bytes * 8, ncols);
    for (size_t byte_idx = 0; byte_idx < (total_bits + 7) / 8; ++byte_idx) {
        uint8_t byte = bitmap[byte_idx];
        while (byte) {
            unsigned lsb_index = __builtin_ctz(static_cast<unsigned>(byte));
            uint32_t id = static_cast<uint32_t>(byte_idx * 8 + lsb_index);
            if (id < ncols) {
                out.push_back(id);
            }
            byte &= (byte - 1);
        }
    }
}

static double DurationMs(std::chrono::high_resolution_clock::time_point a,
                         std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static inline uint32_t align_up(uint32_t x, uint32_t a) { 
    return (x + a - 1) & ~(a - 1); 
}

// Read COO file
bool ReadCOO(const std::string& path, std::vector<std::pair<uint32_t, uint32_t>>& data, 
             uint32_t& max_row, uint32_t& max_col) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    data.clear();
    max_row = 0;
    max_col = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty() || line[0] == '#' || line[0] == '%') {
            continue;
        }
        
        std::istringstream iss(line);
        uint32_t row, col;
        if (iss >> row >> col) {
            data.push_back({row, col});
            max_row = std::max(max_row, row);
            max_col = std::max(max_col, col);
        }
    }
    
    file.close();
    return true;
}

// Read query file (single column of row indices)
bool ReadQuery(const std::string& path, std::vector<uint32_t>& query) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    query.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty() || line[0] == '#' || line[0] == '%') {
            continue;
        }
        
        std::istringstream iss(line);
        uint32_t row;
        if (iss >> row) {
            query.push_back(row);
        }
    }
    
    file.close();
    return true;
}

// Compute ground truth using CPU for a batch of queries
std::vector<std::set<uint32_t>> ComputeGroundTruth(
    const std::vector<std::pair<uint32_t, uint32_t>>& data,
    const std::vector<std::vector<uint32_t>>& queries) {
    
    std::vector<std::set<uint32_t>> results(queries.size());
    
    for (size_t q = 0; q < queries.size(); ++q) {
        std::set<uint32_t> query_set(queries[q].begin(), queries[q].end());
        for (const auto& kv : data) {
            if (query_set.count(kv.first)) {
                results[q].insert(kv.second);
            }
        }
    }
    
    return results;
}

// Find all query files in a directory
std::vector<std::string> FindQueryFiles(const std::string& dir_path) {
    std::vector<std::string> query_files;
    std::regex query_pattern("query_vec_\\d+\\.coo");
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, query_pattern)) {
                    query_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    
    // Sort files for consistent ordering
    std::sort(query_files.begin(), query_files.end());
    return query_files;
}

struct BatchResult {
    size_t batch_idx;
    size_t total_query_size;
    std::vector<size_t> result_sizes;
    double prepare_dense_ms;
    double prepare_sparse_ms;
    double dense_exec_ms;
    double sparse_exec_ms;
    bool dense_correct;
    bool sparse_correct;
};

// Calculate statistics
struct Statistics {
    double mean;
    double median;
    double stddev;
    double min;
    double max;
    double p50;
    double p90;
    double p95;
    double p99;
};

Statistics CalculateStats(std::vector<double> data) {
    Statistics stats;
    if (data.empty()) {
        return Statistics{0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    
    std::sort(data.begin(), data.end());
    
    // Mean
    double sum = 0.0;
    for (double v : data) sum += v;
    stats.mean = sum / data.size();
    
    // Median (P50)
    size_t n = data.size();
    if (n % 2 == 0) {
        stats.median = (data[n/2-1] + data[n/2]) / 2.0;
    } else {
        stats.median = data[n/2];
    }
    stats.p50 = stats.median;
    
    // Standard deviation
    double sq_sum = 0.0;
    for (double v : data) {
        double diff = v - stats.mean;
        sq_sum += diff * diff;
    }
    stats.stddev = std::sqrt(sq_sum / data.size());
    
    // Min and Max
    stats.min = data.front();
    stats.max = data.back();
    
    // Percentiles
    auto percentile = [&](double p) -> double {
        if (data.empty()) return 0.0;
        double idx = p * (data.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(idx));
        size_t upper = static_cast<size_t>(std::ceil(idx));
        if (lower == upper) return data[lower];
        double weight = idx - lower;
        return data[lower] * (1.0 - weight) + data[upper] * weight;
    };
    
    stats.p90 = percentile(0.90);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);
    
    return stats;
}

void PrintStats(const std::string& name, const Statistics& stats) {
    std::cout << name << " Statistics:\n";
    printf("  Mean:   %.3f ms\n", stats.mean);
    printf("  Median: %.3f ms\n", stats.median);
    printf("  Stddev: %.3f ms\n", stats.stddev);
    printf("  Min:    %.3f ms\n", stats.min);
    printf("  Max:    %.3f ms\n", stats.max);
    printf("  P50:    %.3f ms\n", stats.p50);
    printf("  P90:    %.3f ms\n", stats.p90);
    printf("  P95:    %.3f ms\n", stats.p95);
    printf("  P99:    %.3f ms\n", stats.p99);
}

void PrintUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [filter_file] [query_dir] [batch_size]\n";
    std::cout << "  filter_file: Path to filter.coo file (default: ../dataset/adsfilter/filter.coo)\n";
    std::cout << "  query_dir:   Directory containing query files (default: ../dataset/adsfilter)\n";
    std::cout << "  batch_size:  Number of queries per batch (default: 4, must be > 0)\n";
}

int main(int argc, char** argv) {
    std::string filter_path = "../dataset/adsfilter/filter.coo";
    std::string query_dir = "../dataset/adsfilter";
    uint32_t batch_size = 4;

    if (argc >= 2) filter_path = argv[1];
    if (argc >= 3) query_dir = argv[2];
    if (argc >= 4) {
        batch_size = std::atoi(argv[3]);
        if (batch_size == 0) {
            std::cerr << "Error: batch_size must be greater than 0\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  AdsFilter Batch Query Test\n";
    std::cout << "========================================\n";
    std::cout << "Filter file: " << filter_path << std::endl;
    std::cout << "Query directory: " << query_dir << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;

    // Find all query files
    auto query_files = FindQueryFiles(query_dir);
    if (query_files.empty()) {
        std::cerr << "No query files found in " << query_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << query_files.size() << " query files" << std::endl;
    
    // Only process queries that form complete batches
    size_t num_complete_queries = (query_files.size() / batch_size) * batch_size;
    if (num_complete_queries < query_files.size()) {
        std::cout << "WARNING: Truncating to " << num_complete_queries 
                  << " queries (complete batches only)" << std::endl;
        query_files.resize(num_complete_queries);
    }
    
    if (query_files.empty()) {
        std::cerr << "Not enough queries to form a complete batch (need at least " 
                  << batch_size << " queries)" << std::endl;
        return 1;
    }

    // Read filter data (COO sparse matrix)
    std::vector<std::pair<uint32_t, uint32_t>> kvs;
    uint32_t max_row, max_col;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!ReadCOO(filter_path, kvs, max_row, max_col)) {
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read filter data: " << kvs.size() << " entries, "
              << "rows=" << (max_row + 1) << ", cols=" << (max_col + 1)
              << " (" << DurationMs(t0, t1) << " ms)" << std::endl;

    // Build inverted index once
    t0 = std::chrono::high_resolution_clock::now();
    ivf_list list;
    list.BuildFrom(kvs);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Built inverted index (" << DurationMs(t0, t1) << " ms)" << std::endl;

    const uint32_t nrows = max_row + 1;
    const uint32_t ncols = max_col + 1;
    const uint32_t res_bytes_per_query = (ncols + 7) / 8;
    const uint32_t res_bytes_aligned = align_up(res_bytes_per_query, 16u);
    
    // Allocate result buffers
    LenBuffer<OUTPUT, uint8_t> res_dense, res_sparse;
    res_dense.Alloc(batch_size * res_bytes_aligned);
    res_sparse.Alloc(batch_size * res_bytes_aligned);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup: create a dummy query to initialize GPU kernels
    std::cout << "\nPerforming warmup..." << std::endl;
    {
        uint32_t query_bytes = (nrows + 7) / 8;
        uint32_t query_bytes_aligned = align_up(query_bytes, 16u);
        std::vector<uint8_t> warmup_bitmap(batch_size * query_bytes_aligned, 0);
        
        auto warmup_dense_worker = list.PrepareBatch(batch_size, query_bytes_aligned, warmup_bitmap.data());
        cudaMemsetAsync(res_dense.d, 0, res_dense.len, stream);
        BatchListUnion_dense<4>(*warmup_dense_worker, stream, reinterpret_cast<uint32_t*>(res_dense.d), res_bytes_aligned / 4);
        cudaStreamSynchronize(stream);
        
        std::vector<uint32_t> warmup_keys, warmup_qidx;
        for (uint32_t i = 0; i < batch_size; ++i) {
            warmup_keys.push_back(i);
            warmup_qidx.push_back(i);
        }
        auto warmup_sparse_worker = list.Prepare(warmup_keys);
        LenBuffer<INPUT, uint32_t> warmup_qidx_map;
        warmup_qidx_map.AllocFrom(warmup_qidx);
        cudaMemsetAsync(res_sparse.d, 0, res_sparse.len, stream);
        BatchListUnion_sparse(*warmup_sparse_worker, reinterpret_cast<uint32_t*>(res_sparse.d), warmup_qidx_map.d, res_bytes_aligned, stream);
        cudaStreamSynchronize(stream);
        
        std::cout << "Warmup completed." << std::endl;
    }

    // Statistics collection
    std::vector<BatchResult> all_results;
    
    int dense_pass_count = 0;
    int sparse_pass_count = 0;
    
    double total_dense_exec_time = 0.0;
    double total_sparse_exec_time = 0.0;

    // Process queries in batches
    size_t num_batches = (query_files.size() + batch_size - 1) / batch_size;
    
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start_idx = batch_idx * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, query_files.size());
        size_t current_batch_size = end_idx - start_idx;

        std::cout << "\n========================================\n";
        std::cout << "Processing Batch " << (batch_idx + 1) << "/" << num_batches 
                  << " (queries " << start_idx << "-" << (end_idx - 1) << ")" << std::endl;

        // Read queries for this batch (should always be exactly batch_size)
        std::vector<std::vector<uint32_t>> queries(batch_size);
        size_t total_query_keys = 0;
        
        for (size_t i = 0; i < batch_size; ++i) {
            if (!ReadQuery(query_files[start_idx + i], queries[i])) {
                std::cerr << "Failed to read query: " << query_files[start_idx + i] << std::endl;
                return 1;
            }
            total_query_keys += queries[i].size();
            std::cout << "  Query " << (start_idx + i) << ": " << queries[i].size() << " keys" << std::endl;
        }

        // Compute ground truth
        auto gt_results = ComputeGroundTruth(kvs, queries);

        // Prepare dense worker (query bitmap)
        uint32_t query_bytes = (nrows + 7) / 8;
        uint32_t query_bytes_aligned = align_up(query_bytes, 16u);
        std::vector<uint8_t> query_bitmap(batch_size * query_bytes_aligned, 0);
        
        for (size_t i = 0; i < batch_size; ++i) {
            uint8_t* qbm = query_bitmap.data() + i * query_bytes_aligned;
            for (auto r : queries[i]) {
                if (r < nrows) {
                    qbm[r / 8] |= (1u << (r % 8));
                }
            }
        }

        auto t_prep_dense_start = std::chrono::high_resolution_clock::now();
        auto dense_worker = list.PrepareBatch(batch_size, query_bytes_aligned, query_bitmap.data());
        auto t_prep_dense_end = std::chrono::high_resolution_clock::now();
        double dense_prepare_time = DurationMs(t_prep_dense_start, t_prep_dense_end);

        // Prepare sparse worker (flat keys and indices)
        std::vector<uint32_t> flat_keys, flat_qidx;
        for (size_t i = 0; i < batch_size; ++i) {
            for (auto k : queries[i]) {
                flat_keys.push_back(k);
                flat_qidx.push_back(i);
            }
        }

        auto t_prep_sparse_start = std::chrono::high_resolution_clock::now();
        auto sparse_worker = list.Prepare(flat_keys);
        LenBuffer<INPUT, uint32_t> qidx_map;
        qidx_map.AllocFrom(flat_qidx);
        auto t_prep_sparse_end = std::chrono::high_resolution_clock::now();
        double sparse_prepare_time = DurationMs(t_prep_sparse_start, t_prep_sparse_end);

        // Execute Dense
        cudaMemsetAsync(res_dense.d, 0, res_dense.len, stream);
        auto t_dense_start = std::chrono::high_resolution_clock::now();
        BatchListUnion_dense<4>(*dense_worker, stream, reinterpret_cast<uint32_t*>(res_dense.d), res_bytes_aligned / 4);
        cudaStreamSynchronize(stream);
        auto t_dense_end = std::chrono::high_resolution_clock::now();
        double dense_exec_time = DurationMs(t_dense_start, t_dense_end);

        // Execute Sparse
        cudaMemsetAsync(res_sparse.d, 0, res_sparse.len, stream);
        auto t_sparse_start = std::chrono::high_resolution_clock::now();
        BatchListUnion_sparse(*sparse_worker, reinterpret_cast<uint32_t*>(res_sparse.d), qidx_map.d, res_bytes_aligned, stream);
        cudaStreamSynchronize(stream);
        auto t_sparse_end = std::chrono::high_resolution_clock::now();
        double sparse_exec_time = DurationMs(t_sparse_start, t_sparse_end);

        // Copy results back
        LENBUF_CUDA_CHECK(cudaMemcpy(res_dense.h, res_dense.d, res_dense.len, cudaMemcpyDeviceToHost));
        LENBUF_CUDA_CHECK(cudaMemcpy(res_sparse.h, res_sparse.d, res_sparse.len, cudaMemcpyDeviceToHost));

        // Verify correctness
        bool dense_correct = true;
        bool sparse_correct = true;
        std::vector<size_t> result_sizes(current_batch_size);

        for (size_t i = 0; i < current_batch_size; ++i) {
            const uint8_t* dense_bitmap = res_dense.h + i * res_bytes_aligned;
            const uint8_t* sparse_bitmap = res_sparse.h + i * res_bytes_aligned;

            std::vector<uint32_t> dense_ids, sparse_ids;
            CollectIdsFromBitmap(dense_bitmap, res_bytes_aligned, ncols, dense_ids);
            CollectIdsFromBitmap(sparse_bitmap, res_bytes_aligned, ncols, sparse_ids);

            std::set<uint32_t> dense_set(dense_ids.begin(), dense_ids.end());
            std::set<uint32_t> sparse_set(sparse_ids.begin(), sparse_ids.end());

            result_sizes[i] = gt_results[i].size();

            if (dense_set != gt_results[i]) {
                dense_correct = false;
                std::cout << "  Query " << (start_idx + i) << " Dense MISMATCH: "
                          << "expected " << gt_results[i].size() << ", got " << dense_set.size() << std::endl;
            }

            if (sparse_set != gt_results[i]) {
                sparse_correct = false;
                std::cout << "  Query " << (start_idx + i) << " Sparse MISMATCH: "
                          << "expected " << gt_results[i].size() << ", got " << sparse_set.size() << std::endl;
            }
        }

        if (dense_correct) dense_pass_count++;
        if (sparse_correct) sparse_pass_count++;

        // Store results
        BatchResult result;
        result.batch_idx = batch_idx;
        result.total_query_size = total_query_keys;
        result.result_sizes = result_sizes;
        result.prepare_dense_ms = dense_prepare_time;
        result.prepare_sparse_ms = sparse_prepare_time;
        result.dense_exec_ms = dense_exec_time;
        result.sparse_exec_ms = sparse_exec_time;
        result.dense_correct = dense_correct;
        result.sparse_correct = sparse_correct;

        all_results.push_back(result);
        
        // Accumulate execution times
        total_dense_exec_time += dense_exec_time;
        total_sparse_exec_time += sparse_exec_time;

        // Print batch results
        std::cout << "Batch " << (batch_idx + 1) << " Results:" << std::endl;
        std::cout << "  Dense:  Correct=" << (dense_correct ? "YES" : "NO") << std::endl;
        std::cout << "  Sparse: Correct=" << (sparse_correct ? "YES" : "NO") << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(stream);

    // Summary report
    std::cout << "\n========================================\n";
    std::cout << "           SUMMARY REPORT\n";
    std::cout << "========================================\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Total batches processed: " << all_results.size() << "\n";
    std::cout << "Total queries processed: " << query_files.size() << "\n\n";

    // Correctness summary
    std::cout << "Correctness Summary:\n";
    std::cout << "  Dense method:  " << dense_pass_count << "/" << all_results.size() << " batches PASS\n";
    std::cout << "  Sparse method: " << sparse_pass_count << "/" << all_results.size() << " batches PASS\n\n";

    // Throughput calculation
    size_t total_queries = query_files.size();
    double dense_throughput = (total_dense_exec_time > 0) ? (total_queries * 1000.0) / total_dense_exec_time : 0.0;
    double sparse_throughput = (total_sparse_exec_time > 0) ? (total_queries * 1000.0) / total_sparse_exec_time : 0.0;

    std::cout << "========================================\n";
    std::cout << "Throughput Results:\n";
    std::cout << "----------------------------------------\n";
    printf("  Dense method:  %.2f queries/sec (total exec time: %.3f ms)\n", dense_throughput, total_dense_exec_time);
    printf("  Sparse method: %.2f queries/sec (total exec time: %.3f ms)\n", sparse_throughput, total_sparse_exec_time);

    bool all_pass = (dense_pass_count == all_results.size()) && (sparse_pass_count == all_results.size());
    std::cout << "\n========================================\n";
    std::cout << "Final result: " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "========================================\n";

    return all_pass ? 0 : 1;
}
