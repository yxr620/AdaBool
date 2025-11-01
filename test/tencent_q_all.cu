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

constexpr uint32_t kThreadNum = 64;
constexpr uint32_t kKeyStep = 256;

// Dense union implementation
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

// Sparse union implementation
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

static double DurationMs(std::chrono::high_resolution_clock::time_point a,
                         std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
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
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
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
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
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

// Compute ground truth using CPU
std::set<uint32_t> ComputeGroundTruth(const std::vector<std::pair<uint32_t, uint32_t>>& data,
                                      const std::vector<uint32_t>& query) {
    std::set<uint32_t> query_set(query.begin(), query.end());
    std::set<uint32_t> result;
    
    for (const auto& kv : data) {
        if (query_set.count(kv.first) > 0) {
            result.insert(kv.second);
        }
    }
    
    return result;
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
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    
    // Sort files for consistent ordering
    std::sort(query_files.begin(), query_files.end());
    return query_files;
}

struct QueryResult {
    std::string query_file;
    size_t query_size;
    size_t result_size;
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
        return {0, 0, 0, 0, 0, 0, 0, 0, 0};
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
        double pos = p * (n - 1);
        size_t lower = static_cast<size_t>(pos);
        size_t upper = lower + 1;
        if (upper >= n) return data.back();
        double weight = pos - lower;
        return data[lower] * (1 - weight) + data[upper] * weight;
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

int main(int argc, char** argv) {
    std::string filter_path = "result/output/filter.coo";
    std::string query_dir = "result/output";

    if (argc >= 2) filter_path = argv[1];
    if (argc >= 3) query_dir = argv[2];

    std::cout << "Filter file: " << filter_path << std::endl;
    std::cout << "Query directory: " << query_dir << std::endl;

    // Find all query files
    auto query_files = FindQueryFiles(query_dir);
    if (query_files.empty()) {
        std::cerr << "No query files found in " << query_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << query_files.size() << " query files" << std::endl;

    // Read filter data (COO sparse matrix)
    std::vector<std::pair<uint32_t, uint32_t>> kvs;
    uint32_t max_row, max_col;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!ReadCOO(filter_path, kvs, max_row, max_col)) {
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Read filter data: " << kvs.size() << " entries, "
              << "max_row=" << max_row << ", max_col=" << max_col
              << " (" << DurationMs(t0, t1) << " ms)" << std::endl;

    // Build inverted index once
    t0 = std::chrono::high_resolution_clock::now();
    ivf_list list;
    list.BuildFrom(kvs);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Built inverted index (" << DurationMs(t0, t1) << " ms)" << std::endl;

    const uint32_t nrows = max_row + 1;
    const uint32_t ncols = max_col + 1;
    const size_t res_bytes = (ncols + 7) / 8;
    
    // Allocate result buffers
    LenBuffer<OUTPUT, uint8_t> res_dense, res_sparse;
    res_dense.Alloc(res_bytes);
    res_sparse.Alloc(res_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Statistics collection
    std::vector<QueryResult> all_results;
    all_results.reserve(query_files.size());
    
    std::vector<double> dense_prepare_times;
    std::vector<double> sparse_prepare_times;
    std::vector<double> dense_exec_times;
    std::vector<double> sparse_exec_times;
    
    int dense_pass_count = 0;
    int sparse_pass_count = 0;

    // Process each query file
    for (size_t query_idx = 0; query_idx < query_files.size(); ++query_idx) {
        const std::string& query_path = query_files[query_idx];
        std::string query_name = std::filesystem::path(query_path).filename().string();
        
        std::cout << "\n[" << (query_idx + 1) << "/" << query_files.size() 
                  << "] Processing " << query_name << std::endl;

        QueryResult result;
        result.query_file = query_name;

        // Read query data
        std::vector<uint32_t> query_list;
        if (!ReadQuery(query_path, query_list)) {
            std::cerr << "Failed to read query file: " << query_path << std::endl;
            continue;
        }
        result.query_size = query_list.size();

        // Compute ground truth for correctness checking
        auto ground_truth = ComputeGroundTruth(kvs, query_list);
        result.result_size = ground_truth.size();

        // Prepare query bitmap for dense method
        std::vector<uint8_t> query_bitmap((nrows + 7) / 8, 0);
        for (uint32_t idx : query_list) {
            if (idx < nrows) {
                query_bitmap[idx / 8] |= (1 << (idx % 8));
            }
        }

        // Prepare and execute dense method
        t0 = std::chrono::high_resolution_clock::now();
        auto dense_worker = list.Prepare(query_bitmap.size(), query_bitmap.data());
        t1 = std::chrono::high_resolution_clock::now();
        result.prepare_dense_ms = DurationMs(t0, t1);

        cudaMemsetAsync(res_dense.d, 0, res_dense.len, stream);
        t0 = std::chrono::high_resolution_clock::now();
        ListUnion_dense(*dense_worker, reinterpret_cast<uint32_t*>(res_dense.d), stream);
        cudaStreamSynchronize(stream);
        t1 = std::chrono::high_resolution_clock::now();
        result.dense_exec_ms = DurationMs(t0, t1);

        // Prepare and execute sparse method
        t0 = std::chrono::high_resolution_clock::now();
        auto sparse_worker = list.Prepare(query_list);
        t1 = std::chrono::high_resolution_clock::now();
        result.prepare_sparse_ms = DurationMs(t0, t1);

        cudaMemsetAsync(res_sparse.d, 0, res_sparse.len, stream);
        t0 = std::chrono::high_resolution_clock::now();
        ListUnion_sparse(*sparse_worker, reinterpret_cast<uint32_t*>(res_sparse.d), stream);
        cudaStreamSynchronize(stream);
        t1 = std::chrono::high_resolution_clock::now();
        result.sparse_exec_ms = DurationMs(t0, t1);

        // Correctness check for dense
        cudaMemcpy(res_dense.h, res_dense.d, res_dense.len * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        std::vector<uint32_t> dense_results;
        CollectIdsFromBitmap(res_dense.h, res_dense.len, dense_results);
        std::set<uint32_t> dense_set(dense_results.begin(), dense_results.end());
        result.dense_correct = (dense_set == ground_truth);

        // Correctness check for sparse
        cudaMemcpy(res_sparse.h, res_sparse.d, res_sparse.len * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        std::vector<uint32_t> sparse_results;
        CollectIdsFromBitmap(res_sparse.h, res_sparse.len, sparse_results);
        std::set<uint32_t> sparse_set(sparse_results.begin(), sparse_results.end());
        result.sparse_correct = (sparse_set == ground_truth);

        // Collect statistics
        dense_prepare_times.push_back(result.prepare_dense_ms);
        sparse_prepare_times.push_back(result.prepare_sparse_ms);
        dense_exec_times.push_back(result.dense_exec_ms);
        sparse_exec_times.push_back(result.sparse_exec_ms);
        
        if (result.dense_correct) dense_pass_count++;
        if (result.sparse_correct) sparse_pass_count++;

        all_results.push_back(result);
        
        std::cout << "  Query size: " << result.query_size 
                  << ", Result size: " << result.result_size << std::endl;
        std::cout << "  Dense:  prepare=" << result.prepare_dense_ms 
                  << " ms, exec=" << result.dense_exec_ms 
                  << " ms, correct=" << (result.dense_correct ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Sparse: prepare=" << result.prepare_sparse_ms 
                  << " ms, exec=" << result.sparse_exec_ms 
                  << " ms, correct=" << (result.sparse_correct ? "PASS" : "FAIL") << std::endl;
    }

    // Cleanup
    cudaStreamDestroy(stream);

    // Summary report
    std::cout << "\n========================================\n";
    std::cout << "           SUMMARY REPORT\n";
    std::cout << "========================================\n";
    std::cout << "Total queries processed: " << all_results.size() << "\n\n";

    // Correctness summary
    std::cout << "Correctness Summary:\n";
    std::cout << "  Dense method:  " << dense_pass_count << "/" << all_results.size() << " PASS\n";
    std::cout << "  Sparse method: " << sparse_pass_count << "/" << all_results.size() << " PASS\n\n";

    // Dense statistics
    std::cout << "========================================\n";
    std::cout << "Dense Method - Prepare Time:\n";
    PrintStats("", CalculateStats(dense_prepare_times));
    std::cout << "\nDense Method - Execution Time:\n";
    PrintStats("", CalculateStats(dense_exec_times));

    // Sparse statistics
    std::cout << "\n========================================\n";
    std::cout << "Sparse Method - Prepare Time:\n";
    PrintStats("", CalculateStats(sparse_prepare_times));
    std::cout << "\nSparse Method - Execution Time:\n";
    PrintStats("", CalculateStats(sparse_exec_times));

    // Detailed table
    std::cout << "\n========================================\n";
    std::cout << "Detailed Results:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Query                  | QSize    | RSize    | Dense Prep | Dense Exec | Sparse Prep | Sparse Exec | D_Correct | S_Correct\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------\n";
    for (const auto& r : all_results) {
        printf("%-22s | %8zu | %8zu | %10.3f | %10.3f | %11.3f | %11.3f | %-9s | %-9s\n",
               r.query_file.c_str(), r.query_size, r.result_size,
               r.prepare_dense_ms, r.dense_exec_ms,
               r.prepare_sparse_ms, r.sparse_exec_ms,
               r.dense_correct ? "PASS" : "FAIL",
               r.sparse_correct ? "PASS" : "FAIL");
    }

    bool all_pass = (dense_pass_count == all_results.size()) && (sparse_pass_count == all_results.size());
    std::cout << "\n========================================\n";
    std::cout << "Final result: " << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    std::cout << "========================================\n";

    return all_pass ? 0 : 1;
}
