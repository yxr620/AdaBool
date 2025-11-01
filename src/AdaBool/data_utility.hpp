#pragma once

#include <utility>
#include <map>
#include <random>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>


std::tuple<uint64_t, std::multimap<uint32_t, uint32_t>> GenLargeDataset(
    std::mt19937_64 *rngp, uint64_t num_elems, uint64_t max_elems, uint64_t sparsity=10){
    auto &rng = *rngp;
    std::multimap<uint32_t, uint32_t> kvs;
    uint64_t cnt = 0;
    uint64_t max_key = 0;
    std::vector<uint32_t> rand_keys(max_elems);
    for (size_t i = 0; i < rand_keys.size(); ++i)
    {
        rand_keys[i] = static_cast<uint32_t>(i);
    }
    std::uniform_int_distribution<uint32_t> len_dist(1, std::max<uint64_t>(1, max_elems / sparsity));
    for (uint32_t i = 0; i < num_elems; ++i)
    {
        uint32_t len = len_dist(rng);
        cnt += len;
        // std::cout<<"Key: "<<i<<", Len: "<<len<<"cnt: "<<cnt<<std::endl;
        if (cnt >= num_elems)
        {
            max_key = i;
            break;
        }
        for (uint32_t j = 0; j < len; ++j)
        {
            if (j >= max_elems) break;
            std::uniform_int_distribution<uint32_t> dist(j, static_cast<uint32_t>(max_elems - 1));
            uint32_t r = dist(rng);
            std::swap(rand_keys[j], rand_keys[r]);
            kvs.emplace(i, rand_keys[j]);
        }
    }
    return {max_key, kvs};
}

// Zipfian分布辅助类
// 用于生成服从Zipfian分布的随机数，常用于模拟真实世界中的数据倾斜场景
class ZipfianDistribution {
private:
    double alpha_;      // 分布参数：控制偏斜程度，值越大分布越偏斜
    double eta_;        // 归一化系数：1 / theta_
    double theta_;      // 归一化常数：所有项的概率之和
    uint32_t n_;        // 分布范围：[1, n_]
    std::uniform_real_distribution<double> uniform_;  // 均匀分布生成器，用于采样

public:
    // 构造函数
    // @param n: 分布的最大值，生成范围为 [1, n]
    // @param alpha: 偏斜参数，默认1.0（标准Zipf分布）
    //               alpha > 1: 更偏向小值
    //               alpha < 1: 更均匀分布
    ZipfianDistribution(uint32_t n, double alpha = 1.0) 
        : alpha_(alpha), n_(n), uniform_(0.0, 1.0) {
        // 预计算归一化常数 theta_ = sum(1/i^alpha, i=1..n)
        theta_ = 0.0;
        for (uint32_t i = 1; i <= n_; ++i) {
            theta_ += 1.0 / std::pow(static_cast<double>(i), alpha_);
        }
        eta_ = 1.0 / theta_;  // 归一化系数
    }

    // 生成一个服从Zipfian分布的随机数
    // 使用逆变换采样方法：生成均匀分布的u，然后找到对应的i使得累积概率>=u
    template<typename Generator>
    uint32_t operator()(Generator& gen) {
        double u = uniform_(gen);  // 生成[0,1)的均匀随机数
        double sum = 0.0;
        // 累积概率法：遍历直到累积概率超过u
        for (uint32_t i = 1; i <= n_; ++i) {
            sum += eta_ / std::pow(static_cast<double>(i), alpha_);
            if (sum >= u) {
                return i;  // 返回对应的值
            }
        }
        return n_;  // 边界情况返回最大值
    }
};

// 使用Zipfian分布生成大规模稀疏数据集
// 与GenLargeDataset的区别：行长度（每行的非零元素数量）服从Zipfian分布而非均匀分布
//
// @param rngp: 随机数生成器指针
// @param num_elems: 目标总元素数量（所有行的非零元素总和）
// @param max_elems: 列的最大数量（值域范围 [0, max_elems)）
// @param sparsity: 稀疏度参数，行长度最大值 = max_elems / sparsity
// @param alpha: Zipfian分布的偏斜参数，默认1.0
//               - alpha=1.0: 标准Zipf分布，符合80-20规则
//               - alpha>1.0: 更偏斜，大部分行长度很小
//               - alpha<1.0: 更均匀，行长度分布较平均
//
// @return: tuple<max_key, kvs>
//          - max_key: 实际生成的最大行号
//          - kvs: multimap存储(行号, 列号)键值对
std::tuple<uint64_t, std::multimap<uint32_t, uint32_t>> GenLargeDataset_Zipfian(
    std::mt19937_64 *rngp, uint64_t num_elems, uint64_t max_elems, uint64_t sparsity=10, double alpha=1.0){
    auto &rng = *rngp;
    std::multimap<uint32_t, uint32_t> kvs;  // 存储(行号, 列号)对
    uint64_t cnt = 0;       // 已生成的元素总数
    uint64_t max_key = 0;   // 最大行号
    std::vector<uint32_t> rand_keys(max_elems);  // 列索引池，用于随机采样
    
    // 初始化列索引池：[0, 1, 2, ..., max_elems-1]
    for (size_t i = 0; i < rand_keys.size(); ++i) {
        rand_keys[i] = static_cast<uint32_t>(i);
    }
    
    // 创建Zipfian分布生成器用于行长度
    // 行长度范围：[1, max_elems/sparsity]
    uint32_t max_len = std::max<uint64_t>(1, max_elems / sparsity);
    ZipfianDistribution len_dist(max_len, alpha);
    
    // 逐行生成数据，直到达到目标元素数量
    for (uint32_t i = 0; i < num_elems; ++i) {
        uint32_t len = len_dist(rng);  // 从Zipfian分布采样当前行的非零元素数量
        cnt += len;  // 累加元素计数
        
        // 如果已达到或超过目标元素数量，停止生成
        if (cnt >= num_elems) {
            max_key = i;  // 记录最大行号
            break;
        }
        
        // 为当前行生成len个随机列索引
        // 使用Fisher-Yates洗牌算法的部分洗牌版本确保无重复
        for (uint32_t j = 0; j < len; ++j)
        {
            if (j >= max_elems) break; 
            // 从[j, max_elems-1]中随机选择一个索引
            std::uniform_int_distribution<uint32_t> dist(j, static_cast<uint32_t>(max_elems - 1));
            uint32_t r = dist(rng);
            // 交换，保证rand_keys[0..j-1]中的元素不会被重复选中
            std::swap(rand_keys[j], rand_keys[r]);
            // 插入(行号i, 列号rand_keys[j])
            kvs.emplace(i, rand_keys[j]);
        }
    }
    
    return {max_key, kvs};
}

std::tuple<uint64_t, std::multimap<uint32_t, uint32_t>> GenLargeDatasetMT(
    std::mt19937_64 *rngp, uint64_t num_elems, uint64_t max_elems, uint64_t sparsity=10){
    // --- Phase 1: sequentially decide lengths per key (preserves original early-stop logic) ---
    // Original behavior: generate len, update cnt; if cnt >= num_elems -> set max_key to i and break WITHOUT inserting pairs for that key.
    auto &rng = *rngp;
    std::uniform_int_distribution<uint32_t> len_dist(1, std::max<uint64_t>(1, max_elems / sparsity));

    std::vector<uint32_t> lengths; lengths.reserve(1024);
    uint64_t cnt = 0; uint64_t max_key = 0;
    for (uint32_t i = 0; i < num_elems; ++i) {
        uint32_t len = len_dist(rng);
        cnt += len;
        if (cnt >= num_elems) { // stop; do not record length for this key (mirrors original)
            max_key = i;
            break;
        }
        lengths.push_back(len);
    }

    if (lengths.empty()) {
        // Degenerate case: num_elems very small; return empty map, max_key already set.
        return {max_key, {}};
    }

    const uint32_t key_count = static_cast<uint32_t>(lengths.size()); // keys we actually materialize

    // --- Phase 2: parallel generation of (key, value) pairs ---
    // We split key range into simple contiguous chunks; each thread keeps a local partially shuffled index array.
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4; // fallback
    unsigned thread_count = std::min<unsigned>(hw, key_count);
    std::cout<<"hw: "<<hw<<", thread_count: "<<thread_count<<" key_count: "<<key_count<<std::endl;
    if (thread_count <= 1 || key_count < 1024) { // small case: fall back to original single-thread algorithm (re-implemented here)
        std::multimap<uint32_t, uint32_t> kvs;
        std::vector<uint32_t> rand_keys(max_elems);
        for (uint32_t i = 0; i < max_elems; ++i) rand_keys[i] = i;
        // reuse a new rng seeded from original (deterministic relative to earlier draws)
        std::mt19937_64 rng_local(rng());
        for (uint32_t key = 0; key < key_count; ++key) {
            uint32_t len = lengths[key];
            for (uint32_t j = 0; j < len; ++j) {
                if (j >= max_elems) break;
                std::uniform_int_distribution<uint32_t> dist(j, static_cast<uint32_t>(max_elems - 1));
                uint32_t r = dist(rng_local);
                std::swap(rand_keys[j], rand_keys[r]);
                kvs.emplace(key, rand_keys[j]);
            }
        }
        return {max_key, kvs};
    }

    struct ThreadResult { std::vector<std::pair<uint32_t,uint32_t>> kvs; };
    std::vector<ThreadResult> results(thread_count);
    std::vector<std::thread> threads; threads.reserve(thread_count);

    // Pre-generate seeds for deterministic-ish behavior (not strictly matching original sequence, but acceptable)
    std::vector<uint64_t> seeds(thread_count);
    for (unsigned t = 0; t < thread_count; ++t) seeds[t] = rng() ^ (0x9e3779b97f4a7c15ULL + t * 0x5851f42d4c957f2dULL);

    auto worker_fn = [&](unsigned tid){
        // std::cout<<"Thread "<<tid<<" started.\n";
        uint32_t start = (key_count * tid) / thread_count;
        uint32_t end   = (key_count * (tid + 1)) / thread_count;
        if (start >= end) return; // no work
        std::mt19937_64 local_rng(seeds[tid]);
        std::vector<uint32_t> rand_keys(max_elems);
        for (uint32_t i = 0; i < max_elems; ++i) rand_keys[i] = i;
        auto &out = results[tid].kvs;
        // Reserve approximate space (heuristic)
        uint64_t local_tot = 0; for (uint32_t k = start; k < end; ++k) local_tot += lengths[k];
        out.reserve(static_cast<size_t>(std::min<uint64_t>(local_tot, local_tot))); // keep as-is
        for (uint32_t key = start; key < end; ++key) {
            uint32_t len = lengths[key];
            for (uint32_t j = 0; j < len; ++j) {
                if (j >= max_elems) break;
                std::uniform_int_distribution<uint32_t> dist(j, static_cast<uint32_t>(max_elems - 1));
                uint32_t r = dist(local_rng);
                std::swap(rand_keys[j], rand_keys[r]);
                out.emplace_back(key, rand_keys[j]);
            }
        }
    };

    for (unsigned t = 0; t < thread_count; ++t) threads.emplace_back(worker_fn, t);
    for (auto &th : threads) th.join();
    std::cout<<"All threads joined.\n";


    // Merge results
    std::multimap<uint32_t, uint32_t> kvs;
    // Reserve hint: multimap has no reserve; we could accumulate into vector then sort, but keep simple.
    for (unsigned t = 0; t < thread_count; ++t) {
        // std::cout<<"Thread "<<t<<" finished.\n";
        for (auto &pr : results[t].kvs) kvs.emplace(pr.first, pr.second);
    }

    return {max_key, kvs};
}

template <typename KVContainer>
static void BuildRowIndex(
    const KVContainer& kvs,
    uint32_t max_row,
    std::vector<std::vector<uint32_t>>& rows_out) {
    rows_out.assign(max_row + 1, {});
    for (auto &kv : kvs) {
        if (kv.first <= max_row)
            rows_out[kv.first].push_back(kv.second);
        // std::cout<< kv.first << "," << kv.second << "\n";
    }
    for (auto &r : rows_out)
        std::sort(r.begin(), r.end());
}


// New: print full columns (no truncation) for the first kPrintMaxRows non-empty rows
static void PrintMatrixFirstRowsFull(const std::vector<std::vector<uint32_t>>& rows) {
    // print full matrix
    for (size_t r = 0; r < rows.size() && r < 100; ++r) {
        if (rows[r].empty()) continue;
        std::cout << "R" << r << " -> [";
        for (size_t i = 0; i < rows[r].size(); ++i) {
            std::cout << rows[r][i];
            if (i + 1 != rows[r].size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

static void PrintQueryBitmap(const std::vector<uint8_t>& bitmap, uint32_t batch_size = 1) {
    int kQueryBitmapPreview = 100;
    std::cout << "\n--- Query Bitmap Summary ---\n";
    if (batch_size == 0) return;
    const size_t total_bits_all = bitmap.size() * 8;
    const size_t bits_per_item = total_bits_all / batch_size;
    const size_t bytes_per_item = bitmap.size() / batch_size;

    for (uint32_t i = 0; i < batch_size; ++i) {
        if (batch_size > 1) {
            std::cout << "--- Batch item " << i << " ---\n";
        }
        size_t set_bits = 0;
        std::vector<uint32_t> first_ids;
        first_ids.reserve(kQueryBitmapPreview);
        const size_t start_byte = i * bytes_per_item;
        const size_t end_byte = start_byte + bytes_per_item;

        for (size_t byte_i = start_byte; byte_i < end_byte; ++byte_i) {
            uint8_t b = bitmap[byte_i];
            while (b) {
                unsigned lsb = __builtin_ctz(static_cast<unsigned>(b));
                uint32_t id = static_cast<uint32_t>((byte_i - start_byte) * 8 + lsb);
                ++set_bits;
                if (first_ids.size() < kQueryBitmapPreview)
                    first_ids.push_back(id);
                b &= (b - 1);
            }
        }
        std::cout << "Total bits: " << bits_per_item
                  << ", Set bits: " << set_bits
                  << ", Preview (up to " << kQueryBitmapPreview << "): \n";
        for (auto v : first_ids) std::cout << v << ' ';
        if (first_ids.empty()) std::cout << "(none)";
        std::cout << "\n";
    }
}


std::vector<uint8_t> GenResData(const int kBatchSize, const uint64_t kPaddedMaxElems, std::vector<std::vector<uint32_t>> keyss, std::multimap<uint32_t, uint32_t> kvs){
    std::vector<uint8_t> gt_bitmap(kBatchSize * kPaddedMaxElems / 8, 0);

    // Build ground truth on CPU
    for (int qidx = 0; qidx < kBatchSize; ++qidx) {
        for (auto key : keyss[qidx]) {
            auto [beg, end] = kvs.equal_range(key);
            for (auto it = beg; it != end; ++it) {
                uint32_t v = it->second;
                size_t offset = v / 8 + qidx * (kPaddedMaxElems / 8);
                gt_bitmap[offset] |= 1u << (v % 8);
            }
        }
    }
    return gt_bitmap;
}

// 生成Zipfian分布的稀疏矩阵（行长度服从幂律分布）
// @param rows, cols: 矩阵维度
// @param density: 目标密度 (0,1]，实际nnz ≈ density*rows*cols (±10%误差)
// @param alpha: 偏斜度 (1.0=标准Zipf, >1更偏斜, <1更均匀)
// @return: (max_row_id, multimap<row, col>)
// 优化：自适应max_len估算 + 三阶段动态缩放（10%预热, 80%周期调整, 10%精确控制）
std::tuple<uint64_t, std::multimap<uint32_t, uint32_t>> GenLargeDataset_Zipfian_rows(
    std::mt19937_64 *rngp, uint64_t rows, uint64_t cols, double density, double alpha=1.0){
    
    if (density <= 0.0 || density > 1.0) {
        std::cerr << "Warning: density should be in (0, 1], got " << density << std::endl;
        density = std::min(1.0, std::max(1e-6, density));
    }
    
    auto &rng = *rngp;
    std::multimap<uint32_t, uint32_t> kvs;
    
    uint64_t target_nnz = static_cast<uint64_t>(density * rows * cols);
    if (target_nnz == 0) target_nnz = 1;
    
    std::vector<uint32_t> rand_keys(cols);
    for (size_t i = 0; i < rand_keys.size(); ++i) {
        rand_keys[i] = static_cast<uint32_t>(i);
    }
    
    double avg_len = static_cast<double>(target_nnz) / rows;
    
    // 估算max_len使得E[Zipfian(max_len,α)] ≈ avg_len
    // α=1: E[X]≈n/(ln(n)+0.5772), 反解迭代求n
    // α≠1: 使用经验校准因子
    uint32_t max_len;
    if (std::abs(alpha - 1.0) < 0.01) {
        // α≈1: 迭代求解 n = avg_len*(ln(n)+0.5772)
        double n_est = avg_len * 2.0;
        for (int i = 0; i < 3; ++i) {
            n_est = avg_len * (std::log(n_est) + 0.5772);
        }
        max_len = std::max<uint32_t>(1, static_cast<uint32_t>(n_est * 1.1));
    } else {
        // α≠1: 经验校准因子
        double calibration_factor = (alpha > 1.0) ? 
            (2.0 + 0.5 * (alpha - 1.0)) : 
            (2.0 - 0.5 * (1.0 - alpha));
        max_len = std::max<uint32_t>(1, static_cast<uint32_t>(avg_len * calibration_factor));
    }
    
    max_len = std::min<uint32_t>(max_len, static_cast<uint32_t>(cols));
    
    ZipfianDistribution len_dist(max_len, alpha);
    
    uint64_t total_generated = 0;
    const uint32_t checkpoint_interval = rows / 20;  // 每5%调整
    const uint32_t adjustment_start = rows / 10;     // 10%开始
    const uint32_t precise_start = static_cast<uint32_t>(rows * 0.9);  // 90%精确控制
    const double target_ratio = static_cast<double>(target_nnz) / rows;
    double scale_factor = 1.0;
    
    // 三阶段生成: 10%预热 -> 80%周期调整 -> 10%精确控制
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t len = len_dist(rng);
        
        // 阶段2: 周期性调整缩放因子
        if (row > adjustment_start && row % checkpoint_interval == 0) {
            double actual_ratio = static_cast<double>(total_generated) / row;
            scale_factor = std::max(0.7, std::min(1.5, target_ratio / actual_ratio));
        }
        
        if (scale_factor != 1.0 && row > adjustment_start) {
            len = static_cast<uint32_t>(len * scale_factor);
        }
        
        // 阶段3: 精确控制（Zipfian与理想值加权平均）
        if (row >= precise_start) {
            uint64_t remaining_rows = rows - row;
            if (remaining_rows > 0 && total_generated < target_nnz) {
                uint64_t remaining_target = target_nnz - total_generated;
                uint32_t ideal_len = static_cast<uint32_t>(remaining_target / remaining_rows);
                double weight = static_cast<double>(row - precise_start) / (rows - precise_start);
                len = static_cast<uint32_t>(len * (1.0 - weight) + ideal_len * weight);
            }
        }
        
        if (len == 0) continue;
        len = std::min(len, static_cast<uint32_t>(cols));
        
        // Fisher-Yates洗牌生成不重复列索引
        for (uint32_t j = 0; j < len; ++j) {
            if (j >= cols) break;
            std::uniform_int_distribution<uint32_t> dist(j, static_cast<uint32_t>(cols - 1));
            uint32_t r = dist(rng);
            std::swap(rand_keys[j], rand_keys[r]);
            kvs.emplace(row, rand_keys[j]);
            total_generated++;
        }
    }
    
    // 密度偏差检查
    double actual_density = static_cast<double>(total_generated) / (rows * cols);
    double density_error = std::abs(actual_density - density) / density * 100.0;
    
    if (density_error > 10.0) {
        std::cerr << "Warning: Density deviation is " << density_error << "%. "
                  << "Target: " << density << ", Actual: " << actual_density 
                  << " (generated " << total_generated << " out of target " << target_nnz << ")" << std::endl;
    }
    
    uint64_t max_key = rows - 1;
    return {max_key, kvs};
}