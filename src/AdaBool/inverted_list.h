#pragma once

#include <vector>
#include <utility>
#include <map>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "len_buffer.h"
#include "load_balance_transformer.h"
// #include "key_transformer.cu.h"

enum MixedListArrayType
{
    LIST_ARRAY1 = 0,
    LIST_ARRAY4 = 1,
    LIST_ARRAY8 = 2,
    LIST_ARRAY16 = 3,
    LIST_ARRAY32 = 4,
    LIST_BITSET = 5,

    MAX_ARRAY_TYPE,
};

struct KeyBlockMeta
{
    uint64_t len;
    uint32_t block_step;
    LenBuffer<INPUT, uint32_t> key_pos_segments;
    LenBuffer<INPUT, uint32_t> key_block_segments;
    LenBuffer<INPUT, uint8_t> key_buffers;
};

/**
 * @brief Build per-block compressed key metadata used mainly by bitmap (dense) query.
 *
 * Input: a sorted key array (length = len). We partition keys into consecutive
 * blocks of size "block_step" (except the last which may be smaller).
 * For each block we record:
 *   1) A block start key list (key_block_segments): block_seg[i] is the first
 *      key of block i; we also append a sentinel last key (the last element in
 *      the whole key array) so size = #blocks + 1. Thus #blocks = block_seg.size()-1.
 *   2) For each block we decide the minimal number of bytes (1 / 2 / 4) required
 *      to represent any key in that block relative to the block start key. The
 *      decision is based on span = block_seg[i+1] - block_seg[i]:
 *          span < 2^8  -> 1 byte per key offset
 *          span < 2^16 -> 2 bytes per key offset
 *          else        -> 4 bytes per key offset
 *   3) We write the (key - block_start) offsets sequentially (little-endian) to
 *      key_buffers. For each block we only write "num_elem * key_bytes" *effective*
 *      bytes (num_elem is actual key count in that block) and then pad the block
 *      with zeros so that every block occupies exactly (block_step * key_bytes)
 *      bytes in key_buffers. This fixed-size stride per block simplifies GPU side
 *      addressing even though the last block and/or smaller blocks waste a little
 *      space.
 *   4) Because of the padding, "useful" bytes per block differ from the physical
 *      allocated bytes. We therefore maintain an exclusive prefix array
 *      key_pos_segments (pos_seg) whose size equals (block_seg.size()). For block i
 *      the useful offsets (pos_seg[i] .. pos_seg[i+1]) cover exactly
 *      num_elem * key_bytes bytes. pos_seg never counts padding bytes. The total
 *      useful (non-padding) bytes is pos_seg.back().
 *
 * Filled fields in KeyBlockMeta:
 *   meta->len                : total number of keys
 *   meta->block_step         : block granularity used
 *   meta->key_block_segments : block start key list + sentinel last key
 *   meta->key_pos_segments   : prefix sums of useful bytes (size = #blocks + 1)
 *   meta->key_buffers        : padded per-block key offset storage (little-endian)
 *
 * Example layout (block_step=4):
 *   keys: [100,101,255,260,  500,510]
 *   block 0 start=100 span to next start=500 => span=400 -> key_bytes=2
 *     num_elem=4 -> store offsets [0,1,155,160] (8 bytes) then pad to 4*2=8 (no pad)
 *   block 1 start=500 (last, sentinel=510) span=10 -> key_bytes=1
 *     num_elem=2 -> store [0,10] (2 bytes) pad to 4*1=4 bytes (adds 2 zeros)
 *   key_pos_segments: [0, 8, 10]  (useful bytes only)
 *   key_block_segments: [100,500,510]
 *   key_buffers size: 8 + 4 = 12 (includes padding)
 *
 * NOTE: Caller is responsible for freeing previous buffers (LenBuffer::AllocFrom handles it).
 *
 * @param block_step   Target maximum keys per block (stride granularity).
 * @param len          Number of keys.
 * @param keys         Pointer to sorted key array (ascending). If unsorted behavior is undefined.
 * @param meta         Output structure populated in-place.
 */
void ComputeKeyBlockMeta(int block_step, uint64_t len, const uint32_t *keys, KeyBlockMeta *meta)
{
    meta->len = len;
    meta->block_step = block_step;
    if (len == 0)
    {
        return;
    }
    std::vector<uint32_t> block_seg;
    std::vector<uint32_t> pos_seg;
    std::vector<uint8_t> key_buffer;
    for (uint64_t i = 0; i < len; i += block_step)
    {
        block_seg.push_back(keys[i]);
    }
    block_seg.push_back(keys[len - 1]);
    pos_seg.push_back(0);
    auto FillBuffer = [&](uint32_t beg, int size, const uint32_t *ks, int s)
    {
        union
        {
            uint32_t u32;
            uint8_t u8[4];
        } k;
        for (int i = 0; i < size; ++i)
        {
            k.u32 = ks[i] - beg;
            for (int j = 0; j < s; ++j)
            {
                key_buffer.push_back(k.u8[j]);
            }
        }
        // Padding key_buffer so its size is a multiplier of block_step
        key_buffer.resize(key_buffer.size() + (block_step - size) * s, 0);
    };
    for (size_t i = 0; i < block_seg.size() - 1; ++i)
    {
        int span = block_seg[i + 1] - block_seg[i];
        int num_elem = std::min<int>(block_step, len - i * block_step);
        int key_bytes;
        if (span < 256)
        {
            key_bytes = 1;
        }
        else if (span < 65536)
        {
            key_bytes = 2;
        }
        else
        {
            key_bytes = 4;
        }
        pos_seg.push_back(pos_seg.back() + num_elem * key_bytes);
        FillBuffer(block_seg[i], num_elem, keys + i * block_step, key_bytes);
    }
    meta->key_pos_segments.AllocFrom(pos_seg);
    meta->key_block_segments.AllocFrom(block_seg);
    meta->key_buffers.AllocFrom(key_buffer);
}

static const char *ArrayTypeName(int t)
{
    switch (t)
    {
    case LIST_ARRAY1:
        return "LIST_ARRAY1";
    case LIST_ARRAY4:
        return "LIST_ARRAY4";
    case LIST_ARRAY8:
        return "LIST_ARRAY8";
    case LIST_ARRAY16:
        return "LIST_ARRAY16";
    case LIST_ARRAY32:
        return "LIST_ARRAY32";
    case LIST_BITSET:
        return "LIST_BITSET";
    default:
        return "UNKNOWN";
    }
}

struct InvertedListWorker
{
    struct ArrayTask
    {
        LenBuffer<INPUT, uint32_t> indices_beg;
        LenBuffer<INPUT, uint32_t> segments;
        uint32_t *d_value_headers;
        uint8_t *d_value_arrays;
        LoadBalanceTransformer worker;
    };
    std::array<ArrayTask, MAX_ARRAY_TYPE> tasks;

    void Prepare()
    {
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
        {
            tasks[i].worker.Prepare(tasks[i].segments.len - 1, tasks[i].segments.h, 256);
        }
    }

    template <class F, class... Args>
    void Run(cudaStream_t stream, const F &f, Args... args);

    template <int kArrayType, class F, class... Args>
    void RunArrayImpl(cudaStream_t stream, const F &f, Args... args);

    template <class F, class... Args>
    void RunArrays(cudaStream_t stream, const F &f, Args... args);

    template <class F, class... Args>
    void RunBitsets(cudaStream_t stream, const F &f, Args... args);

    /**
     * @brief Debug print current prepared tasks for each array type.
     *        输出每个类型的：query段数量、子任务(work unit)数量、前5个 segments 前缀、
     *        前5个 indices_beg、前8个 (seg_id, offset, len) 拆分结果，方便核对 LBS 切分是否符合预期。
     * @note 仅访问 host 侧缓存 (LenBuffer.h) 中的 h 指针，不触及 device 数据。
     */
    void DebugPrint() const
    {
        std::cout << "===== InvertedListWorker Debug =====\n";
        for (int t = 0; t < MAX_ARRAY_TYPE; ++t)
        {
            const auto &task = tasks[t];
            size_t num_query_segments = task.segments.len ? task.segments.len - 1 : 0; // 原始请求 key 的段数量
            size_t num_work_units = task.worker.seg_ids.len;                           // 负载均衡后子任务数量
            std::cout << "Type=" << ArrayTypeName(t)
                      << " (#querySegs=" << num_query_segments
                      << ", #workUnits=" << num_work_units << ")\n";
            // segments 前几个值
            std::cout << "  segments prefix: ";
            for (size_t i = 0; i < std::min<size_t>(task.segments.len, 6); ++i)
                std::cout << task.segments.h[i] << (i + 1 < std::min<size_t>(task.segments.len, 6) ? ',' : '\0');
            if (task.segments.len > 6)
                std::cout << "...";
            std::cout << "\n  indices_beg first: ";
            for (size_t i = 0; i < std::min<size_t>(task.indices_beg.len, 6); ++i)
                std::cout << task.indices_beg.h[i] << (i + 1 < std::min<size_t>(task.indices_beg.len, 6) ? ',' : '\0');
            if (task.indices_beg.len > 6)
                std::cout << "...";
            std::cout << "\n  work units (seg_id:offset+len): ";
            for (size_t i = 0; i < std::min<size_t>(task.worker.seg_ids.len, 8); ++i)
            {
                std::cout << task.worker.seg_ids.h[i] << ':'
                          << task.worker.seg_offsets.h[i] << '+'
                          << task.worker.seg_lens.h[i];
                if (i + 1 < std::min<size_t>(task.worker.seg_ids.len, 8))
                    std::cout << ' ';
            }
            if (task.worker.seg_ids.len > 8)
                std::cout << " ...";
            std::cout << "\n";
        }
        std::cout << "====================================\n";
    }
};

template <int kKeyStep = 256, int kThreadNum = 64>
struct InvertedListBitmapWorkerImpl
{
    LenBuffer<INPUT, uint32_t> query_bitmap;
    // Prepared dense worker
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *keys{nullptr};
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *key_block_segments{nullptr};
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *headers{nullptr};
    const std::array<LenBuffer<INPUT, uint8_t>, MAX_ARRAY_TYPE> *arrays{nullptr};
    const std::array<KeyBlockMeta, MAX_ARRAY_TYPE> *metas{nullptr};

    template <class State>
    void Prepare(State *state) {}

    template <class F, class... Args>
    void Run(cudaStream_t stream, const F &f, Args... args);

    template <int kArrayType, class F, class... Args>
    void RunArrayImpl(cudaStream_t stream, const F &f, Args... args);

    template <class F, class... Args>
    void RunArrays(cudaStream_t stream, const F &f, Args... args);

    template <class F, class... Args>
    void RunBitsets(cudaStream_t stream, const F &f, Args... args);
};

// Bitmap worker with batch. Call with device callback f(key, value, qidx, args...)
template <int kKeyStep = 256, int kThreadNum = 64>
struct InvertedListBatchWorkerImpl
{
    int batch_size;
    int query_bitmap_len; // 每一个query的bitmap长度，以uint32为单位，后期会使用uint4，因此要128对齐每个query
    const uint32_t *d_query_bitmap; // 实际的请求的bitmap，全部batch的query拼接在一起
    LenBuffer<INPUT, uint32_t> query_bitmap; // 好像是没有用的代码
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *keys;
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *key_block_segments;
    const std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> *headers;
    const std::array<LenBuffer<INPUT, uint8_t>, MAX_ARRAY_TYPE> *arrays;
    const std::array<KeyBlockMeta, MAX_ARRAY_TYPE> *metas;

    template <class State>
    void Prepare(State *state) {}

    template <class F, class... Args>
    void Run(cudaStream_t stream, const F &f, Args... args);

    // template <int kBatchSize, int kArrayType, class F, class... Args>
    // void RunArrayImpl(cudaStream_t stream, const F &f, Args... args);

    // template <int kBatchSize, class F, class... Args>
    // void RunArraysImpl(cudaStream_t stream, const F &f, Args... args);
    template <int kBatchSize>
    void RunArraysImpl(cudaStream_t stream, uint32_t* d_res, uint32_t res_len);

    template <int kBatchSize, class F, class... Args>
    void RunBitsetsImpl(cudaStream_t stream, const F &f, Args... args);

    // template <class F, class... Args>
    // void RunArrays(cudaStream_t stream, const F &f, Args... args);
    void RunArrays(cudaStream_t stream, uint32_t* d_res, uint32_t res_len);

    // template <class F, class... Args>
    // void RunBitsets(cudaStream_t stream, const F &f, Args... args);
    void RunBitsets(cudaStream_t stream);
};

template <int kKeyStep = 256, int kThreadNum = 64>
class ivf_list
{
public:
    /**
     * @brief 构造函数，初始化各类型倒排数组的开关。
     *        array_switches_ 数组用于控制不同 LIST_ARRAY* / LIST_BITSET 类型是否启用。
     */
    ivf_list() { array_switches_ = {true, true, true, true, true, true}; }
    /**
     * @brief 由 (key, value) 对集合构建混合形式的倒排列表结构。
     *
     * 输入数据格式：xs 中的每个 pair<int,int> 视为一个 32bit 值 value 归属于一个 key：
     *   - first  : key (逻辑“主键”，用于 key_segments_ 统计)
     *   - second : 32bit 原值，将其拆分成 (高 24bit | 低 8bit)。
     *              高 24bit 作为 header 高位 (对齐到 0xffffff00)，低 8bit 作为列表中的元素之一。
     * 处理流程：
     *   1. 根据 key 聚合，再根据高 24bit (masked by 0xffffff00) 细分 bucket。
     *   2. 对每个 (key, headerHigh) bucket 内的低 8bit 列表，根据长度选择放入不同存储格式：
     *        LIST_ARRAY1  (<=1 个值，低 8bit 内联在 header 中)
     *        LIST_ARRAY4  (<=5 个值，第 1 个内联，其余最多 4 个写入数组)
     *        LIST_ARRAY8  (<=9 个值)
     *        LIST_ARRAY16 (<=17 个值)
     *        LIST_ARRAY32 (<=33 个值)
     *        LIST_BITSET  (超过 33 个值，使用 256 bit (32B) 位图表示所有可能的低 8bit 出现情况)
     *   3. headers_[t]   保存 (高24bit | 第一个低8bit) 合并后的 32bit。
     *   4. arrays_[t]    保存除第一个低 8bit 以外的后续低 8bit（或 BITSET 位图）。
     *   5. keys_[t]      保存该条倒排记录对应的 key。
     *   6. key_segments_[t] 维护前缀计数：key_segments_[t][k] 与 [k+1] 之间差值为 key=k 的该类型记录数量。
     *
     * @param xs 输入 (key,value) 列表。
     * @note 构建后会调用 Dump() 输出调试信息。
     */
    template <class Container = std::multimap<uint32_t, uint32_t>>
    void BuildFrom(const Container &xs)
    {
        std::map<uint32_t, std::map<uint32_t, std::vector<uint8_t>>> buckets;
        for (auto &&[k, x] : xs)
        {
            buckets[k][x & 0xffffff00].push_back(x & 0xff); // setbit map
        }
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> tmp_key_segments;
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> tmp_keys;
        std::array<std::vector<uint8_t>, MAX_ARRAY_TYPE> tmp_arrays;
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> tmp_headers;
        GroupKeys(&buckets, &tmp_key_segments, &tmp_keys, &tmp_arrays, &tmp_headers);
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i) // copy
        {
            if (i != LIST_ARRAY1)
            {
                arrays_[i].AllocFrom(tmp_arrays[i]);
            }
            key_segments_[i].AllocFrom(tmp_key_segments[i]);
            headers_[i].AllocFrom(tmp_headers[i]);
            keys_[i].AllocFrom(tmp_keys[i]);
        }
        ComputeKeyBlockSegments();
        // debug print
        // Dump();
    }

    /**
     * @brief 组装 buckets 到临时结构 (key_segments / keys / arrays / headers)。
     *
     * @param buckets_p        外部构造的三层 map： key -> (headerHigh -> vector<low8>)。
     * @param tmp_key_segments 输出：各类型的 key 前缀数组；初始化时会 push_back(0)。
     * @param tmp_keys         输出：各类型记录的 key 顺序，与 headers 同索引对应。
     * @param tmp_arrays       输出：各类型的额外 payload（数组或位图）。LIST_ARRAY1 不使用。
     * @param tmp_headers      输出：各类型 32bit header（高24bit + 第一个低8bit）。
     *
     * 逻辑：对每个 key 下的所有 headerHigh 分桶，按 bucket 大小判定所属类型；
     *       统计 key 在该类型出现次数，累计写入 tmp_key_segments[i] 形成前缀。
     *
     * 复杂度：设总元素数 N，桶划分与排序使整体近似 O(N log L)（L 为单 bucket 大小）。
     */
    void GroupKeys(
        std::map<uint32_t, std::map<uint32_t, std::vector<uint8_t>>> *buckets_p,
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> *tmp_key_segments_p,
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> *tmp_keys_p,
        std::array<std::vector<uint8_t>, MAX_ARRAY_TYPE> *tmp_arrays_p,
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> *tmp_headers_p)
    {
        auto &buckets = *buckets_p;
        auto &tmp_key_segments = *tmp_key_segments_p;
        auto &tmp_keys = *tmp_keys_p;
        auto &tmp_arrays = *tmp_arrays_p;
        auto &tmp_headers = *tmp_headers_p;
        auto AddArray = [&](int array_len, uint32_t key, auto &key_cnt,
                            uint32_t header, auto &bucket,
                            MixedListArrayType array_idx)
        {
            ++key_cnt[array_idx];
            std::sort(bucket.begin(), bucket.end());
            tmp_headers[array_idx].push_back(header | bucket[0]);
            tmp_keys[array_idx].push_back(key);
            for (int i = 0; i < array_len; ++i)
            {
                uint8_t b = i + 1 < bucket.size() ? bucket[i + 1] : 0;
                tmp_arrays[array_idx].push_back(b);
            }
        };
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
        {
            tmp_key_segments[i].push_back(0);
        }
        for (auto &&[key, value_bucket] : buckets)
        {
            std::array<int, MAX_ARRAY_TYPE> key_cnt{0};
            for (auto &&[header, bucket] : value_bucket)
            {
                if (bucket.size() <= 1 && array_switches_[LIST_ARRAY1])
                {
                    AddArray(0, key, key_cnt, header, bucket, LIST_ARRAY1);
                }
                else if (bucket.size() <= 5 && array_switches_[LIST_ARRAY4])
                {
                    AddArray(4, key, key_cnt, header, bucket, LIST_ARRAY4);
                }
                else if (bucket.size() <= 9 && array_switches_[LIST_ARRAY8])
                {
                    AddArray(8, key, key_cnt, header, bucket, LIST_ARRAY8);
                }
                else if (bucket.size() <= 17 && array_switches_[LIST_ARRAY16])
                {
                    AddArray(16, key, key_cnt, header, bucket, LIST_ARRAY16);
                }
                else if (bucket.size() <= 33 && array_switches_[LIST_ARRAY32])
                {
                    AddArray(32, key, key_cnt, header, bucket, LIST_ARRAY32);
                }
                else
                {
                    ++key_cnt[LIST_BITSET];
                    tmp_headers[LIST_BITSET].push_back(header);
                    // Bitset
                    std::vector<uint8_t> tmp_bits(256 / 8);
                    for (auto b : bucket)
                    {
                        tmp_bits[b / 8] |= 1u << (b % 8);
                    }
                    tmp_arrays[LIST_BITSET].insert(tmp_arrays[LIST_BITSET].end(),
                                                   tmp_bits.begin(), tmp_bits.end());
                    tmp_keys[LIST_BITSET].push_back(key);
                }
            }
            // Insert
            for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
            {
                while (tmp_key_segments[i].size() <= key)
                {
                    tmp_key_segments[i].push_back(tmp_key_segments[i].back());
                }
                tmp_key_segments[i].push_back(tmp_key_segments[i].back() + key_cnt[i]);
            }
        }
    }

    void ComputeKeyBlockSegments(int step = kKeyStep)
    {
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
        {
            ComputeKeyBlockMeta(kKeyStep, keys_[i].len, keys_[i].h, &key_array_metas_[i]);
        }
    }

    // Prepare inverted list worker for sparse query.
    template <class Container>
    std::unique_ptr<InvertedListWorker> Prepare(const Container &keys) const
    {
        auto ret = std::make_unique<InvertedListWorker>();
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> hit_indices_beg; // key在全局数据中开始的地方
        std::array<std::vector<uint32_t>, MAX_ARRAY_TYPE> hit_segments;    // 请求key集合的block长度求和
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
        {
            hit_segments[i].push_back(0);
        }
        for (auto key : keys)
        {
            for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
            {
                int beg = 0;
                int len = 0;
                if (key < key_segments_[i].len - 1)
                {
                    beg = key_segments_[i].h[key];
                    len = key_segments_[i].h[key + 1] - beg;
                }
                hit_indices_beg[i].push_back(beg);
                hit_segments[i].push_back(hit_segments[i].back() + len);
            }
        }
        for (int i = 0; i < MAX_ARRAY_TYPE; ++i)
        {
            ret->tasks[i].d_value_arrays = arrays_[i].d;
            ret->tasks[i].d_value_headers = headers_[i].d;
            // copy hit_indices_beg, hit_segments
            ret->tasks[i].indices_beg.AllocFrom(hit_indices_beg[i]);
            ret->tasks[i].segments.AllocFrom(hit_segments[i]);

            // // DEBUG
            // std::cout << "idx=" << i << " hit_indices_beg=";
            // for (size_t j = 0; j < hit_indices_beg[i].size(); ++j)
            //     std::cout << " " << hit_indices_beg[i][j];
            // std::cout << " hit_segments=";
            // for (size_t j = 0; j < hit_segments[i].size(); ++j)
            //     std::cout << " " << hit_segments[i][j];
            // std::cout << "\n";
        }

        ret->Prepare();
        // ret->DebugPrint();
        return ret;
    }

    // Bitmap dense query
    std::unique_ptr<InvertedListBitmapWorkerImpl<kKeyStep, kThreadNum>>
    Prepare(uint64_t len, const uint8_t *query_bitmap) const
    {
        auto ret = std::make_unique<InvertedListBitmapWorkerImpl<kKeyStep, kThreadNum>>();
        ret->query_bitmap.Alloc((len + 3) / 4); // LenBuffer<uint32_t>，元素个数 = (len+3)/4
        std::memcpy(ret->query_bitmap.h, query_bitmap, len);
        cudaMemcpy(ret->query_bitmap.d, ret->query_bitmap.h, ret->query_bitmap.len * sizeof(uint32_t), cudaMemcpyHostToDevice);
        ret->keys = &keys_;
        ret->key_block_segments = &key_block_segments_;
        ret->arrays = &arrays_;
        ret->headers = &headers_;
        ret->metas = &key_array_metas_;
        return ret;
    }

    /**
     * @brief 以 Host 端位图批量构建稠密查询 Worker（Batch Bitmap Dense Query）。
     *
     * 功能：
     *   - 为 batch_size 条查询各自的位图（每条长度为 len 字节）分配一块连续缓冲区；
     *   - 将传入的 host 内存 query_bitmap 拷贝到内部的 host 缓冲（ret->query_bitmap.h）；
     *   - 设置 device 指针别名 ret->d_query_bitmap 指向内部 device 缓冲（ret->query_bitmap.d）。
     *
     * 约定：
     *   - 位图按查询为主（query-major）连续排列：总字节数 = batch_size * len；
     *   - len 必须是 4 的倍数（内部以 uint32_t 视角组织并通过 query_bitmap_len=len/4 记录长度）。
     *   - 本函数只做 Host 内存拷贝到内部 host 缓冲，不主动执行 Host→Device 的 cudaMemcpy；
     *     若 LenBuffer 未启用统一内存/自动上传，需在后续执行阶段确保数据已在 device 侧可见。
     *
     * @param batch_size 批量查询数。
     * @param len        每个查询的位图长度（字节）。应为 4 的倍数。
     * @param query_bitmap Host 端指针，大小为 batch_size*len 字节，按查询连续存放。
     * @return 初始化完成的批量稠密查询 Worker。
     */
    std::unique_ptr<InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>> PrepareBatch(
        uint64_t batch_size, uint64_t len,
        const uint8_t *query_bitmap) const
    {
        auto ret = PrepareBatchDevice(batch_size, len, nullptr);
        // ret->query_bitmap.Alloc(batch_size * len / 4);
        // memcpy(ret->query_bitmap.h, query_bitmap, batch_size * len);
        ret->query_bitmap.AllocFrom(reinterpret_cast<const uint32_t *>(query_bitmap), batch_size * len / 4);
        ret->d_query_bitmap = ret->query_bitmap.d;
        return ret;
    }

    /**
     * @brief 以 Device 端位图指针批量构建稠密查询 Worker。
     *
     * 用于位图已在 GPU 上的场景（零拷贝/外部已拷贝完成）。仅设置元数据与指针，不分配或复制位图数据。
     *
     * 约定：
     *   - d_query_bitmap 指向大小为 batch_size*len 字节的 device 连续缓冲（query-major 布局）。
     *   - len 需为 4 的倍数；内部使用 query_bitmap_len=len/4 并以 uint32_t* 视角访问。
     *   - 返回的 Worker 仅持有对 ivf_list 内部各数组（keys/headers/arrays 等）的引用，需保证其生命周期覆盖 Worker 使用期。
     *
     * @param batch_size 批量查询数。
     * @param len        每个查询的位图长度（字节）。应为 4 的倍数。
     * @param d_query_bitmap Device 端位图首地址（可为 nullptr，表示之后再赋值）。
     * @return 初始化完成的批量稠密查询 Worker（不涉及位图数据拷贝）。
     */
    std::unique_ptr<InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>> PrepareBatchDevice(
        uint64_t batch_size, uint64_t len,
        const uint8_t *d_query_bitmap) const
    {
        auto ret = std::make_unique<InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>>();
        ret->batch_size = batch_size;
        ret->query_bitmap_len = len / 4; // uint8_t -> uint32_t
        ret->d_query_bitmap = reinterpret_cast<const uint32_t *>(d_query_bitmap);
        ret->keys = &keys_;
        ret->key_block_segments = &key_block_segments_;
        ret->arrays = &arrays_;
        ret->headers = &headers_;
        ret->metas = &key_array_metas_;
        return ret;
    }

    /**
     * @brief 给定类型返回其 payload（除首元素外的存储部分）定长字节数。
     * @param t MixedListArrayType。
     * @return int 对应数组区长度；LIST_ARRAY1 为 0；BITSET 固定 32 字节。
     */
    static int ArrayPayloadLength(int t)
    {
        switch (t)
        {
        case LIST_ARRAY1:
            return 0; // stored entirely in header low byte
        case LIST_ARRAY4:
            return 4;
        case LIST_ARRAY8:
            return 8;
        case LIST_ARRAY16:
            return 16;
        case LIST_ARRAY32:
            return 32;
        case LIST_BITSET:
            return 256 / 8; // 32
        default:
            return 0;
        }
    }

    /**
     * @brief 打印当前结构的详细调试信息，包括：
     *   - 各类型 key_segments 前缀数组
     *   - headers/keys 数量及 key 范围
     *   - 逐条记录的解析（首值、长度、低 8bit 序列，以及复原的高24|低8 整数列表）
     * @note 用于验证 BuildFrom/GroupKeys 的正确性，生产环境可移除。
     */
    void Dump() const
    {
        std::cout << "================ IVF LIST DEBUG DUMP ================\n";
        size_t total_lists = 0;
        size_t total_bytes = 0;
        for (int t = 0; t < MAX_ARRAY_TYPE; ++t)
            total_lists += headers_[t].len;
        for (int t = 0; t < MAX_ARRAY_TYPE; ++t)
            total_bytes += arrays_[t].len; // payload total bytes (excluding LIST_ARRAY1 which is inline)
        std::cout << "Summary: total_lists=" << total_lists << " total_payload_bytes=" << total_bytes << "\n";
        for (int t = 0; t < MAX_ARRAY_TYPE; ++t)
        {
            std::cout << "-- Array Type: " << ArrayTypeName(t) << " (index " << t << ")\n";
            std::cout << "   key_segments size: " << key_segments_[t].len;
            if (key_segments_[t].len)
            {
                const auto *seg_h = key_segments_[t].h;
                std::cout << " first: " << seg_h[0] << " last: " << seg_h[key_segments_[t].len - 1];
            }
            std::cout << "\n   key_segments: [";
            for (size_t i = 0; i < key_segments_[t].len; ++i)
            {
                if (i)
                    std::cout << ',';
                std::cout << key_segments_[t].h[i];
            }
            std::cout << "]\n";
            std::cout << "   #lists: " << headers_[t].len << "  #keys: " << keys_[t].len;
            if (keys_[t].len)
            {
                const uint32_t *kb = keys_[t].h;
                uint32_t min_key = *std::min_element(kb, kb + keys_[t].len);
                uint32_t max_key = *std::max_element(kb, kb + keys_[t].len);
                std::cout << "  key_range=[" << min_key << ',' << max_key << ']';
            }
            if (t != LIST_ARRAY1)
            {
                std::cout << "  payload_bytes=" << arrays_[t].len;
            }
            std::cout << "\n";
            int payload_len = ArrayPayloadLength(t);
            if (headers_[t].len)
            {
                // Header line for table (extended)
                std::cout << "   idx\tkey\theaderHigh\tfirst\tlen\tvalues\n";
            }
            for (size_t li = 0; li < headers_[t].len; ++li)
            {
                uint32_t header = headers_[t].h[li];
                uint32_t high = header & 0xffffff00u; // upper 24 bits
                uint32_t first_val = header & 0xffu;  // inline low byte
                uint32_t key = keys_[t].h[li];
                std::vector<int> values; // reconstructed list of low bytes
                if (t == LIST_ARRAY1)
                {
                    values.push_back((int)first_val);
                }
                else if (t == LIST_BITSET)
                {
                    size_t offset = li * payload_len;
                    for (int b = 0; b < 256; ++b)
                    {
                        int byte_index = b / 8;
                        int bit_index = b % 8;
                        if (offset + byte_index < arrays_[t].len && (arrays_[t].h[offset + byte_index] & (1u << bit_index)))
                        {
                            values.push_back(b);
                        }
                    }
                }
                else
                {
                    size_t offset = li * payload_len;
                    values.push_back((int)first_val);
                    for (int j = 0; j < payload_len; ++j)
                    {
                        if (offset + j >= arrays_[t].len) // safety guard
                            break;
                        uint8_t v = arrays_[t].h[offset + j];
                        if (v == 0)
                            break; // stop at first zero as padding
                        values.push_back((int)v);
                    }
                }
                std::cout << "   "
                          << std::setw(3) << li << '\t'
                          << std::setw(4) << key << '\t'
                          << "0x" << std::hex << std::setw(6) << std::setfill('0') << high << std::dec << std::setfill(' ') << '\t'
                          << std::setw(5) << (int)first_val << '\t'
                          << std::setw(3) << values.size() << '\t';
                std::cout << '[';
                for (size_t vi = 0; vi < values.size(); ++vi)
                {
                    if (vi)
                        std::cout << ',';
                    std::cout << values[vi];
                }
                std::cout << ']';
                // For clarity show combined full 32-bit numbers (high|low)
                std::cout << " full=[";
                for (size_t vi = 0; vi < values.size(); ++vi)
                {
                    if (vi)
                        std::cout << ',';
                    uint32_t full = (high) | (uint32_t)(uint8_t)values[vi];
                    std::cout << full;
                }
                std::cout << "]\n";
            }
        }
        // Separate section for KeyBlockMeta details
        std::cout << "---------------- KeyBlockMeta Summary ----------------\n";
        for (int t = 0; t < MAX_ARRAY_TYPE; ++t)
        {
            const auto &m = key_array_metas_[t];
            std::cout << "Type=" << ArrayTypeName(t)
                      << " len=" << m.len
                      << " block_step=" << m.block_step;
            if (m.key_block_segments.len)
            {
                size_t block_cnt = m.key_block_segments.len - 1;
                std::cout << " blocks=" << block_cnt;
            }
            std::cout << " useful_bytes=" << (m.key_pos_segments.len ? m.key_pos_segments.h[m.key_pos_segments.len - 1] : 0)
                      << " buffer_bytes=" << m.key_buffers.len << "\n";
            // block starts (first few + last)
            std::cout << "  block_starts: [";
            size_t show_bs = std::min<size_t>(m.key_block_segments.len, 6);
            for (size_t i = 0; i < show_bs; ++i)
            {
                if (i)
                    std::cout << ',';
                std::cout << m.key_block_segments.h[i];
            }
            if (m.key_block_segments.len > show_bs)
                std::cout << ",...";
            std::cout << "]\n";
            // pos segments
            std::cout << "  pos_segments : [";
            size_t show_ps = std::min<size_t>(m.key_pos_segments.len, 6);
            for (size_t i = 0; i < show_ps; ++i)
            {
                if (i)
                    std::cout << ',';
                std::cout << m.key_pos_segments.h[i];
            }
            if (m.key_pos_segments.len > show_ps)
                std::cout << ",...";
            std::cout << "]\n";
            // Show first block offsets (optional)
            if (m.len && m.key_block_segments.len > 1)
            {
                // decode first block
                uint32_t start_key = m.key_block_segments.h[0];
                // uint32_t next_start = m.key_block_segments.h[1];
                uint32_t num_elem = (uint32_t)std::min<uint64_t>(m.block_step, m.len);
                uint32_t useful_bytes = m.key_pos_segments.len > 1 ? (m.key_pos_segments.h[1] - m.key_pos_segments.h[0]) : 0;
                uint32_t key_bytes = num_elem ? (useful_bytes / num_elem) : 0;
                std::cout << "  first_block_offsets: {start=" << start_key << ", num=" << num_elem << ", key_bytes=" << key_bytes << ", offsets=[";
                size_t show_keys = std::min<uint32_t>(num_elem, 8);
                size_t block_physical_bytes = (size_t)m.block_step * key_bytes;
                const uint8_t *buf = m.key_buffers.h; // first block at offset 0
                for (size_t kj = 0; kj < show_keys; ++kj)
                {
                    if (kj)
                        std::cout << ',';
                    uint32_t off = 0;
                    if (key_bytes)
                        std::memcpy(&off, buf + kj * key_bytes, key_bytes);
                    std::cout << off;
                }
                std::cout << "]} (physical_block_bytes=" << block_physical_bytes << ")\n";
            }
        }
        std::cout << "------------------------------------------------------\n";
        std::cout << "=====================================================\n";
    }

protected:
    std::array<uint32_t, MAX_ARRAY_TYPE> num_array_lists_;
    std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> key_segments_;
    std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> headers_;
    std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> keys_;
    std::array<LenBuffer<INPUT, uint8_t>, MAX_ARRAY_TYPE> arrays_;

    // Mainly for dense query. Maintain the range of keys for each block
    std::array<LenBuffer<INPUT, uint32_t>, MAX_ARRAY_TYPE> key_block_segments_;
    std::array<bool, MAX_ARRAY_TYPE> array_switches_;
    std::array<KeyBlockMeta, MAX_ARRAY_TYPE> key_array_metas_;
};

template <int kArrayType>
__device__ void ProcessArrayFunctor(
    uint32_t key, uint32_t idx, const uint32_t *d_value_headers,
    const uint8_t *d_value_arrays, uint32_t *d_res);

template <int kArrayType>
__device__ void ProcessArrayFunctor_BatchExpand(
    uint32_t key, uint32_t idx, const uint32_t *d_value_headers,
    const uint8_t *d_value_arrays, uint32_t *d_res,
    uint32_t qidx, uint32_t res_len);