#pragma once

#include <cinttypes>
#include <utility>
#include <cstdio>

#include "cub/cub.cuh"
#include "inverted_list.h"
#include "cuda_op_utility.h"

template <int kBlockSize = 64, int kTotalWork = 256>
__device__ SimpleArray<uint32_t, kTotalWork / kBlockSize> LoadBPKeys(
    int key_beg, int key_end, int pos_beg, const uint8_t *d_key_buffers)
{
    constexpr int kWorkPerThread = kTotalWork / kBlockSize;
    int tid = threadIdx.x;
    int key_span = key_end - key_beg;
    auto *key_buffer = d_key_buffers + pos_beg;
    SimpleArray<uint32_t, kTotalWork / kBlockSize> ret;
    if (key_span < 256)
    {
        auto keys = DeviceLoadBytes<kWorkPerThread, uint8_t>(key_buffer, tid);
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            ret[i] = keys[i] + key_beg;
        }
        return ret;
    }
    else if (key_span < 65536)
    {
        auto keys = DeviceLoadBytes<kWorkPerThread, uint16_t>(key_buffer, tid);
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            ret[i] = keys[i] + key_beg;
        }
        return ret;
    }
    else
    {
        ret = DeviceLoadBytes<kWorkPerThread, uint32_t>(key_buffer, tid);
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            ret[i] = ret[i] + key_beg;
        }
        return ret;
    }
}

// The assigned shared memory is not large enough to use uint4 to load efficiently
inline __device__ void LoadMultipleBitmap(int b, int len_u4, const void *a,
                                          int lda, uint32_t *shared_b)
{
    auto *a_u32 = reinterpret_cast<const uint32_t *>(a);
    int len = len_u4 * 4;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    // Load to shared_memory
    for (int idx = tid; idx < b * len; idx += nt)
    {
        int qidx = idx % b;
        int bitmap_idx = idx / b;
        auto reg = a_u32[bitmap_idx + qidx * lda * 4];
        shared_b[bitmap_idx * b + qidx] = reg;
    }
}

template <int kBatchSize, int kBlockSize, int kTotalWork>
__global__ void KeyBatchBPTransformKernel_Expanded1(
    uint64_t len,
    const uint32_t *d_key_pos_seg,
    const uint32_t *d_key_block_seg,
    const uint8_t *d_key_buffers,
    int bitmap_len,
    const uint32_t *d_bitmaps,
    const uint32_t *headers,
    uint32_t *d_res,
    uint32_t res_len)
{
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1; // default 3
    constexpr int kWorkPerThread = kTotalWork / kBlockSize; // default 4

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uint32_t key_beg = d_key_block_seg[bid];
    uint32_t key_end = d_key_block_seg[bid + 1];
    uint32_t pos_beg = d_key_pos_seg[bid];

    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;

    int bitmap_len_u4 = bitmap_len / 4;

    // Load four keys for each thread, eg. 0,1,2,3 for block 0 thread 0
    auto keys = LoadBPKeys<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers);
    // print keys
    // printf("[Block %d, Thread %d] Keys: %u, %u, %u, %u\n",
    //        bid, tid, keys[0], keys[1], keys[2], keys[3]);

    // 共享内存（仅 Dense 路径使用）
    __shared__ struct
    {
        // 词位主序 × 批量次序（每个词位连着kBatchSize个uint32）
        uint32_t bitmap[kSharedBitmapLenU4 * 4 * kBatchSize];
    } shared;

    auto *d_bitmap_u4 = reinterpret_cast<const uint4 *>(d_bitmaps);

    if (threadIdx.x == 0)
    {
        // printf("KeyBatchBPTransformKernel_Expanded1 [Block %d] Starts. Key range: [%u, %u), Data pos: %u "
        //        "bitmap_beg_u4: %d, bitmap_end_u4: %d, span_u4: %d, kSharedBitmapLenU4: %d\n",
        //        bid, key_beg, key_end, pos_beg,
        //        bitmap_beg_u4, bitmap_end_u4, span_u4, kSharedBitmapLenU4);
    }

    if (span_u4 >= kSharedBitmapLenU4) // Sparse：直接从全局内存按query逐个 __ldg 取词位
    {
        // printf("span_u4(%d) >= kSharedBitmapLenU4(%d) use Sparse path\n",
        //        span_u4, kSharedBitmapLenU4);
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t word_idx = (key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));

            // 将k个query的该词位装入寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                // d_bitmaps布局：每个query连续 bitmap_len 个uint32
                reg_qks[qid] = __ldg(d_bitmaps + word_idx + qid * bitmap_len);
            }

            // 测试bit并回调
            // 懒加载
            // bool loaded = false;
            // uint32_t value = 0;
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Sparse Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    uint32_t value = __ldg(headers + idx); // 这个地方可以考虑懒加载，
                    // if (!loaded) { value = __ldg(headers + idx); loaded = true; }
                    // atomicOr(d_res + v / 32 + qidx * res_len, 1u << (v % 32));
                    atomicOr(d_res + value / 32 + qid * res_len, 1u << (value % 32));
                }
            }
        }
    }
    else // Dense：先把本block覆盖范围的词位全部搬到共享内存（按“词位主序 × 批量次序”布局）
    {
        // printf("span_u4(%d) < kSharedBitmapLenU4(%d) use Dense path\n",
        //        span_u4, kSharedBitmapLenU4);
        LoadMultipleBitmap(kBatchSize, span_u4, d_bitmap_u4 + bitmap_beg_u4, bitmap_len_u4, shared.bitmap);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t shift_key = key - bitmap_beg_u4 * 128; // 移到本block共享片内的偏移
            uint32_t word_idx = (shift_key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));    // original

            // 该词位在共享内存中的基址（后面紧跟kBatchSize个query的uint32）
            const uint32_t *qk = shared.bitmap + (word_idx * kBatchSize);
            // 装到寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                reg_qks[qid] = qk[qid];
            }

            // 测试bit并回调
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Dense Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    uint32_t value = __ldg(headers + idx);
                    // atomicOr(d_res + v / 32 + qidx * res_len, 1u << (v % 32));
                    atomicOr(d_res + value / 32 + qid * res_len, 1u << (value % 32));
                }
            }
        }
    }
}

template <int kBatchSize, int kBlockSize, int kTotalWork, int kArrayType>
__global__ void KeyBatchBPTransformKernel_Expanded(
    uint64_t len,
    const uint32_t *d_key_pos_seg,
    const uint32_t *d_key_block_seg,
    const uint8_t *d_key_buffers,
    int bitmap_len,
    const uint32_t *d_bitmaps,
    const uint32_t *headers,
    const uint8_t *arrays,
    uint32_t *d_res,
    uint32_t res_len)
{
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1; // default 3
    constexpr int kWorkPerThread = kTotalWork / kBlockSize; // default 4

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uint32_t key_beg = d_key_block_seg[bid];
    uint32_t key_end = d_key_block_seg[bid + 1];
    uint32_t pos_beg = d_key_pos_seg[bid];

    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;

    int bitmap_len_u4 = bitmap_len / 4;

    // Load four keys for each thread, eg. 0,1,2,3 for block 0 thread 0
    auto keys = LoadBPKeys<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers);
    // print keys
    // printf("[Block %d, Thread %d] Keys: %u, %u, %u, %u\n",
    //        bid, tid, keys[0], keys[1], keys[2], keys[3]);

    // 共享内存（仅 Dense 路径使用）
    __shared__ struct
    {
        // 词位主序 × 批量次序（每个词位连着kBatchSize个uint32）
        uint32_t bitmap[kSharedBitmapLenU4 * 4 * kBatchSize];
    } shared;

    auto *d_bitmap_u4 = reinterpret_cast<const uint4 *>(d_bitmaps);

    if (threadIdx.x == 0)
    {
        // printf("KeyBatchBPTransformKernel_Expanded [Block %d] Starts. Key range: [%u, %u), Data pos: %u "
        //        "bitmap_beg_u4: %d, bitmap_end_u4: %d, span_u4: %d, kSharedBitmapLenU4: %d\n",
        //        bid, key_beg, key_end, pos_beg,
        //        bitmap_beg_u4, bitmap_end_u4, span_u4, kSharedBitmapLenU4);
    }

    if (span_u4 >= kSharedBitmapLenU4)
    {
        // printf("span_u4(%d) >= kSharedBitmapLenU4(%d) use Sparse path\n",
        //        span_u4, kSharedBitmapLenU4);
        // Sparse：直接从全局内存按query逐个 __ldg 取词位
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t word_idx = (key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));

            // 将k个query的该词位装入寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                // d_bitmaps布局：每个query连续 bitmap_len 个uint32
                reg_qks[qid] = __ldg(d_bitmaps + word_idx + qid * bitmap_len);
            }

            // 测试bit并回调
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Sparse Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    ProcessArrayFunctor_BatchExpand<kArrayType>(key, idx, headers, arrays, d_res, qid, res_len);
                }
            }
        }
    }
    else
    {
        // printf("span_u4(%d) < kSharedBitmapLenU4(%d) use Dense path\n",
        //        span_u4, kSharedBitmapLenU4);
        // Dense：先把本block覆盖范围的词位全部搬到共享内存（按“词位主序 × 批量次序”布局）
        LoadMultipleBitmap(kBatchSize, span_u4, d_bitmap_u4 + bitmap_beg_u4, bitmap_len_u4, shared.bitmap);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t shift_key = key - bitmap_beg_u4 * 128; // 移到本block共享片内的偏移
            uint32_t word_idx = (shift_key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));    // original
            // uint32_t bit_mask = (1u << (shift_key & 0x1f)); // AI changed

            // 该词位在共享内存中的基址（后面紧跟kBatchSize个query的uint32）
            const uint32_t *qk = shared.bitmap + (word_idx * kBatchSize);

            // 装到寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                reg_qks[qid] = qk[qid];
            }

            // 测试bit并回调
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Dense Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    // key(key), idx(idx), qidx(qid)
                    // paf(key, idx, d_value_headers, d_value_arrays, lambda);
                    ProcessArrayFunctor_BatchExpand<kArrayType>(key, idx, headers, arrays, d_res, qid, res_len);
                }
            }
        }
    }
}


template <int kBatchSize, int kBlockSize, int kTotalWork>
__global__ void KeyBatchBPTransformKernel_ExpandedBitset(
    uint64_t len,
    const uint32_t *d_key_pos_seg,
    const uint32_t *d_key_block_seg,
    const uint8_t *d_key_buffers,
    int bitmap_len,
    const uint32_t *d_bitmaps,
    const uint32_t *headers,
    const uint8_t *arrays,
    uint32_t *d_res,
    uint32_t res_len)
{
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1; // default 3
    constexpr int kWorkPerThread = kTotalWork / kBlockSize; // default 4

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    uint32_t key_beg = d_key_block_seg[bid];
    uint32_t key_end = d_key_block_seg[bid + 1];
    uint32_t pos_beg = d_key_pos_seg[bid];

    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;

    int bitmap_len_u4 = bitmap_len / 4;

    // Load four keys for each thread, eg. 0,1,2,3 for block 0 thread 0
    auto keys = LoadBPKeys<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers);
    // print keys
    // printf("[Block %d, Thread %d] Keys: %u, %u, %u, %u\n",
    //        bid, tid, keys[0], keys[1], keys[2], keys[3]);

    // 共享内存（仅 Dense 路径使用）
    __shared__ struct
    {
        // 词位主序 × 批量次序（每个词位连着kBatchSize个uint32）
        uint32_t bitmap[kSharedBitmapLenU4 * 4 * kBatchSize];
    } shared;

    auto *d_bitmap_u4 = reinterpret_cast<const uint4 *>(d_bitmaps);

    if (threadIdx.x == 0)
    {
        // printf("KeyBatchBPTransformKernel_ExpandedBitset [Block %d] Starts. Key range: [%u, %u), Data pos: %u "
        //        "bitmap_beg_u4: %d, bitmap_end_u4: %d, span_u4: %d, kSharedBitmapLenU4: %d\n",
        //        bid, key_beg, key_end, pos_beg,
        //        bitmap_beg_u4, bitmap_end_u4, span_u4, kSharedBitmapLenU4);
    }

    if (span_u4 >= kSharedBitmapLenU4)
    {
        // printf("span_u4(%d) >= kSharedBitmapLenU4(%d) use Sparse path\n",
        //        span_u4, kSharedBitmapLenU4);
        // Sparse：直接从全局内存按query逐个 __ldg 取词位
#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t word_idx = (key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));

            // 将k个query的该词位装入寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                // d_bitmaps布局：每个query连续 bitmap_len 个uint32
                reg_qks[qid] = __ldg(d_bitmaps + word_idx + qid * bitmap_len);
            }

            // 测试bit并回调
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Sparse Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    using T = uint4;
                    auto *d_bitsets_t = reinterpret_cast<const T *>(arrays); 
                    uint32_t new_header = __ldg(headers + idx);
                    T bits[2];
                    bits[0] = __ldg(d_bitsets_t + idx * 2);
                    bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
                    auto *bits_u32 = reinterpret_cast<const uint32_t *>(&bits);
                #pragma unroll
                    for (uint32_t j = 0; j < 256; j += 32) {
                        uint32_t vl = bits_u32[j / 32];
                        uint32_t vh = new_header | j;
                        atomicOr(d_res + vh / 32 + qid * res_len, vl);
                    }
                }
            }
        }
    }
    else
    {
        // printf("span_u4(%d) < kSharedBitmapLenU4(%d) use Dense path\n",
        //        span_u4, kSharedBitmapLenU4);
        // Dense：先把本block覆盖范围的词位全部搬到共享内存（按“词位主序 × 批量次序”布局）
        LoadMultipleBitmap(kBatchSize, span_u4, d_bitmap_u4 + bitmap_beg_u4, bitmap_len_u4, shared.bitmap);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < kWorkPerThread; ++i)
        {
            int idx = tid * kWorkPerThread + i + bid * kTotalWork;
            if (idx >= len)
            {
                return;
            }
            uint32_t key = keys[i];
            uint32_t shift_key = key - bitmap_beg_u4 * 128; // 移到本block共享片内的偏移
            uint32_t word_idx = (shift_key >> 5);
            uint32_t bit_mask = (1u << (key & 0x1f));    // original
            // uint32_t bit_mask = (1u << (shift_key & 0x1f)); // AI changed

            // 该词位在共享内存中的基址（后面紧跟kBatchSize个query的uint32）
            const uint32_t *qk = shared.bitmap + (word_idx * kBatchSize);

            // 装到寄存器
            uint32_t reg_qks[kBatchSize];
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                reg_qks[qid] = qk[qid];
            }

            // 测试bit并回调
#pragma unroll
            for (int qid = 0; qid < kBatchSize; ++qid)
            {
                if (reg_qks[qid] & bit_mask)
                {
                    // printf("Dense Match ! [Block %d, Thread %d] Match found! Key: %d, Idx: %d, Qid: %d\n",
                    //        blockIdx.x, threadIdx.x, key, idx, qid);
                    // f(key, idx, qid, std::forward<Args>(args)...);
                    // key(key), idx(idx), qidx(qid), d_value_headers(headers), d_bitsets(arrays)
                    // pbf(key, idx, d_value_headers, d_bitsets, lambda);
                    using T = uint4;
                    auto *d_bitsets_t = reinterpret_cast<const T *>(arrays); 
                    uint32_t new_header = __ldg(headers + idx);
                    T bits[2];
                    bits[0] = __ldg(d_bitsets_t + idx * 2);
                    bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
                    auto *bits_u32 = reinterpret_cast<const uint32_t *>(&bits);
                #pragma unroll
                    for (uint32_t j = 0; j < 256; j += 32) {
                        uint32_t vl = bits_u32[j / 32];
                        uint32_t vh = new_header | j;

                        atomicOr(d_res + vh / 32 + qid * res_len, vl);
                    }
                }
            }
        }
    }
}


template <int kBatchSize, int kBlockSize, int kTotalWork>
void RunKeyBatchBPTransformKernel1(
    cudaStream_t stream, const KeyBlockMeta &meta, int bitmap_len,
    const uint32_t *d_bitmaps, uint32_t *headers,
    uint32_t *d_res, uint32_t res_len)
{
    if (meta.block_step != kTotalWork || kBlockSize > kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d) or kBlockSize(%d) > kTotalWork(%d)\n",
               meta.block_step, kTotalWork, kBlockSize, kTotalWork);
        return;
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBatchBPTransformKernel_Expanded1<kBatchSize, kBlockSize, kTotalWork><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            bitmap_len, d_bitmaps, headers, d_res, res_len);
    }
}

template <int kBatchSize, int kBlockSize, int kTotalWork, int kArrayType>
void RunKeyBatchBPTransformKernel(
    cudaStream_t stream, const KeyBlockMeta &meta, int bitmap_len,
    const uint32_t *d_bitmaps, uint32_t *headers, uint8_t *arrays,
    uint32_t *d_res, uint32_t res_len)
{
    if (meta.block_step != kTotalWork || kBlockSize > kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d) or kBlockSize(%d) > kTotalWork(%d)\n",
               meta.block_step, kTotalWork, kBlockSize, kTotalWork);
        return;
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBatchBPTransformKernel_Expanded<kBatchSize, kBlockSize, kTotalWork, kArrayType><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            bitmap_len, d_bitmaps, headers, arrays, d_res, res_len);
    }
}


template <int kBatchSize, int kBlockSize, int kTotalWork>
void RunKeyBatchBPTransformKernelBitset(
    cudaStream_t stream, const KeyBlockMeta &meta, int bitmap_len,
    const uint32_t *d_bitmaps, uint32_t *headers, uint8_t *arrays,
    uint32_t *d_res, uint32_t res_len)
{
    if (meta.block_step != kTotalWork || kBlockSize > kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d) or kBlockSize(%d) > kTotalWork(%d)\n",
               meta.block_step, kTotalWork, kBlockSize, kTotalWork);
        return;
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBatchBPTransformKernel_ExpandedBitset<kBatchSize, kBlockSize, kTotalWork><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            bitmap_len, d_bitmaps, headers, arrays, d_res, res_len);
    }
}