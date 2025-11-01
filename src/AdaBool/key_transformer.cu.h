#pragma once

#include <cinttypes>
#include <utility>
#include <cstdio>

#include "cub/cub.cuh"
#include "inverted_list.h"
#include "cuda_op_utility.h"

template <int kBlockSize = 64, int kTotalWork = 256, class F>
__device__ void ForBPKeyIdxs(int key_beg, int key_end, int pos_beg,
                             const uint8_t* d_key_buffers, const F& f) {
  constexpr int kWorkPerThread = kTotalWork / kBlockSize;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int key_span = key_end - key_beg;
  auto* key_buffer = d_key_buffers + pos_beg;
  union {
    SimpleArray<uint8_t, kWorkPerThread> u8;
    SimpleArray<uint16_t, kWorkPerThread> u16;
    SimpleArray<uint32_t, kWorkPerThread> u32;
  } keys;
  int offset = tid * kWorkPerThread + bid * kTotalWork;
  if (key_span < 256) {
    keys.u8 = DeviceLoadBytes<kWorkPerThread, uint8_t>(key_buffer, tid);
#pragma unroll
    for (int i = 0; i < kWorkPerThread; ++i) {
      f(keys.u8[i] + key_beg, offset + i);
    }
  } else if (key_span < 65536) {
    keys.u16 = DeviceLoadBytes<kWorkPerThread, uint16_t>(key_buffer, tid);
#pragma unroll
    for (int i = 0; i < kWorkPerThread; ++i) {
      f(keys.u16[i] + key_beg, offset + i);
    }
  } else {
    keys.u32 = DeviceLoadBytes<kWorkPerThread, uint32_t>(key_buffer, tid);
#pragma unroll
    for (int i = 0; i < kWorkPerThread; ++i) {
      f(keys.u32[i] + key_beg, offset + i);
    }
  }
}


template <int kBlockSize, int kTotalWork, int kArrayType>
__global__ void KeyBPTransformKernelBitset(
    uint64_t len, const uint32_t* d_key_pos_seg, const uint32_t* d_key_block_seg,
    const uint8_t* d_key_buffers, const uint32_t* d_bitmaps, const uint32_t* headers,
    const uint8_t *array, uint32_t* d_res) {
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1;
    int bid = blockIdx.x;
    int key_beg = d_key_block_seg[bid];
    int key_end = d_key_block_seg[bid + 1];
    int pos_beg = d_key_pos_seg[bid];
    // Only thread 0 in each block prints block-level info to avoid spam
    if (threadIdx.x == 0) {
        // printf("[Block %d] Starts. Key range: [%d, %d), Data pos: %d\n",
        //     bid, key_beg, key_end, pos_beg);
    }
    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;
    __shared__ struct {
        uint4 bitmap[kSharedBitmapLenU4];
    } shared;
    auto* d_bitmap_u4 = reinterpret_cast<const uint4*>(d_bitmaps);

    if (span_u4 < kSharedBitmapLenU4)
    {
        // Dense block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Dense path. span_u4(%d) < kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }
        for (int i = threadIdx.x; i < span_u4; i += blockDim.x)
        {
            shared.bitmap[i] = d_bitmap_u4[bitmap_beg_u4 + i];
        }
        __syncthreads();
        auto *shared_bitmap_shifted = reinterpret_cast<const uint32_t *>(shared.bitmap - bitmap_beg_u4);

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = shared_bitmap_shifted[key >> 5];
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                // key(key), idx(idx), d_values(headers), d_bitset(array)
                using T = uint4; // 128 bits
                auto *d_bitsets_t = reinterpret_cast<const T *>(array);
                uint32_t header = __ldg(headers + idx);
                T bits[2];
                bits[0] = __ldg(d_bitsets_t + idx * 2);
                bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
                auto *bits_u32 = reinterpret_cast<const uint32_t *>(bits);
            #pragma unroll
                for (uint32_t j = 0; j < 256; j+=32) {
                    uint32_t v = header | j;
                    atomicOr(d_res + v / 32, bits_u32[j/32]);
                }
            }
        });
    }
    else
    {
        // Sparse block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Sparse path. span_u4(%d) >= kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = __ldg(d_bitmaps + (key >> 5));
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                using T = uint4; // 128 bits
                auto *d_bitsets_t = reinterpret_cast<const T *>(array);
                uint32_t header = __ldg(headers + idx);
                T bits[2];
                bits[0] = __ldg(d_bitsets_t + idx * 2);
                bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
                auto *bits_u32 = reinterpret_cast<const uint32_t *>(bits);
            #pragma unroll
                for (uint32_t j = 0; j < 256; j+=32) {
                    uint32_t v = header | j;
                    atomicOr(d_res + v / 32, bits_u32[j/32]);
                }

            }
        });
    }
}


template <int kBlockSize, int kTotalWork, int kArrayType>
__global__ void KeyBPTransformKernel(
    uint64_t len, const uint32_t* d_key_pos_seg, const uint32_t* d_key_block_seg,
    const uint8_t* d_key_buffers, const uint32_t* d_bitmaps, const uint32_t* headers,
    const uint8_t *array, uint32_t* d_res) {
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1;
    int bid = blockIdx.x;
    int key_beg = d_key_block_seg[bid];
    int key_end = d_key_block_seg[bid + 1];
    int pos_beg = d_key_pos_seg[bid];
    // Only thread 0 in each block prints block-level info to avoid spam
    if (threadIdx.x == 0) {
        // printf("[Block %d] Starts. Key range: [%d, %d), Data pos: %d\n",
        //     bid, key_beg, key_end, pos_beg);
    }
    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;
    __shared__ struct {
        uint4 bitmap[kSharedBitmapLenU4];
    } shared;
    auto* d_bitmap_u4 = reinterpret_cast<const uint4*>(d_bitmaps);

    if (span_u4 < kSharedBitmapLenU4)
    {
        // Dense block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Dense path. span_u4(%d) < kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }
        for (int i = threadIdx.x; i < span_u4; i += blockDim.x)
        {
            shared.bitmap[i] = d_bitmap_u4[bitmap_beg_u4 + i];
        }
        __syncthreads();
        auto *shared_bitmap_shifted = reinterpret_cast<const uint32_t *>(shared.bitmap - bitmap_beg_u4);

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = shared_bitmap_shifted[key >> 5];
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                // key(key), idx(idx), d_values(headers)
                ProcessArrayFunctor<kArrayType>(key, idx, headers, array, d_res);
            }
        });
    }
    else
    {
        // Sparse block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Sparse path. span_u4(%d) >= kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = __ldg(d_bitmaps + (key >> 5));
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                ProcessArrayFunctor<kArrayType>(key, idx, headers, array, d_res);
            }
        });
    }
}

template <int kBlockSize, int kTotalWork>
__global__ void KeyBPTransformKernel1(
    uint64_t len, const uint32_t* d_key_pos_seg, const uint32_t* d_key_block_seg,
    const uint8_t* d_key_buffers, const uint32_t* d_bitmaps, const uint32_t* headers,
    uint32_t* d_res) {
    constexpr int kSharedBitmapLenU4 = kTotalWork / 128 + 1; // 256/128 + 1= 3
    int bid = blockIdx.x;
    int key_beg = d_key_block_seg[bid];
    int key_end = d_key_block_seg[bid + 1];
    int pos_beg = d_key_pos_seg[bid];
    int bitmap_beg_u4 = key_beg / 128;
    int bitmap_end_u4 = key_end / 128 + 1;
    int span_u4 = bitmap_end_u4 - bitmap_beg_u4;
    if (threadIdx.x == 0) {
        // printf("[Block %d] Starts. Key range: [%d, %d), Data pos: %d"
        //     "bitmap_beg_u4: %d, bitmap_end_u4: %d, span_u4: %d\n",
            // bid, key_beg, key_end, pos_beg,
            // bitmap_beg_u4, bitmap_end_u4, span_u4);
    }
    __shared__ struct {
        uint4 bitmap[kSharedBitmapLenU4];
    } shared;
    auto* d_bitmap_u4 = reinterpret_cast<const uint4*>(d_bitmaps);

    if (span_u4 < kSharedBitmapLenU4)
    {
        // Dense block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Dense path. span_u4(%d) < kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }
        for (int i = threadIdx.x; i < span_u4; i += blockDim.x)
        {
            shared.bitmap[i] = d_bitmap_u4[bitmap_beg_u4 + i];
        }
        __syncthreads();
        auto *shared_bitmap_shifted = reinterpret_cast<const uint32_t *>(shared.bitmap - bitmap_beg_u4);

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = shared_bitmap_shifted[key >> 5];
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                // key(key), idx(idx), d_values(headers)
                uint32_t value = __ldg(headers + idx);
                atomicOr(d_res + value / 32, 1u << (value % 32));
            }
        });
    }
    else
    {
        // Sparse block
        if (threadIdx.x == 0)
        {
            // printf("[Block %d] Sparse path. span_u4(%d) >= kSharedBitmapLenU4(%d)\n",
            //         bid, span_u4, kSharedBitmapLenU4);
        }

        ForBPKeyIdxs<kBlockSize, kTotalWork>(key_beg, key_end, pos_beg, d_key_buffers,
                                            [&](int key, int idx)
        {
            if (idx >= len)
            {
                return;
            }
            auto qk = __ldg(d_bitmaps + (key >> 5));
            auto cond = qk & (1u << (key & 0x1f));
            if (cond)
            {
                // printf("[Block %d, Thread %d] Match found! Key: %d, Idx: %d\n",
                //         blockIdx.x, threadIdx.x, key, idx);
                // f(key, idx, std::forward<Args>(args)...);
                uint32_t value = __ldg(headers + idx);
                atomicOr(d_res + value / 32, 1u << (value % 32));
                
            }
        });
    }
}


// Global kernel cannot have rvalue arguments
template <int kBlockSize, int kTotalWork>
void RunKeyBPTransformKernel1(cudaStream_t stream, const KeyBlockMeta &meta,
                             const uint32_t *d_bitmaps, const uint32_t *headers,
                             uint32_t *d_res)
{
    if (meta.block_step != kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d)\n", meta.block_step, kTotalWork);
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBPTransformKernel1<kBlockSize, kTotalWork><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            d_bitmaps, headers, d_res);
    }
}


// Global kernel cannot have rvalue arguments
template <int kBlockSize, int kTotalWork, int kArrayType>
void RunKeyBPTransformKernel(cudaStream_t stream, const KeyBlockMeta &meta,
                             const uint32_t *d_bitmaps, const uint32_t *headers,
                             const uint8_t *arrays, uint32_t *d_res)
{
    if (meta.block_step != kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d)\n", meta.block_step, kTotalWork);
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBPTransformKernel<kBlockSize, kTotalWork, kArrayType><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            d_bitmaps, headers, arrays, d_res);
    }
}


// Global kernel cannot have rvalue arguments
template <int kBlockSize, int kTotalWork, int kArrayType>
void RunKeyBPTransformKernelBitset(cudaStream_t stream, const KeyBlockMeta &meta,
                             const uint32_t *d_bitmaps, const uint32_t *headers,
                             const uint8_t *arrays, uint32_t *d_res)
{
    if (meta.block_step != kTotalWork)
    {
        printf("Error: meta.block_step(%d) != kTotalWork(%d)\n", meta.block_step, kTotalWork);
    }
    auto block_len = meta.key_block_segments.len;
    if (block_len > 0)
    {
        KeyBPTransformKernelBitset<kBlockSize, kTotalWork, kArrayType><<<block_len - 1, kBlockSize, 0, stream>>>(
            meta.len, meta.key_pos_segments.d, meta.key_block_segments.d, meta.key_buffers.d,
            d_bitmaps, headers, arrays, d_res);
    }
}
