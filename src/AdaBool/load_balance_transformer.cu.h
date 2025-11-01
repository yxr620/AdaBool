
#pragma once

#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>  // for debug printf

#include "len_buffer.h"
#include "load_balance_transformer.h"
#include "cuda_op_utility.h"
#include "inverted_list.cu.h"

__global__ void ListUnionKernel1(
    uint32_t *d_res, const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];


    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        // id (seg_id), offset + idx (seg_offset), d_indices_beg, d_values(d_value_header)
        uint32_t idx_beg = __ldg(d_indices_beg + id); 
        uint32_t idx = idx_beg + offset + i;
        uint32_t value = __ldg(d_value_header + idx);
        // seg_id, value, d_res
        // atomicOr(d_res + v / 32, 1u << (v % 32));
        atomicOr(d_res + value / 32, 1u << (value % 32));

        // printf("Thread b:%d, t:%d, ListUnionKernel, seg_id: %d, seg_offset: %d, idx_beg: %d, idx: %d, value: %d\n",
        //        blockIdx.x, threadIdx.x, id, offset + i, idx_beg, idx, value);
    }
}

template <int kArrayType>
__global__ void ListUnionKernel(
    uint32_t *d_res, const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];


    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        // id (seg_id), offset + i (seg_offset), d_indices_beg, d_values(d_value_header), d_value_arrays
        uint32_t idx_beg = __ldg(d_indices_beg + id);
        uint32_t idx = idx_beg + offset + i;
        // seg_id, idx, d_value_headers, d_value_arrays
        ProcessArrayFunctor<kArrayType>(id, idx, d_value_header, d_value_arrays, d_res);
    }
}

__global__ void ListUnionKernelBitset(
    uint32_t *d_res, const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        //ProcessBitsetFunctor(segment) id (seg_id), offset + i (seg_offset), d_indices_beg, d_values(d_value_header), d_value_arrays
        uint32_t idx_beg = __ldg(d_indices_beg + id);
        uint32_t idx = idx_beg + offset + i;
        //ProcessBitsetFunctor(key idx) key(id), idx, d_value_headers, d_bitsets(d_value_arrays)
        using T = uint4; // 128 bits
        auto *d_bitsets_t = reinterpret_cast<const T *>(d_value_arrays);
        uint32_t header = __ldg(d_value_header + idx);
        T bits[2];
        bits[0] = __ldg(d_bitsets_t + idx * 2);
        bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
        auto *bits_u32 = reinterpret_cast<const uint32_t*>(&bits);
    #pragma unroll
        for (uint32_t j = 0; j < 256; j+=32) {
            uint32_t v = header | j;
            atomicOr(d_res + v / 32, bits_u32[j/32]); // This is already the Fast version
            // atomicOr(d_res + v / 32, bits_u32[j/32]);
        }
    }
}

template <int kBlockSize>
void LoadBalanceTransformer::RunListUnion1(
    cudaStream_t stream, uint32_t *d_res,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const
{
    if (seg_ids.len > 0)
    {
        ListUnionKernel1<<<seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, seg_ids.d, seg_offsets.d, seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}

template <int kArrayType, int kBlockSize>
void LoadBalanceTransformer::RunListUnion(
    cudaStream_t stream, uint32_t *d_res,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const
{
    if (seg_ids.len > 0)
    {
        ListUnionKernel<kArrayType><<<seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, seg_ids.d, seg_offsets.d, seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}

template <int kBlockSize>
void LoadBalanceTransformer::RunListUnionBitset(
    cudaStream_t stream, uint32_t *d_res,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const
{
    if (seg_ids.len > 0)
    {
        ListUnionKernelBitset<<<seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, seg_ids.d, seg_offsets.d, seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}


// ==========================================================================================
// 1. Batch-Enabled CUDA Kernels
//    (Based on kernels from load_balance_transformer.cu.h, extended for batch processing)
// ==========================================================================================

__global__ void SparseBatchListUnionKernel1(
    uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int seg_id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];

    uint32_t qidx = d_qidx[seg_id];
    uint32_t* d_res_for_batch = d_res + qidx * res_len_per_batch_u32;

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        uint32_t idx_beg = __ldg(d_indices_beg + seg_id);
        uint32_t idx = idx_beg + offset + i;
        uint32_t value = __ldg(d_value_header + idx);
        atomicOr(d_res_for_batch + value / 32, 1u << (value % 32));
    }
}

template <int kArrayType>
__global__ void SparseBatchListUnionKernel(
    uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int seg_id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];

    uint32_t qidx = d_qidx[seg_id];
    uint32_t* d_res_for_batch = d_res + qidx * res_len_per_batch_u32;

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        uint32_t idx_beg = __ldg(d_indices_beg + seg_id);
        uint32_t idx = idx_beg + offset + i;
        // The key for ProcessArrayFunctor is the original segment id
        ProcessArrayFunctor<kArrayType>(seg_id, idx, d_value_header, d_value_arrays, d_res_for_batch);
    }
}

__global__ void SparseBatchListUnionKernelBitset(
    uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_ids, const uint32_t *d_offsets, const uint32_t *d_lens,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    int bid = blockIdx.x;
    int seg_id = d_ids[bid];
    int offset = d_offsets[bid];
    int len = d_lens[bid];

    uint32_t qidx = d_qidx[seg_id];
    uint32_t* d_res_for_batch = d_res + qidx * res_len_per_batch_u32;

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        // ############ FIX 1: Corrected `id` to `seg_id` here ############
        uint32_t idx_beg = __ldg(d_indices_beg + seg_id);
        uint32_t idx = idx_beg + offset + i;

        using T = uint4; // 128 bits
        auto *d_bitsets_t = reinterpret_cast<const T *>(d_value_arrays);
        uint32_t header = __ldg(d_value_header + idx);
        T bits[2];
        bits[0] = __ldg(d_bitsets_t + idx * 2);
        bits[1] = __ldg(d_bitsets_t + idx * 2 + 1);
        auto *bits_u32 = reinterpret_cast<const uint32_t*>(&bits);

        // ############ FIX 2: Moved constant calculation out of the loop ############
        uint32_t v_h = header & 0xffffff00u;
        uint32_t* base_addr = d_res_for_batch + (v_h / 32);

    #pragma unroll
        for (uint32_t j = 0; j < 8; j++) { // 256 bits = 8 * 32 bits
            // ############ FIX 3: Correctly apply the atomicOr for each 32-bit chunk ############
            atomicOr(base_addr + j, bits_u32[j]);
        }
    }
}


// ==========================================================================================
// 2. Batch-Enabled Kernel Launchers
// ==========================================================================================

template <int kBlockSize = 256>
void RunSparseBatchListUnion1(
    const LoadBalanceTransformer& worker, cudaStream_t stream, uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    if (worker.seg_ids.len > 0)
    {
        SparseBatchListUnionKernel1<<<worker.seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, d_qidx, res_len_per_batch_u32, worker.seg_ids.d, worker.seg_offsets.d, worker.seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}

template <int kArrayType, int kBlockSize = 256>
void RunSparseBatchListUnion(
    const LoadBalanceTransformer& worker, cudaStream_t stream, uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    if (worker.seg_ids.len > 0)
    {
        SparseBatchListUnionKernel<kArrayType><<<worker.seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, d_qidx, res_len_per_batch_u32, worker.seg_ids.d, worker.seg_offsets.d, worker.seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}

template <int kBlockSize = 256>
void RunSparseBatchListUnionBitset(
    const LoadBalanceTransformer& worker, cudaStream_t stream, uint32_t *d_res, const uint32_t *d_qidx, uint32_t res_len_per_batch_u32,
    const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays)
{
    if (worker.seg_ids.len > 0)
    {
        SparseBatchListUnionKernelBitset<<<worker.seg_ids.len, kBlockSize, 0, stream>>>(
            d_res, d_qidx, res_len_per_batch_u32, worker.seg_ids.d, worker.seg_offsets.d, worker.seg_lens.d,
            d_indices_beg, d_value_header, d_value_arrays);
    }
}

