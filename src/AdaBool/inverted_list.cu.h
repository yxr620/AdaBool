#pragma once

#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio> // for debug printf

#include "inverted_list.h"
#include "cuda_op_utility.h"
#include "key_batch_transformer.cu.h"

template <int>
struct ArraySizeTrait;

template <>
struct ArraySizeTrait<LIST_ARRAY1>
{
    static const int kArraySize = 1;
};

template <>
struct ArraySizeTrait<LIST_ARRAY4>
{
    using CopyType = uint;
    static const int kMinNum = 1;
    static const int kArraySize = 4;
};

template <>
struct ArraySizeTrait<LIST_ARRAY8>
{
    using CopyType = uint2;
    static const int kMinNum = 5;
    static const int kArraySize = 8;
};

template <>
struct ArraySizeTrait<LIST_ARRAY16>
{
    using CopyType = uint4;
    static const int kMinNum = 9;
    static const int kArraySize = 16;
};

template <>
struct ArraySizeTrait<LIST_ARRAY32>
{
    using CopyType = ulonglong4;
    static const int kMinNum = 17;
    static const int kArraySize = 32;
};

template <>
struct ArraySizeTrait<LIST_BITSET>
{
    static const int kArraySize = 8;
};

template <int kArrayType>
__device__ void ProcessArrayFunctor(
    uint32_t key, uint32_t idx, const uint32_t *d_value_headers,
    const uint8_t *d_value_arrays, uint32_t *d_res)
{
    const int kArraySize = ArraySizeTrait<kArrayType>::kArraySize;
    using T = typename ArraySizeTrait<kArrayType>::CopyType;
    constexpr int kCopyBytes = sizeof(T);
    constexpr int kMinNum = ArraySizeTrait<kArrayType>::kMinNum;
    uint32_t header = __ldg(d_value_headers + idx);
    uint32_t h = header & 0xffffff00u;
    auto ls_u8 = DeviceLoadBytes<kCopyBytes, uint8_t>(d_value_arrays, idx);
    // key, header,
    atomicOr(d_res + header / 32, 1u << (header % 32));
    // printf("ProcessArrayFunctor<%d> thread d:%d, t:%d, <key:%d, idx:%d> <header:%d>\n",
    //    kArrayType, threadIdx.x, blockIdx.x, key, idx, header);
#pragma unroll
    for (int i = 0; i < kMinNum; ++i)
    {
        uint32_t value = h | ls_u8[i];
        atomicOr(d_res + value / 32, 1u << (value % 32));
        // printf("thread d:%d, t:%d, value: %d\n", threadIdx.x, blockIdx.x, value);
    }
#pragma unroll
    for (int i = kMinNum; i < kArraySize; ++i)
    {
        if (!ls_u8[i])
        {
            return;
        }
        uint32_t value = h | ls_u8[i];
        atomicOr(d_res + value / 32, 1u << (value % 32));
    }
}

template <int kArrayType>
__device__ void ProcessArrayFunctor_BatchExpand(
    uint32_t key, uint32_t idx, const uint32_t *d_value_headers,
    const uint8_t *d_value_arrays, uint32_t *d_res,
    uint32_t qidx, uint32_t res_len)
{
    const int kArraySize = ArraySizeTrait<kArrayType>::kArraySize;
    using T = typename ArraySizeTrait<kArrayType>::CopyType;
    constexpr int kCopyBytes = sizeof(T);
    constexpr int kMinNum = ArraySizeTrait<kArrayType>::kMinNum;
    uint32_t header = __ldg(d_value_headers + idx);
    uint32_t h = header & 0xffffff00u;
    auto ls_u8 = DeviceLoadBytes<kCopyBytes, uint8_t>(d_value_arrays, idx);
    // key, header,
    atomicOr(d_res + header / 32 + qidx * res_len, 1u << (header % 32));
    // printf("ProcessArrayFunctor<%d> thread d:%d, t:%d, <key:%d, idx:%d> <header:%d>\n",
    //    kArrayType, threadIdx.x, blockIdx.x, key, idx, header);
#pragma unroll
    for (int i = 0; i < kMinNum; ++i)
    {
        uint32_t value = h | ls_u8[i];
        atomicOr(d_res + value / 32 + qidx * res_len, 1u << (value % 32));
        // printf("thread d:%d, t:%d, value: %d\n", threadIdx.x, blockIdx.x, value);
    }
#pragma unroll
    for (int i = kMinNum; i < kArraySize; ++i)
    {
        if (!ls_u8[i])
        {
            return;
        }
        uint32_t value = h | ls_u8[i];
        atomicOr(d_res + value / 32 + qidx * res_len, 1u << (value % 32));
    }
}

template <int kKeyStep, int kThreadNum>
template <int kBatchSize>
void InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>::RunArraysImpl(
    cudaStream_t stream, uint32_t *d_res, uint32_t res_len)
{
    auto &headers0 = headers->at(LIST_ARRAY1);
    auto &meta0 = metas->at(LIST_ARRAY1);
    RunKeyBatchBPTransformKernel1<kBatchSize, kThreadNum, kKeyStep>(
        stream, meta0, query_bitmap_len, d_query_bitmap, headers0.d, d_res, res_len);

    auto &headers1 = headers->at(LIST_ARRAY4);
    auto &arrays1 = arrays->at(LIST_ARRAY4);
    auto &meta1 = metas->at(LIST_ARRAY4);
    RunKeyBatchBPTransformKernel<kBatchSize, kThreadNum, kKeyStep, LIST_ARRAY4>(
        stream, meta1, query_bitmap_len, d_query_bitmap, headers1.d, arrays1.d, d_res, res_len);

    auto &headers2 = headers->at(LIST_ARRAY8);
    auto &arrays2 = arrays->at(LIST_ARRAY8);
    auto &meta2 = metas->at(LIST_ARRAY8);
    RunKeyBatchBPTransformKernel<kBatchSize, kThreadNum, kKeyStep, LIST_ARRAY8>(
        stream, meta2, query_bitmap_len, d_query_bitmap, headers2.d, arrays2.d, d_res, res_len);
    auto &headers3 = headers->at(LIST_ARRAY16);
    auto &arrays3 = arrays->at(LIST_ARRAY16);
    auto &meta3 = metas->at(LIST_ARRAY16);
    RunKeyBatchBPTransformKernel<kBatchSize, kThreadNum, kKeyStep, LIST_ARRAY16>(
        stream, meta3, query_bitmap_len, d_query_bitmap, headers3.d, arrays3.d, d_res, res_len);
    auto &headers4 = headers->at(LIST_ARRAY32);
    auto &arrays4 = arrays->at(LIST_ARRAY32);
    auto &meta4 = metas->at(LIST_ARRAY32);
    RunKeyBatchBPTransformKernel<kBatchSize, kThreadNum, kKeyStep, LIST_ARRAY32>(
        stream, meta4, query_bitmap_len, d_query_bitmap, headers4.d, arrays4.d, d_res, res_len);

    // Bitmap
    auto &headers5 = headers->at(LIST_BITSET);
    auto &arrays5 = arrays->at(LIST_BITSET);
    auto &meta5 = metas->at(LIST_BITSET);
    RunKeyBatchBPTransformKernelBitset<kBatchSize, kThreadNum, kKeyStep>(
        stream, meta5, query_bitmap_len, d_query_bitmap, headers5.d, arrays5.d, d_res, res_len);
    


}

// Since it has template parameters including a device functor,
// it cannot be represented as function pointers.
// Otherwise the device functor cannot be inlined.
// Those implementations that are guaranteed to compile to function table lookup are
// either too complicated or require higher standard of C++
// Thus an explicit function table is used here
#define FTABLE_ENTRY_NOARGS(f, i)     \
    case i:                           \
        f<i>(stream, d_res, res_len); \
        break
#define GENERATE_FTABLE_NOARGS(val, f) \
    switch (val)                       \
    {                                  \
        FTABLE_ENTRY_NOARGS(f, 1);     \
        FTABLE_ENTRY_NOARGS(f, 2);     \
        FTABLE_ENTRY_NOARGS(f, 3);     \
        FTABLE_ENTRY_NOARGS(f, 4);     \
        FTABLE_ENTRY_NOARGS(f, 5);     \
        FTABLE_ENTRY_NOARGS(f, 6);     \
        FTABLE_ENTRY_NOARGS(f, 7);     \
        FTABLE_ENTRY_NOARGS(f, 8);     \
        FTABLE_ENTRY_NOARGS(f, 9);     \
        FTABLE_ENTRY_NOARGS(f, 10);    \
        FTABLE_ENTRY_NOARGS(f, 11);    \
        FTABLE_ENTRY_NOARGS(f, 12);    \
        FTABLE_ENTRY_NOARGS(f, 13);    \
        FTABLE_ENTRY_NOARGS(f, 14);    \
        FTABLE_ENTRY_NOARGS(f, 15);    \
        FTABLE_ENTRY_NOARGS(f, 16);    \
        FTABLE_ENTRY_NOARGS(f, 17);    \
        FTABLE_ENTRY_NOARGS(f, 18);    \
        FTABLE_ENTRY_NOARGS(f, 19);    \
        FTABLE_ENTRY_NOARGS(f, 20);    \
        FTABLE_ENTRY_NOARGS(f, 21);    \
        FTABLE_ENTRY_NOARGS(f, 22);    \
        FTABLE_ENTRY_NOARGS(f, 23);    \
        FTABLE_ENTRY_NOARGS(f, 24);    \
        FTABLE_ENTRY_NOARGS(f, 25);    \
        FTABLE_ENTRY_NOARGS(f, 26);    \
        FTABLE_ENTRY_NOARGS(f, 27);    \
        FTABLE_ENTRY_NOARGS(f, 28);    \
        FTABLE_ENTRY_NOARGS(f, 29);    \
        FTABLE_ENTRY_NOARGS(f, 30);    \
        FTABLE_ENTRY_NOARGS(f, 31);    \
        FTABLE_ENTRY_NOARGS(f, 32);    \
    default:                           \
        break;                         \
    }

template <int kKeyStep, int kThreadNum>
void InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>::RunArrays(cudaStream_t stream, uint32_t *d_res, uint32_t res_len)
{
    GENERATE_FTABLE_NOARGS(batch_size, RunArraysImpl);
}

// template <int kKeyStep, int kThreadNum>
// void InvertedListBatchWorkerImpl<kKeyStep, kThreadNum>::RunBitsets(cudaStream_t stream)
// {
//     GENERATE_FTABLE(batch_size, RunBitsetsImpl, f);
// }

#undef GENERATE_FTABLE
#undef FTABLE_ENTRY
