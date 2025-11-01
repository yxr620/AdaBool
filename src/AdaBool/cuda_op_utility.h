#include <cuda_runtime.h>
#include <cstdint>

#pragma once

__device__ __forceinline__ unsigned int LaneMaskLt() {
  unsigned int ret;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));
  return ret;
}

template <int multiplier = 1>
__device__ inline int atomicAggInc(int* ctr) {
  unsigned int active = __activemask();
  int leader = __ffs(active) - 1;
  int change = __popc(active) * multiplier;
  int rank = __popc(active & LaneMaskLt());
  int warp_res;
  if (rank == 0) {
    warp_res = atomicAdd(ctr, change);
  }
  warp_res = __shfl_sync(active, warp_res, leader);
  return warp_res + rank * multiplier;
}

template <class T, int kSize>
struct SimpleArray {
  T x[kSize];
  __device__ T& operator[](int i) {
    return x[i];
  }
  __device__ const T& operator[](int i) const {
    return x[i];
  }
};

template <class T, int kNumBytes>
struct NumBytesTrait {
  typedef int4 CopyType;
  typedef SimpleArray<T, kNumBytes / sizeof(T)> ReturnType;
  static_assert(kNumBytes / sizeof(T) > 0, "unsupported #bytes");
};

template <class T>
struct NumBytesTrait<T, 1> {
  typedef int8_t CopyType;
  typedef SimpleArray<T, 1 / sizeof(T)> ReturnType;
};

template <class T>
struct NumBytesTrait<T, 2> {
  typedef int16_t CopyType;
  typedef SimpleArray<T, 2 / sizeof(T)> ReturnType;
};

template <class T>
struct NumBytesTrait<T, 4> {
  typedef int CopyType;
  typedef SimpleArray<T, 4 / sizeof(T)> ReturnType;
};

template <class T>
struct NumBytesTrait<T, 8> {
  typedef int2 CopyType;
  typedef SimpleArray<T, 8 / sizeof(T)> ReturnType;
};

template <int kNumElems = 16, class T = uint32_t>
__device__ typename NumBytesTrait<T, kNumElems * sizeof(T)>::ReturnType DeviceLoadBytes(
    const void* data, int offset = 0) {
  constexpr int kNumBytes = kNumElems * sizeof(T);
  using CopyType = typename NumBytesTrait<T, kNumBytes>::CopyType;
  auto* data_cp = reinterpret_cast<const CopyType*>(data);
  typename NumBytesTrait<T, kNumBytes>::ReturnType ret;
  auto* ret_cp = reinterpret_cast<CopyType*>(&ret);
  constexpr int kRetElem = sizeof(ret) / sizeof(*data_cp);
#pragma unroll
  for (int i = 0; i < kRetElem; ++i) {
    ret_cp[i] = __ldg(data_cp + offset * kRetElem + i);
  }
  return ret;
}
