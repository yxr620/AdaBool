#pragma once

#include <vector>
#include <algorithm>
#include <memory>

#include "len_buffer.h"

struct LoadBalanceTransformer
{
    LenBuffer<INPUT, uint32_t> seg_ids;
    LenBuffer<INPUT, uint32_t> seg_offsets;
    LenBuffer<INPUT, uint32_t> seg_lens;

    /**
     * @brief Prepare load-balanced work units for a list of variable-length segments. (CPU pre-processing)
     *
     * 中文概述：
     *   给定一组按前缀形式表示的段边界数组 h_segments[0..num_segments]（长度为 num_segments+1，
     *   第 i 段范围是 [h_segments[i], h_segments[i+1]) ），本函数在 CPU 上把每个段再按固定 step 大小
     *   切分成若干更小的“子任务”（work unit），目的是：
     *     1. 让后续 GPU Kernel 可以在多个 thread block 间更均匀地分配工作（避免因为个别段过长导致负载不均）。
     *     2. 便于使用统一的 (seg_id, offset, len) 三元组访问真实数据；len<=step，最后一个子任务可能不足 step。
     *
     * 参数说明：
     *   num_segments : 原始段数量（等于 h_segments 中前缀计数长度 - 1）。
     *   h_segments   : host 端数组，长度为 num_segments+1，满足非降序；相邻差值 = 该段真实长度。
     *   step         : 期望的最大子块大小（默认 4096，可根据数据规模或 kernel 期望并行度调节）。
     *
     * 生成的三个 LenBuffer：
     *   seg_ids    : 每个拆分后子任务所属的原段编号 i。
     *   seg_offsets: 在该原段内的偏移（单位：元素个数），沿 [0, step, 2*step, ...] 直到段末尾。
     *   seg_lens   : 子任务实际处理的长度（<= step，最后一个子任务可能小于 step）。
     *
     * 处理流程：
     *   for i in [0, num_segments):
     *       beg = h_segments[i]; end = h_segments[i+1]; len = end - beg
     *       按 0, step, 2*step, ... < len 产生多个分块；对每个 offset 记录一条三元组。
     *
     * 复杂度：O( sum_i ceil(len_i / step) )。额外内存开销与产生的子任务数线性相关。
     *
     * 使用场景：
     *   在 kernel 中用 thread block id 索引 work unit j，读取 seg_ids[j] / seg_offsets[j] / seg_lens[j]
     *   再映射到原数据地址 = base_of_segment(seg_ids[j]) + seg_offsets[j]，处理 seg_lens[j] 个元素。
     *
     * 注意事项：
     *   - 假设 h_segments 是有效的前缀数组且 end >= beg。
     *   - 若 step 过小会产生很多子任务，调度开销增大；过大则负载不均衡可能加剧，需要根据实际分布调优。
     */
    void Prepare(size_t num_segments, uint32_t *h_segments, int step = 4096)
    {
        std::vector<uint32_t> seg_ids_vec;
        std::vector<uint32_t> seg_offsets_vec;
        std::vector<uint32_t> seg_lens_vec;
        for (int i = 0; i < num_segments; ++i)
        {
            int beg = h_segments[i];
            int end = h_segments[i + 1];
            for (int idx = 0; idx < end - beg; idx += step)
            {
                seg_ids_vec.push_back(i);
                seg_offsets_vec.push_back(idx);
                seg_lens_vec.push_back(std::min(end - beg - idx, step));
            }
        }
        seg_ids.AllocFrom(seg_ids_vec);
        seg_offsets.AllocFrom(seg_offsets_vec);
        seg_lens.AllocFrom(seg_lens_vec);
    }

    template <int kBlockSize = 1024, class F, class... Args>
    void Run(cudaStream_t stream, const F &f, Args... args);


    template <int kBlockSize = 1024>
    void RunListUnion1(
        cudaStream_t stream, uint32_t *d_res,
        const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const;

    template <int kArrayType, int kBlockSize = 1024>
    void RunListUnion(
        cudaStream_t stream, uint32_t *d_res,
        const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const;

    template <int kBlockSize = 1024>
    void RunListUnionBitset(
        cudaStream_t stream, uint32_t *d_res,
        const uint32_t *d_indices_beg, const uint32_t *d_value_header, const uint8_t *d_value_arrays) const;

    // No need to prepare. Use GPU to compute the work assignment. May alloc
    // larger device buffer then prepare & run
    template <class State, class F, class... Args>
    void LbsRun(cudaStream_t stream, State *state, int count, int num_segments,
                int *d_segments, const F &f, Args... args);
};
