#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// MemType
enum MemType
{
    UNKNOWN = 0, // cannot compile
    HOST = 1,
    DEVICE = 2,
    INPUT = 3,
    OUTPUT = 4,
};

// Simple CUDA error wrapper (no-op if CUDA not available)
#ifdef __CUDACC__
inline void _check_cuda(cudaError_t e, const char *file, int line)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)e, cudaGetErrorString(e), file, line);
        std::abort();
    }
}
#define LENBUF_CUDA_CHECK(expr) _check_cuda((expr), __FILE__, __LINE__)
#else
#define LENBUF_CUDA_CHECK(expr) ((void)0)
#endif

template <MemType mtype, class T>
struct LenBuffer;

// HOST: only host memory
template <class T>
struct LenBuffer<HOST, T>
{
    size_t len{0};
    T *h{nullptr};

    LenBuffer() = default;
    explicit LenBuffer(size_t size) { Alloc(size); }
    ~LenBuffer() { reset(); }

    LenBuffer(const LenBuffer &) = delete;
    LenBuffer &operator=(const LenBuffer &) = delete;
    LenBuffer(LenBuffer &&o) noexcept { swap(o); }
    LenBuffer &operator=(LenBuffer &&o) noexcept
    {
        if (this != &o)
        {
            reset();
            swap(o);
        }
        return *this;
    }

    T *Alloc(size_t size)
    {
        reset();
        if (size == 0)
            return nullptr;
        len = size;
        h = static_cast<T *>(std::malloc(len * sizeof(T)));
        if (!h)
            throw std::bad_alloc();
        return h;
    }
    void reset()
    {
        if (h)
            std::free(h);
        h = nullptr;
        len = 0;
    }
    void swap(LenBuffer &o) noexcept
    {
        std::swap(len, o.len);
        std::swap(h, o.h);
    }
    const T *begin() const { return h; }
    const T *end() const { return h + len; }

    // Allocate and copy from pointer range (host only)
    T *AllocFrom(const T *src, size_t n)
    {
        if (!src || n == 0)
        {
            reset();
            return nullptr;
        }
        Alloc(n);
        std::copy(src, src + n, h);
        return h;
    }
    // Allocate and copy from std::vector
    T *AllocFrom(const std::vector<T> &v)
    {
        return AllocFrom(v.data(), v.size());
    }
};

// DEVICE: only device memory
template <class T>
struct LenBuffer<DEVICE, T>
{
    size_t len{0};
    T *d{nullptr};

    LenBuffer() = default;
    explicit LenBuffer(size_t size) { Alloc(size); }
    ~LenBuffer() { reset(); }
    LenBuffer(const LenBuffer &) = delete;
    LenBuffer &operator=(const LenBuffer &) = delete;
    LenBuffer(LenBuffer &&o) noexcept { swap(o); }
    LenBuffer &operator=(LenBuffer &&o) noexcept
    {
        if (this != &o)
        {
            reset();
            swap(o);
        }
        return *this;
    }

    T *Alloc(size_t size)
    {
        reset();
        if (size == 0)
            return nullptr;
        len = size;
#ifdef __CUDACC__
        LENBUF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), len * sizeof(T)));
#else
        // Fallback: allocate host memory if CUDA not present (still marks as device logically)
        d = static_cast<T *>(std::malloc(len * sizeof(T)));
        if (!d)
            throw std::bad_alloc();
#endif
        return d;
    }
    void reset()
    {
#ifdef __CUDACC__
        if (d)
            cudaFree(d);
#else
        if (d)
            std::free(d);
#endif
        d = nullptr;
        len = 0;
    }
    void swap(LenBuffer &o) noexcept
    {
        std::swap(len, o.len);
        std::swap(d, o.d);
    }

    // Allocate device buffer and copy from host pointer
    T *AllocFrom(const T *src, size_t n)
    {
    if (!src || n == 0)
    {
        reset();
        return nullptr;
    }
    Alloc(n);
#ifdef __CUDACC__
    LENBUF_CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(T), cudaMemcpyHostToDevice));
#else
    // Fallback: treat d as host memory already allocated above
    std::copy(src, src + n, d);
#endif
    return d;
    }
    T *AllocFrom(const std::vector<T> &v) { return AllocFrom(v.data(), v.size()); }
};

// INPUT: host + device (used for uploading data)
template <class T>
struct LenBuffer<INPUT, T>
{
    size_t len{0};
    T *h{nullptr};
    T *d{nullptr};

    LenBuffer() = default;
    explicit LenBuffer(size_t size) { Alloc(size); }
    ~LenBuffer() { reset(); }
    LenBuffer(const LenBuffer &) = delete;
    LenBuffer &operator=(const LenBuffer &) = delete;
    LenBuffer(LenBuffer &&o) noexcept { swap(o); }
    LenBuffer &operator=(LenBuffer &&o) noexcept
    {
        if (this != &o)
        {
            reset();
            swap(o);
        }
        return *this;
    }

    T *Alloc(size_t size)
    {
        reset();
        if (size == 0)
            return nullptr;
        len = size;
        h = static_cast<T *>(std::malloc(len * sizeof(T)));
        if (!h)
            throw std::bad_alloc();
#ifdef __CUDACC__
        LENBUF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), len * sizeof(T)));
#else
        d = nullptr; // no device buffer without CUDA
#endif
        return h;
    }
    void reset()
    {
#ifdef __CUDACC__
        if (d)
            cudaFree(d);
#else
        if (d)
            std::free(d); // Only allocated if no CUDA (fallback not used here currently)
#endif
        if (h)
            std::free(h);
        d = nullptr;
        h = nullptr;
        len = 0;
    }
    void swap(LenBuffer &o) noexcept
    {
        std::swap(len, o.len);
        std::swap(h, o.h);
        std::swap(d, o.d);
    }
    const T *begin() const { return h; }
    const T *end() const { return h + len; }

    // Allocate host(+device) and copy from host pointer into both
    T *AllocFrom(const T *src, size_t n)
    {
        if (!src || n == 0)
        {
            reset();
            return nullptr;
        }
        Alloc(n);
        std::copy(src, src + n, h);
#ifdef __CUDACC__
        if (d)
            LENBUF_CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
#endif
        return h;
    }
    T *AllocFrom(const std::vector<T> &v) { return AllocFrom(v.data(), v.size()); }
};

// OUTPUT: host + device (used for downloading results)
template <class T>
struct LenBuffer<OUTPUT, T>
{
    size_t len{0};
    T *h{nullptr};
    T *d{nullptr};

    LenBuffer() = default;
    explicit LenBuffer(size_t size) { Alloc(size); }
    ~LenBuffer() { reset(); }
    LenBuffer(const LenBuffer &) = delete;
    LenBuffer &operator=(const LenBuffer &) = delete;
    LenBuffer(LenBuffer &&o) noexcept { swap(o); }
    LenBuffer &operator=(LenBuffer &&o) noexcept
    {
        if (this != &o)
        {
            reset();
            swap(o);
        }
        return *this;
    }

    T *Alloc(size_t size)
    {
        reset();
        if (size == 0)
            return nullptr;
        len = size;
        h = static_cast<T *>(std::malloc(len * sizeof(T)));
        if (!h)
            throw std::bad_alloc();
#ifdef __CUDACC__
        LENBUF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d), len * sizeof(T)));
#else
        d = nullptr; // no device buffer without CUDA
#endif
        return h;
    }
    void reset()
    {
#ifdef __CUDACC__
        if (d)
            cudaFree(d);
#else
        if (d)
            std::free(d);
#endif
        if (h)
            std::free(h);
        d = nullptr;
        h = nullptr;
        len = 0;
    }
    void swap(LenBuffer &o) noexcept
    {
        std::swap(len, o.len);
        std::swap(h, o.h);
        std::swap(d, o.d);
    }
    const T *begin() const { return h; }
    const T *end() const { return h + len; }

    // Allocate host(+device) and copy from host pointer into both (OUTPUT side typically download target but allow init)
    T *AllocFrom(const T *src, size_t n)
    {
        if (!src || n == 0)
        {
            reset();
            return nullptr;
        }
        Alloc(n);
        std::copy(src, src + n, h);
#ifdef __CUDACC__
        if (d)
            LENBUF_CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
#endif
        return h;
    }
    T *AllocFrom(const std::vector<T> &v) { return AllocFrom(v.data(), v.size()); }
};

inline bool IsNonEmpty(bool b) { return b; }

template <class T>
bool IsNonEmpty(const T &b) { return b.len != 0; }

template <class T, class... Ts>
bool IsNonEmpty(const T &b, const Ts &...bs) { return IsNonEmpty(b) && IsNonEmpty(bs...); }

#define CHECK_NON_EMPTY(args...)                                \
    do                                                          \
    {                                                           \
        if (!::gdt::sunfish::IsNonEmpty(args))                  \
        {                                                       \
            assert(!"LenBuffer: expected non-empty buffer(s)"); \
        }                                                       \
    } while (false)
