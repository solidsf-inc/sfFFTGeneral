// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sfFFT {
namespace detail {
__device__ __forceinline__ float2 rotate(float2 value, float cosine, float sine) {
    return make_float2(fmaf(-value.y, sine, value.x * cosine),
                       fmaf(value.x, sine, value.y * cosine));
}

__device__ __forceinline__ unsigned shared_index(unsigned index) {
    return index ^ ((index >> 4) & 15u) ^ ((index >> 8) & 15u) ^ ((index >> 12) & 15u);
}

__device__ __forceinline__ void twiddle_pi(float x, float* sine, float* cosine) {
    const int quadrant = __float2int_rn(2.0f * x);
    const float r = fmaf(-0.5f, float(quadrant), x);
    const float squared = r * r;
    // Exact dyadic quadrant reduction leaves |r| <= 1/4; Taylor degrees 11 and 12.
    float sp = fmaf(squared, -0.007370430945714348f, 0.08214588661112819f);
    sp = fmaf(squared, sp, -0.5992645293207919f);
    sp = fmaf(squared, sp, 2.550164039877345f);
    sp = fmaf(squared, sp, -5.167712780049969f);
    sp = fmaf(squared, sp, 3.141592653589793f);
    float cp = fmaf(squared, 0.001929574309403922f, -0.02580689139001405f);
    cp = fmaf(squared, cp, 0.23533063035889312f);
    cp = fmaf(squared, cp, -1.3352627688545893f);
    cp = fmaf(squared, cp, 4.058712126416768f);
    cp = fmaf(squared, cp, -4.934802200544679f);
    const float s = r * sp, c = fmaf(squared, cp, 1.0f);
    const float flip = quadrant & 2 ? -1.0f : 1.0f;
    *sine = (quadrant & 1 ? c : s) * flip;
    *cosine = (quadrant & 1 ? -s : c) * flip;
}

template<unsigned Bits, unsigned Threads>
static __global__ __launch_bounds__(Threads) void fused(float2* data, std::size_t batches, bool inverse) {
    constexpr unsigned n = 1u << Bits;
    constexpr unsigned values = n < Threads ? 1 : n / Threads;
    extern __shared__ float2 scratch[];
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread & 31u;
    const float sign = inverse ? 2.0f : -2.0f;
    for (std::size_t batch = blockIdx.x; batch < batches; batch += gridDim.x) {
        float2 local[values];
        #pragma unroll
        for (unsigned r = 0; r < values; ++r)
            local[r] = thread + r * Threads < n ? data[batch * n + thread + r * Threads]
                                               : make_float2(0.0f, 0.0f);
        #pragma unroll
        for (unsigned width = n; width > Threads; width >>= 1) {
            const unsigned step = width / (2 * Threads);
            #pragma unroll
            for (unsigned r = 0; r < step; ++r) {
                float sine, cosine;
                twiddle_pi(sign * float(thread + r * Threads) / float(width), &sine, &cosine);
                #pragma unroll
                for (unsigned base = 0; base < values; base += 2 * step) {
                    const float2 a = local[base + r], b = local[base + r + step];
                    local[base + r] = make_float2(a.x + b.x, a.y + b.y);
                    local[base + r + step] = rotate(make_float2(a.x - b.x, a.y - b.y), cosine, sine);
                }
            }
        }
        #pragma unroll
        for (unsigned width = n < Threads ? n : Threads; width > 32; width >>= 1) {
            const unsigned half = width / 2;
            if constexpr (n == 8192) {
                const unsigned pair = thread / width * half + (thread & (half - 1));
                if (thread & half) {
                    #pragma unroll
                    for (unsigned r = 0; r < values; ++r)
                        scratch[shared_index(pair + r * (Threads / 2))] = local[r];
                }
                __syncthreads();
                if (!(thread & half)) {
                    float sine, cosine;
                    twiddle_pi(sign * float(thread & (half - 1)) / float(width), &sine, &cosine);
                    #pragma unroll
                    for (unsigned r = 0; r < values; ++r) {
                        const unsigned index = shared_index(pair + r * (Threads / 2));
                        const float2 a = local[r], b = scratch[index];
                        local[r] = make_float2(a.x + b.x, a.y + b.y);
                        scratch[index] = rotate(make_float2(a.x - b.x, a.y - b.y), cosine, sine);
                    }
                }
                __syncthreads();
                if (thread & half) {
                    #pragma unroll
                    for (unsigned r = 0; r < values; ++r)
                        local[r] = scratch[shared_index(pair + r * (Threads / 2))];
                }
                __syncthreads();
            } else {
                float sine, cosine;
                twiddle_pi(sign * float(thread & (half - 1)) / float(width), &sine, &cosine);
                #pragma unroll
                for (unsigned r = 0; r < values; ++r) scratch[shared_index(thread + r * Threads)] = local[r];
                __syncthreads();
                #pragma unroll
                for (unsigned r = 0; r < values; ++r) {
                    const float2 a = local[r], b = scratch[shared_index((thread ^ half) + r * Threads)];
                    local[r] = thread & half ? rotate(make_float2(b.x - a.x, b.y - a.y), cosine, sine)
                                             : make_float2(a.x + b.x, a.y + b.y);
                }
                __syncthreads();
            }
        }
        float warp_sine = 0.0f, warp_cosine = 1.0f;
        if constexpr (n >= 8)
            twiddle_pi(sign * float(lane & 15u) / 32.0f, &warp_sine, &warp_cosine);
        #pragma unroll
        for (unsigned width = n < 32 ? n : 32; width >= 2; width >>= 1) {
            const unsigned half = width / 2;
            float sine, cosine;
            if (width == 2) { sine = 0.0f; cosine = 1.0f; }
            else if (width == 4) {
                sine = lane & 1u ? sign * 0.5f : 0.0f;
                cosine = lane & 1u ? 0.0f : 1.0f;
            } else if (width == 32) { sine = warp_sine; cosine = warp_cosine; }
            else {
                const unsigned source = (lane & (half - 1)) * (32 / width);
                sine = __shfl_sync(0xffffffffu, warp_sine, source);
                cosine = __shfl_sync(0xffffffffu, warp_cosine, source);
            }
            #pragma unroll
            for (unsigned r = 0; r < values; ++r) {
                const float2 a = local[r];
                const float2 b = make_float2(__shfl_xor_sync(0xffffffffu, a.x, half),
                                            __shfl_xor_sync(0xffffffffu, a.y, half));
                local[r] = lane & half ? rotate(make_float2(b.x - a.x, b.y - a.y), cosine, sine)
                                       : make_float2(a.x + b.x, a.y + b.y);
            }
        }
        const float scale = inverse ? 1.0f / n : 1.0f;
        if constexpr (n == 8192) {
            #pragma unroll
            for (unsigned part = 0; part < 2; ++part) {
                if ((thread & 1u) == part) {
                    #pragma unroll
                    for (unsigned r = 0; r < values; ++r) {
                        const unsigned reversed = __brev(thread + r * Threads) >> (32 - Bits);
                        scratch[shared_index(reversed & (n / 2 - 1))] = make_float2(local[r].x * scale, local[r].y * scale);
                    }
                }
                __syncthreads();
                #pragma unroll
                for (unsigned r = 0; r < values / 2; ++r) {
                    const unsigned index = thread + r * Threads;
                    data[batch * n + part * (n / 2) + index] = scratch[shared_index(index)];
                }
                __syncthreads();
            }
        } else {
            #pragma unroll
            for (unsigned r = 0; r < values; ++r) {
                const unsigned index = thread + r * Threads;
                if (index < n) {
                    const unsigned reversed = __brev(index) >> (32 - Bits);
                    scratch[shared_index(reversed)] = make_float2(local[r].x * scale, local[r].y * scale);
                }
            }
            if constexpr (Threads == 32) __syncwarp();
            else __syncthreads();
            #pragma unroll
            for (unsigned r = 0; r < values; ++r) {
                const unsigned index = thread + r * Threads;
                if (index < n) data[batch * n + index] = scratch[shared_index(index)];
            }
            if constexpr (Threads == 32) __syncwarp();
            else __syncthreads();
        }
    }
}

template<unsigned Bits>
inline cudaError_t launch_fused(float2* data, std::size_t batches, bool inverse, cudaStream_t stream) {
    constexpr unsigned n = 1u << Bits;
    constexpr unsigned threads = n <= 256 ? 32 : (n == 512 ? 256 : 512);
    constexpr unsigned shared_bytes = n * sizeof(float2) / (n == 8192 ? 2 : 1);
    const unsigned grid = static_cast<unsigned>(batches < 65535 ? batches : 65535);
    fused<Bits, threads><<<grid, threads, shared_bytes, stream>>>(data, batches, inverse);
    return cudaGetLastError();
}

static __global__ void reorder(float2* data, std::size_t n, std::size_t batches, unsigned bits) {
    const std::size_t total = n * batches;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
         i < total; i += std::size_t(blockDim.x) * gridDim.x) {
        const std::size_t k = i % n;
        const std::size_t reversed = __brev(static_cast<unsigned>(k)) >> (32 - bits);
        if (k < reversed) {
            const std::size_t j = i - k + reversed;
            const float2 a = data[i];
            data[i] = data[j];
            data[j] = a;
        }
    }
}

static __global__ void stage(float2* data, std::size_t n, std::size_t batches,
                             std::size_t width, bool inverse) {
    const std::size_t half = width / 2;
    const std::size_t per_batch = n / 2;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
         i < per_batch * batches; i += std::size_t(blockDim.x) * gridDim.x) {
        const std::size_t local = i % per_batch;
        const std::size_t k = local % half;
        const std::size_t a = (i / per_batch) * n + (local / half) * width + k;
        const std::size_t b = a + half;
        float s, c;
        const float angle = (inverse ? 2.0f : -2.0f)
                          * static_cast<float>(k) / static_cast<float>(width);
        sincospif(angle, &s, &c);
        const float2 u = data[a], v = data[b];
        const float vr = v.x * c - v.y * s;
        const float vi = v.x * s + v.y * c;
        data[a] = make_float2(u.x + vr, u.y + vi);
        data[b] = make_float2(u.x - vr, u.y - vi);
    }
}

static __global__ void normalize(float2* data, std::size_t total, float scale) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
         i < total; i += std::size_t(blockDim.x) * gridDim.x) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

inline unsigned blocks(std::size_t count) {
    const std::size_t needed = (count - 1) / 256 + 1;
    return static_cast<unsigned>(needed < 65535 ? needed : 65535);
}

template<unsigned Columns>
static __global__ __launch_bounds__(256) void workspace_columns(const float2* data, float2* workspace,
                                                               std::size_t batches, bool inverse) {
    constexpr unsigned rows = 128, n = Columns * rows, tiles = Columns / 8;
    __shared__ float2 scratch[8 * rows];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const float sign = inverse ? 2.0f : -2.0f;
    for (std::size_t task = blockIdx.x; task < batches * tiles; task += gridDim.x) {
        const std::size_t batch = task / tiles;
        const unsigned column = unsigned(task % tiles) * 8 + x;
        float2 local[4];
        #pragma unroll
        for (unsigned r = 0; r < 4; ++r) local[r] = data[batch * n + column + Columns * (y + 32 * r)];
        #pragma unroll
        for (unsigned width = rows; width > 32; width >>= 1) {
            const unsigned step = width / 64;
            #pragma unroll
            for (unsigned r = 0; r < step; ++r) {
                float sine, cosine;
                twiddle_pi(sign * float(y + 32 * r) / width, &sine, &cosine);
                #pragma unroll
                for (unsigned base = 0; base < 4; base += 2 * step) {
                    const float2 a = local[base + r], b = local[base + r + step];
                    local[base + r] = make_float2(a.x + b.x, a.y + b.y);
                    local[base + r + step] = rotate(make_float2(a.x - b.x, a.y - b.y), cosine, sine);
                }
            }
        }
        #pragma unroll
        for (unsigned width = 32; width >= 8; width >>= 1) {
            const unsigned half = width / 2;
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) scratch[shared_index((y + 32 * r) * 8 + x)] = local[r];
            __syncthreads();
            float sine, cosine;
            twiddle_pi(sign * float(y & (half - 1)) / width, &sine, &cosine);
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                const float2 a = local[r], b = scratch[shared_index(((y ^ half) + 32 * r) * 8 + x)];
                local[r] = y & half ? rotate(make_float2(b.x - a.x, b.y - a.y), cosine, sine)
                                    : make_float2(a.x + b.x, a.y + b.y);
            }
            __syncthreads();
        }
        #pragma unroll
        for (unsigned width = 4; width >= 2; width >>= 1) {
            const unsigned half = width / 2;
            const float sine = width == 4 && (y & 1u) ? sign * 0.5f : 0.0f;
            const float cosine = width == 4 && (y & 1u) ? 0.0f : 1.0f;
            #pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                const float2 a = local[r];
                const float2 b = make_float2(__shfl_xor_sync(0xffffffffu, a.x, half * 8),
                                            __shfl_xor_sync(0xffffffffu, a.y, half * 8));
                local[r] = y & half ? rotate(make_float2(b.x - a.x, b.y - a.y), cosine, sine)
                                    : make_float2(a.x + b.x, a.y + b.y);
            }
        }
        #pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
            const unsigned k2 = __brev(y + 32 * r) >> 25;
            float sine, cosine;
            twiddle_pi(sign * float(column * k2) / n, &sine, &cosine);
            workspace[batch * n + column + Columns * k2] = rotate(local[r], cosine, sine);
        }
        __syncthreads();
    }
}

template<unsigned Columns>
static __global__ __launch_bounds__(128) void workspace_rows(const float2* workspace, float2* data,
                                                           std::size_t batches, bool inverse) {
    constexpr unsigned rows = 128, n = Columns * rows, values = Columns / 16;
    constexpr unsigned bits = Columns == 64 ? 6 : 7;
    __shared__ float2 scratch[8 * Columns];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const float sign = inverse ? 2.0f : -2.0f;
    for (std::size_t task = blockIdx.x; task < batches * (rows / 8); task += gridDim.x) {
        const std::size_t batch = task / (rows / 8);
        const unsigned row_base = unsigned(task % (rows / 8)) * 8;
        float2 local[values];
        #pragma unroll
        for (unsigned r = 0; r < values; ++r)
            local[r] = workspace[batch * n + Columns * (row_base + y) + x + 16 * r];
        #pragma unroll
        for (unsigned width = Columns; width > 16; width >>= 1) {
            const unsigned step = width / 32;
            #pragma unroll
            for (unsigned r = 0; r < step; ++r) {
                float sine, cosine;
                twiddle_pi(sign * float(x + 16 * r) / width, &sine, &cosine);
                #pragma unroll
                for (unsigned base = 0; base < values; base += 2 * step) {
                    const float2 a = local[base + r], b = local[base + r + step];
                    local[base + r] = make_float2(a.x + b.x, a.y + b.y);
                    local[base + r + step] = rotate(make_float2(a.x - b.x, a.y - b.y), cosine, sine);
                }
            }
        }
        float root_sine, root_cosine;
        twiddle_pi(sign * float(x & 7u) / 16.0f, &root_sine, &root_cosine);
        #pragma unroll
        for (unsigned width = 16; width >= 2; width >>= 1) {
            const unsigned half = width / 2;
            float sine, cosine;
            if (width == 2) { sine = 0; cosine = 1; }
            else if (width == 4) { sine = x & 1u ? sign * 0.5f : 0; cosine = x & 1u ? 0 : 1; }
            else {
                const unsigned source = (x & (half - 1)) * (16 / width);
                sine = __shfl_sync(0xffffffffu, root_sine, source, 16);
                cosine = __shfl_sync(0xffffffffu, root_cosine, source, 16);
            }
            #pragma unroll
            for (unsigned r = 0; r < values; ++r) {
                const float2 a = local[r];
                const float2 b = make_float2(__shfl_xor_sync(0xffffffffu, a.x, half, 16),
                                            __shfl_xor_sync(0xffffffffu, a.y, half, 16));
                local[r] = x & half ? rotate(make_float2(b.x - a.x, b.y - a.y), cosine, sine)
                                    : make_float2(a.x + b.x, a.y + b.y);
            }
        }
        const float scale = inverse ? 1.0f / n : 1.0f;
        #pragma unroll
        for (unsigned r = 0; r < values; ++r) {
            const unsigned k1 = __brev(x + 16 * r) >> (32 - bits);
            scratch[shared_index(k1 * 8 + y)] = make_float2(local[r].x * scale, local[r].y * scale);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned r = 0; r < values; ++r) {
            const unsigned index = x + 16 * y + 128 * r;
            data[batch * n + row_base + (index & 7u) + rows * (index / 8)] = scratch[shared_index(index)];
        }
        __syncthreads();
    }
}

template<unsigned Columns>
inline cudaError_t launch_workspace(float2* data, float2* workspace, std::size_t batches,
                                   bool inverse, cudaStream_t stream) {
    const auto grid = [](std::size_t count) { return static_cast<unsigned>(count < 65535 ? count : 65535); };
    workspace_columns<Columns><<<grid(batches * (Columns / 8)), dim3(8, 32), 0, stream>>>(data, workspace, batches, inverse);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    workspace_rows<Columns><<<grid(batches * 16), dim3(16, 8), 0, stream>>>(workspace, data, batches, inverse);
    return cudaGetLastError();
}
}

// Contiguous [batch][n] complex FP32; forward sign -1, inverse sign +1 with 1/n scaling.
inline cudaError_t transform(float2* data, std::size_t n, std::size_t batches,
                             bool inverse, cudaStream_t stream = nullptr) {
    if (!data || !n || !batches || (n & (n - 1)) || n > (std::size_t(1) << 24)
        || batches > std::numeric_limits<std::size_t>::max() / (n * sizeof(float2)))
        return cudaErrorInvalidValue;
    if (n == 1) return cudaSuccess;
    switch (n) {
        case 2: return detail::launch_fused<1>(data, batches, inverse, stream);
        case 4: return detail::launch_fused<2>(data, batches, inverse, stream);
        case 8: return detail::launch_fused<3>(data, batches, inverse, stream);
        case 16: return detail::launch_fused<4>(data, batches, inverse, stream);
        case 32: return detail::launch_fused<5>(data, batches, inverse, stream);
        case 64: return detail::launch_fused<6>(data, batches, inverse, stream);
        case 128: return detail::launch_fused<7>(data, batches, inverse, stream);
        case 256: return detail::launch_fused<8>(data, batches, inverse, stream);
        case 512: return detail::launch_fused<9>(data, batches, inverse, stream);
        case 1024: return detail::launch_fused<10>(data, batches, inverse, stream);
        case 2048: return detail::launch_fused<11>(data, batches, inverse, stream);
        case 4096: return detail::launch_fused<12>(data, batches, inverse, stream);
        case 8192: return detail::launch_fused<13>(data, batches, inverse, stream);
    }
    unsigned bits = 0;
    for (std::size_t width = n; width > 1; width >>= 1) ++bits;
    detail::reorder<<<detail::blocks(n * batches), 256, 0, stream>>>(data, n, batches, bits);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    for (std::size_t width = 2; width <= n; width <<= 1) {
        detail::stage<<<detail::blocks(n / 2 * batches), 256, 0, stream>>>(data, n, batches, width, inverse);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    if (inverse) {
        detail::normalize<<<detail::blocks(n * batches), 256, 0, stream>>>(data, n * batches, 1.0f / n);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

inline std::size_t workspace_size(std::size_t n, std::size_t batches) {
    if ((n != 8192 && n != 16384) || !batches
        || batches > std::numeric_limits<std::size_t>::max() / (n * sizeof(float2))) return 0;
    return n * batches * sizeof(float2);
}

inline cudaError_t transform_with_workspace(float2* data, float2* workspace, std::size_t workspace_bytes,
                                            std::size_t n, std::size_t batches, bool inverse,
                                            cudaStream_t stream = nullptr) {
    if (!data || !n || !batches || (n & (n - 1)) || n > (std::size_t(1) << 24)
        || batches > std::numeric_limits<std::size_t>::max() / (n * sizeof(float2))) return cudaErrorInvalidValue;
    if (n != 8192 && n != 16384) return transform(data, n, batches, inverse, stream);
    const std::size_t needed = workspace_size(n, batches);
    if (!workspace || workspace_bytes < needed) return cudaErrorInvalidValue;
    const auto a = reinterpret_cast<std::uintptr_t>(data), b = reinterpret_cast<std::uintptr_t>(workspace);
    if ((a | b) % alignof(float2) || needed > std::numeric_limits<std::uintptr_t>::max() - a
        || needed > std::numeric_limits<std::uintptr_t>::max() - b || (a < b + needed && b < a + needed))
        return cudaErrorInvalidValue;
    return n == 8192 ? detail::launch_workspace<64>(data, workspace, batches, inverse, stream)
                     : detail::launch_workspace<128>(data, workspace, batches, inverse, stream);
}
}
