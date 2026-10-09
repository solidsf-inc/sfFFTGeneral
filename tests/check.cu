// SPDX-License-Identifier: MIT
#include "../src/kernel.cuh"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#ifdef SFFFT_CHECK_CUFFT
#include <cufft.h>
#endif

struct Accuracy {
    std::size_t length, batches;
    bool inverse, normwise, finite, pointwise_pass, passed, planned;
    unsigned pattern;
    double max_absolute, max_scaled, relative_l2;
};
static std::vector<Accuracy> accuracy;
static double root_max_error = 0;
static unsigned root_max_ulp = 0;

static __global__ void twiddle_values(float2* values) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 16384) return;
    float sine, cosine;
    sfFFT::detail::twiddle_pi((index < 8192 ? -2.0f : 2.0f) * float(index & 8191u) / 16384.0f, &sine, &cosine);
    values[index] = make_float2(cosine, sine);
}

static bool check_twiddles() {
    std::vector<float2> values(16384);
    float2* device = nullptr;
    bool ok = cudaMalloc(&device, values.size() * sizeof(float2)) == cudaSuccess;
    if (ok) {
        twiddle_values<<<64, 256>>>(device);
        ok = cudaGetLastError() == cudaSuccess
          && cudaMemcpy(values.data(), device, values.size() * sizeof(float2), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (device) cudaFree(device);
    if (!ok) return false;
    for (unsigned i = 0; i < values.size(); ++i) {
        const unsigned k = i & 8191u;
        const int sign = i < 8192 ? -1 : 1;
        const double angle = sign * 6.2831853071795864769 * k / 16384;
        const double s = k == 0 ? 0 : (k == 4096 ? double(sign) : std::sin(angle));
        const double c = k == 0 ? 1 : (k == 4096 ? 0 : std::cos(angle));
        const double error = std::hypot(values[i].x - c, values[i].y - s);
        if (!std::isfinite(error)) return false;
        root_max_error = std::fmax(root_max_error, error);
        for (const auto pair : {std::pair<float, float>{values[i].x, float(c)}, {values[i].y, float(s)}}) {
            if (pair.first == pair.second) continue;
            std::uint32_t actual, expected;
            std::memcpy(&actual, &pair.first, sizeof(actual));
            std::memcpy(&expected, &pair.second, sizeof(expected));
            if ((actual >> 31) != (expected >> 31)) return false;
            const unsigned ulp = actual > expected ? actual - expected : expected - actual;
            if (ulp > root_max_ulp) root_max_ulp = ulp;
        }
    }
    return root_max_error <= 1.5e-7 && root_max_ulp <= 2;
}

static bool check(std::size_t n, std::size_t batches, bool inverse, unsigned pattern = 0, bool planned = false) {
    std::vector<float2> input(n * batches), expected(n * batches), output(n * batches);
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (pattern == 1) {
            input[i] = i % n == 0 ? make_float2(0.25f, -0.5f)
                     : i % n == 1 ? make_float2(0.75f + (i / n) * 0.03125f, 0.125f)
                                  : make_float2(0.0f, 0.0f);
        } else if (pattern == 2) {
            const float sign = i % 2 ? -1.0f : 1.0f;
            input[i] = make_float2(0.125f + sign * 0.25f, 0.0625f + sign * 0.125f);
        } else input[i] = make_float2(float(int(i % 13) - 6) / 7, float(int(i % 7) - 3) / 5);
    }
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t batch = 0; batch < batches; ++batch) {
        if (pattern) {
            for (std::size_t k = 0; k < n; ++k) {
                double re = 0, im = 0;
                if (pattern == 1) {
                    const double angle = (inverse ? 2 : -2) * pi * k / n;
                    const double a = 0.75 + batch * 0.03125, b = 0.125;
                    re = 0.25 + a * std::cos(angle) - b * std::sin(angle);
                    im = -0.5 + a * std::sin(angle) + b * std::cos(angle);
                } else if (k == 0) { re = n * 0.125; im = n * 0.0625; }
                else if (k == n / 2) { re = n * 0.25; im = n * 0.125; }
                const double scale = inverse ? 1.0 / n : 1.0;
                expected[batch * n + k] = make_float2(float(re * scale), float(im * scale));
            }
        } else if (n > 256) {
            std::vector<std::complex<double>> work(n);
            for (std::size_t i = 0, reversed = 0; i < n; ++i) {
                work[reversed] = {input[batch * n + i].x, input[batch * n + i].y};
                std::size_t bit = n >> 1;
                while (reversed & bit) { reversed ^= bit; bit >>= 1; }
                reversed ^= bit;
            }
            for (std::size_t width = 2; width <= n; width *= 2) {
                const double angle = (inverse ? 2 : -2) * pi / width;
                const std::complex<double> step(std::cos(angle), std::sin(angle));
                for (std::size_t base = 0; base < n; base += width) {
                    std::complex<double> twiddle(1.0, 0.0);
                    for (std::size_t k = 0; k < width / 2; ++k) {
                        const auto a = work[base + k], b = work[base + k + width / 2] * twiddle;
                        work[base + k] = a + b;
                        work[base + k + width / 2] = a - b;
                        twiddle *= step;
                    }
                }
            }
            for (std::size_t k = 0; k < n; ++k) {
                const double scale = inverse ? 1.0 / n : 1.0;
                expected[batch * n + k] = make_float2(float(work[k].real() * scale), float(work[k].imag() * scale));
            }
        } else for (std::size_t k = 0; k < n; ++k) {
            double re = 0, im = 0;
            for (std::size_t j = 0; j < n; ++j) {
                const double angle = (inverse ? 2 : -2) * pi * double(k * j) / n;
                const auto a = input[batch * n + j];
                re += a.x * std::cos(angle) - a.y * std::sin(angle);
                im += a.x * std::sin(angle) + a.y * std::cos(angle);
            }
            const double scale = inverse ? 1.0 / n : 1.0;
            expected[batch * n + k] = make_float2(float(re * scale), float(im * scale));
        }
    }
    float2* device = nullptr;
    float2* workspace = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = cudaMalloc(&device, input.size() * sizeof(float2)) == cudaSuccess;
    const std::size_t workspace_bytes = planned ? sfFFT::workspace_size(n, batches) : 0;
    if (workspace_bytes) ok = ok && cudaMalloc(&workspace, workspace_bytes) == cudaSuccess;
    ok = ok && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
    ok = ok && cudaMemcpyAsync(device, input.data(), input.size() * sizeof(float2), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    ok = ok && (planned ? sfFFT::transform_with_workspace(device, workspace, workspace_bytes, n, batches, inverse, stream)
                        : sfFFT::transform(device, n, batches, inverse, stream)) == cudaSuccess;
    ok = ok && cudaMemcpyAsync(output.data(), device, input.size() * sizeof(float2), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
    const auto inspect = [&](const std::vector<float2>& actual, const char* implementation) {
        bool pointwise_pass = true, finite = true;
        double max_absolute = 0, max_scaled = 0, error_squared = 0, expected_squared = 0;
        std::size_t worst = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double error = std::hypot(double(actual[i].x) - expected[i].x, double(actual[i].y) - expected[i].y);
            const double magnitude = std::hypot(double(expected[i].x), double(expected[i].y));
            const double scaled = error / (1 + magnitude);
            finite = finite && std::isfinite(error);
            pointwise_pass = pointwise_pass && std::isfinite(error) && scaled <= 3e-5;
            if (scaled > max_scaled) { max_scaled = scaled; worst = i; }
            max_absolute = std::fmax(max_absolute, error);
            error_squared += error * error;
            expected_squared += magnitude * magnitude;
        }
        const double relative_l2 = std::sqrt(error_squared / expected_squared);
        const bool normwise = !pattern && !inverse && n >= 8192;
        const bool passed = finite && (normwise ? relative_l2 <= 1e-6 : pointwise_pass);
        if (implementation[0] == 's') accuracy.push_back({n, batches, inverse, normwise, finite,
            pointwise_pass, passed, planned, pattern, max_absolute, max_scaled, relative_l2});
#ifdef SFFFT_CHECK_CUFFT
        if (n >= 512) std::fprintf(stderr,
            "{\"length\":%zu,\"batches\":%zu,\"inverse\":%s,\"implementation\":\"%s\","
            "\"max_absolute_error\":%.17g,\"max_scaled_error\":%.17g,\"relative_l2_error\":%.17g,"
            "\"pointwise_gate\":3e-5,\"pointwise_gate_pass\":%s,\"worst_index\":%zu,"
            "\"expected\":[%.9g,%.9g],\"actual\":[%.9g,%.9g]}\n",
            n, batches, inverse ? "true" : "false", implementation, max_absolute, max_scaled,
            relative_l2, pointwise_pass ? "true" : "false", worst,
            expected[worst].x, expected[worst].y, actual[worst].x, actual[worst].y);
#endif
        return passed;
    };
    if (ok) ok = inspect(output, "sfFFT");
#ifdef SFFFT_CHECK_CUFFT
    if (device && stream && n >= 8192) {
        cufftHandle plan = 0;
        bool vendor = cufftPlan1d(&plan, static_cast<int>(n), CUFFT_C2C, static_cast<int>(batches)) == CUFFT_SUCCESS;
        vendor = vendor && cufftSetStream(plan, stream) == CUFFT_SUCCESS;
        vendor = vendor && cudaMemcpyAsync(device, input.data(), input.size() * sizeof(float2), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        vendor = vendor && cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(device), reinterpret_cast<cufftComplex*>(device),
                                      inverse ? CUFFT_INVERSE : CUFFT_FORWARD) == CUFFT_SUCCESS;
        vendor = vendor && cudaMemcpyAsync(output.data(), device, output.size() * sizeof(float2), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        vendor = vendor && cudaStreamSynchronize(stream) == cudaSuccess;
        if (vendor) {
            if (inverse) for (auto& value : output) { value.x /= n; value.y /= n; }
            inspect(output, "cuFFT");
        } else std::fprintf(stderr, "cuFFT diagnostic failed for n=%zu batches=%zu inverse=%d\n", n, batches, inverse);
        if (plan) cufftDestroy(plan);
    }
#endif
    if (device) cudaFree(device);
    if (workspace) cudaFree(workspace);
    if (stream) cudaStreamDestroy(stream);
    return ok;
}

int main() {
    int devices = 0;
    const bool device = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
    float2 placeholder{};
    std::vector<std::pair<const char*, bool>> cases = {
        {"cuda_device", device},
        {"bounded_twiddle_roots", device && check_twiddles()},
        {"null_rejected", sfFFT::transform(nullptr, 8, 1, false) == cudaErrorInvalidValue},
        {"zero_rejected", sfFFT::transform(&placeholder, 0, 1, false) == cudaErrorInvalidValue},
        {"non_power_two_rejected", sfFFT::transform(&placeholder, 6, 1, false) == cudaErrorInvalidValue},
        {"zero_batch_rejected", sfFFT::transform(&placeholder, 8, 0, false) == cudaErrorInvalidValue},
        {"oversize_rejected", sfFFT::transform(&placeholder, std::size_t(1) << 25, 1, false) == cudaErrorInvalidValue},
        {"overflow_rejected", sfFFT::transform(&placeholder, 8, std::numeric_limits<std::size_t>::max(), false) == cudaErrorInvalidValue},
        {"workspace_size", sfFFT::workspace_size(8192, 3) == 8192 * 3 * sizeof(float2)},
        {"workspace_size_overflow", sfFFT::workspace_size(8192, std::numeric_limits<std::size_t>::max()) == 0},
        {"workspace_null_rejected", sfFFT::transform_with_workspace(&placeholder, nullptr, 8192 * sizeof(float2), 8192, 1, false) == cudaErrorInvalidValue},
        {"workspace_short_rejected", sfFFT::transform_with_workspace(&placeholder, &placeholder, 8192 * sizeof(float2) - 1, 8192, 1, false) == cudaErrorInvalidValue},
        {"workspace_alias_rejected", sfFFT::transform_with_workspace(&placeholder, &placeholder, 8192 * sizeof(float2), 8192, 1, false) == cudaErrorInvalidValue},
        {"workspace_overlap_rejected", sfFFT::transform_with_workspace(&placeholder, reinterpret_cast<float2*>(reinterpret_cast<std::uintptr_t>(&placeholder) + sizeof(float2)), 8192 * sizeof(float2), 8192, 1, false) == cudaErrorInvalidValue},
        {"singleton", device && check(1, 3, false)},
        {"singleton_inverse", device && check(1, 3, true)},
        {"forward_two", device && check(2, 1, false)},
        {"forward_batched_32", device && check(32, 3, false)},
        {"inverse_batched_64", device && check(64, 2, true)},
        {"forward_tail_128", device && check(128, 5, false)},
        {"forward_warp_256", device && check(256, 32, false)},
        {"inverse_warp_256", device && check(256, 3, true)},
        {"forward_block_512", device && check(512, 3, false)},
        {"inverse_block_1024", device && check(1024, 2, true)},
        {"forward_block_2048", device && check(2048, 3, false)},
        {"inverse_block_4096", device && check(4096, 2, true)},
        {"forward_block_8192", device && check(8192, 8, false)},
        {"inverse_block_8192", device && check(8192, 3, true)},
        {"forward_fallback_16384", device && check(16384, 2, false)},
        {"inverse_fallback_16384", device && check(16384, 2, true)},
        {"forward_impulses_8192", device && check(8192, 3, false, 1)},
        {"inverse_impulses_8192", device && check(8192, 3, true, 1)},
        {"forward_modes_8192", device && check(8192, 3, false, 2)},
        {"forward_impulses_16384", device && check(16384, 2, false, 1)},
        {"forward_modes_16384", device && check(16384, 2, false, 2)},
        {"workspace_forward_8192", device && check(8192, 8, false, 0, true)},
        {"workspace_inverse_8192", device && check(8192, 3, true, 0, true)},
        {"workspace_impulses_8192", device && check(8192, 3, false, 1, true)},
        {"workspace_modes_8192", device && check(8192, 3, false, 2, true)},
        {"workspace_forward_16384", device && check(16384, 3, false, 0, true)},
        {"workspace_inverse_16384", device && check(16384, 2, true, 0, true)},
        {"workspace_impulses_16384", device && check(16384, 2, false, 1, true)},
        {"workspace_modes_16384", device && check(16384, 2, false, 2, true)}
    };
    bool ok = true;
    for (const auto& c : cases) ok = ok && c.second;
    std::printf("{\"package\":\"sfFFT\",\"status\":\"%s\",\"cases\":[", ok ? "PASS" : "FAIL");
    for (std::size_t i = 0; i < cases.size(); ++i)
        std::printf("%s{\"name\":\"%s\",\"status\":\"%s\"}", i ? "," : "", cases[i].first, cases[i].second ? "PASS" : "FAIL");
    std::printf("],\"accuracy\":[");
    for (std::size_t i = 0; i < accuracy.size(); ++i) {
        const auto& a = accuracy[i];
        std::printf("%s{\"length\":%zu,\"batches\":%zu,\"inverse\":%s,\"pattern\":%u,"
                    "\"criterion\":\"%s\",\"caller_workspace\":%s,\"all_finite\":%s,\"pointwise_limit\":3e-5,"
                    "\"relative_l2_limit\":1e-6,\"pointwise_pass\":%s,\"passed\":%s,"
                    "\"max_absolute_error\":", i ? "," : "", a.length, a.batches, a.inverse ? "true" : "false", a.pattern,
                    a.normwise ? "relative_l2" : "pointwise", a.planned ? "true" : "false", a.finite ? "true" : "false",
                    a.pointwise_pass ? "true" : "false", a.passed ? "true" : "false");
        std::printf(std::isfinite(a.max_absolute) ? "%.17g" : "null", a.max_absolute);
        std::printf(",\"max_scaled_error\":");
        std::printf(std::isfinite(a.max_scaled) ? "%.17g" : "null", a.max_scaled);
        std::printf(",\"relative_l2_error\":");
        std::printf(std::isfinite(a.relative_l2) ? "%.17g" : "null", a.relative_l2);
        std::printf("}");
    }
    std::printf("],\"twiddle_accuracy\":{\"max_complex_error\":%.17g,\"max_component_ulp\":%u,"
                "\"complex_error_limit\":1.5e-7,\"component_ulp_limit\":2}}\n", root_max_error, root_max_ulp);
    return ok ? 0 : 1;
}
