# sfFFT general complex transform kernels

Fresh MIT CUDA implementation of a batched, in-place complex FP32 radix-2 FFT.
The API accepts contiguous `float2 data[batch][n]`, positive batch count and
power-of-two `n` from 1 through 2^24. Forward uses exp(-2 pi i j k / n);
inverse uses exp(+2 pi i j k / n) and divides by n.

```cpp
#include "src/kernel.cuh"
cudaError_t error = sfFFT::transform(device_data, n, batches, false, stream);
```

The separately published sfFFT repository implements a specialized causal
convolution path using tensor cores. This package supplies a general complex
FFT API. It has a different contract and implementation; benchmark results
from that published repository do not measure this package.

Its published 6.1x result uses 6,144 real sequences at L=2048 with BF16
input/output, FP16 transform intermediates and FP32 tensor-core accumulation,
compared with a pad/R2C/multiply/C2R/slice pipeline. The general C2C
measurements use FP32 complex arithmetic and much smaller batches. Fusion
is still essential for this contract: lengths 2 through 8192 now use one
kernel launch, keep butterfly values in registers, use warp shuffles for
the final five stages, and exchange values between warps through shared
memory. Lengths through 256 fit in one warp. The 8192 path uses a
32 KiB half-exchange buffer; all paths need at most 32 KiB, so no
per-call shared-memory opt-in or context cache is needed. Bit reversal
and inverse scaling are fused into the final coalesced output stores.
Larger lengths retain the staged global-memory fallback.

An explicit caller-workspace API additionally supports two-pass transforms
at 8192 and 16384. It factors n as 64 x 128 or 128 x 128: independent
128-point column transforms write the intermediate spectrum and twiddle
products into workspace, then 64/128-point row transforms write natural-order
output. Small blocks distribute one transform across many SMs; the caller
provides `n*batches*sizeof(float2)` workspace bytes, without hidden allocation.

```cpp
const std::size_t bytes = sfFFT::workspace_size(n, batches);
cudaError_t error = sfFFT::transform_with_workspace(device_data, device_workspace,
                                                  bytes, n, batches, inverse, stream);
```

Workspace and data must have disjoint, aligned device storage. The API rejects
null, short and overlapping workspace for these two sizes; other valid sizes
delegate to the existing allocation-free API and require zero workspace.
Both buffers must remain live until stream completion. Workspace preparation
and allocation are excluded from benchmarks, like cuFFT plan/workspace setup;
caller-workspace rows are labelled separately from in-place rows.

## Measured GB10 scope

The caller-workspace path has a workload crossover: it reaches cuFFT latency
at 8192 with batch 8 and is 1.394x faster at 16384 with batch 8, but cuFFT
remains faster for the measured larger 8192 batches. These are matched
complex FP32 forward transforms, using seven alternating CUDA-event samples
on one GB10 at stock clocks. Allocation, input resets, transfers and cuFFT
plan/workspace preparation are excluded for both implementations.

| Path | Length | Batch | Caller workspace | SF median us | cuFFT median us | cuFFT / SF |
|---|---:|---:|---:|---:|---:|---:|
| In-place | 256 | 32 | 0 | 4.000 | 3.936 | 0.984x |
| In-place | 8192 | 8 | 0 | 14.176 | 8.032 | 0.567x |
| Caller workspace | 8192 | 8 | 512 KiB | 8.096 | 8.032 | 0.992x |
| Caller workspace | 8192 | 32 | 2 MiB | 14.176 | 10.048 | 0.709x |
| Caller workspace | 8192 | 64 | 4 MiB | 35.680 | 18.144 | 0.509x |
| Caller workspace | 16384 | 8 | 1 MiB | 10.144 | 14.144 | 1.394x |

Ratios above 1 mean SF is faster. The 8192 batch-8 workspace result is
parity; it does not establish a speedup. These measurements do not establish
an advantage for unmeasured batches, lengths or chips. Hardware performance
on Hopper and the other Blackwell chips remains unmeasured. The workspace
path uses eight rows per output tile: a measured wider-tile experiment
improved the 8192 batch-64 case by only about 3% and regressed the 16384
batch-8 case by about 20%, so it was rejected.
Fused twiddles reduce exactly representable dyadic turns into quarters,
then evaluate fresh degree-11 sine / degree-12 cosine Taylor polynomials
with FP32 FMA. Their argument is bounded by pi/4. This avoids general
trigonometric range handling, large temporary register sets and FP32
rounding of pi before reduction. The staged fallback uses `sincospif`.

The source inventory found native CPU f64 radix-2 transforms used by wavetable
audio and circuit spectra, but no general native CUDA FFT. The kernel here is
new source based on the mathematical radix-2 decomposition. No published
sfFFT code, private rvllm code, or third-party FFT implementation was copied.

CUDA core operations support preparation for Hopper `sm_90` and Blackwell
`sm_100`, `sm_103`, and GB10 `sm_121`; compile each target with a
CUDA toolkit that recognizes it. GPU runtime qualification is separate from
architecture compilation. The header does not use tensor ISA or CUDA libraries.

Launches use the supplied stream, allocate no memory and do not synchronize.
The return value checks argument validity and kernel enqueue errors; the caller
must check stream completion for execution failures. The caller supplies a valid
device allocation of at least `batches*n*sizeof(float2)` bytes. Inputs may include
NaN/Inf, which propagate through FP32 arithmetic. No bitwise reproducibility,
correct rounding, or performance advantage over cuFFT is claimed.

`tests/check.cu` compares forward and inverse results with a scalar double DFT
through length 256, and with an independent scalar double DIT FFT at larger
sizes. It covers singleton, batched, warp and shared-memory paths, including
8192 forward/inverse and the 16384 staged fallback on a non-default stream.
Periodic forward inputs at 8192 and 16384 use a relative L2 error limit of
1e-6 and require every output to be finite. Strong cancellation makes the
small-transform pointwise criterion inappropriate for those two new cases:
the earlier sincospif candidate's GB10 diagnostic recorded SF and cuFFT
8192 relative L2 errors of 1.414e-7 and 2.471e-7,
yet their maximum error/(1+|bin|) is 8.21e-5 and 1.412e-4. The 16384
figures are 1.445e-7 / 2.587e-7 in L2 and 7.945e-5 / 1.652e-4 pointwise.
The JSON retains both measures and the selected criterion. All previously
tested cases retain their 3e-5 pointwise gate. New analytic impulse and
DC/Nyquist-mode cases at both large sizes also require that pointwise gate,
including every low-magnitude bin, so the normwise cases do not conceal
phase, ordering, normalization or leakage errors.
An additional CUDA test checks all 16384 dyadic roots in both directions,
covering every root used by the fused sizes, against scalar double sine/cosine.
It requires at most 2 component ULPs and 1.5e-7 complex absolute error.
Negative controls reject null pointers, zero dimensions, non-power-of-two sizes,
oversize transforms and overflowing storage sizes. It emits one JSON object and
returns nonzero on any failure. Run with the package build tooling; runtime
results must be captured on each qualified architecture.
Defining `SFFFT_CHECK_CUFFT` and linking `-lcufft` emits maximum
absolute/scaled and relative L2 errors on stderr for larger cases, including
cuFFT on the identical 8192/16384 inputs, without changing the sfFFT gate;
the ordinary standalone test has no cuFFT dependency.
Workspace tests cover both transform sizes, both signs, analytic impulses and
modes, storage sizing/overflow, null/short buffers, exact and partial overlap.

## Standalone build

CUDA 13.0 or newer is required for the full target matrix.

```sh
make -j1
make test
# A single device family can be selected explicitly:
make ARCHS=90 BUILD_DIR=build-hopper
```

Default targets: H100/H200 (`sm_90`), B200 (`sm_100`), B300/GB300 (`sm_103`),
GB10 (`sm_121`), plus portable Hopper PTX. Do not add `--use_fast_math`;
that changes the numerical contract. Compile success and numerical
qualification are reported separately from performance benchmarking.

## Published source and GB10 measurements

The standalone source repository is [solidsf-inc/sfFFTGeneral](https://github.com/solidsf-inc/sfFFTGeneral). The [solidSF CUDA kernel catalog](https://solidsf.com/kernels) contains the formal report, all measured workloads and raw timing samples.

| Workload | SF median us | Reference median us | Reference / SF |
| --- | ---: | ---: | ---: |
| complex32 forward; N=256, batch=32 in-place | 4.000 | 3.936 | 0.984x |
| complex32 forward; N=8192, batch=8 in-place | 14.176 | 8.032 | 0.567x |
| complex32 forward; N=256, batch=256 in-place | 3.936 | 3.904 | 0.992x |
| complex32 forward; N=256, batch=1024 in-place | 5.888 | 5.984 | 1.016x |
| complex32 forward; N=8192, batch=32 in-place | 14.144 | 10.080 | 0.713x |
| complex32 forward; N=8192, batch=64 in-place | 24.416 | 14.144 | 0.579x |
| complex32 forward caller workspace; N=8192, batch=8 caller workspace | 8.096 | 8.032 | 0.992x |
| complex32 forward caller workspace; N=8192, batch=32 caller workspace | 14.176 | 10.048 | 0.709x |
| complex32 forward caller workspace; N=8192, batch=64 caller workspace | 35.680 | 18.144 | 0.509x |
| complex32 forward caller workspace; N=16384, batch=8 caller workspace | 10.144 | 14.144 | 1.394x |

References: NVIDIA cuFFT C2C forward.

These are seven-sample medians of alternating CUDA-event measurements on one GB10 at stock clocks. Setup, allocations, transfers, plan creation and destructive-input resets are excluded; stream timing can include API enqueue gaps. Ratios above one mean SF is faster; near-unity medians do not establish a meaningful advantage. No aggregate or universal speedup is claimed.

All 43 package numerical/API cases passed on GB10. All four cubin targets and compute_90 PTX were verified. H100, H200, B200, B300 and GB300 hardware runtime and performance remain unmeasured.
The caller-workspace path uses N*batch*sizeof(float2) extra storage at 8192/16384; its allocation is excluded. The existing [sfFFT](https://github.com/solidsf-inc/sfFFT) fused-convolution repository is a separate API and workload.
