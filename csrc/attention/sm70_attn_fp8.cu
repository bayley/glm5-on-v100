// SM70 fp8-e4m3 weight-only attention-projection kernels for GLM/DeepSeek
// DSA models on V100 (see NOTES.md "attention fp8" section).
//
// The GLM-5.1 AWQ checkpoint keeps self_attn + indexer projections in
// fp16/bf16 (~82.6 MB/layer/GPU at TP=8); the official GLM-5.1-FP8 release
// ships these exact tensors as fp8-e4m3 with [128,128] fp32 block scales,
// so 8-bit storage is quality-sanctioned by the vendor. These kernels let us
// store the weights as plain row-major fp8 [N, K] + fp32 scale_inv
// [ceil(N/128), ceil(K/128)] (the official checkpoint layout) and:
//   - sm70_attn_fp8_gemv:   y[M,N] = x[M,K] @ W^T for decode (M <= 8),
//     ~630 GB/s vs ~600 GB/s cuBLAS fp16 GEMV at 2x fewer bytes
//     (1.9x faster at M=1 on the 4-GEMV attention set).
//   - sm70_attn_fp8_dequant: W -> fp16 scratch, for prefill/big-M where
//     cuBLAS fp16 GEMM wins.
//
// e4m3 -> fp16 conversion trick (SM70 has no fp8 hardware): for byte v,
// __ushort_as_half(((v & 0x80) << 8) | ((v & 0x7f) << 7)) == value(v) / 256
// EXACTLY, including e4m3 denormals (which map onto fp16 denormals scaled
// by the same 2^-8). The 256 factor is folded into the block scale.
// e4m3 NaN (0x7f/0xff) is not handled (weights must be finite).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

namespace {

#define DEV_INLINE __device__ __forceinline__

DEV_INLINE __half fp8_raw_to_half(uint8_t v) {
  // value / 256 (exact; see file header)
  return __ushort_as_half(
      (unsigned short)(((v & 0x80) << 8) | ((v & 0x7f) << 7)));
}

DEV_INLINE __half2 fp8x2_to_half2(uint8_t lo, uint8_t hi) {
  return __halves2half2(fp8_raw_to_half(lo), fp8_raw_to_half(hi));
}

// ---------------------------------------------------------------------------
// GEMV: grid.x = N/8, block = (32, 8): one warp per output row.
// x is staged in smem in KTILE chunks; per inner step a lane handles 8
// elements: weights via 8B uint2 loads (a warp reads 256B contiguous), x via
// 16B uint4 smem loads at 16B lane stride (conflict-free: the four 8-lane
// phases of a 128-bit smem access each cover all 32 banks exactly once).
// hfma2 accumulation within the 8-chunk, fp32 across chunks; the [128,128]
// block scale is applied per 8-chunk (8 | 128, so a chunk never straddles
// two scale blocks).
// ---------------------------------------------------------------------------
template <int M>
__global__ void __launch_bounds__(256) sm70_attn_fp8_gemv_kernel(
    const uint8_t* __restrict__ W,  // [N, K] row-major fp8-e4m3 bits
    const float* __restrict__ S,    // [ceil(N/128), KB] scale_inv
    const __half* __restrict__ X,   // [M, K]
    __half* __restrict__ Y,         // [M, N]
    const int N, const int K, const int KB) {
  constexpr int KTILE = (M <= 2) ? 2048 : 1024;  // keep >= 4 blocks/SM
  extern __shared__ __half xs[];  // [M][KTILE]

  const int row = blockIdx.x * 8 + threadIdx.y;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * 32 + lane;
  const float* srow = S + (size_t)(row >> 7) * KB;

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KTILE) {
    const int tile = min(KTILE, K - k0);
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; m++) {
      const __half* xg = X + (size_t)m * K + k0;
      for (int i = tid * 8; i < tile; i += 256 * 8) {
        *reinterpret_cast<uint4*>(xs + m * KTILE + i) =
            *reinterpret_cast<const uint4*>(xg + i);
      }
    }
    __syncthreads();

    const uint8_t* wrow = W + (size_t)row * K + k0;
    for (int c = lane * 8; c < tile; c += 32 * 8) {
      const uint2 wv = *reinterpret_cast<const uint2*>(wrow + c);
      const float s = srow[(k0 + c) >> 7] * 256.f;
      const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
      __half2 w2[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
        w2[i] = fp8x2_to_half2(wb[2 * i], wb[2 * i + 1]);
#pragma unroll
      for (int m = 0; m < M; m++) {
        const uint4 xv = *reinterpret_cast<const uint4*>(xs + m * KTILE + c);
        const __half2* x2 = reinterpret_cast<const __half2*>(&xv);
        __half2 h = __hmul2(w2[0], x2[0]);
        h = __hfma2(w2[1], x2[1], h);
        h = __hfma2(w2[2], x2[2], h);
        h = __hfma2(w2[3], x2[3], h);
        const float2 p = __half22float2(h);
        acc[m] += s * (p.x + p.y);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; m++) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      acc[m] += __shfl_down_sync(0xffffffff, acc[m], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; m++) Y[(size_t)m * N + row] = __float2half(acc[m]);
  }
}

// ---------------------------------------------------------------------------
// Dequant to fp16 scratch (for the cuBLAS prefill path). One thread = one
// 16B chunk (16 fp8 -> 32B fp16); a chunk never straddles scale blocks.
// ---------------------------------------------------------------------------
__global__ void sm70_attn_fp8_dequant_kernel(const uint8_t* __restrict__ W,
                                             const float* __restrict__ S,
                                             __half* __restrict__ out,
                                             const int64_t total,  // N*K/16
                                             const int K, const int KB) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t off = idx * 16;
  const int row = off / K;
  const int k = off % K;
  const float s = S[(size_t)(row >> 7) * KB + (k >> 7)] * 256.f;
  const uint4 wv = *reinterpret_cast<const uint4*>(W + off);
  const uint8_t* wb = reinterpret_cast<const uint8_t*>(&wv);
  __half2 o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const __half2 w2 = fp8x2_to_half2(wb[2 * i], wb[2 * i + 1]);
    const float2 p = __half22float2(w2);
    o[i] = __floats2half2_rn(p.x * s, p.y * s);
  }
  *reinterpret_cast<uint4*>(out + off) = *reinterpret_cast<uint4*>(o);
  *reinterpret_cast<uint4*>(out + off + 8) = *reinterpret_cast<uint4*>(o + 4);
}

}  // namespace

void sm70_attn_fp8_gemv(torch::Tensor& y, const torch::Tensor& x,
                        const torch::Tensor& w, const torch::Tensor& s) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && s.is_cuda() && y.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
  TORCH_CHECK(y.scalar_type() == at::kHalf, "y must be fp16");
  TORCH_CHECK(w.scalar_type() == at::kByte, "w must be uint8 (e4m3 bits)");
  TORCH_CHECK(s.scalar_type() == at::kFloat, "s must be fp32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2 && s.dim() == 2);
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  const int64_t KB = s.size(1);
  TORCH_CHECK(M >= 1 && M <= 8, "sm70_attn_fp8_gemv: M must be 1..8, got ", M);
  TORCH_CHECK(K % 16 == 0 && N % 8 == 0, "bad N/K alignment");
  TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N);
  TORCH_CHECK(s.size(0) == (N + 127) / 128 && KB == (K + 127) / 128,
              "scale shape mismatch");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y.is_contiguous() &&
              s.is_contiguous());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / 8), block(32, 8);
  const int ktile = (M <= 2) ? 2048 : 1024;
  const int smem = M * ktile * sizeof(__half);
  auto* wp = reinterpret_cast<const uint8_t*>(w.data_ptr());
  auto* sp = s.data_ptr<float>();
  auto* xp = reinterpret_cast<const __half*>(x.data_ptr());
  auto* yp = reinterpret_cast<__half*>(y.data_ptr());
#define SM70_ATTN_FP8_GEMV_CASE(MM)                             \
  case MM:                                                      \
    sm70_attn_fp8_gemv_kernel<MM><<<grid, block, smem, stream>>>( \
        wp, sp, xp, yp, N, K, KB);                              \
    break;
  switch (M) {
    SM70_ATTN_FP8_GEMV_CASE(1)
    SM70_ATTN_FP8_GEMV_CASE(2)
    SM70_ATTN_FP8_GEMV_CASE(3)
    SM70_ATTN_FP8_GEMV_CASE(4)
    SM70_ATTN_FP8_GEMV_CASE(5)
    SM70_ATTN_FP8_GEMV_CASE(6)
    SM70_ATTN_FP8_GEMV_CASE(7)
    SM70_ATTN_FP8_GEMV_CASE(8)
  }
#undef SM70_ATTN_FP8_GEMV_CASE
}

void sm70_attn_fp8_dequant(torch::Tensor& out, const torch::Tensor& w,
                           const torch::Tensor& s) {
  TORCH_CHECK(out.is_cuda() && w.is_cuda() && s.is_cuda());
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be fp16");
  TORCH_CHECK(w.scalar_type() == at::kByte, "w must be uint8 (e4m3 bits)");
  TORCH_CHECK(s.scalar_type() == at::kFloat, "s must be fp32");
  TORCH_CHECK(out.dim() == 2 && w.dim() == 2 && s.dim() == 2);
  const int64_t N = w.size(0), K = w.size(1), KB = s.size(1);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(out.size(0) == N && out.size(1) == K);
  TORCH_CHECK(s.size(0) == (N + 127) / 128 && KB == (K + 127) / 128,
              "scale shape mismatch");
  TORCH_CHECK(out.is_contiguous() && w.is_contiguous() && s.is_contiguous());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t total = N * K / 16;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  sm70_attn_fp8_dequant_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(w.data_ptr()), s.data_ptr<float>(),
      reinterpret_cast<__half*>(out.data_ptr()), total, K, KB);
}
