// SPDX-License-Identifier: Apache-2.0
// SM70 (Volta / V100) dense MLA decode attention.
//
// Ported from the llama.cpp fork's Volta MLA kernel
// (ggml/src/ggml-cuda/sparse-attn.cu, sparse_attn_kernel_wmma_g1): a
// "gather-once" nvcuda::wmma flash-decode. One block per decode token processes
// ALL heads (MQA-shared latent KV gathered ONCE per KV tile, reused across every
// head-tile), flash online-softmax over the KV dimension, WMMA 16x16x16 tiles for
// both QK (contract D_K) and PV (contract the KV tile), f16 inputs / f32 accum.
//
// Differences vs the llama.cpp version:
//   * Dense causal decode instead of sparse top-k: every KV position
//     0..seq_len-1 is attended (validity = j < seq_len), so there is no idx[]
//     gather — the KV row for position j is resolved through vLLM's PAGED cache
//     via block_table[j / PAGE] * PAGE + (j % PAGE).
//   * Inputs are fp16 (vLLM q / paged latent cache) rather than fp32 Q / fp16 KV.
//   * Emits o [B, H, D_V] fp16 and lse [B, H] fp32 (to match the Triton MLA path).
//
// Specialized for the GLM-5.1 MLA-MQA decode shapes: D_K=576 (kv_lora 512 +
// qk_rope 64), D_V=512, n_head=64.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700) || !defined(__CUDA_ARCH__)
#define SM70_MLA_WMMA_AVAILABLE
#endif

#ifdef SM70_MLA_WMMA_AVAILABLE
#include <mma.h>
namespace wmma = nvcuda::wmma;
#endif

namespace {

constexpr int TOPK_NBIN = 4096;  // 12-bit keys per level (2 levels = 24 bits)
constexpr int TOPK_THREADS = 256;

__device__ __forceinline__ uint32_t fp32_monotone(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---- shared helpers for the decode top-k machinery ----


// vectorized f16x4 load from the paged latent cache (row-major [.., D_K]).
static __device__ __forceinline__ float4 mla_load4_half(const half* p,
                                                        int64_t i4) {
  const int2 raw = *(const int2*)(p + 4 * i4);
  const half2 a = *(const half2*)&raw.x;
  const half2 b = *(const half2*)&raw.y;
  const float2 fa = __half22float2(a);
  const float2 fb = __half22float2(b);
  return make_float4(fa.x, fa.y, fb.x, fb.y);
}

// ---- fp8 (e4m3) cache support ----
//
// e4m3 -> fp16 WITHOUT hardware cvt (Volta has none): for an e4m3 byte v,
// __ushort_as_half(((v & 0x80) << 8) | ((v & 0x7f) << 7)) == value(v) / 256
// EXACTLY, including e4m3 denormals (same trick as sm70_attn_fp8.cu). The
// x256 is folded into the dequant scale at read time.
static __device__ __forceinline__ half sm70_fp8_to_half_div256(uint8_t v) {
  const unsigned short bits =
      (unsigned short)(((v & 0x80) << 8) | ((v & 0x7f) << 7));
  return __ushort_as_half(bits);
}

// fp8_ds_mla latent-cache row (656 bytes, the upstream DeepSeek V3.2 layout,
// written by concat_and_cache_ds_mla_kernel):
//   bytes [0,512):    512 e4m3 NoPE latents, 4 tiles of 128
//   bytes [512,528):  4 fp32 per-tile scales (dequant = fp8_val * scale)
//   bytes [528,656):  64 fp16 RoPE values (verbatim)
// Load 4 dims starting at dim d (multiple of 4) of the logical 576-wide row.
constexpr int DSMLA_ROW_BYTES = 656;
static __device__ __forceinline__ float4 dsmla_load4(const uint8_t* rowp,
                                                     int d) {
  if (d < 512) {
    const uchar4 v = *reinterpret_cast<const uchar4*>(rowp + d);
    const float s =
        *reinterpret_cast<const float*>(rowp + 512 + ((d >> 7) << 2)) * 256.0f;
    return make_float4(__half2float(sm70_fp8_to_half_div256(v.x)) * s,
                       __half2float(sm70_fp8_to_half_div256(v.y)) * s,
                       __half2float(sm70_fp8_to_half_div256(v.z)) * s,
                       __half2float(sm70_fp8_to_half_div256(v.w)) * s);
  }
  // RoPE: fp16 at byte 528; rows are 16B-aligned (656 = 41*16) and d is a
  // multiple of 4, so this is an aligned 8-byte load.
  const int2 raw = *reinterpret_cast<const int2*>(rowp + 528 + ((d - 512) << 1));
  const half2 a = *(const half2*)&raw.x;
  const half2 b = *(const half2*)&raw.y;
  const float2 fa = __half22float2(a);
  const float2 fb = __half22float2(b);
  return make_float4(fa.x, fa.y, fb.x, fb.y);
}

// fp8 indexer-K cache row (SM70-private inline layout, 132 bytes):
//   bytes [0,128):    128 e4m3 values (dequant = fp8_val * scale)
//   bytes [128,132):  1 fp32 per-token scale
// Written by sm70_indexer_k_store_fp8_kernel below.
constexpr int IDXK_FP8_ROW_BYTES = 132;
static __device__ __forceinline__ float4 idxk_fp8_load4(const uint8_t* rowp,
                                                        int d) {
  const uchar4 v = *reinterpret_cast<const uchar4*>(rowp + d);
  const float s = *reinterpret_cast<const float*>(rowp + 128) * 256.0f;
  return make_float4(__half2float(sm70_fp8_to_half_div256(v.x)) * s,
                     __half2float(sm70_fp8_to_half_div256(v.y)) * s,
                     __half2float(sm70_fp8_to_half_div256(v.z)) * s,
                     __half2float(sm70_fp8_to_half_div256(v.w)) * s);
}

// unpack TWO e4m3 bytes (pre-spread into the low bytes of each 16-bit lane of
// x01, e.g. via __byte_perm(a, 0, 0x4140)) into a half2 of value/256 each —
// the packed-SIMD form of sm70_fp8_to_half_div256 (2 values in ~4 bit ops,
// no multiplies). Used by the latency-bound decode indexer scan where the
// per-element fp32 dequant was ~30% instruction overhead.
static __device__ __forceinline__ half2 sm70_fp8x2_to_half2_div256(
    uint32_t x01) {
  const uint32_t bits =
      ((x01 & 0x00800080u) << 8) | ((x01 & 0x007f007fu) << 7);
  half2 out;
  *reinterpret_cast<uint32_t*>(&out) = bits;
  return out;
}

// ---------------------------------------------------------------------------
// DSA lightning-indexer logits (fp16, Volta). Replaces DeepGEMM fp8_mqa_logits.
// For each KV position p (0..seq_len-1):
//   score[p] = sum_{h<n_ihead} relu( sum_d q[h,d]*k_p[d] ) * weights[h]
// q: [B, n_ihead, HD] fp16 ; weights: [B, n_ihead] fp16 (pre-scaled by the
// caller, as in vLLM's Indexer.forward). k: paged indexer-K cache, flat rows
// [num_blocks*PAGE, HD] fp16. out: score [B, max_kv] fp32 (invalid/pad = -inf).
// Decode-path layout: one WARP per position, LANE = HEAD (N_IHEAD == 32).
// q is staged in shared memory ONCE per block (coalesced 8 KB load; padded
// rows so the lane=head register fill is bank-conflict-free) — the naive
// per-lane global q load re-read 64 KB per block, which dominated at long
// context. The lane's q row (64 half2) then lives in registers; the k row is
// staged per-warp in smem (2 half2 per lane) and read back with UNIFORM
// (broadcast) addresses, with 4 interleaved half2 accumulators to break the
// FMA dependency chain and an fp32 flush every 16 dims to bound fp16
// accumulation error. The k row for the NEXT position is prefetched into
// registers while the current one computes, and block-table entries are
// staged in smem, so the two dependent global loads per position are
// pipelined instead of latency-exposed (the old kernel's ~60 us/layer floor).
template <int HD, int N_IHEAD, int PPB, bool K_FP8>
__global__ void __launch_bounds__(256) sm70_indexer_logits_kernel(
    const half* __restrict__ Q, const void* __restrict__ Kv,
    const half* __restrict__ W, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ score,
    uint32_t* __restrict__ hist,  // optional [B, TOPK_NBIN]: fused top-k hist
    const int n_kv_rows, const int max_blocks, const int page_size,
    const int max_kv) {
  static_assert(N_IHEAD == 32, "lane-per-head layout needs 32 indexer heads");
  constexpr int NWARP = 256 / 32;
  constexpr int PPW = PPB / NWARP;  // positions per warp
  constexpr int HD2 = HD / 2;       // 64 half2
  constexpr int QPAD = HD2 + 1;     // pad q rows: bank-conflict-free by-head
  const int tok = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;  // = head index

  const int seq_len = seq_lens[tok];
  float* score_row = score + (size_t)tok * max_kv;

  const int p_lo = blockIdx.y * PPB;
  const int p_hi = min(p_lo + PPB, max_kv);
  if (p_lo >= p_hi) return;

  // Blocks fully beyond the causal context only need the -inf padding (kept
  // for the non-fused top-k safety); skip the 8 KB q/block-table staging and
  // the whole pipelined scan. At a 200k max_model_len decode graph the grid
  // is ~1584 blocks wide regardless of the live context — without this the
  // out-of-range blocks' q staging alone costs ~1 ms/token across 78 layers.
  if (p_lo >= seq_len) {
    for (int i = p_lo + (int)threadIdx.x; i < p_hi; i += 256)
      score_row[i] = -INFINITY;
    return;
  }

  __shared__ half2 s_q[N_IHEAD * QPAD];
  __shared__ half2 s_k[NWARP][HD2];
  __shared__ int32_t s_bt[PPB / 8 + 2];  // generous for page_size >= 8

  // stage this block's block-table entries (<= PPB/page + 1 pages)
  const int pg0 = p_lo / page_size;
  const int npg = (p_hi - 1) / page_size - pg0 + 1;
  if ((int)threadIdx.x < npg)
    s_bt[threadIdx.x] =
        (pg0 + (int)threadIdx.x < max_blocks)
            ? block_table[(size_t)tok * max_blocks + pg0 + threadIdx.x]
            : -1;
  // stage q coalesced into padded smem rows
  {
    const half2* q2 =
        reinterpret_cast<const half2*>(Q + (size_t)tok * N_IHEAD * HD);
    for (int i = threadIdx.x; i < N_IHEAD * HD2; i += 256)
      s_q[(i / HD2) * QPAD + (i % HD2)] = q2[i];
  }
  __syncthreads();

  // this lane's head: q row into registers (padded smem -> no bank conflicts)
  half2 q_reg[HD2];
#pragma unroll
  for (int i = 0; i < HD2; i++) q_reg[i] = s_q[lane * QPAD + i];
  const float w = __half2float(W[(size_t)tok * N_IHEAD + lane]);

  const half2 zero2 = __float2half2_rn(0.0f);

  // software pipeline: k(p+NWARP) loads into regs while k(p) computes.
  // page_size is a power of two (checked host-side): use shift/mask — the
  // per-position runtime integer div/mod cost ~40 cycles each.
  const int page_shift = 31 - __clz(page_size);
  const int page_mask = page_size - 1;
  // For K_FP8 the k row is kept UNSCALED (raw e4m3 value/256 as half) through
  // the dot, and the per-token fp32 scale (x256) is applied ONCE to the dot
  // result — no per-element dequant multiplies in this latency/issue-bound
  // scan. sc is unused in the fp16 path.
  auto load_k = [&](int p, half2& k0, half2& k1, float& sc) {
    k0 = zero2;
    k1 = zero2;
    sc = 0.0f;
    if (p < p_hi && p < seq_len) {
      const int blk = s_bt[(p >> page_shift) - pg0];
      const int64_t row = ((int64_t)blk << page_shift) + (p & page_mask);
      if (row >= 0 && row < n_kv_rows) {
        if constexpr (K_FP8) {
          // ONE coalesced uchar4 per lane (dims 4*lane .. 4*lane+3): a single
          // 128-byte transaction per position (vs 256 B for fp16). Bytes are
          // spread to 16-bit lanes with byte_perm and bit-shifted to
          // half(value/256) — no dequant multiplies (see the epilogue).
          const uint8_t* rp =
              reinterpret_cast<const uint8_t*>(Kv) + row * IDXK_FP8_ROW_BYTES;
          const uint32_t a = *reinterpret_cast<const uint32_t*>(rp + 4 * lane);
          k0 = sm70_fp8x2_to_half2_div256(__byte_perm(a, 0, 0x4140));
          k1 = sm70_fp8x2_to_half2_div256(__byte_perm(a, 0, 0x4342));
          sc = *reinterpret_cast<const float*>(rp + 128) * 256.0f;
        } else {
          const half2* k2 = reinterpret_cast<const half2*>(
              reinterpret_cast<const half*>(Kv) + row * HD);
          k0 = k2[lane];
          k1 = k2[lane + 32];
        }
      }
    }
  };

  int p = p_lo + warp;
  // depth-2 software pipeline: the ~64-LDS/64-HFMA2 dot (~150-200 cycles) is
  // shorter than the ~400-500-cycle k gather, so one prefetch slot leaves the
  // load partially exposed; two in flight cover it.
  half2 k0, k1, m0, m1, n0, n1;
  float sk, sm_, sn;
  load_k(p, k0, k1, sk);
  load_k(p + NWARP, m0, m1, sm_);
#pragma unroll 1  // the dot body is large; full unroll thrashes the i-cache
  for (int it = 0; it < PPW; it++, p += NWARP) {
    if (p >= p_hi) break;
    load_k(p + 2 * NWARP, n0, n1, sn);  // prefetch two positions ahead
    if (p >= seq_len) {
      if (lane == 0) score_row[p] = -INFINITY;
    } else {
      // publish this position's k row to the warp's smem buffer
      if constexpr (K_FP8) {
        // fp8 load order: lane holds dims 4*lane..4*lane+3
        s_k[warp][2 * lane] = k0;
        s_k[warp][2 * lane + 1] = k1;
      } else {
        s_k[warp][lane] = k0;
        s_k[warp][lane + 32] = k1;
      }
      __syncwarp();
      // dot from uniform (broadcast) smem addresses, vectorized as float2
      // (2 half2 per LDS.64); 4 interleaved half2 accumulators (4 deep each)
      // break the FMA dependency chain; fp32 flush every 8 half2 (16 dims)
      // per accumulator pair bounds fp16 accumulation error.
      const float2* kf = reinterpret_cast<const float2*>(&s_k[warp][0]);
      float dot = 0.0f;
#pragma unroll
      for (int c = 0; c < HD2 / 2; c += 8) {  // 8 float2 = 16 half2 = 32 dims
        half2 a0 = zero2, a1 = zero2, a2 = zero2, a3 = zero2;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float2 f = kf[c + 2 * i];
          const float2 g = kf[c + 2 * i + 1];
          a0 = __hfma2(q_reg[2 * (c + 2 * i)],
                       *reinterpret_cast<const half2*>(&f.x), a0);
          a1 = __hfma2(q_reg[2 * (c + 2 * i) + 1],
                       *reinterpret_cast<const half2*>(&f.y), a1);
          a2 = __hfma2(q_reg[2 * (c + 2 * i + 1)],
                       *reinterpret_cast<const half2*>(&g.x), a2);
          a3 = __hfma2(q_reg[2 * (c + 2 * i + 1) + 1],
                       *reinterpret_cast<const half2*>(&g.y), a3);
        }
        dot += __low2float(a0) + __high2float(a0);
        dot += __low2float(a1) + __high2float(a1);
        dot += __low2float(a2) + __high2float(a2);
        dot += __low2float(a3) + __high2float(a3);
      }
      __syncwarp();  // s_k reused next iteration
      // relu * head weight, then sum over the 32 heads (lanes). For K_FP8 the
      // dot was computed on value/256 halves: apply the per-token scale (x256
      // folded) ONCE here (scale > 0, so it commutes with the relu).
      if constexpr (K_FP8) dot *= sk;
      float v = (dot > 0.0f ? dot : 0.0f) * w;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
      if (lane == 0) {
        score_row[p] = v;
        // fused top-k histogram (decode path): saves the top-k kernel a full
        // re-read + histogram pass over the score row.
        if (hist != nullptr)
          atomicAdd(&hist[(size_t)tok * TOPK_NBIN + (fp32_monotone(v) >> 20)],
                    1u);
      }
    }
    k0 = m0;
    k1 = m1;
    sk = sm_;
    m0 = n0;
    m1 = n1;
    sm_ = sn;
  }
}

#ifdef SM70_MLA_WMMA_AVAILABLE
// FlashAttention-style WMMA indexer logits. Same math as the scalar kernel:
//   score[q,p] = sum_h relu( q[q,h,:] . k_p[:] ) * weights[q,h]
// but tiled so each gathered K row is reused across a whole query tile (QT
// tokens) and all N_IHEAD heads, and the q.k dot uses Volta tensor cores.
// This is the long-context prefill win: the scalar kernel re-reads each K row
// once PER QUERY TOKEN (512x at a 512-token prefill chunk) with scalar FMA;
// this reads K once per (query-tile, key-tile) and contracts HD=128 on WMMA.
//
// grid = (cdiv(B, QT), cdiv(max_kv, KT)); block = WARPS*32.
//   Q: [B, N_IHEAD, HD] fp16 ; W: [B, N_IHEAD] fp16 ; K: paged flat [rows, HD].
//   block_table: [B, max_blocks] int32 (per token; a query tile shares one
//   sequence in prefill, so row qbase's table is used for the gather).
//   score: [B, max_kv] fp32 (p >= seq_len[q] -> -inf).
template <int HD, int N_IHEAD, int QT, int KT, int WARPS, bool K_FP8>
__global__ void __launch_bounds__(WARPS * 32) sm70_indexer_logits_wmma_kernel(
    const half* __restrict__ Q, const void* __restrict__ Kv,
    const half* __restrict__ W, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ score,
    const int B, const int n_kv_rows, const int max_blocks,
    const int page_size, const int max_kv) {
  const int qtile = blockIdx.x;
  const int ktile = blockIdx.y;
  const int qbase = qtile * QT;         // first query token
  const int pbase = ktile * KT;         // first key position
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  constexpr int DK16 = HD / 16;         // 128/16 = 8 k-tiles for the QK contraction
  constexpr int NKT = KT / 16;          // key-position tiles (WMMA N)
  constexpr int PADK = HD;              // HD=128 multiple of 8 -> ok WMMA stride

  extern __shared__ char smem_raw[];
  half* s_k = (half*)smem_raw;                       // [KT x PADK] gathered K tile
  half* s_q = (half*)(s_k + (size_t)KT * PADK);      // [QT x HD] one head's Q tile
  float* s_sc = (float*)(s_q + (size_t)QT * HD);     // [QT x KT] score accumulator
  float* s_pv = (float*)(s_sc + (size_t)QT * KT);    // [WARPS x QT*KT] QK partials

  // Gather the KT K rows ONCE for the whole query tile. All QT queries in a
  // prefill chunk share the same sequence, so use qbase's block_table row.
  const int32_t* bt_row = block_table + (size_t)qbase * max_blocks;
  for (int i = tid; i < KT * (HD / 4); i += WARPS * 32) {
    const int r = i / (HD / 4);
    const int d4 = i % (HD / 4);
    const int p = pbase + r;
    float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
    if (p < max_kv) {
      const int blk = bt_row[p / page_size];
      const int64_t row = (int64_t)blk * page_size + (p % page_size);
      if (row >= 0 && row < n_kv_rows) {
        if constexpr (K_FP8) {
          val = idxk_fp8_load4(
              reinterpret_cast<const uint8_t*>(Kv) + row * IDXK_FP8_ROW_BYTES,
              d4 * 4);
        } else {
          val = mla_load4_half(reinterpret_cast<const half*>(Kv),
                               row * (HD / 4) + d4);
        }
      }
    }
    const int d = d4 * 4;
    half* kd = s_k + r * PADK + d;
    kd[0] = __float2half(val.x); kd[1] = __float2half(val.y);
    kd[2] = __float2half(val.z); kd[3] = __float2half(val.w);
  }
  // zero the score accumulator
  for (int i = tid; i < QT * KT; i += WARPS * 32) s_sc[i] = 0.0f;
  __syncthreads();

  // per head: QK (WMMA, contract HD) -> relu -> * w[q,h] -> accumulate.
  constexpr int MT = QT / 16;           // query-row WMMA M tiles
  for (int h = 0; h < N_IHEAD; h++) {
    // stage this head's Q tile [QT x HD] into shared
    for (int i = tid; i < QT * HD; i += WARPS * 32) {
      const int q = i / HD, d = i % HD;
      const int gq = qbase + q;
      s_q[i] = (gq < B) ? Q[((size_t)gq * N_IHEAD + h) * HD + d]
                        : __float2half(0.0f);
    }
    __syncthreads();

    // QK: acc = s_q[QT x HD] * s_k[KT x HD]^T, contract HD (DK16 tiles),
    // split the MT x NKT output tiles across warps. Each warp owns a distinct
    // (query-row, key-col) 16x16 tile, so it can relu+weight+accumulate into
    // s_sc for that tile race-free.
    for (int t = warp; t < MT * NKT; t += WARPS) {
      const int mt = t / NKT;
      const int nt = t % NKT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
#pragma unroll
      for (int kt = 0; kt < DK16; kt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
        wmma::load_matrix_sync(Qf, s_q + (mt * 16) * HD + kt * 16, HD);
        wmma::load_matrix_sync(Kf, s_k + (nt * 16) * PADK + kt * 16, PADK);
        wmma::mma_sync(acc, Qf, Kf, acc);
      }
      // store this [16 x 16] tile to per-warp scratch, then relu+weight+accum.
      float* scr = s_pv + warp * (16 * 16);
      wmma::store_matrix_sync(scr, acc, 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 16 * 16; e += 32) {
        const int qr = mt * 16 + (e >> 4);  // query row within the QT tile
        const int kc = e & 15;              // key col within this tile
        const int gq = qbase + qr;
        const int kcol = nt * 16 + kc;
        float dot = scr[e];
        if (dot > 0.0f && gq < B) {
          const float w = __half2float(W[(size_t)gq * N_IHEAD + h]);
          s_sc[qr * KT + kcol] += dot * w;
        }
      }
    }
    __syncthreads();
  }

  // write scores with causal mask (p >= seq_len[q] -> -inf; also p >= max_kv).
  for (int i = tid; i < QT * KT; i += WARPS * 32) {
    const int q = i / KT, kcol = i % KT;
    const int gq = qbase + q;
    const int p = pbase + kcol;
    if (gq >= B || p >= max_kv) continue;
    const int sl = seq_lens[gq];
    score[(size_t)gq * max_kv + p] = (p < sl) ? s_sc[i] : -INFINITY;
  }
}
#endif  // SM70_MLA_WMMA_AVAILABLE

// ---------------------------------------------------------------------------
// PACKED-Q register-accumulator WMMA indexer (the fast prefill path).
//
// Microbenched 2026-07-07 (scratchpad idxk/bench.cu, 16x sweep over
// tilings/layouts/unrolls + ncu): the kernel above is L1-WAVEFRONT-bound
// (ncu: L1/TEX 97%, tensor pipes ~11%) because (a) the per-head epilogue
// round-trips every 16x16 accumulator tile through shared memory into a
// shared score accumulator, (b) Q is re-staged into smem per (head, block)
// with 2 barriers per head, and (c) A-fragment loads from strided Q layouts
// touch 16 non-adjacent 32 B sectors per fragment (16 L1 wavefronts vs the
// 4 minimum). This kernel fixes all three:
//   - Q is PRE-PACKED by the caller as Qp[NH][ceil(B/16)][8][16][16] so every
//     A fragment is 512 CONTIGUOUS bytes (4 L1 wavefronts);
//   - the 8 K B-fragments per warp-tile are hoisted into registers across
//     the whole 32-head loop (head-invariant);
//   - the relu(dot)*w epilogue and the cross-head score accumulation run
//     entirely in registers using the (empirically derived, self-checked at
//     first launch) Volta m16n16k16 f32 accumulator fragment layout;
//   - no __syncthreads after the initial K-tile gather; the head loop is
//     fully unrolled with a single sequential accumulator per tile, which
//     keeps the mma accumulation order IDENTICAL to the kernel above
//     (bit-exact scores -> identical top-k selection).
// Measured (V100, B=128): 32 -> 44 TFLOPS over spans 8k..49k vs 6.5-7.6 for
// the kernel above (~5-6.5x). Fixed tiling QT=32/KT=32/WARPS=4 won the sweep.
//
// grid = (cdiv(B,32), cdiv(max_kv,32)); block = 128 threads.
#ifdef SM70_MLA_WMMA_AVAILABLE
// Volta wmma m16n16k16 f32 accumulator layout: (lane, elem) -> row/col within
// the 16x16 tile. Derived empirically (basis-vector probes); validated at
// first launch by sm70_idx_layout_probe_kernel — a mismatch aborts loudly
// rather than silently corrupting scores on a future toolchain.
__constant__ int c_idx_accrow[256] = {
    0,0,2,2,0,0,2,2,1,1,3,3,1,1,3,3,0,0,2,2,0,0,2,2,1,1,3,3,1,1,3,3,
    8,8,10,10,8,8,10,10,9,9,11,11,9,9,11,11,8,8,10,10,8,8,10,10,9,9,11,11,9,9,11,11,
    0,0,2,2,0,0,2,2,1,1,3,3,1,1,3,3,0,0,2,2,0,0,2,2,1,1,3,3,1,1,3,3,
    8,8,10,10,8,8,10,10,9,9,11,11,9,9,11,11,8,8,10,10,8,8,10,10,9,9,11,11,9,9,11,11,
    4,4,6,6,4,4,6,6,5,5,7,7,5,5,7,7,4,4,6,6,4,4,6,6,5,5,7,7,5,5,7,7,
    12,12,14,14,12,12,14,14,13,13,15,15,13,13,15,15,12,12,14,14,12,12,14,14,13,13,15,15,13,13,15,15,
    4,4,6,6,4,4,6,6,5,5,7,7,5,5,7,7,4,4,6,6,4,4,6,6,5,5,7,7,5,5,7,7,
    12,12,14,14,12,12,14,14,13,13,15,15,13,13,15,15,12,12,14,14,12,12,14,14,13,13,15,15,13,13,15,15};
__constant__ int c_idx_acccol[256] = {
    0,1,0,1,4,5,4,5,0,1,0,1,4,5,4,5,2,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,
    0,1,0,1,4,5,4,5,0,1,0,1,4,5,4,5,2,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,
    8,9,8,9,12,13,12,13,8,9,8,9,12,13,12,13,10,11,10,11,14,15,14,15,10,11,10,11,14,15,14,15,
    8,9,8,9,12,13,12,13,8,9,8,9,12,13,12,13,10,11,10,11,14,15,14,15,10,11,10,11,14,15,14,15,
    0,1,0,1,4,5,4,5,0,1,0,1,4,5,4,5,2,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,
    0,1,0,1,4,5,4,5,0,1,0,1,4,5,4,5,2,3,2,3,6,7,6,7,2,3,2,3,6,7,6,7,
    8,9,8,9,12,13,12,13,8,9,8,9,12,13,12,13,10,11,10,11,14,15,14,15,10,11,10,11,14,15,14,15,
    8,9,8,9,12,13,12,13,8,9,8,9,12,13,12,13,10,11,10,11,14,15,14,15,10,11,10,11,14,15,14,15};

// re-derive the accumulator layout on device and compare with the tables.
__global__ void sm70_idx_layout_probe_kernel(int* ok) {
  __shared__ half a[256], b[256];
  const int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < 256; i += 32) {
    a[i] = __float2half(0.f);
    b[i] = __float2half(0.f);
  }
  __syncwarp();
  if (lane < 16) {
    a[lane * 16] = __float2half((float)(lane + 1));  // A[i][0] = i+1
    b[lane * 16] = __float2half((float)(lane + 101));  // B[0][j] = j+101
  }
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> Cf;
  wmma::load_matrix_sync(Af, a, 16);
  wmma::load_matrix_sync(Bf, b, 16);
  wmma::fill_fragment(Cf, 0.f);
  wmma::mma_sync(Cf, Af, Bf, Cf);  // C[i][j] = (i+1)*(j+101)
  bool good = (Cf.num_elements == 8);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float expect = (float)(c_idx_accrow[lane * 8 + i] + 1) *
                         (float)(c_idx_acccol[lane * 8 + i] + 101);
    good &= (Cf.x[i] == expect);
  }
  if (!__all_sync(0xffffffff, good) && lane == 0) atomicExch(ok, 0);
}

template <bool K_FP8>
__global__ void __launch_bounds__(128) sm70_indexer_logits_packed_kernel(
    const half* __restrict__ Qp, const void* __restrict__ Kv,
    const half* __restrict__ W, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ score,
    const int B, const int n_kv_rows, const int max_blocks,
    const int page_size, const int max_kv) {
  constexpr int HD = 128, N_IHEAD = 32;
  constexpr int QT = 32, KT = 32, WARPS = 4;
  constexpr int DK16 = HD / 16, NKT = KT / 16, MT = QT / 16;
  const int qtile = blockIdx.x, ktile = blockIdx.y;
  const int qbase = qtile * QT, pbase = ktile * KT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  __shared__ half s_k[KT * HD];          // gathered K tile
  __shared__ half s_w[N_IHEAD * QT];     // W transposed [h][q]

  // gather the KT K rows (fp16: 16 B vectors; fp8: dequant in 8 B steps)
  const int32_t* bt_row = block_table + (size_t)qbase * max_blocks;
  if constexpr (K_FP8) {
    for (int i = tid; i < KT * (HD / 4); i += WARPS * 32) {
      const int r = i / (HD / 4), d4 = i % (HD / 4);
      const int p = pbase + r;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (p < max_kv) {
        const int blk = bt_row[p / page_size];
        const int64_t row = (int64_t)blk * page_size + (p % page_size);
        if (row >= 0 && row < n_kv_rows)
          val = idxk_fp8_load4(
              reinterpret_cast<const uint8_t*>(Kv) + row * IDXK_FP8_ROW_BYTES,
              d4 * 4);
      }
      half* kd = s_k + r * HD + d4 * 4;
      kd[0] = __float2half(val.x); kd[1] = __float2half(val.y);
      kd[2] = __float2half(val.z); kd[3] = __float2half(val.w);
    }
  } else {
    const half* kvh = reinterpret_cast<const half*>(Kv);
    for (int i = tid; i < KT * (HD / 8); i += WARPS * 32) {
      const int r = i / (HD / 8), d8 = i % (HD / 8);
      const int p = pbase + r;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (p < max_kv) {
        const int blk = bt_row[p / page_size];
        const int64_t row = (int64_t)blk * page_size + (p % page_size);
        if (row >= 0 && row < n_kv_rows)
          v = reinterpret_cast<const float4*>(kvh + row * HD)[d8];
      }
      reinterpret_cast<float4*>(s_k + r * HD)[d8] = v;
    }
  }
  for (int i = tid; i < QT * N_IHEAD; i += WARPS * 32) {
    const int h = i / QT, q = i % QT;
    const int gq = qbase + q;
    s_w[i] = (gq < B) ? W[(size_t)gq * N_IHEAD + h] : __float2half(0.f);
  }
  __syncthreads();

  const int mt = warp / NKT, nt = warp % NKT;
  // Qp[h][qt16][kt][16][16]; this warp's 16-query tile. A block spans MT=2
  // 16-row tiles but the packed Q only has ceil(B/16): for B in (0,16] mod 32
  // the block's SECOND tile lies wholly past the array — every row is >= B
  // (write-masked anyway), but the fragment loads would read OUT OF BOUNDS
  // (probabilistic Xid 31; hit live on an 11-token prompt). Bail AFTER the
  // barrier (no block syncs follow, so an early-exiting warp is safe).
  const int n_qt = (B + 15) / 16;
  const int gqt = qtile * MT + mt;
  if (gqt >= n_qt) return;

  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf[DK16];
#pragma unroll
  for (int kt = 0; kt < DK16; kt++)
    wmma::load_matrix_sync(Kf[kt], s_k + (nt * 16) * HD + kt * 16, HD);

  int erow[8];
#pragma unroll
  for (int i = 0; i < 8; i++) erow[i] = c_idx_accrow[lane * 8 + i];

  float facc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) facc[i] = 0.f;

  const half* qp0 = Qp + (size_t)gqt * DK16 * 256;
  const size_t hstride = (size_t)n_qt * DK16 * 256;
#pragma unroll
  for (int h = 0; h < N_IHEAD; h++) {
    const half* qtile_ptr = qp0 + h * hstride;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (int kt = 0; kt < DK16; kt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
      wmma::load_matrix_sync(Qf, qtile_ptr + kt * 256, 16);
      wmma::mma_sync(acc, Qf, Kf[kt], acc);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float dot = acc.x[i];
      if (dot > 0.0f)
        facc[i] += dot * __half2float(s_w[h * QT + mt * 16 + erow[i]]);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int ecol_i = c_idx_acccol[lane * 8 + i];
    const int gq = qbase + mt * 16 + erow[i];
    const int p = pbase + nt * 16 + ecol_i;
    if (gq >= B || p >= max_kv) continue;
    score[(size_t)gq * max_kv + p] =
        (p < seq_lens[gq]) ? facc[i] : -INFINITY;
  }
}
#endif  // SM70_MLA_WMMA_AVAILABLE

// ---------------------------------------------------------------------------
// fp16 indexer-K store. CUDA-graph-safe scatter (no host sync, no dynamic
// shapes, race-free). One thread block per token; threads copy head_dim
// elements. Tokens with slot_mapping < 0 (CUDA-graph padding) are skipped
// entirely on device, so no real KV row is touched. k: [n_tok, HD] fp16,
// kv_cache flat rows [n_rows, HD] fp16, slot_mapping: [n_tok] int (row index).
__global__ void sm70_indexer_k_store_kernel(
    const half* __restrict__ K, const int32_t* __restrict__ slot,
    half* __restrict__ KV, const int n_tok, const int head_dim,
    const int64_t n_rows) {
  const int tok = blockIdx.x;
  if (tok >= n_tok) return;
  const int64_t row = (int64_t)slot[tok];
  if (row < 0 || row >= n_rows) return;  // skip padding / OOB
  const half* src = K + (size_t)tok * head_dim;
  half* dst = KV + row * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dst[d] = src[d];
}

// fp8 (e4m3) indexer-K store: same CUDA-graph-safe per-token scatter, but
// quantizing to the 132-byte inline row layout [128 e4m3 | fp32 scale]
// (scale = amax/448, dequant = fp8_val * scale — matching the upstream
// per-token dynamic scaling). One block of HD=128 threads per token; block
// amax via warp shuffles + smem. __nv_cvt_float_to_fp8 is software-emulated
// on pre-SM89, so this compiles and runs on Volta.
__global__ void sm70_indexer_k_store_fp8_kernel(
    const half* __restrict__ K, const int32_t* __restrict__ slot,
    uint8_t* __restrict__ KV, const int n_tok, const int64_t n_rows) {
  constexpr int HD = 128;
  const int tok = blockIdx.x;
  if (tok >= n_tok) return;
  const int64_t row = (int64_t)slot[tok];
  if (row < 0 || row >= n_rows) return;  // skip padding / OOB
  const int d = threadIdx.x;             // 0..127
  const float v = __half2float(K[(size_t)tok * HD + d]);

  __shared__ float s_amax[HD / 32];
  float a = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
  if ((d & 31) == 0) s_amax[d >> 5] = a;
  __syncthreads();
  const float amax = fmaxf(fmaxf(s_amax[0], s_amax[1]),
                           fmaxf(s_amax[2], s_amax[3]));
  const float scale = fmaxf(amax / 448.0f, FLT_MIN);

  uint8_t* rp = KV + row * IDXK_FP8_ROW_BYTES;
  rp[d] = (uint8_t)__nv_cvt_float_to_fp8(v / scale, __NV_SATFINITE, __NV_E4M3);
  if (d == 0) *reinterpret_cast<float*>(rp + HD) = scale;
}

#ifdef SM70_MLA_WMMA_AVAILABLE
// grid = (n_tok=B, n_head/HB); block = WARPS*32 threads.
// Each block handles HB heads (blockIdx.y * HB .. +HB) of one decode token.
// Q  : [B, n_head, D_K]  fp16
// KV : paged latent cache, flat rows [num_blocks*PAGE, D_K] fp16 (V = first D_V)
// block_table : [B, max_blocks] int32 (physical block ids)
// seq_lens    : [B] int32 (context length incl. the current decode token)
// O  : [B, n_head, D_V] fp16
// LSE: [B, n_head] fp32
template <int D_K, int D_V, int HB, int TN, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) sm70_mla_decode_kernel(
    const half* __restrict__ Q, const half* __restrict__ KV,
    const int32_t* __restrict__ block_table, const int32_t* __restrict__ seq_lens,
    half* __restrict__ O, float* __restrict__ LSE, const int n_head,
    const int n_kv_rows, const int max_blocks, const int page_size,
    const float scale) {
  const int tok = blockIdx.x;
  const int head_group = blockIdx.y;  // which block of HB heads
  const int hbase = head_group * HB;  // first global head handled here
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  constexpr int NTILES = TN / 16;
  constexpr int DK16 = D_K / 16;  // 36
  constexpr int DV16 = D_V / 16;  // 32
  constexpr int PADK = D_K;       // D_K=576 is a multiple of 8 -> WMMA stride ok
  constexpr int HT = HB / 16;     // head-tiles within this block

  extern __shared__ char smem_raw[];
  half* s_k = (half*)smem_raw;                              // [TN x PADK]
  half* s_p = s_k + (size_t)TN * PADK;                      // [16 x TN]
  float* s_sc = (float*)(s_p + (size_t)16 * TN);            // [16 x TN]
  half* s_o = (half*)(s_sc + (size_t)16 * TN);              // [HB x D_V]
  float* s_pv = (float*)(s_o + (size_t)HB * D_V);           // [WARPS x max(256,16*TN)]
  half* s_qs = (half*)(s_pv + (size_t)WARPS * (16 * TN > 256 ? 16 * TN : 256));  // [WARPS x 256]
  float* s_m = (float*)(s_qs + (size_t)WARPS * 256);        // [HB]
  float* s_l = s_m + HB;                                    // [HB]

  const int seq_len = seq_lens[tok];  // number of valid KV positions (causal)
  const int32_t* bt_row = block_table + (size_t)tok * max_blocks;

  for (int i = tid; i < HB; i += WARPS * 32) {
    s_m[i] = -1e30f;
    s_l[i] = 0.0f;
  }
  for (int i = tid; i < HB * D_V; i += WARPS * 32) s_o[i] = __float2half(0.0f);
  // Q base for this token's head-group.
  const half* Qbase = Q + (size_t)tok * n_head * D_K + (size_t)hbase * D_K;
  __syncthreads();

  for (int base = 0; base < seq_len; base += TN) {
    // Gather TN latent KV rows ONCE (shared by all head-tiles). Row for KV
    // position j = block_table[j / page]*page + j % page.
    for (int i = tid; i < TN * (D_K / 4); i += WARPS * 32) {
      const int r = i / (D_K / 4);
      const int d4 = i % (D_K / 4);
      const int j = base + r;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (j < seq_len) {
        const int blk = bt_row[j / page_size];
        const int64_t row = (int64_t)blk * page_size + (j % page_size);
        if (row >= 0 && row < n_kv_rows)
          val = mla_load4_half(KV, row * (D_K / 4) + d4);
      }
      const int d = d4 * 4;
      half* kdst = s_k + r * PADK + d;
      kdst[0] = __float2half(val.x);
      kdst[1] = __float2half(val.y);
      kdst[2] = __float2half(val.z);
      kdst[3] = __float2half(val.w);
    }
    __syncthreads();

    for (int ht = 0; ht < HT; ht++) {
      const int h0 = ht * 16;               // local head offset within block
      const half* Qtok = Qbase + (size_t)h0 * D_K;

      // QK: s_sc[16 x TN] = Q[16 x D_K] * s_k[TN x D_K]^T. Split the D_K
      // contraction (DK16 k-tiles) across warps; partials -> per-warp s_pv,
      // then reduced into s_sc.
      {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NTILES];
#pragma unroll
        for (int nt = 0; nt < NTILES; nt++) wmma::fill_fragment(acc[nt], 0.0f);
        for (int kt = warp; kt < DK16; kt += WARPS) {
          half* qs = s_qs + warp * 256;
#pragma unroll
          for (int e = lane; e < 256; e += 32) {
            const int h = e >> 4, c = e & 15;
            qs[e] = __hmul(Qtok[h * D_K + kt * 16 + c], __float2half(scale));
          }
          __syncwarp();
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
          wmma::load_matrix_sync(Qf, qs, 16);
#pragma unroll
          for (int nt = 0; nt < NTILES; nt++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
            wmma::load_matrix_sync(Kf, s_k + (nt * 16) * PADK + kt * 16, PADK);
            wmma::mma_sync(acc[nt], Qf, Kf, acc[nt]);
          }
        }
#pragma unroll
        for (int nt = 0; nt < NTILES; nt++)
          wmma::store_matrix_sync(s_pv + warp * (16 * TN) + nt * 16, acc[nt], TN,
                                  wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < 16 * TN; i += WARPS * 32) {
          float sum = 0.0f;
#pragma unroll
          for (int w = 0; w < WARPS; w++) sum += s_pv[w * (16 * TN) + i];
          s_sc[i] = sum;
        }
      }
      __syncthreads();

      // flash softmax over TN cols, per head-row; rescale s_o[h0+h].
      for (int h = warp; h < 16; h += WARPS) {
        const int hg = h0 + h;
        float m_cur = s_m[hg];
        for (int c = lane; c < TN; c += 32) {
          const int j = base + c;
          const bool valid = (j < seq_len);
          float v = valid ? s_sc[h * TN + c] : -INFINITY;
          s_sc[h * TN + c] = v;
          m_cur = fmaxf(m_cur, v);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffff, m_cur, off));
        const float m_prev = s_m[hg];
        const float alpha = expf(m_prev - m_cur);
        float lsum = 0.0f;
        for (int c = lane; c < TN; c += 32) {
          float v = s_sc[h * TN + c];
          float p = (v == -INFINITY) ? 0.0f : expf(v - m_cur);
          s_p[h * TN + c] = __float2half(p);
          lsum += p;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          lsum += __shfl_xor_sync(0xffffffff, lsum, off);
        if (lane == 0) {
          s_m[hg] = m_cur;
          s_l[hg] = s_l[hg] * alpha + lsum;
        }
        for (int d = lane; d < D_V; d += 32)
          s_o[hg * D_V + d] =
              __float2half(__half2float(s_o[hg * D_V + d]) * alpha);
      }
      __syncthreads();

      // PV: s_o[16 x D_V] += s_p[16 x TN] * s_v[TN x D_V]  (V = first D_V of s_k)
      for (int vt = warp; vt < DV16; vt += WARPS) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int nt = 0; nt < NTILES; nt++) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Pf;
          wmma::load_matrix_sync(Pf, s_p + nt * 16, TN);
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Vf;
          wmma::load_matrix_sync(Vf, s_k + (nt * 16) * PADK + vt * 16, PADK);
          wmma::mma_sync(acc, Pf, Vf, acc);
        }
        float* scratch = s_pv + warp * 256;
        wmma::store_matrix_sync(scratch, acc, 16, wmma::mem_row_major);
#pragma unroll
        for (int e = lane; e < 256; e += 32) {
          const int rr = e >> 4, cc = e & 15;
          s_o[(h0 + rr) * D_V + vt * 16 + cc] = __float2half(
              __half2float(s_o[(h0 + rr) * D_V + vt * 16 + cc]) + scratch[e]);
        }
      }
      __syncthreads();
    }
  }

  // finalize: o = acc / l ; lse = m + log(l)
  for (int hg = warp; hg < HB; hg += WARPS) {
    const float l = s_l[hg];
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    const int g_head = hbase + hg;  // global head index
    half* oh = O + (size_t)tok * n_head * D_V + (size_t)g_head * D_V;
    for (int d = lane; d < D_V; d += 32)
      oh[d] = __float2half(__half2float(s_o[hg * D_V + d]) * inv);
    if (lane == 0) {
      LSE[(size_t)tok * n_head + g_head] =
          (l > 0.0f) ? (s_m[hg] + logf(l)) : -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// Sparse (top-k) MLA decode. Ported from llama.cpp sparse_attn_kernel_wmma
// (the head-tile nvcuda::wmma kernel, the V100 winner at TN=32/WARPS=16). One
// block per (token, head-tile of 16 heads, kv-split); grid = (n_tok, n_head/16,
// n_splits). Iterates the TOPK indices selected by the DSA indexer over its
// split's slice [split*span, (split+1)*span), gathering each selected latent KV
// row from the PAGED cache. Cost is O(topk) -> context-INDEPENDENT past topk.
// The kv-split axis (blockIdx.z) raises occupancy for low-batch/few-head decode
// (B=1, TP=8 -> 8 heads -> only 1 block without splits, ~1% occupancy on 80 SMs;
// see NOTES). Each split writes a PARTIAL online-softmax state (o, m, l) to
// scratch; sm70_sparse_mla_merge_kernel combines them.
//   Q   : [B, n_head, D_K] fp16
//   KV  : paged latent cache, flat rows [num_blocks*PAGE, D_K] fp16 (V=first D_V)
//   idx : [B, topk] int32  -- logical KV positions (indexer top-k), -1 = unused
//   block_table : [B, max_blocks] int32
//   seq_lens    : [B] int32 (causal upper bound on the logical position)
//   P_O : [B, n_head, n_splits, D_V] fp32 partial outputs (pre-normalized accum)
//   P_ML: [B, n_head, n_splits, 2]   fp32 partial (m, l) per split
template <int D_K, int D_V, int TN, int WARPS, bool KV_FP8>
__global__ void __launch_bounds__(WARPS * 32) sm70_sparse_mla_decode_kernel(
    const half* __restrict__ Q, const void* __restrict__ KVv,
    const int32_t* __restrict__ idx, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, float* __restrict__ P_O,
    float* __restrict__ P_ML, const int n_head, const int n_kv_rows,
    const int max_blocks, const int page_size, const int topk,
    const int n_splits, const float scale) {
  const int tok = blockIdx.x;
  const int ht = blockIdx.y;   // head-tile index (each = 16 heads)
  const int h0 = ht * 16;      // first (global) head of this tile
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  constexpr int NTILES = TN / 16;
  constexpr int DK16 = D_K / 16;
  constexpr int DV16 = D_V / 16;
  constexpr int PADK = D_K + 8;

  // This block's slice of the topk axis (aligned to TN so tiles don't straddle).
  int span = (topk + n_splits - 1) / n_splits;
  span = ((span + TN - 1) / TN) * TN;  // round up to a multiple of TN
  const int base_lo = split * span;
  int base_hi = base_lo + span;
  if (base_hi > topk) base_hi = topk;

  extern __shared__ char smem_raw[];
  half* s_k = (half*)smem_raw;                          // [TN x PADK]
  half* s_p = s_k + (size_t)TN * PADK;                  // [16 x TN]
  float* s_sc = (float*)(s_p + (size_t)16 * TN);        // [16 x TN]
  half* s_o = (half*)(s_sc + (size_t)16 * TN);          // [16 x D_V]
  float* s_pv = (float*)(s_o + (size_t)16 * D_V);       // [WARPS x max(256,16*TN)]
  half* s_qs = (half*)(s_pv + (size_t)WARPS * (16 * TN > 256 ? 16 * TN : 256));

  __shared__ float s_m[16];
  __shared__ float s_l[16];

  const int seq_len = seq_lens[tok];
  const int32_t* idx_row = idx + (size_t)tok * topk;
  const int32_t* bt_row = block_table + (size_t)tok * max_blocks;

  for (int i = tid; i < 16; i += WARPS * 32) {
    s_m[i] = -1e30f;
    s_l[i] = 0.0f;
  }
  for (int i = tid; i < 16 * D_V; i += WARPS * 32) s_o[i] = __float2half(0.0f);
  const half* Qtok = Q + (size_t)tok * n_head * D_K + (size_t)h0 * D_K;
  __syncthreads();

  for (int base = base_lo; base < base_hi; base += TN) {

    // gather TN selected latent KV rows into shared f16 (via paged block_table)
    for (int i = tid; i < TN * (D_K / 4); i += WARPS * 32) {
      const int r = i / (D_K / 4);
      const int d4 = i % (D_K / 4);
      const int j = base + r;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      if (j < topk) {
        const int pos = idx_row[j];  // logical KV position selected by indexer
        if (pos >= 0 && pos < seq_len) {
          const int blk = bt_row[pos / page_size];
          const int64_t row = (int64_t)blk * page_size + (pos % page_size);
          if (row >= 0 && row < n_kv_rows) {
            if constexpr (KV_FP8) {
              val = dsmla_load4(reinterpret_cast<const uint8_t*>(KVv) +
                                    row * DSMLA_ROW_BYTES,
                                d4 * 4);
            } else {
              val = mla_load4_half(reinterpret_cast<const half*>(KVv),
                                   row * (D_K / 4) + d4);
            }
          }
        }
      }
      const int d = d4 * 4;
      half* kdst = s_k + r * PADK + d;
      kdst[0] = __float2half(val.x);
      kdst[1] = __float2half(val.y);
      kdst[2] = __float2half(val.z);
      kdst[3] = __float2half(val.w);
    }
    __syncthreads();

    // QK: s_sc[16 x TN] = Q[16 x D_K] * s_k[TN x D_K]^T
    {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NTILES];
#pragma unroll
      for (int nt = 0; nt < NTILES; nt++) wmma::fill_fragment(acc[nt], 0.0f);
      for (int kt = warp; kt < DK16; kt += WARPS) {
        half* qs = s_qs + warp * 256;
#pragma unroll
        for (int e = lane; e < 256; e += 32) {
          const int h = e >> 4, c = e & 15;
          // Guard partial head-tiles (e.g. TP splits 64 heads into 8/rank):
          // heads >= n_head are padded with 0 so QK is harmless; their outputs
          // are not written in the finalize loop below.
          half qv = (h0 + h < n_head)
                        ? __hmul(Qtok[h * D_K + kt * 16 + c], __float2half(scale))
                        : __float2half(0.0f);
          qs[e] = qv;
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
        wmma::load_matrix_sync(Qf, qs, 16);
#pragma unroll
        for (int nt = 0; nt < NTILES; nt++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
          wmma::load_matrix_sync(Kf, s_k + (nt * 16) * PADK + kt * 16, PADK);
          wmma::mma_sync(acc[nt], Qf, Kf, acc[nt]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < NTILES; nt++)
        wmma::store_matrix_sync(s_pv + warp * (16 * TN) + nt * 16, acc[nt], TN,
                                wmma::mem_row_major);
      __syncthreads();
      for (int i = tid; i < 16 * TN; i += WARPS * 32) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; w++) sum += s_pv[w * (16 * TN) + i];
        s_sc[i] = sum;
      }
    }
    __syncthreads();

    // flash softmax over the TN columns, per head-row (one warp per head)
    for (int h = warp; h < 16; h += WARPS) {
      float m_cur = s_m[h];
      for (int c = lane; c < TN; c += 32) {
        const int j = base + c;
        bool valid = false;
        if (j < topk) {
          const int pos = idx_row[j];
          valid = (pos >= 0 && pos < seq_len);
        }
        float v = valid ? s_sc[h * TN + c] : -INFINITY;
        s_sc[h * TN + c] = v;
        m_cur = fmaxf(m_cur, v);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffff, m_cur, off));
      const float m_prev = s_m[h];
      const float alpha = expf(m_prev - m_cur);
      float lsum = 0.0f;
      for (int c = lane; c < TN; c += 32) {
        float v = s_sc[h * TN + c];
        float p = (v == -INFINITY) ? 0.0f : expf(v - m_cur);
        s_p[h * TN + c] = __float2half(p);
        lsum += p;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
      if (lane == 0) {
        s_m[h] = m_cur;
        s_l[h] = s_l[h] * alpha + lsum;
      }
      for (int d = lane; d < D_V; d += 32)
        s_o[h * D_V + d] = __float2half(__half2float(s_o[h * D_V + d]) * alpha);
    }
    __syncthreads();

    // PV: s_o[16 x D_V] += s_p[16 x TN] * s_v[TN x D_V]  (V = first D_V of s_k)
    for (int vt = warp; vt < DV16; vt += WARPS) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
#pragma unroll
      for (int nt = 0; nt < NTILES; nt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Pf;
        wmma::load_matrix_sync(Pf, s_p + nt * 16, TN);
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Vf;
        wmma::load_matrix_sync(Vf, s_k + (nt * 16) * PADK + vt * 16, PADK);
        wmma::mma_sync(acc, Pf, Vf, acc);
      }
      float* scratch = s_pv + warp * 256;
      wmma::store_matrix_sync(scratch, acc, 16, wmma::mem_row_major);
#pragma unroll
      for (int e = lane; e < 256; e += 32) {
        const int rr = e >> 4, cc = e & 15;
        s_o[rr * D_V + vt * 16 + cc] = __float2half(
            __half2float(s_o[rr * D_V + vt * 16 + cc]) + scratch[e]);
      }
    }
    __syncthreads();
  }

  // finalize: write this split's PARTIAL online-softmax state (unnormalized
  // accumulator s_o, running max m, running sum l) to scratch. The merge kernel
  // combines splits. (skip padded heads of a partial tile)
  for (int h = warp; h < 16; h += WARPS) {
    const int g_head = h0 + h;
    if (g_head >= n_head) continue;
    const size_t po_base =
        (((size_t)tok * n_head + g_head) * n_splits + split) * D_V;
    for (int d = lane; d < D_V; d += 32)
      P_O[po_base + d] = __half2float(s_o[h * D_V + d]);
    if (lane == 0) {
      const size_t pml = ((size_t)tok * n_head + g_head) * n_splits + split;
      P_ML[pml * 2 + 0] = s_m[h];
      P_ML[pml * 2 + 1] = s_l[h];
    }
  }
}

// Merge partial split states into the final O / LSE (flash-decode reduction).
// grid = (B, n_head, cdiv(D_V, MERGE_DCHUNK)); block = MERGE_DCHUNK threads,
// ONE output element per thread. Reads P_O [B,H,S,D_V], P_ML [B,H,S,2].
//
// The D_V axis is split across gridDim.z for occupancy: the old (B, n_head)
// grid was 8 blocks at B=1/TP=8 (~10% of V100's 80 SMs) and its cost was
// data-dependent (splits with l==0 skip their P_O reads), which showed up as
// fake "context scaling" (28 us/layer short-ctx -> 55 us/layer at 16k when
// the top-k window is full). D_V=512 / 64 = 8 chunks -> 64 blocks at B=1.
// Per-split merge weights are computed once into shared memory; empty splits
// (w==0) still skip their P_O reads.
constexpr int MERGE_DCHUNK = 64;
constexpr int MERGE_MAX_SPLITS = 128;  // s_w capacity; host-checked
template <int D_V>
__global__ void sm70_sparse_mla_merge_kernel(
    const float* __restrict__ P_O, const float* __restrict__ P_ML,
    half* __restrict__ O, float* __restrict__ LSE, const int n_head,
    const int n_splits) {
  const int tok = blockIdx.x;
  const int g_head = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float s_ml[MERGE_MAX_SPLITS * 2];
  __shared__ float s_w[MERGE_MAX_SPLITS];

  const float* ml = P_ML + (((size_t)tok * n_head + g_head) * n_splits) * 2;
  for (int s = tid; s < n_splits; s += blockDim.x) {
    s_ml[s * 2 + 0] = ml[s * 2 + 0];
    s_ml[s * 2 + 1] = ml[s * 2 + 1];
  }
  __syncthreads();

  // global max over splits (redundant per thread; n_splits smem reads, cheap)
  float m = -1e30f;
  for (int s = 0; s < n_splits; s++) m = fmaxf(m, s_ml[s * 2 + 0]);
  // per-split merge weight (0 for empty splits so their P_O reads are skipped)
  for (int s = tid; s < n_splits; s += blockDim.x) {
    const float ls = s_ml[s * 2 + 1];
    s_w[s] = (ls > 0.0f) ? expf(s_ml[s * 2 + 0] - m) : 0.0f;
  }
  __syncthreads();
  // combined denominator
  float l = 0.0f;
  for (int s = 0; s < n_splits; s++) l += s_ml[s * 2 + 1] * s_w[s];
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;

  const int d = blockIdx.z * blockDim.x + tid;
  if (d < D_V) {
    const size_t o_base = ((size_t)tok * n_head + g_head) * n_splits * D_V;
    float acc = 0.0f;
    for (int s = 0; s < n_splits; s++) {
      const float w = s_w[s];
      if (w != 0.0f) acc += P_O[o_base + (size_t)s * D_V + d] * w;
    }
    O[((size_t)tok * n_head + g_head) * D_V + d] = __float2half(acc * inv);
  }
  if (blockIdx.z == 0 && tid == 0)
    LSE[(size_t)tok * n_head + g_head] =
        (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}
#endif  // SM70_MLA_WMMA_AVAILABLE

// ---------------------------------------------------------------------------
// Fused decode top-k select (replaces persistent_topk on the B<=4 decode path).
//
// idx[b, :] = positions of the top-K scores among p < seq_lens[b], -1 padded.
// Single kernel, grid (n_tiles, B), 256 threads; all blocks of a row are
// co-resident (grid is clamped to measured occupancy) and synchronize via
// global arrival counters (persistent_topk's pattern):
//   phase 1: per-block smem histogram of the top-12 monotone-fp32 bits of
//            each valid score (4096 bins), merged into a global per-row
//            histogram.
//   phase 2: every block redundantly suffix-scans the 4096-bin histogram
//            (copied to smem) to find the threshold bin T
//            (count(key > T) < K), then compacts its tile:
//                  key > T  -> idx via a global cursor (exact set),
//                  key == T -> per-tile candidate list in POSITION ORDER.
//   phase 3: block x==0 refines bin T with a second-level 4096-bin smem
//            histogram (monotone bits 8..19) over the candidates (iterated
//            FLAT via binary search over the tile prefix offsets): key2 > T2
//            emitted (exact set), the remaining slots filled with key2 == T2
//            candidates by LOWEST POSITION (deterministic). 24 of 32 key bits
//            are exact; residual ties are near-equal scores.
// When seq_len <= K the threshold logic selects every valid position (T = -1),
// so short-context selection is exactly complete, like persistent_topk.
// (TOPK_NBIN / TOPK_THREADS / fp32_monotone are defined at the top of this
// namespace — the indexer kernel's fused histogram epilogue uses them too.)
// ---------------------------------------------------------------------------

// Suffix-scan an nbin-bin histogram (global or smem pointer): find threshold
// bin T such that count(bin > T) < K <= count(bin >= T); T = -1 if the total
// is <= K (select everything). Must be called by all 256 threads; s_scan is
// 256+2 words of smem scratch. Returns (T, count strictly above T).
template <int NBIN>
__device__ __forceinline__ void topk_find_threshold(
    const uint32_t* hist, uint32_t* s_scan, int K, int* out_T,
    uint32_t* out_above) {
  const int tid = threadIdx.x;
  constexpr int chunk = NBIN / TOPK_THREADS;
  // per-thread chunk sum (bins [tid*chunk, (tid+1)*chunk))
  uint32_t cs = 0;
#pragma unroll
  for (int i = 0; i < chunk; i++) cs += hist[tid * chunk + i];
  s_scan[tid] = cs;
  __syncthreads();
  // thread 0: serial suffix over 256 chunks (bins ascend; "above" = higher)
  if (tid == 0) {
    uint32_t run = 0;
    for (int c = TOPK_THREADS - 1; c >= 0; c--) {
      uint32_t v = s_scan[c];
      s_scan[c] = run;  // suffix strictly above chunk c
      run += v;
    }
    s_scan[TOPK_THREADS] = run;                    // total
    s_scan[TOPK_THREADS + 1] = 0xffffffffu;        // T sentinel (= -1)
  }
  __syncthreads();
  const uint32_t total = s_scan[TOPK_THREADS];
  uint32_t my_above = 0;
  int my_T = -2;
  if (total > (uint32_t)K) {
    // exactly one bin satisfies above < K <= above + hist[bin]
    uint32_t above = s_scan[tid];  // strictly above my chunk's top bin
    for (int i = chunk - 1; i >= 0; i--) {
      const int bin = tid * chunk + i;
      const uint32_t h = hist[bin];
      if (above < (uint32_t)K && above + h >= (uint32_t)K) {
        my_T = bin;
        my_above = above;
        break;
      }
      above += h;
    }
  }
  __syncthreads();
  if (my_T >= 0) {
    s_scan[TOPK_THREADS + 1] = (uint32_t)my_T;
    s_scan[TOPK_THREADS] = my_above;  // reuse: count above T
  }
  __syncthreads();
  const uint32_t traw = s_scan[TOPK_THREADS + 1];
  *out_T = (traw == 0xffffffffu) ? -1 : (int)traw;
  *out_above = (*out_T < 0) ? 0 : s_scan[TOPK_THREADS];
  __syncthreads();
}

__global__ void __launch_bounds__(TOPK_THREADS) sm70_decode_topk_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ seq_lens,
    int32_t* __restrict__ idx_out, uint32_t* __restrict__ counters,  // [B][2]
    const uint32_t* __restrict__ hist,  // [B][NBIN] prebuilt by the indexer
    uint32_t* __restrict__ hist2,       // [B][NBIN] level-2, built here
    uint32_t* __restrict__ tile_cnt,    // [B][NT]
    int32_t* __restrict__ cand,  // [B][NT][tile_size]
    const int max_kv, const int K, const int n_tiles, const int tile_size) {
  const int tile = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ uint32_t s_hist[TOPK_NBIN];
  __shared__ uint32_t s_scan[TOPK_THREADS + 2];
  __shared__ uint32_t s_wcnt[8], s_wbase[8];
  __shared__ uint32_t s_tcnt[8], s_tbase[8];  // tie compaction
  __shared__ uint32_t s_run;      // running ordered-compaction count
  __shared__ uint32_t s_abase;    // chunk base slot for "above" emissions
  __shared__ uint32_t s_toff[65]; // phase-3 tile prefix offsets (NT <= 64)

  const int seq_len = seq_lens[b];
  const float* row = score + (size_t)b * max_kv;
  uint32_t* my_ctr = counters + b * 2;  // [arrive, out_cursor]
  volatile uint32_t* v_ctr = (volatile uint32_t*)my_ctr;
  const uint32_t* my_hist = hist + (size_t)b * TOPK_NBIN;
  uint32_t* my_hist2 = hist2 + (size_t)b * TOPK_NBIN;

  const int lo = tile * tile_size;
  const int hi = min(lo + tile_size, max_kv);
  const int valid_hi = min(hi, seq_len);

  // ---- phase 1: threshold from the PREBUILT histogram (the indexer's fused
  // epilogue wrote it; stream order guarantees it is complete), then compact
  // this block's tile. Ties also feed the level-2 histogram (bits 8..19). ----
  for (int i = tid; i < TOPK_NBIN; i += TOPK_THREADS) s_hist[i] = my_hist[i];
  __syncthreads();
  int T;
  uint32_t above_T;
  topk_find_threshold<TOPK_NBIN>(s_hist, s_scan, K, &T, &above_T);

  if (tid == 0) s_run = 0;
  __syncthreads();
  int32_t* my_cand = cand + ((size_t)b * n_tiles + tile) * tile_size;
  // Only iterate the causally-valid span of this tile (positions >= seq_len
  // are never candidates). Round up to TOPK_THREADS so the __syncthreads
  // inside stay uniform across the block; tiles fully beyond seq_len do zero
  // iterations and fall through to the barrier. At a 200k frozen decode-graph
  // geometry this skips ~all iterations for short/mid contexts.
  const int hi_iter =
      min(hi, ((valid_hi - lo + TOPK_THREADS - 1) / TOPK_THREADS) *
                      TOPK_THREADS +
                  lo);
  for (int base = lo; base < hi_iter; base += TOPK_THREADS) {
    const int p = base + tid;
    const bool valid = (p < valid_hi);
    uint32_t mono = 0;
    if (valid) mono = fp32_monotone(row[p]);
    const int key = valid ? (int)(mono >> 20) : -1;
    const bool is_above = valid && (T < 0 || key > T);
    const bool is_tie = valid && T >= 0 && key == T;
    // one ballot round for both classes; "above" reserves a chunk range with
    // ONE cursor atomic (per-element atomics on the shared cursor serialize
    // ~K emissions across all blocks — measured 3-8x slower at 16k+).
    const unsigned ma = __ballot_sync(0xffffffff, is_above);
    const unsigned mt = __ballot_sync(0xffffffff, is_tie);
    if (lane == 0) {
      s_wcnt[warp] = __popc(ma);
      s_tcnt[warp] = __popc(mt);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t runa = 0, runt = s_run;
      for (int wi = 0; wi < 8; wi++) {
        s_wbase[wi] = runa;
        runa += s_wcnt[wi];
        s_tbase[wi] = runt;
        runt += s_tcnt[wi];
      }
      s_abase = runa ? atomicAdd(&my_ctr[1], runa) : 0u;
      s_run = runt;
    }
    __syncthreads();
    if (is_above) {
      const uint32_t off = s_wbase[warp] + __popc(ma & ((1u << lane) - 1));
      idx_out[(size_t)b * K + s_abase + off] = p;
    }
    if (is_tie) {
      const uint32_t off = s_tbase[warp] + __popc(mt & ((1u << lane) - 1));
      my_cand[off] = p;  // position-ordered within the tile
      atomicAdd(&my_hist2[(mono >> 8) & 0xFFFu], 1u);
    }
    __syncthreads();
  }
  if (tid == 0) tile_cnt[(size_t)b * n_tiles + tile] = s_run;
  __threadfence();
  if (tid == 0) atomicAdd(&my_ctr[0], 1u);
  if (tile != 0) return;

  // ---- phase 3 (block x==0 only): fill the remaining slots from bin T ----
  if (tid == 0)
    while (v_ctr[0] < (uint32_t)n_tiles) {
    }
  __syncthreads();
  __threadfence();

  const uint32_t above_total = v_ctr[1];  // volatile: other SMs wrote via atomics
  const int R = K - (int)above_total;
  if (R <= 0 || T < 0) return;  // full, or everything already selected

  // tile prefix offsets + total candidate count
  if (tid == 0) {
    uint32_t run = 0;
    for (int t = 0; t < n_tiles; t++) {
      s_toff[t] = run;
      run += tile_cnt[(size_t)b * n_tiles + t];
    }
    s_toff[n_tiles] = run;
  }
  __syncthreads();
  const uint32_t C = s_toff[n_tiles];
  if (C == 0) return;

  // Candidates are iterated FLAT (global candidate index g -> (tile, local)
  // via binary search over s_toff): per-tile chunk loops paid several
  // __syncthreads per tile even for tiny tiles, which dominated phase 3.
  // g ascends in POSITION ORDER (tiles ascend; per-tile lists are ordered).
  auto cand_pos = [&](uint32_t g) -> int {
    int t_lo = 0, t_hi = n_tiles - 1;
    while (t_lo < t_hi) {  // last tile with s_toff[t] <= g
      const int mid = (t_lo + t_hi + 1) >> 1;
      if (s_toff[mid] <= g)
        t_lo = mid;
      else
        t_hi = mid - 1;
    }
    return cand[((size_t)b * n_tiles + t_lo) * tile_size + (g - s_toff[t_lo])];
  };

  // level-2 threshold from the histogram built during compaction. If C <= R
  // it returns T2 = -1 (select every candidate) — no special case needed.
  for (int i = tid; i < TOPK_NBIN; i += TOPK_THREADS) s_hist[i] = my_hist2[i];
  __syncthreads();
  int T2;
  uint32_t above2;
  topk_find_threshold<TOPK_NBIN>(s_hist, s_scan, R, &T2, &above2);

  // emit key2 > T2 (exact set) and the final R2 = R - above2 tie slots
  // (lowest position first). Both classes use chunk-aggregated ordered
  // compaction (single block; no global atomics needed).
  const int R2 = R - (int)above2;
  if (tid == 0) {
    s_abase = above_total;  // next slot for the exact (key2 > T2) set
    s_run = 0;              // ordered tie rank counter
  }
  __syncthreads();
  const uint32_t tie_base = above_total + above2;  // ties after the exact set
  for (uint32_t base = 0; base < C; base += TOPK_THREADS) {
    const uint32_t g = base + tid;
    const bool v = (g < C);
    int pos = -1, key2 = -1;
    if (v) {
      pos = cand_pos(g);
      key2 = (int)((fp32_monotone(row[pos]) >> 8) & 0xFFFu);
    }
    const bool is_above2 = v && key2 > T2;
    const bool is_tie2 = v && key2 == T2;
    const unsigned ma = __ballot_sync(0xffffffff, is_above2);
    const unsigned mt = __ballot_sync(0xffffffff, is_tie2);
    if (lane == 0) {
      s_wcnt[warp] = __popc(ma);
      s_tcnt[warp] = __popc(mt);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t runa = s_abase, runt = s_run;
      for (int wi = 0; wi < 8; wi++) {
        s_wbase[wi] = runa;
        runa += s_wcnt[wi];
        s_tbase[wi] = runt;
        runt += s_tcnt[wi];
      }
      s_abase = runa;
      s_run = runt;
    }
    __syncthreads();
    if (is_above2) {
      const uint32_t off = s_wbase[warp] + __popc(ma & ((1u << lane) - 1));
      idx_out[(size_t)b * K + off] = pos;
    }
    if (is_tie2) {
      const uint32_t rank = s_tbase[warp] + __popc(mt & ((1u << lane) - 1));
      if (rank < (uint32_t)R2)
        idx_out[(size_t)b * K + tie_base + rank] = pos;
    }
    __syncthreads();
  }
}

}  // namespace

// out: o [B, n_head, D_V] fp16, lse [B, n_head] fp32.
void sm70_mla_decode(torch::Tensor& o, torch::Tensor& lse,
                     const torch::Tensor& q, const torch::Tensor& kv_cache,
                     const torch::Tensor& block_table,
                     const torch::Tensor& seq_lens, double scale) {
#ifdef SM70_MLA_WMMA_AVAILABLE
  TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda());
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  TORCH_CHECK(kv_cache.scalar_type() == at::kHalf, "kv_cache must be fp16");
  TORCH_CHECK(o.scalar_type() == at::kHalf, "o must be fp16");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be fp32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt);

  constexpr int D_K = 576;
  constexpr int D_V = 512;
  constexpr int HB = 32;  // heads per block (2 head-groups of 32 -> fits 96KB)
  constexpr int TN = 32;
  constexpr int WARPS = 8;

  const int B = q.size(0);
  const int n_head = q.size(1);
  TORCH_CHECK(q.size(2) == D_K, "q last dim must be 576");
  TORCH_CHECK(n_head % HB == 0, "n_head must be a multiple of 32");
  TORCH_CHECK(o.size(2) == D_V, "o last dim must be 512");

  // kv_cache flat rows [num_blocks*page, D_K].
  const int page_size = kv_cache.size(-2);
  const int64_t n_kv_rows = kv_cache.numel() / D_K;
  const int max_blocks = block_table.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int PADK = D_K;
  const size_t smem =
      (size_t)TN * PADK * sizeof(half) +               // s_k
      (size_t)16 * TN * sizeof(half) +                 // s_p
      (size_t)16 * TN * sizeof(float) +                // s_sc
      (size_t)HB * D_V * sizeof(half) +                // s_o
      (size_t)WARPS * (16 * TN > 256 ? 16 * TN : 256) * sizeof(float) +  // s_pv
      (size_t)WARPS * 256 * sizeof(half) +             // s_qs
      (size_t)2 * HB * sizeof(float);                  // s_m, s_l

  auto kern = sm70_mla_decode_kernel<D_K, D_V, HB, TN, WARPS>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    attr_set = true;
  }
  const dim3 grid(B, n_head / HB, 1);
  kern<<<grid, WARPS * 32, smem, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr()),
      reinterpret_cast<const half*>(kv_cache.data_ptr()),
      block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(o.data_ptr()), lse.data_ptr<float>(), n_head,
      (int)n_kv_rows, max_blocks, page_size, (float)scale);
#else
  TORCH_CHECK(false, "sm70_mla_decode requires SM70+ (WMMA) build");
#endif
}

// Sparse (top-k) MLA decode. idx: [B, topk] int32 logical KV positions.
// kv_cache: fp16 paged rows [num_blocks, PAGE, 576] OR the uint8 fp8_ds_mla
// layout [num_blocks, PAGE, 656] (dequantized on the fly in the gather).

// ---------------------------------------------------------------------------
// PREFILL-mode sparse MLA attention (register-accumulator rewrite, 2026-07-07).
//
// The decode kernel above phase-locks 16 warps through a 16-way k-split with
// 32 KB of smem partials, re-stages Q from global per (key-tile, kt-step),
// and accumulates PV in an fp16 smem tile RMW'd twice per key-tile. Fine for
// B<=4 decode (graph-captured, occupancy from topk splits) but only ~4.5
// TFLOPS at prefill B=1024. This kernel (microbenched in scratchpad
// idxk/sparse_bench.cu vs an fp32 ground truth; 2.51x, and ~5x MORE accurate
// because PV accumulates in fp32 registers):
//   - Q arrives PRE-PACKED+SCALED as [B, 36, 16, 16] fragment tiles;
//   - per-tile row pointers staged once (not 3 dependent global loads per
//     16 B of gather payload);
//   - WIDE 16-byte gather with SIMD e4m3 pair dequant + one scale load per
//     chunk (dsmla_load4 is decode-tuned: 4-byte loads, 12.5% sector eff);
//   - K tile double-buffered (gather overlaps compute across warps);
//   - QK: warp-owned (nt, kslice) fragments, 4-way partials FUSED into the
//     softmax pass; PV: fp32 REGISTER accumulators (warp-owned dv-tiles),
//     online-softmax rescale in-register via the derived Volta acc layout;
//   - direct O/LSE output (no split scratch, no merge pass).
// Requires n_head <= 16 (one head-tile; TP4=16, TP8=8). Fixed TN=32/WARPS=16/
// KSPLIT=4 (sweep winner). Selected for B >= 32 in the host fn below;
// VLLM_SM70_SPARSE_PREFILL_V2=0 falls back to the decode kernel + merge.
#ifdef SM70_MLA_WMMA_AVAILABLE
// ==== V5: v4 + WIDE gather: 16-byte fp8 chunks with SIMD pair dequant and
// one scale load per chunk. dsmla_load4 (decode-tuned) issues 4-BYTE global
// loads (12.5% sector efficiency, ~160 transactions/row); this does ~41. ====
template <bool KV_FP8>
__global__ void __launch_bounds__(512) sm70_sparse_mla_prefill_kernel(
    const half* __restrict__ Qp, const void* __restrict__ KVv,
    const int32_t* __restrict__ idxb, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, half* __restrict__ O,
    float* __restrict__ LSE, const int n_head, const int n_kv_rows,
    const int max_blocks, const int page_size, const int topk,
    const float scale) {
  constexpr int TN = 32, WARPS = 16, KSPLIT = 4;
  constexpr int DK = 576, DV = 512;
  const uint8_t* KV = reinterpret_cast<const uint8_t*>(KVv);
  constexpr int ROW_BYTES = KV_FP8 ? DSMLA_ROW_BYTES : DK * 2;
  constexpr int NCHUNK = KV_FP8 ? 40 : (DK * 2) / 16;  // 16 B chunks per row
  constexpr int NTILES = TN / 16;
  constexpr int DK16 = DK / 16, DV16 = DV / 16;
  constexpr int PADK = DK + 8;
  constexpr int NPV = DV16 / WARPS;
  constexpr int SLOTS = NTILES * KSPLIT;
  constexpr int KCHUNK = DK16 / KSPLIT;
  static_assert(DV16 % WARPS == 0 && SLOTS <= WARPS && DK16 % KSPLIT == 0);

  const int tok = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  extern __shared__ char smem_raw[];
  half* s_k = (half*)smem_raw;                          // [2][TN x PADK]
  float* s_part = (float*)(s_k + (size_t)2 * TN * PADK);// [SLOTS x 256]
  float* s_sc = s_part + (KSPLIT > 1 ? (size_t)SLOTS * 256 : 0);  // [16 x TN]
  half* s_p = (half*)(s_sc + (size_t)16 * TN);          // [16 x TN]
  long* s_rp = (long*)(s_p + (size_t)16 * TN + 16);     // [2][TN] row ptrs
  __shared__ float s_m[16], s_l[16], s_alpha[16];

  const int seq_len = seq_lens[tok];
  const int32_t* idx_row = idxb + (size_t)tok * topk;
  const int32_t* bt_row = block_table + (size_t)tok * max_blocks;
  const half* qp = Qp + (size_t)tok * DK16 * 256;

  for (int i = tid; i < 16; i += WARPS * 32) {
    s_m[i] = -1e30f;
    s_l[i] = 0.f;
    s_alpha[i] = 1.f;
  }
  int erow[8];
#pragma unroll
  for (int i = 0; i < 8; i++) erow[i] = c_idx_accrow[lane * 8 + i];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[NPV];
#pragma unroll
  for (int t = 0; t < NPV; t++) wmma::fill_fragment(oacc[t], 0.f);

  // row pointers + gather for tile 0 into buffer 0
  auto stage_rowptrs = [&](int base, int buf) {
    for (int r = tid; r < TN; r += WARPS * 32) {
      const int j = base + r;
      long row = -1;
      if (j < topk) {
        const int pos = idx_row[j];
        if (pos >= 0 && pos < seq_len) {
          const long rr =
              (long)bt_row[pos / page_size] * page_size + pos % page_size;
          if (rr >= 0 && rr < n_kv_rows) row = rr;
        }
      }
      s_rp[buf * TN + r] = row;
    }
  };
  auto gather = [&](int buf) {
    half* dst = s_k + (size_t)buf * TN * PADK;
    const long* rp = s_rp + buf * TN;
    // fp8: 40 chunks of 16 B per row (32 fp8-dim chunks + 8 rope chunks);
    // fp16: 72 direct 16 B copies.
    for (int i = tid; i < TN * NCHUNK; i += WARPS * 32) {
      const int r = i / NCHUNK, c = i % NCHUNK;
      const long row = rp[r];
      const uint8_t* rowp = KV + row * (long)ROW_BYTES;
      if (!KV_FP8) {
        uint4 w = {0, 0, 0, 0};
        if (row >= 0) w = *reinterpret_cast<const uint4*>(rowp + c * 16);
        *reinterpret_cast<uint4*>(dst + r * PADK + c * 8) = w;
      } else if (c < 32) {
        half2 h8[8];
        if (row >= 0) {
          const uint4 w = *reinterpret_cast<const uint4*>(rowp + c * 16);
          const float s =
              *reinterpret_cast<const float*>(rowp + 512 + ((c >> 3) << 2)) *
              256.0f;
          const half2 hs = __float2half2_rn(s);
          const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for (int k = 0; k < 4; k++) {
            h8[k * 2] = __hmul2(
                sm70_fp8x2_to_half2_div256(__byte_perm(ws[k], 0, 0x4140)), hs);
            h8[k * 2 + 1] = __hmul2(
                sm70_fp8x2_to_half2_div256(__byte_perm(ws[k], 0, 0x4342)), hs);
          }
        } else {
#pragma unroll
          for (int k = 0; k < 8; k++) h8[k] = __float2half2_rn(0.f);
        }
        *reinterpret_cast<uint4*>(dst + r * PADK + c * 16) =
            *reinterpret_cast<const uint4*>(h8);
        *reinterpret_cast<uint4*>(dst + r * PADK + c * 16 + 8) =
            *reinterpret_cast<const uint4*>(h8 + 4);
      } else {
        const int cc = c - 32;
        uint4 w = {0, 0, 0, 0};
        if (row >= 0)
          w = *reinterpret_cast<const uint4*>(rowp + 528 + cc * 16);
        *reinterpret_cast<uint4*>(dst + r * PADK + 512 + cc * 8) = w;
      }
    }
  };

  const int ntiles_total = (topk + TN - 1) / TN;
  stage_rowptrs(0, 0);
  __syncthreads();
  gather(0);
  if (ntiles_total > 1) stage_rowptrs(TN, 1);
  __syncthreads();

  for (int ti = 0; ti < ntiles_total; ti++) {
    const int base = ti * TN;
    const int buf = ti & 1;
    // issue next tile's gather into the other buffer (row ptrs were staged
    // before the last barrier), overlapping this tile's compute
    if (ti + 1 < ntiles_total) gather(buf ^ 1);
    const half* kbuf = s_k + (size_t)buf * TN * PADK;

    // QK on current buffer
    if (warp < SLOTS) {
      const int nt = warp / KSPLIT, ks = warp % KSPLIT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int k = 0; k < KCHUNK; k++) {
        const int kt = ks * KCHUNK + k;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
        wmma::load_matrix_sync(Qf, qp + kt * 256, 16);
        wmma::load_matrix_sync(Kf, kbuf + (nt * 16) * PADK + kt * 16, PADK);
        wmma::mma_sync(acc, Qf, Kf, acc);
      }
      if (KSPLIT > 1)
        wmma::store_matrix_sync(s_part + warp * 256, acc, 16,
                                wmma::mem_row_major);
      else
        wmma::store_matrix_sync(s_sc + nt * 16, acc, TN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int h = warp; h < 16; h += WARPS) {
      float m_cur = s_m[h];
      for (int c = lane; c < TN; c += 32) {
        const int j = base + c;
        const bool valid = (j < topk) && (s_rp[buf * TN + c] >= 0);
        float v = -INFINITY;
        if (valid) {
          if (KSPLIT > 1) {
            const int nt = c / 16, cc = c % 16;
            v = 0.f;
#pragma unroll
            for (int ks = 0; ks < KSPLIT; ks++)
              v += s_part[(nt * KSPLIT + ks) * 256 + h * 16 + cc];
          } else {
            v = s_sc[h * TN + c];
          }
        }
        s_sc[h * TN + c] = v;
        m_cur = fmaxf(m_cur, v);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffff, m_cur, off));
      const float m_prev = s_m[h];
      const float alpha = __expf(m_prev - m_cur);
      float lsum = 0.f;
      for (int c = lane; c < TN; c += 32) {
        float v = s_sc[h * TN + c];
        float p = (v == -INFINITY) ? 0.f : expf(v - m_cur);
        s_p[h * TN + c] = __float2half(p);
        lsum += p;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
      if (lane == 0) {
        s_m[h] = m_cur;
        s_l[h] = s_l[h] * alpha + lsum;
        s_alpha[h] = alpha;
      }
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < NPV; t++) {
      const int vt = warp * NPV + t;
#pragma unroll
      for (int i = 0; i < 8; i++) oacc[t].x[i] *= s_alpha[erow[i]];
#pragma unroll
      for (int nt = 0; nt < NTILES; nt++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Vf;
        wmma::load_matrix_sync(Pf, s_p + nt * 16, TN);
        wmma::load_matrix_sync(Vf, kbuf + (nt * 16) * PADK + vt * 16, PADK);
        wmma::mma_sync(oacc[t], Pf, Vf, oacc[t]);
      }
    }
    // stage tile ti+2's row ptrs into THIS buffer's slot (its validity data
    // was consumed by the softmax above); the tail barrier publishes them.
    if (ti + 2 < ntiles_total) stage_rowptrs((ti + 2) * TN, buf);
    __syncthreads();
  }

  // finalize via bounce through s_k buffer 0
  float* s_out = (float*)s_k;
#pragma unroll
  for (int t = 0; t < NPV; t++)
    wmma::store_matrix_sync(s_out + (warp * NPV + t) * 16, oacc[t], DV,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 16 * DV; i += WARPS * 32) {
    const int h = i / DV, d = i % DV;
    if (h >= n_head) continue;
    const float l = s_l[h] > 0.f ? s_l[h] : 1.f;
    O[((size_t)tok * n_head + h) * DV + d] = __float2half(s_out[i] / l);
  }
  for (int h = tid; h < n_head && h < 16; h += WARPS * 32)
    LSE[(size_t)tok * n_head + h] = s_m[h] + logf(s_l[h] > 0.f ? s_l[h] : 1.f);
}


// ==== Warp-specialized producer/consumer prefill (VLLM_SM70_SPARSE_PREFILL_WS,
// default OFF): 24 warps = 16 producer (stage_rowptrs + gather only) + 8
// consumer (QK/softmax/PV, mapping identical to the v5 kernel but over 8
// warps: SLOTS=8 QK slots -> 1/warp, 2 heads/warp softmax, NPV=4 dv-tiles).
// Named-barrier full/empty ping-pong per K buffer decouples gather latency
// from compute: v5's remaining stall was each thread's own gather load->smem
// store chain sitting in front of its WMMA work (double buffering cannot fix
// in-thread ordering; NOTES.md "SPARSE-ATTN PREFILL KERNEL REWRITE").
// Microbench (B=1024, ctx 48k, topk 2048, fp8): 10.96 -> 15.73 TFLOPS
// (+43%), bit-identical output ordering to v5 (same per-accumulator MMA
// sequence). Barrier ids: 1+buf=full, 3+buf=empty, 5=consumer-internal,
// 6=producer-internal, 7=exit park. KNOWN HAZARD (fixed, keep on ports):
// when ntiles is odd the LAST tile lives in buffer 0 which the finalize
// s_out bounce aliases -> consumer barrier before the bounce is REQUIRED.
// compute-sanitizer racecheck false-positives on this kernel (it does not
// model named barriers and its replay breaks them); use synccheck+memcheck.
static __device__ __forceinline__ void sm70_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
static __device__ __forceinline__ void sm70_bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

template <bool KV_FP8>
__global__ void __launch_bounds__(768) sm70_sparse_mla_prefill_ws_kernel(
    const half* __restrict__ Qp, const void* __restrict__ KVv,
    const int32_t* __restrict__ idxb, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, half* __restrict__ O,
    float* __restrict__ LSE, const int n_head, const int n_kv_rows,
    const int max_blocks, const int page_size, const int topk,
    const float scale) {
  constexpr int TN = 32, WARPS = 24, KSPLIT = 4, PWARPS = 16;
  constexpr int CWARPS = WARPS - PWARPS;
  constexpr int DK = 576, DV = 512;
  const uint8_t* KV = reinterpret_cast<const uint8_t*>(KVv);
  constexpr int ROW_BYTES = KV_FP8 ? DSMLA_ROW_BYTES : DK * 2;
  constexpr int NCHUNK = KV_FP8 ? 40 : (DK * 2) / 16;  // 16 B chunks per row
  constexpr int NTILES = TN / 16;
  constexpr int DK16 = DK / 16, DV16 = DV / 16;
  constexpr int PADK = DK + 8;
  constexpr int NPV = DV16 / CWARPS;
  constexpr int SLOTS = NTILES * KSPLIT;
  constexpr int KCHUNK = DK16 / KSPLIT;
  constexpr int NT_ALL = WARPS * 32, NT_C = CWARPS * 32, NT_P = PWARPS * 32;
  static_assert(DV16 % CWARPS == 0 && SLOTS <= CWARPS && DK16 % KSPLIT == 0);

  const int tok = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool is_prod = warp >= CWARPS;

  extern __shared__ char smem_raw[];
  half* s_k = (half*)smem_raw;                          // [2][TN x PADK]
  float* s_part = (float*)(s_k + (size_t)2 * TN * PADK);// [SLOTS x 256]
  float* s_sc = s_part + (size_t)SLOTS * 256;           // [16 x TN]
  half* s_p = (half*)(s_sc + (size_t)16 * TN);          // [16 x TN]
  long* s_rp = (long*)(s_p + (size_t)16 * TN + 16);     // [2][TN] row ptrs
  __shared__ float s_m[16], s_l[16], s_alpha[16];

  const int seq_len = seq_lens[tok];
  const int32_t* idx_row = idxb + (size_t)tok * topk;
  const int32_t* bt_row = block_table + (size_t)tok * max_blocks;
  const half* qp = Qp + (size_t)tok * DK16 * 256;
  const int ntiles_total = (topk + TN - 1) / TN;

  if (is_prod) {
    // ---------------- producer warps ----------------
    const int ptid = tid - NT_C;
    for (int ti = 0; ti < ntiles_total; ti++) {
      const int buf = ti & 1;
      if (ti >= 2) sm70_bar_sync(3 + buf, NT_ALL);  // consumers done w/ buf
      for (int r = ptid; r < TN; r += NT_P) {
        const int j = ti * TN + r;
        long row = -1;
        if (j < topk) {
          const int pos = idx_row[j];
          if (pos >= 0 && pos < seq_len) {
            const long rr =
                (long)bt_row[pos / page_size] * page_size + pos % page_size;
            if (rr >= 0 && rr < n_kv_rows) row = rr;
          }
        }
        s_rp[buf * TN + r] = row;
      }
      sm70_bar_sync(6, NT_P);  // publish row ptrs to sibling producers
      half* dst = s_k + (size_t)buf * TN * PADK;
      const long* rp = s_rp + buf * TN;
      for (int i = ptid; i < TN * NCHUNK; i += NT_P) {
        const int r = i / NCHUNK, c = i % NCHUNK;
        const long row = rp[r];
        const uint8_t* rowp = KV + row * (long)ROW_BYTES;
        if (!KV_FP8) {
          uint4 w = {0, 0, 0, 0};
          if (row >= 0) w = *reinterpret_cast<const uint4*>(rowp + c * 16);
          *reinterpret_cast<uint4*>(dst + r * PADK + c * 8) = w;
        } else if (c < 32) {
          half2 h8[8];
          if (row >= 0) {
            const uint4 w = *reinterpret_cast<const uint4*>(rowp + c * 16);
            const float s =
                *reinterpret_cast<const float*>(rowp + 512 + ((c >> 3) << 2)) *
                256.0f;
            const half2 hs = __float2half2_rn(s);
            const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int k = 0; k < 4; k++) {
              h8[k * 2] = __hmul2(
                  sm70_fp8x2_to_half2_div256(__byte_perm(ws[k], 0, 0x4140)),
                  hs);
              h8[k * 2 + 1] = __hmul2(
                  sm70_fp8x2_to_half2_div256(__byte_perm(ws[k], 0, 0x4342)),
                  hs);
            }
          } else {
#pragma unroll
            for (int k = 0; k < 8; k++) h8[k] = __float2half2_rn(0.f);
          }
          *reinterpret_cast<uint4*>(dst + r * PADK + c * 16) =
              *reinterpret_cast<const uint4*>(h8);
          *reinterpret_cast<uint4*>(dst + r * PADK + c * 16 + 8) =
              *reinterpret_cast<const uint4*>(h8 + 4);
        } else {
          const int cc = c - 32;
          uint4 w = {0, 0, 0, 0};
          if (row >= 0)
            w = *reinterpret_cast<const uint4*>(rowp + 528 + cc * 16);
          *reinterpret_cast<uint4*>(dst + r * PADK + 512 + cc * 8) = w;
        }
      }
      __threadfence_block();
      sm70_bar_arrive(1 + buf, NT_ALL);  // buf is full
    }
    // Park until consumers finish (exiting warps while consumer barrier
    // generations are in flight is asking for trouble; this is free).
    sm70_bar_sync(7, NT_ALL);
    return;
  }

  // ---------------- consumer warps ----------------
  for (int i = tid; i < 16; i += NT_C) {
    s_m[i] = -1e30f;
    s_l[i] = 0.f;
    s_alpha[i] = 1.f;
  }
  int erow[8];
#pragma unroll
  for (int i = 0; i < 8; i++) erow[i] = c_idx_accrow[lane * 8 + i];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[NPV];
#pragma unroll
  for (int t = 0; t < NPV; t++) wmma::fill_fragment(oacc[t], 0.f);
  sm70_bar_sync(5, NT_C);  // publish s_m/s_l/s_alpha init

  for (int ti = 0; ti < ntiles_total; ti++) {
    const int base = ti * TN;
    const int buf = ti & 1;
    sm70_bar_sync(1 + buf, NT_ALL);  // wait buf full
    const half* kbuf = s_k + (size_t)buf * TN * PADK;

    if (warp < SLOTS) {
      const int nt = warp / KSPLIT, ks = warp % KSPLIT;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int k = 0; k < KCHUNK; k++) {
        const int kt = ks * KCHUNK + k;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
        wmma::load_matrix_sync(Qf, qp + kt * 256, 16);
        wmma::load_matrix_sync(Kf, kbuf + (nt * 16) * PADK + kt * 16, PADK);
        wmma::mma_sync(acc, Qf, Kf, acc);
      }
      wmma::store_matrix_sync(s_part + warp * 256, acc, 16,
                              wmma::mem_row_major);
    }
    sm70_bar_sync(5, NT_C);

    for (int h = warp; h < 16; h += CWARPS) {
      float m_cur = s_m[h];
      for (int c = lane; c < TN; c += 32) {
        const int j = base + c;
        const bool valid = (j < topk) && (s_rp[buf * TN + c] >= 0);
        float v = -INFINITY;
        if (valid) {
          const int nt = c / 16, cc = c % 16;
          v = 0.f;
#pragma unroll
          for (int ks = 0; ks < KSPLIT; ks++)
            v += s_part[(nt * KSPLIT + ks) * 256 + h * 16 + cc];
        }
        s_sc[h * TN + c] = v;
        m_cur = fmaxf(m_cur, v);
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffff, m_cur, off));
      const float m_prev = s_m[h];
      const float alpha = __expf(m_prev - m_cur);
      float lsum = 0.f;
      for (int c = lane; c < TN; c += 32) {
        float v = s_sc[h * TN + c];
        float p = (v == -INFINITY) ? 0.f : expf(v - m_cur);
        s_p[h * TN + c] = __float2half(p);
        lsum += p;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
      if (lane == 0) {
        s_m[h] = m_cur;
        s_l[h] = s_l[h] * alpha + lsum;
        s_alpha[h] = alpha;
      }
    }
    sm70_bar_sync(5, NT_C);

#pragma unroll
    for (int t = 0; t < NPV; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) oacc[t].x[i] *= s_alpha[erow[i]];
#pragma unroll
    for (int nt = 0; nt < NTILES; nt++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Pf;
      wmma::load_matrix_sync(Pf, s_p + nt * 16, TN);
#pragma unroll
      for (int t = 0; t < NPV; t++) {
        const int vt = warp * NPV + t;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Vf;
        wmma::load_matrix_sync(Vf, kbuf + (nt * 16) * PADK + vt * 16, PADK);
        wmma::mma_sync(oacc[t], Pf, Vf, oacc[t]);
      }
    }
    sm70_bar_arrive(3 + buf, NT_ALL);  // buf consumed (V reads done)
  }

  // ALL consumer warps must finish PV before anyone overwrites s_k with
  // output: when ntiles_total is odd the LAST tile lives in buffer 0 and
  // the s_out bounce below aliases it (v5's loop-tail __syncthreads gave
  // this for free; the arrive above is non-blocking). Removing this barrier
  // corrupts ~30% of odd-ntile launches.
  sm70_bar_sync(5, NT_C);
  float* s_out = (float*)s_k;
#pragma unroll
  for (int t = 0; t < NPV; t++)
    wmma::store_matrix_sync(s_out + (warp * NPV + t) * 16, oacc[t], DV,
                            wmma::mem_row_major);
  sm70_bar_sync(5, NT_C);
  for (int i = tid; i < 16 * DV; i += NT_C) {
    const int h = i / DV, d = i % DV;
    if (h >= n_head) continue;
    const float l = s_l[h] > 0.f ? s_l[h] : 1.f;
    O[((size_t)tok * n_head + h) * DV + d] = __float2half(s_out[i] / l);
  }
  for (int h = tid; h < n_head && h < 16; h += NT_C)
    LSE[(size_t)tok * n_head + h] = s_m[h] + logf(s_l[h] > 0.f ? s_l[h] : 1.f);
  sm70_bar_arrive(7, NT_ALL);  // release parked producers
}


// ==== Fused indexer shard-merge (VLLM_SM70_IDXER_FUSED_MERGE): replaces the
// prefill TP shard-merge epilogue (permute/contiguous + torch.topk[R,tp*K]
// + gather/isinf/where/cast/copy — ~1.3 ms/layer of glue at tp=4, K=2048)
// with one kernel reading the all-gather payload directly.
// Input allf [tp, R, 2K] fp32: per shard, per row, K vals then K idx
// (int32 bitcast; -1 = invalid, val = -inf). Selection: top-K by 64-bit key
// (monotonic(val) << 32) | (ncand-1-col) where col = s*K + c — keys are
// DISTINCT so an MSB-first byte-radix walk yields the exact K-th key with a
// deterministic tie-break identical to torch.topk's column order on the old
// [R, tp*K] value layout. -inf selections emit -1 (excluded positions).
// Output: selected positions sorted ASCENDING per row (old path emitted
// value-descending; same set => same attention up to fp reassociation, and
// ascending positions gather pages coalesced).
template <int K>
__global__ void __launch_bounds__(512) sm70_indexer_shard_merge_kernel(
    const float* __restrict__ allf, const int32_t* __restrict__ dst_rows,
    int32_t* __restrict__ out, const int64_t out_stride, const int R,
    const int tp, const int dst_base) {
  // Register-resident design: each of 512 threads owns ncand/512 (<=16)
  // candidate keys in registers; radix passes 2..8 only visit candidates
  // still matching the prefix (per-thread bitmask), so the K-th-key search
  // costs ~1.2 full scans instead of 8. Shared memory holds only the
  // selection buffer + histogram (~9.3 KB).
  constexpr int NT = 512;
  constexpr int NPER_MAX = 16;  // tp=4: 8192/512
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int ncand = tp * K;
  const int nper = ncand / NT;  // 8 (tp=2) or 16 (tp=4)

  __shared__ unsigned int s_hist[256];
  __shared__ int32_t s_sel[K];
  __shared__ unsigned int s_cnt;
  __shared__ unsigned long long s_prefix;
  __shared__ unsigned int s_target;

  // load owned candidates -> monotonic keys tagged with inverted column
  unsigned long long key[NPER_MAX];
#pragma unroll
  for (int i = 0; i < NPER_MAX; i++) {
    if (i >= nper) break;
    const int col = tid + i * NT;
    const int s = col / K, c = col % K;
    const float v = allf[((size_t)s * R + r) * (2 * K) + c];
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // monotonic ascending
    key[i] = ((unsigned long long)u << 32) | (unsigned int)(ncand - 1 - col);
  }
  if (tid == 0) {
    s_prefix = 0ull;
    s_target = K;
    s_cnt = 0;
  }

  // MSB-first byte radix over the 64-bit keys (distinct -> exact K-th key)
  unsigned int act = (nper >= 32) ? ~0u : ((1u << nper) - 1u);
  for (int b = 7; b >= 0; b--) {
    for (int i = tid; i < 256; i += NT) s_hist[i] = 0;
    __syncthreads();
    const int sh = b * 8;
    const unsigned long long hi_mask = (b == 7) ? 0ull : (~0ull << (sh + 8));
    const unsigned long long pfx = s_prefix;
#pragma unroll
    for (int i = 0; i < NPER_MAX; i++) {
      if (i >= nper) break;
      if (!(act & (1u << i))) continue;
      if ((key[i] & hi_mask) != (pfx & hi_mask)) {
        act &= ~(1u << i);  // fell out of the prefix; never returns
        continue;
      }
      atomicAdd(&s_hist[(key[i] >> sh) & 0xFF], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned int cum = 0, tgt = s_target;
      int v = 255;
      for (; v >= 0; v--) {
        if (cum + s_hist[v] >= tgt) break;
        cum += s_hist[v];
      }
      s_prefix = pfx | ((unsigned long long)(unsigned int)v << sh);
      s_target = tgt - cum;
    }
    __syncthreads();
  }
  const unsigned long long T = s_prefix;

  // collect selected (key >= T; exactly K since keys are distinct).
  // mono(-inf) = 0x007FFFFF -> excluded position, emit -1.
#pragma unroll
  for (int i = 0; i < NPER_MAX; i++) {
    if (i >= nper) break;
    if (key[i] >= T) {
      const unsigned int pos = atomicAdd(&s_cnt, 1u);
      int32_t idx = -1;
      if ((unsigned int)(key[i] >> 32) > 0x007FFFFFu) {
        const int col = tid + i * NT;
        const int s = col / K, c = col % K;
        idx = __float_as_int(allf[((size_t)s * R + r) * (2 * K) + K + c]);
      }
      s_sel[pos] = idx;
    }
  }
  __syncthreads();

  // bitonic sort ascending (K power of two; -1s land first).
  // NOTE: an unstable LSD radix sort was tried (0.68 ms vs 0.88) and is
  // WRONG — LSD requires a stable scatter even with distinct keys.
  for (int k2 = 2; k2 <= K; k2 <<= 1) {
    for (int j = k2 >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < K; i += NT) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const int32_t a = s_sel[i], c2 = s_sel[ixj];
          const bool up = ((i & k2) == 0);
          if ((a > c2) == up) {
            s_sel[i] = c2;
            s_sel[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }

  const int dst = (dst_base >= 0) ? (dst_base + r) : dst_rows[r];
  for (int i = tid; i < K; i += NT)
    out[(size_t)dst * out_stride + i] = s_sel[i];
}

// ==== Stage B: P2P shard exchange (VLLM_SM70_IDXER_P2P_MERGE). Replaces the
// NCCL LL all-gather (1.03 ms/layer for the 16 MB payload; LL protocol caps
// ~12 GB/s) with direct NVLink reads of peers' IPC-mapped shard slots.
// Protocol per sharded chunk (seq strictly increasing, identical on all TP
// ranks — same scheduler output): slot = seq & 1 (double buffer);
//   [slot_wait: peers merged_seq >= seq-2]  (gates slot reuse)
//   staging copies into MY slot (stream-ordered python copies)
//   [flag_set staged_seq = seq]             (fence + release)
//   [merge_p2p: all blocks spin peers' staged_seq >= seq, read shards
//    directly, select + sort exactly as sm70_indexer_shard_merge]
//   [flag_set merged_seq = seq]
// Flags live in a tiny IPC alloc per rank: int32[2] = {staged, merged}.
// Spins hang on peer failure (same class as an NCCL hang) — debuggable.
static __global__ void sm70_ipc_flag_set_kernel(int* flag, int seq) {
  __threadfence_system();
  *(volatile int*)flag = seq;
}

static __global__ void sm70_ipc_slot_wait_kernel(ulonglong4 flag_ptrs, int tp,
                                                 int min_seq) {
  if (threadIdx.x < 4 && threadIdx.x < tp) {
    const unsigned long long fp = (&flag_ptrs.x)[threadIdx.x];
    volatile int* merged = (volatile int*)(fp) + 1;
    while (*merged < min_seq) {
    }
  }
  __syncthreads();
  __threadfence();
}

template <int K>
__global__ void __launch_bounds__(512) sm70_indexer_shard_merge_p2p_kernel(
    ulonglong4 data_ptrs, ulonglong4 flag_ptrs,
    const int32_t* __restrict__ dst_rows, int32_t* __restrict__ out,
    const int64_t out_stride, const int R, const int tp, const int dst_base,
    const int seq, const long slot_off) {
  constexpr int NT = 512;
  constexpr int NPER_MAX = 16;
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const int ncand = tp * K;
  const int nper = ncand / NT;

  __shared__ unsigned int s_hist[256];
  __shared__ int32_t s_sel[K];
  __shared__ unsigned int s_cnt;
  __shared__ unsigned long long s_prefix;
  __shared__ unsigned int s_target;

  // acquire: all peers staged this seq
  if (tid < 4 && tid < tp) {
    volatile int* staged = (volatile int*)((&flag_ptrs.x)[tid]);
    while (*staged < seq) {
    }
  }
  __syncthreads();
  __threadfence_system();

  unsigned long long key[NPER_MAX];
#pragma unroll
  for (int i = 0; i < NPER_MAX; i++) {
    if (i >= nper) break;
    const int col = tid + i * NT;
    const int s = col / K, c = col % K;
    const float* shard = (const float*)((&data_ptrs.x)[s] + slot_off);
    const float v = shard[(size_t)r * (2 * K) + c];
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    key[i] = ((unsigned long long)u << 32) | (unsigned int)(ncand - 1 - col);
  }
  if (tid == 0) {
    s_prefix = 0ull;
    s_target = K;
    s_cnt = 0;
  }

  unsigned int act = (nper >= 32) ? ~0u : ((1u << nper) - 1u);
  for (int b = 7; b >= 0; b--) {
    for (int i = tid; i < 256; i += NT) s_hist[i] = 0;
    __syncthreads();
    const int sh = b * 8;
    const unsigned long long hi_mask = (b == 7) ? 0ull : (~0ull << (sh + 8));
    const unsigned long long pfx = s_prefix;
#pragma unroll
    for (int i = 0; i < NPER_MAX; i++) {
      if (i >= nper) break;
      if (!(act & (1u << i))) continue;
      if ((key[i] & hi_mask) != (pfx & hi_mask)) {
        act &= ~(1u << i);
        continue;
      }
      atomicAdd(&s_hist[(key[i] >> sh) & 0xFF], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned int cum = 0, tgt = s_target;
      int v = 255;
      for (; v >= 0; v--) {
        if (cum + s_hist[v] >= tgt) break;
        cum += s_hist[v];
      }
      s_prefix = pfx | ((unsigned long long)(unsigned int)v << sh);
      s_target = tgt - cum;
    }
    __syncthreads();
  }
  const unsigned long long T = s_prefix;

#pragma unroll
  for (int i = 0; i < NPER_MAX; i++) {
    if (i >= nper) break;
    if (key[i] >= T) {
      const unsigned int pos = atomicAdd(&s_cnt, 1u);
      int32_t idx = -1;
      if ((unsigned int)(key[i] >> 32) > 0x007FFFFFu) {
        const int col = tid + i * NT;
        const int s = col / K, c = col % K;
        const float* shard = (const float*)((&data_ptrs.x)[s] + slot_off);
        idx = __float_as_int(shard[(size_t)r * (2 * K) + K + c]);
      }
      s_sel[pos] = idx;
    }
  }
  __syncthreads();

  for (int k2 = 2; k2 <= K; k2 <<= 1) {
    for (int j = k2 >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < K; i += NT) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const int32_t a = s_sel[i], c2 = s_sel[ixj];
          const bool up = ((i & k2) == 0);
          if ((a > c2) == up) {
            s_sel[i] = c2;
            s_sel[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }

  const int dst = (dst_base >= 0) ? (dst_base + r) : dst_rows[r];
  for (int i = tid; i < K; i += NT)
    out[(size_t)dst * out_stride + i] = s_sel[i];
}

std::vector<torch::Tensor> sm70_ipc_alloc(int64_t bytes) {
  void* ptr = nullptr;
  cudaError_t e = cudaMalloc(&ptr, bytes);
  TORCH_CHECK(e == cudaSuccess,
              "sm70_ipc_alloc cudaMalloc failed: ", cudaGetErrorString(e));
  cudaMemset(ptr, 0, bytes);
  cudaIpcMemHandle_t handle;
  e = cudaIpcGetMemHandle(&handle, ptr);
  TORCH_CHECK(e == cudaSuccess,
              "cudaIpcGetMemHandle failed: ", cudaGetErrorString(e));
  int device = -1;
  cudaGetDevice(&device);
  auto buf = torch::from_blob(
      ptr, {bytes}, [](void* p) { cudaFree(p); },
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA,
                                                         device));
  auto h = torch::empty({(int64_t)sizeof(cudaIpcMemHandle_t)},
                        torch::TensorOptions().dtype(torch::kUInt8));
  memcpy(h.data_ptr(), &handle, sizeof(handle));
  return {buf, h};
}

int64_t sm70_ipc_open(const torch::Tensor& handle) {
  TORCH_CHECK(handle.device().is_cpu() &&
              handle.numel() == (int64_t)sizeof(cudaIpcMemHandle_t));
  cudaIpcMemHandle_t h;
  memcpy(&h, handle.data_ptr(), sizeof(h));
  void* ptr = nullptr;
  cudaError_t e =
      cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
  TORCH_CHECK(e == cudaSuccess,
              "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(e));
  return (int64_t)(uintptr_t)ptr;
}

static ulonglong4 sm70_pack_ptrs(const torch::Tensor& t, int tp) {
  ulonglong4 p = {0, 0, 0, 0};
  const int64_t* d = t.data_ptr<int64_t>();
  if (tp > 0) p.x = (unsigned long long)d[0];
  if (tp > 1) p.y = (unsigned long long)d[1];
  if (tp > 2) p.z = (unsigned long long)d[2];
  if (tp > 3) p.w = (unsigned long long)d[3];
  return p;
}

void sm70_ipc_flag_set(int64_t flag_ptr, int64_t which, int64_t seq) {
  auto stream = at::cuda::getCurrentCUDAStream();
  sm70_ipc_flag_set_kernel<<<1, 1, 0, stream>>>(
      (int*)(uintptr_t)flag_ptr + which, (int)seq);
}

void sm70_ipc_slot_wait(const torch::Tensor& flag_ptrs, int64_t tp,
                        int64_t min_seq) {
  auto stream = at::cuda::getCurrentCUDAStream();
  sm70_ipc_slot_wait_kernel<<<1, 32, 0, stream>>>(
      sm70_pack_ptrs(flag_ptrs, (int)tp), (int)tp, (int)min_seq);
}

void sm70_indexer_shard_merge_p2p(torch::Tensor& out,
                                  const torch::Tensor& data_ptrs,
                                  const torch::Tensor& flag_ptrs,
                                  const torch::Tensor& dst_rows,
                                  int64_t dst_base, int64_t topk, int64_t tp,
                                  int64_t R, int64_t seq, int64_t slot_off) {
  TORCH_CHECK(out.scalar_type() == at::kInt && out.dim() == 2);
  TORCH_CHECK(topk == 2048, "p2p shard merge supports K=2048 only");
  TORCH_CHECK(tp >= 2 && tp <= 4 && (tp * topk) % 512 == 0);
  TORCH_CHECK(dst_base >= 0 || dst_rows.numel() == R);
  TORCH_CHECK(data_ptrs.device().is_cpu() && flag_ptrs.device().is_cpu());
  constexpr int K = 2048;
  auto kern = sm70_indexer_shard_merge_p2p_kernel<K>;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<(int)R, 512, 0, stream>>>(
      sm70_pack_ptrs(data_ptrs, (int)tp), sm70_pack_ptrs(flag_ptrs, (int)tp),
      dst_base >= 0 ? nullptr : dst_rows.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(), out.stride(0), (int)R, (int)tp, (int)dst_base,
      (int)seq, (long)slot_off);
}

void sm70_indexer_shard_merge(torch::Tensor& out, const torch::Tensor& allf,
                              const torch::Tensor& dst_rows, int64_t dst_base,
                              int64_t topk) {
  TORCH_CHECK(allf.is_cuda() && allf.is_contiguous() && allf.dim() == 3,
              "allf must be contiguous [tp, R, 2K] cuda");
  TORCH_CHECK(allf.scalar_type() == at::kFloat, "allf must be fp32");
  TORCH_CHECK(out.scalar_type() == at::kInt && out.dim() == 2);
  const int tp = allf.size(0);
  const int R = allf.size(1);
  TORCH_CHECK(topk == 2048 && allf.size(2) == 2 * topk,
              "shard merge kernel supports K=2048 only");
  TORCH_CHECK(tp >= 2 && tp <= 4, "shard merge kernel supports tp in [2,4]");
  TORCH_CHECK(dst_base >= 0 || dst_rows.numel() == R,
              "need dst_rows when dst_base < 0");
  constexpr int K = 2048;
  TORCH_CHECK((tp * K) % 512 == 0, "ncand must be a multiple of 512");
  auto kern = sm70_indexer_shard_merge_kernel<K>;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(allf));
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<R, 512, 0, stream>>>(
      allf.data_ptr<float>(),
      dst_base >= 0 ? nullptr : dst_rows.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(), out.stride(0), R, tp, (int)dst_base);
}

// pack + pre-scale Q for the prefill kernel: [B, n_head<=16, 576] ->
// [B, 36, 16, 16] fragment tiles (padded heads zeroed, scale folded).
__global__ void sm70_sparse_prefill_pack_q(const half* __restrict__ q,
                                           half* __restrict__ qp, int B,
                                           int n_head, float scale) {
  const long total = (long)B * 36 * 256;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int d_in = i % 16;
    const int h = (i / 16) % 16;
    const int kt = (i / 256) % 36;
    const long b = i / (36 * 256);
    const int d = kt * 16 + d_in;
    qp[i] = (h < n_head)
                ? __hmul(q[((long)b * n_head + h) * 576 + d],
                         __float2half(scale))
                : __float2half(0.f);
  }
}
#endif  // SM70_MLA_WMMA_AVAILABLE

void sm70_sparse_mla_decode(torch::Tensor& o, torch::Tensor& lse,
                            const torch::Tensor& q,
                            const torch::Tensor& kv_cache,
                            const torch::Tensor& idx,
                            const torch::Tensor& block_table,
                            const torch::Tensor& seq_lens, double scale) {
#ifdef SM70_MLA_WMMA_AVAILABLE
  TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda());
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  const bool kv_fp8 = kv_cache.scalar_type() == at::kByte;
  TORCH_CHECK(kv_fp8 || kv_cache.scalar_type() == at::kHalf,
              "kv_cache must be fp16 or uint8 (fp8_ds_mla)");
  TORCH_CHECK(o.scalar_type() == at::kHalf, "o must be fp16");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "lse must be fp32");
  TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt);

  constexpr int D_K = 576;
  constexpr int D_V = 512;
  constexpr int TN = 32;
  constexpr int WARPS = 16;  // V100 winner (TN=32/WARPS=16, occ 25%)

  const int B = q.size(0);
  const int n_head = q.size(1);
  const int topk = idx.size(1);
  TORCH_CHECK(q.size(2) == D_K, "q last dim must be 576");
  TORCH_CHECK(n_head > 0, "n_head must be > 0");
  TORCH_CHECK(o.size(2) == D_V, "o last dim must be 512");
  if (kv_fp8) {
    TORCH_CHECK(kv_cache.size(-1) == DSMLA_ROW_BYTES,
                "fp8 kv_cache last dim must be 656 (fp8_ds_mla layout)");
  }

  const int page_size = kv_cache.size(-2);
  const int64_t n_kv_rows =
      kv_cache.numel() / (kv_fp8 ? DSMLA_ROW_BYTES : D_K);
  const int max_blocks = block_table.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

  // PREFILL fast path (see sm70_sparse_mla_prefill_kernel above)
  static int prefill_v2 = []() {
    const char* v = getenv("VLLM_SM70_SPARSE_PREFILL_V2");
    return v ? atoi(v) : 1;
  }();
  // Warp-specialized prefill kernel (16 producer + 8 consumer warps, named
  // barriers; +43% over the v5 kernel in isolation). Default OFF pending
  // e2e sign-off; VLLM_SM70_SPARSE_PREFILL_WS=1 enables.
  static int prefill_ws = []() {
    const char* v = getenv("VLLM_SM70_SPARSE_PREFILL_WS");
    return v ? atoi(v) : 0;
  }();
  if (prefill_v2 && B >= 32 && n_head <= 16) {
    constexpr int PPADK = D_K + 8;
    const size_t psmem =
        (size_t)2 * 32 * PPADK * sizeof(half) +   // s_k double buffer
        (size_t)8 * 256 * sizeof(float) +         // s_part (SLOTS=8)
        (size_t)16 * 32 * sizeof(float) +         // s_sc
        (size_t)16 * 32 * sizeof(half) + 32 +     // s_p + pad
        (size_t)2 * 32 * sizeof(long);            // s_rp
    auto pkern_f16 = prefill_ws ? sm70_sparse_mla_prefill_ws_kernel<false>
                                : sm70_sparse_mla_prefill_kernel<false>;
    auto pkern_fp8 = prefill_ws ? sm70_sparse_mla_prefill_ws_kernel<true>
                                : sm70_sparse_mla_prefill_kernel<true>;
    static bool pattr_set = false;
    if (!pattr_set) {
      cudaFuncSetAttribute(pkern_f16,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)psmem);
      cudaFuncSetAttribute(pkern_fp8,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)psmem);
      pattr_set = true;
    }
    auto qp = at::empty({B, 36, 16, 16},
                        at::TensorOptions().dtype(at::kHalf).device(q.device()));
    sm70_sparse_prefill_pack_q<<<256, 256, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr()),
        reinterpret_cast<half*>(qp.data_ptr()), B, n_head, (float)scale);
    dim3 pgrid(B, 1, 1);
    const int pthreads = prefill_ws ? 768 : 512;
    auto plaunch = [&](auto kern) {
      kern<<<pgrid, pthreads, psmem, stream>>>(
          reinterpret_cast<const half*>(qp.data_ptr()), kv_cache.data_ptr(),
          idx.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
          seq_lens.data_ptr<int32_t>(),
          reinterpret_cast<half*>(o.data_ptr()), lse.data_ptr<float>(),
          n_head, (int)n_kv_rows, max_blocks, page_size, topk, (float)scale);
    };
    if (kv_fp8) {
      plaunch(pkern_fp8);
    } else {
      plaunch(pkern_f16);
    }
    return;
  }

  constexpr int PADK = D_K + 8;
  const size_t smem =
      (size_t)TN * PADK * sizeof(half) +               // s_k
      (size_t)16 * TN * sizeof(half) +                 // s_p
      (size_t)16 * TN * sizeof(float) +                // s_sc
      (size_t)16 * D_V * sizeof(half) +                // s_o
      (size_t)WARPS * (16 * TN > 256 ? 16 * TN : 256) * sizeof(float) +  // s_pv
      (size_t)WARPS * 256 * sizeof(half);              // s_qs

  auto kern_f16 = sm70_sparse_mla_decode_kernel<D_K, D_V, TN, WARPS, false>;
  auto kern_fp8 = sm70_sparse_mla_decode_kernel<D_K, D_V, TN, WARPS, true>;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kern_f16, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    cudaFuncSetAttribute(kern_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    attr_set = true;
  }

  // Choose the kv-split count so the grid fills the GPU. At B=1/TP=8 (8 heads ->
  // 1 head-tile) the un-split kernel launches ONE block on 80 SMs (~1% occ,
  // ~1.0 ms/layer); splitting the topk axis lifts occupancy ~linearly (measured
  // ~15x at 16 splits). Target ~2*SM_count blocks total, capped at topk/TN tiles.
  const int n_head_tiles = (n_head + 15) / 16;
  const int max_splits = (topk + TN - 1) / TN;  // one tile per split minimum
  int target_blocks = 160;  // ~2x V100 SM count
  int n_splits = target_blocks / (B * n_head_tiles);
  if (n_splits < 1) n_splits = 1;
  if (n_splits > max_splits) n_splits = max_splits;

  // partial-state scratch: P_O [B,H,S,D_V] fp32, P_ML [B,H,S,2] fp32.
  auto fopt = at::TensorOptions().dtype(at::kFloat).device(q.device());
  torch::Tensor p_o = at::empty({B, n_head, n_splits, D_V}, fopt);
  torch::Tensor p_ml = at::empty({B, n_head, n_splits, 2}, fopt);

  const dim3 grid(B, n_head_tiles, n_splits);
  auto launch = [&](auto kern) {
    kern<<<grid, WARPS * 32, smem, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr()), kv_cache.data_ptr(),
        idx.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(), p_o.data_ptr<float>(),
        p_ml.data_ptr<float>(), n_head, (int)n_kv_rows, max_blocks, page_size,
        topk, n_splits, (float)scale);
  };
  if (kv_fp8) {
    launch(kern_fp8);
  } else {
    launch(kern_f16);
  }

  TORCH_CHECK(n_splits <= MERGE_MAX_SPLITS, "n_splits exceeds merge capacity");
  const dim3 mgrid(B, n_head, (D_V + MERGE_DCHUNK - 1) / MERGE_DCHUNK);
  sm70_sparse_mla_merge_kernel<D_V><<<mgrid, MERGE_DCHUNK, 0, stream>>>(
      p_o.data_ptr<float>(), p_ml.data_ptr<float>(),
      reinterpret_cast<half*>(o.data_ptr()), lse.data_ptr<float>(), n_head,
      n_splits);
#else
  TORCH_CHECK(false, "sm70_sparse_mla_decode requires SM70+ (WMMA) build");
#endif
}

// DSA indexer logits (Volta). q:[B,n_ihead,HD] w:[B,n_ihead] fp16,
// k_cache: paged [num_blocks,PAGE,HD] fp16 OR [num_blocks,PAGE,132] uint8
// (fp8 e4m3 + fp32 per-token scale, see sm70_indexer_k_store_fp8_kernel),
// score:[B,max_kv] fp32 out.
// hist (optional, int32 [B, 4096], PRE-ZEROED): fused top-k level-1 histogram
// of the monotone-fp32 top-12 bits of each valid score — feeds
// sm70_decode_topk so it can skip its own histogram pass. Decode path only
// (the B>=8 WMMA prefill dispatch ignores it, checked below).
void sm70_indexer_logits(torch::Tensor& score, const torch::Tensor& q,
                         const torch::Tensor& k_cache,
                         const torch::Tensor& weights,
                         const torch::Tensor& block_table,
                         const torch::Tensor& seq_lens,
                         const std::optional<torch::Tensor>& hist) {
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be fp16");
  const bool k_fp8 = k_cache.scalar_type() == at::kByte;
  TORCH_CHECK(k_fp8 || k_cache.scalar_type() == at::kHalf,
              "k_cache must be fp16 or uint8 (fp8)");
  TORCH_CHECK(weights.scalar_type() == at::kHalf, "weights must be fp16");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt);

  constexpr int HD = 128;
  const int B = q.size(0);
  const int n_ihead = q.size(1);
  TORCH_CHECK(q.size(2) == HD, "indexer head_dim must be 128");
  TORCH_CHECK(n_ihead == 32, "kernel specialized for 32 indexer heads");
  if (k_fp8) {
    TORCH_CHECK(k_cache.size(-1) == IDXK_FP8_ROW_BYTES,
                "fp8 indexer k_cache last dim must be 132");
  }
  const int max_kv = score.size(1);
  const int page_size = k_cache.size(-2);
  TORCH_CHECK((page_size & (page_size - 1)) == 0,
              "indexer cache page size must be a power of two");
  const int64_t n_kv_rows =
      k_cache.numel() / (k_fp8 ? IDXK_FP8_ROW_BYTES : HD);
  const int max_blocks = block_table.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto stream = at::cuda::getCurrentCUDAStream();

#ifdef SM70_MLA_WMMA_AVAILABLE
  // For a multi-token batch (prefill), use the FlashAttention-style WMMA kernel:
  // it gathers each K row ONCE per (query-tile, key-tile) and reuses it across
  // QT queries and all heads on tensor cores, instead of the scalar kernel's
  // per-token K re-read. This is the long-context prefill win. Decode (B==1)
  // has no query reuse -> keep the split scalar kernel (better occupancy there).
  if (B >= 8) {
    TORCH_CHECK(!hist.has_value(),
                "sm70_indexer_logits: fused hist is decode-only (B < 8)");
    // Tile variant, env-selectable for A/B (VLLM_SM70_IDX_WMMA_TILE):
    //   0 = QT16/KT64/WARPS8  (original: NKT=4 < WARPS -> half the warps idle
    //       in the per-head QK+epilogue loop; Q re-staged 32x128KB per block)
    //   1 = QT16/KT64/WARPS4  (all warps active, 3 blocks/SM)
    //   2 = QT32/KT64/WARPS8  (all warps active via 2 M-tiles, K amortized 2x)
    //   3 = QT16/KT128/WARPS8 (all warps active via 8 N-tiles, Q traffic /2)
    //   4 = QT32/KT128/WARPS8 (Q traffic /2 + K amortized 2x; 64KB smem)
    static int variant = []() {
      const char* v = getenv("VLLM_SM70_IDX_WMMA_TILE");
      return v ? atoi(v) : 0;
    }();
    auto run = [&](auto qt_c, auto kt_c, auto warps_c) {
      constexpr int QT = decltype(qt_c)::value;
      constexpr int KT = decltype(kt_c)::value;
      constexpr int WARPS = decltype(warps_c)::value;
      constexpr int PADK = HD;
      const size_t smem =
          (size_t)KT * PADK * sizeof(half) +           // s_k
          (size_t)QT * HD * sizeof(half) +             // s_q
          (size_t)QT * KT * sizeof(float) +            // s_sc
          (size_t)WARPS * 16 * 16 * sizeof(float);     // s_pv
      auto kern_f16 =
          sm70_indexer_logits_wmma_kernel<HD, 32, QT, KT, WARPS, false>;
      auto kern_fp8 =
          sm70_indexer_logits_wmma_kernel<HD, 32, QT, KT, WARPS, true>;
      static bool attr_set = false;
      if (!attr_set) {
        cudaFuncSetAttribute(kern_f16,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        cudaFuncSetAttribute(kern_fp8,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        attr_set = true;
      }
      const dim3 grid((B + QT - 1) / QT, (max_kv + KT - 1) / KT, 1);
      auto launch = [&](auto kern) {
        kern<<<grid, WARPS * 32, smem, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr()), k_cache.data_ptr(),
            reinterpret_cast<const half*>(weights.data_ptr()),
            block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
            score.data_ptr<float>(), B, (int)n_kv_rows, max_blocks, page_size,
            max_kv);
      };
      if (k_fp8) {
        launch(kern_fp8);
      } else {
        launch(kern_f16);
      }
    };
    using i16 = std::integral_constant<int, 16>;
    using i32c = std::integral_constant<int, 32>;
    using i64 = std::integral_constant<int, 64>;
    using i128 = std::integral_constant<int, 128>;
    using i4 = std::integral_constant<int, 4>;
    using i8 = std::integral_constant<int, 8>;
    switch (variant) {
      case 1: run(i16{}, i64{}, i4{}); break;
      case 2: run(i32c{}, i64{}, i8{}); break;
      case 3: run(i16{}, i128{}, i8{}); break;
      case 4: run(i32c{}, i128{}, i8{}); break;
      default: run(i16{}, i64{}, i8{}); break;
    }
    return;
  }
#endif

  // Tile the KV/position axis across blockIdx.y so the O(seq_len) logits scan
  // is spread over many blocks/SMs. PPB=128 positions/block (8 warps x 16)
  // balances grid width (128 blocks at 16k) against the per-block one-time
  // q staging cost.
  constexpr int PPB = 128;
  uint32_t* hist_ptr = nullptr;
  if (hist.has_value()) {
    TORCH_CHECK(hist->scalar_type() == at::kInt && hist->is_contiguous() &&
                    hist->size(0) == B && hist->size(1) == TOPK_NBIN,
                "hist must be contiguous int32 [B, 4096]");
    hist_ptr = reinterpret_cast<uint32_t*>(hist->data_ptr());
  }
  const int n_pos_tiles = (max_kv + PPB - 1) / PPB;
  const dim3 grid(B, n_pos_tiles, 1);
  auto launch = [&](auto kern) {
    kern<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr()), k_cache.data_ptr(),
        reinterpret_cast<const half*>(weights.data_ptr()),
        block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        score.data_ptr<float>(), hist_ptr, (int)n_kv_rows, max_blocks,
        page_size, max_kv);
  };
  if (k_fp8) {
    launch(sm70_indexer_logits_kernel<HD, 32, PPB, true>);
  } else {
    launch(sm70_indexer_logits_kernel<HD, 32, PPB, false>);
  }
}

// Packed-Q register-accumulator WMMA indexer logits (see the kernel comment).
// q_packed: [32, ceil(B/16), 8, 16, 16] fp16 contiguous — produced by the
// caller from q [B, 32, 128] via view/permute (zero-padded to a 16 multiple).
// Other args as sm70_indexer_logits. Scores are BIT-IDENTICAL to the wmma
// kernel above (same mma accumulation order).
void sm70_indexer_logits_packed(torch::Tensor& score,
                                const torch::Tensor& q_packed,
                                const torch::Tensor& k_cache,
                                const torch::Tensor& weights,
                                const torch::Tensor& block_table,
                                const torch::Tensor& seq_lens) {
#ifndef SM70_MLA_WMMA_AVAILABLE
  TORCH_CHECK(false, "sm70_indexer_logits_packed requires WMMA support");
#else
  TORCH_CHECK(q_packed.scalar_type() == at::kHalf && q_packed.is_contiguous(),
              "q_packed must be contiguous fp16");
  const bool k_fp8 = k_cache.scalar_type() == at::kByte;
  TORCH_CHECK(k_fp8 || k_cache.scalar_type() == at::kHalf,
              "k_cache must be fp16 or uint8 (fp8)");
  TORCH_CHECK(weights.scalar_type() == at::kHalf, "weights must be fp16");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt);
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt);
  constexpr int HD = 128;
  const int B = score.size(0);
  const int n_qt = (B + 15) / 16;
  TORCH_CHECK(q_packed.dim() == 5 && q_packed.size(0) == 32 &&
                  q_packed.size(1) == n_qt && q_packed.size(2) == HD / 16 &&
                  q_packed.size(3) == 16 && q_packed.size(4) == 16,
              "q_packed must be [32, ceil(B/16), 8, 16, 16]");
  const int max_kv = score.size(1);
  const int page_size = k_cache.size(-2);
  TORCH_CHECK((page_size & (page_size - 1)) == 0,
              "indexer cache page size must be a power of two");
  const int64_t n_kv_rows =
      k_cache.numel() / (k_fp8 ? IDXK_FP8_ROW_BYTES : HD);
  const int max_blocks = block_table.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_packed));
  auto stream = at::cuda::getCurrentCUDAStream();

  // one-time self-check: the baked-in Volta accumulator fragment layout must
  // match what this driver/toolchain actually produces.
  static bool layout_checked = false;
  if (!layout_checked) {
    auto ok = torch::ones({1}, torch::dtype(torch::kInt32)
                                   .device(q_packed.device()));
    sm70_idx_layout_probe_kernel<<<1, 32, 0, stream>>>(ok.data_ptr<int>());
    TORCH_CHECK(ok.cpu().item<int>() == 1,
                "sm70_indexer_logits_packed: Volta wmma accumulator fragment "
                "layout mismatch — rebuild with the layout re-derived");
    layout_checked = true;
  }

  const dim3 grid((B + 31) / 32, (max_kv + 31) / 32, 1);
  auto launch = [&](auto kern) {
    kern<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const half*>(q_packed.data_ptr()),
        k_cache.data_ptr(),
        reinterpret_cast<const half*>(weights.data_ptr()),
        block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        score.data_ptr<float>(), B, (int)n_kv_rows, max_blocks, page_size,
        max_kv);
  };
  if (k_fp8) {
    launch(sm70_indexer_logits_packed_kernel<true>);
  } else {
    launch(sm70_indexer_logits_packed_kernel<false>);
  }
#endif
}

// Fused decode top-k select: idx [B, K] int32 <- top-K positions of
// score [B, max_kv] fp32 among p < seq_lens[b]; -1 padded. hist is the
// level-1 histogram PREBUILT by sm70_indexer_logits's fused epilogue (int32
// [B, 4096], zeroed by the caller before the indexer runs). See the kernel
// comment for the algorithm/exactness contract. CUDA-graph-safe: static
// shapes, device-side selection, workspace zeroed via stream memset nodes.
void sm70_decode_topk(torch::Tensor& idx, const torch::Tensor& score,
                      const torch::Tensor& seq_lens,
                      const torch::Tensor& hist, int64_t k) {
  TORCH_CHECK(score.is_cuda() && idx.is_cuda());
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
  TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(score.is_contiguous() && idx.is_contiguous());
  const int B = score.size(0);
  const int max_kv = score.size(1);
  const int K = (int)k;
  TORCH_CHECK(idx.size(0) == B && idx.size(1) == K, "idx must be [B, K]");
  TORCH_CHECK(K >= 1, "k must be >= 1");
  TORCH_CHECK(hist.scalar_type() == at::kInt && hist.is_contiguous() &&
                  hist.size(0) == B && hist.size(1) == TOPK_NBIN,
              "hist must be contiguous int32 [B, 4096]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(score));
  auto stream = at::cuda::getCurrentCUDAStream();

  // grid: all blocks of the launch MUST be co-resident (spin barriers).
  static int block_cap = 0;
  if (block_cap == 0) {
    int occ = 0, sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                           at::cuda::current_device());
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, sm70_decode_topk_kernel, TOPK_THREADS, 0);
    TORCH_CHECK(err == cudaSuccess && occ >= 1,
                "sm70_decode_topk occupancy query failed");
    block_cap = sms * occ;
  }
  int n_tiles = std::min(64, (max_kv + TOPK_THREADS - 1) / TOPK_THREADS);
  while (B * n_tiles > block_cap && n_tiles > 1) n_tiles /= 2;
  TORCH_CHECK(B * n_tiles <= block_cap,
              "sm70_decode_topk grid exceeds co-residency cap (B=", B, ")");
  int tile_size = (max_kv + n_tiles - 1) / n_tiles;
  tile_size = (tile_size + TOPK_THREADS - 1) / TOPK_THREADS * TOPK_THREADS;
  n_tiles = (max_kv + tile_size - 1) / tile_size;

  // workspace: [counters 2xB | hist2 Bx4096 | tile_cnt BxNT] u32 (zeroed) +
  //            [cand B x NT x tile_size] i32 (uninitialized).
  const size_t zero_words = (size_t)B * (2 + TOPK_NBIN + n_tiles);
  const size_t total_words = zero_words + (size_t)B * n_tiles * tile_size;
  auto ws = at::empty({(int64_t)(total_words * 4)},
                      at::TensorOptions().dtype(at::kByte).device(score.device()));
  uint32_t* wp = reinterpret_cast<uint32_t*>(ws.data_ptr());
  uint32_t* counters = wp;
  uint32_t* hist2 = wp + (size_t)B * 2;
  uint32_t* tile_cnt = hist2 + (size_t)B * TOPK_NBIN;
  int32_t* cand = reinterpret_cast<int32_t*>(wp + zero_words);

  cudaMemsetAsync(wp, 0, zero_words * 4, stream);
  cudaMemsetAsync(idx.data_ptr(), 0xFF, (size_t)B * K * 4, stream);  // -1 fill

  const dim3 grid(n_tiles, B, 1);
  sm70_decode_topk_kernel<<<grid, TOPK_THREADS, 0, stream>>>(
      score.data_ptr<float>(), seq_lens.data_ptr<int32_t>(),
      idx.data_ptr<int32_t>(), counters,
      reinterpret_cast<const uint32_t*>(hist.data_ptr()), hist2, tile_cnt,
      cand, max_kv, K, n_tiles, tile_size);
}

// indexer-K store: scatter k rows into kv_cache at slot_mapping (skip <0).
// kv_cache fp16 -> plain fp16 copy; kv_cache uint8 -> quantize to the 132-byte
// fp8 inline layout [128 e4m3 | fp32 per-token scale].
void sm70_indexer_k_store(torch::Tensor& kv_cache, const torch::Tensor& k,
                          const torch::Tensor& slot_mapping) {
  TORCH_CHECK(kv_cache.is_cuda() && k.is_cuda());
  const bool k_fp8 = kv_cache.scalar_type() == at::kByte;
  TORCH_CHECK(k_fp8 || kv_cache.scalar_type() == at::kHalf,
              "kv_cache must be fp16 or uint8 (fp8)");
  TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be fp16");
  const int head_dim = k.size(-1);
  const int n_tok = k.size(0);
  TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
  if (n_tok == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (k_fp8) {
    TORCH_CHECK(head_dim == 128, "fp8 indexer store requires head_dim == 128");
    TORCH_CHECK(kv_cache.size(-1) == IDXK_FP8_ROW_BYTES,
                "fp8 indexer kv_cache last dim must be 132");
    const int64_t n_rows = kv_cache.numel() / IDXK_FP8_ROW_BYTES;
    sm70_indexer_k_store_fp8_kernel<<<n_tok, 128, 0, stream>>>(
        reinterpret_cast<const half*>(k.data_ptr()),
        slot_mapping.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), n_tok, n_rows);
    return;
  }
  const int64_t n_rows = kv_cache.numel() / head_dim;
  const int threads = head_dim < 256 ? head_dim : 256;
  sm70_indexer_k_store_kernel<<<n_tok, threads, 0, stream>>>(
      reinterpret_cast<const half*>(k.data_ptr()), slot_mapping.data_ptr<int32_t>(),
      reinterpret_cast<half*>(kv_cache.data_ptr()), n_tok, head_dim, n_rows);
}

