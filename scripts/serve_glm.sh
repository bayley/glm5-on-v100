#!/bin/bash
# Serve GLM-5.1-AWQ on 16x V100 (two 8x V100 nodes via Ray) with the OpenAI-compatible
# server on port 8080, using the SM70 DSA sparse-MLA path (16k context, decode
# CUDA graphs). Mirrors the working benchmark config (see ../docs/GOTCHAS.md).
#
# PREREQS (see ../docs/GOTCHAS.md "Cluster / Ray operations"):
#   - Ray head+worker already started from $VLLM_VENV/bin/ray on
#     BOTH nodes with the correct per-node NCCL/GLOO ifnames. Verify:
#       $VLLM_VENV/bin/ray status   # -> 0.0/16.0 GPU
#   - cuBLAS 12.9 symlink + repo/.so in place on both nodes.
#
# Usage:
#   bash serve_glm.sh          # DEFAULT: MTP-1 spec decode, 183k ctx, TP4xPP4
#                              # (35.6/35.8/34.0 tok/s @0.3/1.6/3.2k decode)
#   MTP=0 bash serve_glm.sh    # plain 200k config (no spec decode; use for
#                              # full-length ingestion or MTP bisects)
#   PORT=8000 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=8 bash serve_glm.sh
#   TP_SIZE=8 PP_SIZE=2 bash serve_glm.sh   # old topology (less KV, +5% decode)
#
# CONCURRENCY vs CONTEXT: default is now MAX_MODEL_LEN=202752 (the model's
# full trained 200k context) with MAX_NUM_SEQS=4 sharing the ~231k-token KV
# pool (fp8 latent cache; 1.14x concurrency at full 200k - one 200k request
# plus small ones fit together; the scheduler queues beyond that). Measured
# (2026-07-06, fp8 KV): cold prefill 566 tok/s @96k / 340 @200k; decode 18.6
# tok/s @96k / 16.7 @200k / 22.4 short - all within noise of the old 100k
# fp16 config at 100k, plus the entire 100k-200k range unlocked. Needle
# retrieval verified at 96k/178k/200k. Mixed workloads (one long + shorts
# arriving mid-prefill, all concurrent) are needle-verified correct - required
# TWO multi-request sparse-prefill fixes (2026-07-06, see ../patches/MANIFEST.md (multi-request
# indexer fixes): cu_seqlen_ks offset subtraction + per-request-segment indexer
# launches.
# Known limitation: >=2 concurrent LONG prefills interleave through the
# shared 512-token chunk budget and aggregate prefill drops to ~385 tok/s
# (vs 1179 single-stream at 16k) - fine for chat turns with prefix caching,
# poor for bulk long-document ingestion.
set -euo pipefail

cd "${VLLM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

MODEL="${MODEL:?set MODEL to your GLM-5.1-AWQ checkpoint path}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
# 202752 = the model's max_position_embeddings (its full trained context).
# Requires the fp8 latent KV cache (KV_FP8=1, default below): KV pool is
# 231,232 tokens at util 0.95 = 1.14x concurrency at full 200k. Validated
# 2026-07-06: 200,079-token needle PASS; decode 16.7 tok/s @200k / 18.6-18.8
# @96k / 22.4 short; cold prefill 566 tok/s @96k, 340 @200k (the O(ctx^2)
# DSA indexer-selection term). With KV_FP8=0 the fp16 pool is 149,760 tokens
# — set MAX_MODEL_LEN accordingly.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"
# MTP speculative decoding: THE DEFAULT since 2026-07-08 (MTP=0 restores
# the plain 200k config). Decode +24-35% at every depth (35.6/35.8/34.0
# tok/s @0.3/1.6/3.2k vs 27.3/26.5/25.5; 21.4 @180k vs 19.3 @200k) at the
# cost of ~15k context (183k vs 198k usable). The draft layer costs
# ~1.9 GiB on the last PP stage plus a 20th KV+indexer cache layer, which
# would leave only a 78,016-token pool; RECOVERED by shifting one MoE
# layer off the binding last stage (PP_PARTITION=22,19,19,18: -> 161,984
# tokens) and the fp8 indexer-K cache (IDXK_FP8=1: x1.157 -> 187,520) —
# both MTP-mode defaults below. The IDXK_FP8 decode-scan cost is fully
# offset by the lighter last stage. Validated: 180k needle PASS, cold
# prefill 1881 tok/s @180k, temp-0 outputs identical to eager, 1085-token
# generation clean, acceptance ~58% (MTP-1). Concurrency: 4 streams = 70.8
# tok/s aggregate @2k / 56.9 @24k (2.0x single-stream aggregate).
MTP="${MTP:-1}"
if [[ "$MTP" == "1" ]]; then
    IDXK_FP8="${IDXK_FP8:-1}"
    if [[ "$MAX_MODEL_LEN" == "202752" ]]; then
        if [[ "${EAGER:-0}" == "1" ]]; then
            # Bringup/debug mode; measured pool without the partition shift.
            MAX_MODEL_LEN=98304
        else
            # 186368 = the 187,520-token pool minus the GEMM tuner's
            # warmup workspace (~700 tokens; estimator max 186,688).
            MAX_MODEL_LEN=186368
        fi
    fi
fi
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# 1024 (2026-07-07): 200k prefill 160.7 -> 142.5 s (1403 tok/s) and 96k
# needle 1287 -> 1497 tok/s vs 512 — MoE expert weights are re-read per
# CHUNK, so bigger chunks amortize them (fixed cost 706 -> 602 us/tok). KV
# pool only drops 247,360 -> 244,735 tokens (still >= the 202,752 floor).
# Tradeoff: decode steps can't run mid-chunk, so concurrent decode pauses
# ~0.7 s during a long prefill (was ~0.4 s at 512).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
# Topology (TP_SIZE*PP_SIZE must be 16): DEFAULT IS NOW TP=4 x PP=4.
# Measured 2026-07-06 (same method, util 0.95, fp8 attn, sparse MLA+prefill):
#              TP8xPP2        TP4xPP4
#   KV @16k    57,984 tok     124,992 tok   (2.16x - MLA KV is REPLICATED per
#                                            TP rank; only PP divides layers)
#   decode     42.7-44.5      44.1-47.2 ms/tok  (-5%, ctx 330..15.9k, flat)
#   prefill    ~775-884       1179 tok/s    (+33%, cold 15.7k prompt)
#   weights    25.3/25.9      24.6/24.9 GiB/GPU (replicated attn proj halve)
# Decode barely moves because the per-layer critical path is dominated by
# REPLICATED GEMVs (unchanged per rank), the sparse-attn WMMA head-tile was
# half-padded at TP8's 8 heads (16 heads fill it for free), and 4-rank
# all-reduces have less straggler skew - the 2x terms (sharded GEMVs, EP MoE)
# are a minority of layer time. TP groups are contiguous world ranks, so PP4
# puts stages 0,1 on the head node and 2,3 on the worker (still one cross-node PP hop).
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-4}"
# Util: PP4 needs 0.95 (the last-stage sampler warmup re-allocates the ~896
# MiB chunked-prefill-workspace simulation - 64k tokens x 16 heads x 448 dims
# fp16 at TP=4 - after KV has consumed the budget; OOM at 0.965). PP2 was
# validated at 0.965 (0.97 died: lazy post-profile NCCL callocs found no slack
# once the partition was balanced). Shrinking that workspace cap (64k tokens
# is sized for "8 full-length requests" upstream) is a known follow-up that
# would reclaim ~0.5-0.7 GiB/GPU on both axes.
if [[ "$PP_SIZE" == "4" ]]; then
    DEFAULT_UTIL=0.95
else
    DEFAULT_UTIL=0.965
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-$DEFAULT_UTIL}"
SERVED_NAME="${SERVED_NAME:-glm-5.1-awq}"
# Prefix caching (default ON): Open WebUI re-sends the WHOLE conversation
# (system prompt + tools ~3k tokens + all turns) every request; cached prefix
# blocks turn that re-prefill into a cache hit, so TTFT per turn only pays for
# the NEW tokens (at ~600-900 tok/s prefill this saves seconds per turn).
# Benchmarks disable it to measure prefill honestly (PREFIX_CACHING=0).
PREFIX_CACHING="${PREFIX_CACHING:-1}"

# --- Cluster networking (HEAD node values; see docs/GOTCHAS.md) ---
export VLLM_HOST_IP="${VLLM_HOST_IP:?set to this node.s IP (the one Ray/NCCL should use)}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:?set to this node.s ethernet ifname}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?set to this node.s ethernet ifname}"
export NCCL_IB_HCA="${NCCL_IB_HCA:?set to this node.s IB HCA (nccl uses it for cross-node)}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-0}"

# --- SM70 DSA sparse-MLA + decode CUDA graphs (the fast 16k path) ---
# SPARSE=0 switches to the dense-MLA reference path (correctness bisects;
# requires KV_FP8=0 — dense paths can't read the fp8 latent layout).
export VLLM_SM70_GLM_SPARSE_MLA="${SPARSE:-1}"
export VLLM_SM70_SPARSE_DECODE_MIN_CTX=2048
export VLLM_SM70_GLM_SPARSE_PREFILL="${SPARSE:-1}"
# WMMA prefill-indexer tile shape (2 = QT32/KT64, ~12% faster than the
# original 0 = QT16/KT64 at 100k+ ctx; bit-exact, see ../docs/
# ENV_FLAGS.md). IDX_TILE=0 restores the original tiling. Only used when the
# packed kernel below is disabled.
export VLLM_SM70_IDX_WMMA_TILE="${IDX_TILE:-2}"
# Packed-Q register-accumulator WMMA prefill indexer (default ON): ~5-6x the
# staged WMMA kernel (32->44 TFLOPS vs ~7), bit-identical scores. ncu showed
# the staged kernel L1-wavefront-bound; this keeps the whole per-head
# epilogue + cross-head accumulation in registers and loads Q as contiguous
# 512 B packed fragments. IDX_PACKED=0 falls back to the staged kernel.
export VLLM_SM70_IDX_PACKED_WMMA="${IDX_PACKED:-1}"
# TP key-sharded prefill indexer scoring: each TP rank scores 1/tp of the
# key positions + local top-k; all-gather + merge = exact global top-k at
# 1/tp the per-GPU cost. The indexer was 84% of GPU time at 200k prefill.
# IDX_TP_SHARD=0 restores replicated scoring.
export VLLM_SM70_IDXER_TP_SHARD_PREFILL="${IDX_TP_SHARD:-1}"
# Warp-specialized sparse-attn prefill kernel (DEFAULT ON 2026-07-09,
# user-approved; SPARSE_WS=0 restores the v5 kernel): 16 producer + 8
# consumer warps with named-barrier ping-pong, decoupling the fp8 KV gather
# from the WMMA pipeline. Kernel 10.96 -> 15.73 TFLOPS (+43%) in isolation,
# op-level 1.33x, outputs BIT-IDENTICAL to the v5 kernel. With
# MOE_PREFILL_TUNE below: 64k cold prefill 2118 -> ~2400 tok/s (+13-15%),
# 96k needle PASS, decode parity.
export VLLM_SM70_SPARSE_PREFILL_WS="${SPARSE_WS:-1}"
# Fused indexer shard-merge (DEFAULT ON 2026-07-09, user-approved;
# FUSED_MERGE=0 reverts to the torch.topk epilogue): one kernel replaces
# the per-layer TP shard-merge glue (permute + topk + gather/where +
# copies, ~15 launches). Same selected key set (tie-break matches
# torch.topk), positions emitted ascending. ~+2.4% cold prefill.
export VLLM_SM70_IDXER_FUSED_MERGE="${FUSED_MERGE:-1}"
# P2P shard exchange (DEFAULT ON 2026-07-09, user-approved after a
# 6-round alternating A/B: +2.3% 64k prefill, 26.59 -> 25.99 s, 17/18
# on-samples beat every off-sample; 96k/180k needle PASS): replaces the
# merge's NCCL all-gather with direct NVLink reads via IPC-mapped
# double-buffered slots + seq flags. Falls back to NCCL on init failure
# or oversize chunks. P2P_MERGE=0 reverts.
export VLLM_SM70_IDXER_P2P_MERGE="${P2P_MERGE:-1}"
# Prefill-shape MoE GEMM tuning (DEFAULT ON 2026-07-09, user-approved;
# MOE_PREFILL_TUNE=0 disables): lets the TurboMind measure-tuner cover the
# full-chunk grouped-GEMM shape (mnbt x top_k = 8192 slots) in addition to
# the decode window. The default heuristic picks 128/64-row CTA-M tiles off
# the aggregate M and pads every expert's ~32 rows; measured tiles
# (16/32-row CTA-M) take W13+W2 6.70 -> 4.08 ms/layer (W2 alone 2.45x).
# EXACT-shape gate only: tail chunks keep default tiles — a broad cap
# raise caused ~9 s one-time kMeasure stalls MID-REQUEST on every novel
# tail-chunk size. Tuned at startup by the profile-run warmup.
if [[ "${MOE_PREFILL_TUNE:-1}" == "1" ]]; then
    export VLLM_SM70_AWQ_MOE_TUNE_PREFILL_SLOTS=$((MAX_NUM_BATCHED_TOKENS * 8))
fi
export VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1
# CUDA-graph capture size must be >= max_num_seqs (batch sizes get captured).
export VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE="${MAX_NUM_SEQS}"
export VLLM_SM70_MLA_PREFILL_KV_BLOCK=128
# fp8-e4m3 weight-only storage for the fp16 attention/indexer projections
# (quantized at load; the AWQ on disk is untouched). -1.46 GiB/GPU weights,
# 2.6x KV capacity, +8.6% decode tok/s, -1.6% prefill. Official GLM-5.1-FP8
# ships these tensors at this exact precision. ATTN_FP8=0 to disable.
export VLLM_SM70_ATTN_FP8="${ATTN_FP8:-1}"
# TurboMind GEMM small-shape tuning during warmup (DEFAULT ON 2026-07-08):
# tunes the M=1..8 dispatch for the AWQ MoE + dense GEMMs — the multi-token
# MTP verify shapes (M = 2*streams) ran untuned dispatch before. Measured
# @2k ctx: single 35.8 -> 37.3 tok/s; 2 streams 49.9 -> 60.3 aggregate
# (+21%); 4 streams 70.8 -> 93.7 (+32%); 4x24k 56.9 -> 79.8 (+40%). No
# measurable startup cost (absorbed in warmup). TUNE=0 restores untuned
# dispatch. Costs ~700 tokens of KV pool (tuner workspace) — the MTP
# default maxlen accounts for it.
export VLLM_SM70_AWQ_TUNE_SMALL_SHAPES="${TUNE:-1}"
# Independent-batch pipeline interleaving (INTERLEAVE=N; DEFAULT 2 since
# 2026-07-09): schedule decode requests in up to N independent waves so
# the PP batch queue keeps N steps in flight across pipeline stages
# (fills the per-stage idle of lockstep decode). Measured @2k ctx vs
# lockstep: 2 streams 60.3 -> 62.5 aggregate, 4 streams 93.7 -> 103.2,
# 4x24k 79.8 -> 87.0; single-stream parity (~36). G=4 tested: within
# noise of G=2 but higher variance (pipeline depth is bounded by each
# request's own return latency). Boundary-tested: joins/leaves, temp-0
# determinism, prefill-mid-decode, temp>0, uneven waves. INTERLEAVE=0
# restores lockstep scheduling. See envs.py for mechanics.
export VLLM_SM70_PP_INTERLEAVE_GROUPS="${INTERLEAVE:-2}"
# Extend the fp8-e4m3 weight-only storage to the fp16 MLP weights: shared
# expert (75 MoE layers) + dense-MLP layers 0-2. Same load-time e4m3
# machinery/kernels as ATTN_FP8 (requires it). Measured 2026-07-07:
# -1.5 ms/tok decode at every ctx, KV pool +4.6% (241,984 tok), prefill
# unchanged; teacher-forced ppl 7.3108 -> 7.3399 (+0.4%, edge of noise),
# temp-0 answers identical. DEFAULT ON (user-approved 2026-07-07).
# MLP_FP8=0 to restore fp16 MLP weights.
export VLLM_SM70_MLP_FP8="${MLP_FP8:-1}"

# fp8 latent KV cache (upstream fp8_ds_mla layout, 656 B/token vs fp16's
# 1152 B): ~1.5x KV capacity; the Volta sparse WMMA kernels dequantize in
# their gathers at no measured speed cost (sparse decode kernel is actually
# slightly FASTER — fewer gather bytes). Requires the sparse prefill (dense
# MLA paths can't read the layout; enforced at startup). KV_FP8=0 to disable.
export VLLM_SM70_MLA_KV_FP8="${KV_FP8:-1}"
# fp8 indexer-K cache (132 B/token vs fp16's 256 B): a further ~9% KV
# capacity, but the latency-bound decode indexer scan pays ~+20% (isolated,
# ~-3% decode tok/s at 100k ctx; prefill WMMA indexer unchanged). Default OFF
# — enable if the KV pool, not decode speed, is the binding constraint.
export VLLM_SM70_INDEXER_K_FP8="${IDXK_FP8:-0}"
# Shrink the MLA chunked-prefill workspace token cap (upstream 64k): the
# profile run reserves workspace_tokens x 16 heads x 448 dims x 2 B (~896 MiB
# at 64k/TP=4) which comes straight out of the KV budget. The SM70 sparse
# prefill doesn't use this workspace (it's only the dense-fallback context
# gather), so 16k (~224 MiB) frees ~0.65 GiB/GPU for KV while still covering
# the sparse-prefill transient score buffers (<=~110 MiB after per-128-row
# sub-chunking). MLA_WORKSPACE_TOKENS=0 restores upstream.
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_TOKENS="${MLA_WORKSPACE_TOKENS:-16384}"
# EP-safe single-token MoE decode route (2026-07-07, default ON in the
# engine): B=1 decode remaps global->local expert ids on device and skips
# non-local experts entirely instead of running the multi-token
# permute/per-expert-dispatch machinery. +8% decode tok/s at every context
# (e.g. 2048 ctx 18.3 -> 19.7 tok/s offline harness), prefill untouched.
# EP_SINGLE_TOKEN=0 to restore the old multi-token EP decode path.
export VLLM_SM70_AWQ_MOE_EP_SINGLE_TOKEN="${EP_SINGLE_TOKEN:-1}"

# --- KV-capacity reclaim (see ../docs/GOTCHAS.md "Memory ceiling") ---
# 1) CUDA-graph memory estimator: KEEP ON (default). Investigated 2026-07-06:
#    the startup log's "0.06 GiB (actual) vs 0.58 GiB (estimated), 883%" is
#    MISLEADING - the estimator's own throwaway capture leaves ~0.5 GiB of
#    persistent workspaces resident, so the later "actual" capture only shows
#    the marginal cost. The real end-to-end capture cost IS ~0.58 GiB (verified:
#    with the estimator off, capture reported 0.58 GiB and the engine OOM'd at
#    util 0.97 (KV alloc), 0.96 (MoE warmup scratch), AND 0.95 (448 MiB sampler
#    warmup) - late-allocating NCCL buffers + capture eat exactly what the
#    estimator reserves). Estimator-off only works around util<=0.94, which
#    nets LESS KV than estimator-on at 0.97. Shrinking the actual capture
#    footprint is the real (future) lever. CUDAGRAPH_MEM_ESTIMATOR=0 to A/B.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${CUDAGRAPH_MEM_ESTIMATOR:-1}"
# 2) The default even PP split counts LAYERS, not bytes: layers 0-2 are dense
#    (~0.05 GiB/GPU) vs MoE (~0.64 GiB/GPU at EP8), so the even split loads the
#    later stages ~1.9 GiB heavier and they bind the KV budget. Balance the MOE
#    count per stage instead (stage0's 3 dense layers ride along ~free).
#    PP2: 40,38 measured optimal at util 0.965 (39,39=44k / 41,37=57k /
#    40,38=58.0k KV tokens) - the exact optimum trades PP0's higher per-token
#    KV cost (more layers) against PP1's ~0.5 GiB logits/sampler peak.
#    PP4: 21,19,19,19 = 18/19/19/19 MoE per stage. Values must sum to 78.
if [[ "$PP_SIZE" == "4" ]]; then
    DEFAULT_PARTITION="21,19,19,19"
    # MTP chat mode: the draft layer (~1.9 GiB + a KV layer) lands on the
    # LAST stage and makes it the binding KV rank; shifting one MoE layer
    # to stage0 more than DOUBLES the pool (78,016 -> 161,984 tokens) and
    # also improves prefill balance (1881 tok/s @180k vs ~1400 baseline).
    if [[ "$MTP" == "1" ]]; then
        DEFAULT_PARTITION="22,19,19,18"
    fi
else
    DEFAULT_PARTITION="40,38"
fi
export VLLM_PP_LAYER_PARTITION="${PP_PARTITION:-$DEFAULT_PARTITION}"

# --- Debug prompt capture (GLM_DUMP_PROMPTS=1 bash serve_glm.sh) ---
# Dumps one JSON per /v1/chat/completions & /v1/completions request into
# GLM_DUMP_DIR: the ORIGINAL client body (Open WebUI's system prompt, tools,
# sampling fields) + the RENDERED chat-template prompt + exact token ids.
# Inspect/replay with ./replay_prompt.py <dump.json> [--show|--mode tokens].
if [[ "${GLM_DUMP_PROMPTS:-0}" == "1" ]]; then
    export VLLM_DEBUG_DUMP_PROMPTS_DIR="${GLM_DUMP_DIR:-/tmp/glm_prompt_dumps}"
    mkdir -p "$VLLM_DEBUG_DUMP_PROMPTS_DIR"
    echo "[serve] prompt capture ON -> $VLLM_DEBUG_DUMP_PROMPTS_DIR"
fi

echo "[serve] model=$MODEL host=$HOST port=$PORT max_model_len=$MAX_MODEL_LEN"
echo "[serve] max_num_seqs=$MAX_NUM_SEQS util=$GPU_MEM_UTIL sparse_mla+prefill=on"

PREFIX_FLAG="--enable-prefix-caching"
if [[ "$PREFIX_CACHING" != "1" ]]; then
    PREFIX_FLAG="--no-enable-prefix-caching"
fi
echo "[serve] prefix caching: $PREFIX_CACHING ($PREFIX_FLAG)"

# EP=0 (DEFAULT since 2026-07-07) disables expert parallelism: experts are
# TP-sharded (each rank computes a quarter-slice of every active expert).
# Same total weight bytes as EP, but per-rank decode work is IDENTICAL
# across ranks — EP's per-token expert-placement variance made every
# all-reduce wait on the straggler rank. Measured (prod 200k config,
# MLP_FP8=1): decode +11-15% at EVERY context (27.3/26.5/25.5 t/s at
# 0.3/1.6/3.2k; 21.9 @96k; 19.3 @200k), prefill parity (579 @96k, 345
# @200k), KV pool 247,360 tok (best yet). Validated: 96k+200k needle PASS,
# mixed 96k-long + 3 mid-prefill shorts all correct. With EP=0 the legacy
# TP-only single-token MoE fastpaths are enabled for decode. EP=1 restores
# expert parallelism.
MOE_EP="${EP:-0}"
EP_FLAG="--enable-expert-parallel"
if [[ "$MOE_EP" != "1" ]]; then
    EP_FLAG="--no-enable-expert-parallel"
    # Overridable for bisecting (EP0_FASTPATHS=0 disables the whole set).
    EP0_FP="${EP0_FASTPATHS:-1}"
    export VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH="${VLLM_SM70_MOE_SINGLE_TOKEN_FASTPATH:-$EP0_FP}"
    export VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH="${VLLM_SM70_MOE_SINGLE_TOKEN_PERMUTE_FASTPATH:-$EP0_FP}"
    export VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH="${VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_STAGE_FASTPATH:-$EP0_FP}"
    export VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH="${VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W13_FASTPATH:-$EP0_FP}"
    export VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH="${VLLM_SM70_MOE_SINGLE_TOKEN_INDEXED_W2_FASTPATH:-$EP0_FP}"
fi
echo "[serve] expert parallel: $MOE_EP ($EP_FLAG)"

# --- torch profiler (PROFILER_DIR=/path bash serve_glm.sh) ---
# Enables the /start_profile + /stop_profile endpoints; each worker writes its
# trace under PROFILER_DIR on ITS OWN node (worker-node ranks write there).
# PROFILER_MAX_ITERS bounds a window to N engine steps after /start_profile
# (auto-stops; 0 = record until /stop_profile). Stack/shape recording is off
# to keep overhead low on the 78-layer model.
PROF_ARGS=()
if [[ -n "${PROFILER_DIR:-}" ]]; then
    mkdir -p "$PROFILER_DIR"
    PROF_ARGS+=(
        --profiler-config.profiler=torch
        --profiler-config.torch_profiler_dir="$PROFILER_DIR"
        --profiler-config.torch_profiler_with_stack=false
        --profiler-config.ignore_frontend=true
        --profiler-config.max_iterations="${PROFILER_MAX_ITERS:-0}"
    )
    echo "[serve] torch profiler ON -> $PROFILER_DIR (max_iters=${PROFILER_MAX_ITERS:-0})"
fi

# --- MTP speculative decoding ("chat mode", default OFF) ---
# MTP=1 drafts with the checkpoint's own MTP layer (model.layers.78, a full
# DSA block incl. indexer; num_nextn_predict_layers=1). The cost lands on the
# LAST PP stage only: ~1.9 GiB/GPU weights (draft MoE + its own embed_tokens
# copy under PP>1) plus one extra KV+indexer cache layer, which shrinks the
# KV pool — run with a reduced MAX_MODEL_LEN (chat mode), not the 200k
# default. MTP_NUM_TOKENS>1 re-runs the single MTP layer per extra draft
# position (acceptance decays per position).
# (MTP resolved near the top of the script, default 1.)
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-1}"
SPEC_ARGS=()
if [[ "$MTP" == "1" ]]; then
    # NOTE: keep the padded/async drafter path (the fork's native MTP
    # design). disable_padded_drafter_batch=true force-disables async
    # scheduling, and this fork then never delivers the previous sampled
    # token to the workers (new_token_ids_lens=[0] relies on the async PP
    # broadcast) — the verify batch shifts one row and verification is
    # garbage. Non-last PP ranks reconstruct draft ids from the scheduler
    # output (see gpu_model_runner._prepare_input_ids CPU fallback).
    SPEC_ARGS+=(--speculative-config
        "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_NUM_TOKENS}}")
    # Confirm layers.78 draft weights + the top-level embed_tokens copy load.
    export VLLM_DEBUG_MTP_LOAD="${VLLM_DEBUG_MTP_LOAD:-1}"
    echo "[serve] MTP speculative decoding ON (num_speculative_tokens=$MTP_NUM_TOKENS)"
fi

# EAGER=1 forces --enforce-eager (disables the SM70 decode CUDA graphs).
# Bringup/measurement aid: the qlen=1+k MTP verify step is not graph-capturable
# until the spec-as-decode TRITON_MLA work lands (capture asserts
# max_query_len <= reorder_batch_threshold). Decode is ~3x slower eager.
EAGER_ARGS=()
if [[ "${EAGER:-0}" == "1" ]]; then
    EAGER_ARGS+=(--enforce-eager)
    echo "[serve] enforce_eager ON (decode CUDA graphs disabled)"
fi

# NOTE: use the api_server MODULE, not the `vllm serve` CLI. This repo is loaded
# via the _repo_vllm.pth editable trick (no installed dist metadata), so the CLI
# entrypoint crashes on importlib.metadata.version("vllm"). The module bypasses
# that. Same flags as `vllm serve`.
#
# --default-chat-template-kwargs: this fork defaults enable_thinking=False
# (Qwen-centric); GLM-5.1's own chat template defaults to thinking ON, so
# restore that. Per-request {"chat_template_kwargs":{"enable_thinking":false}}
# still disables it. --reasoning-parser glm45 splits reasoning_content from
# content for Open WebUI etc.
#
# --tool-call-parser glm47 (NOT glm45): GLM-5.1's template emits
# `<tool_call>{name}<arg_key>...` with the function name directly followed by
# <arg_key> (no newline). The glm45 parser's regex requires a newline after
# the name and would fail; glm47 handles the no-separator form (and
# zero-argument calls), matching GLM-5.1 exactly. --enable-auto-tool-choice
# lets the model decide when to call tools.
exec "${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --dtype float16 \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    "$EP_FLAG" \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser glm45 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    "${PROF_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    "$PREFIX_FLAG"
