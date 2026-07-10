# Patch manifest

Per-file description of everything in `glm5-v100-sm70.patch` (generated
against upstream 1Cat-vLLM `v1.2.1`,
commit `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`). Use this to re-apply
hunks contextually if the patch does not apply to your base, or to understand
the change surface before applying it.

**Cross-cutting summary:** almost everything is ADDITIVE behind
`VLLM_SM70_*` env flags and/or `device_capability.major == 7` checks. The
unconditional behavior changes are all bug fixes:

- the fp16-amax fix in `cache_kernels.cu` (upstream fp8 cache-write bug),
- the output-token trim `elif`→`if` + drafter/PP-rank guards in
  `gpu_model_runner.py` (correctness under PP + async spec decode),
- MLA guards on two speculative-config auto-defaults in `arg_utils.py`,
- the Eagle/MTP + PP>1 verification relaxation in `speculative.py`,
- the GLM tool-parser zero-argument streaming fix,
- the AWQ MoE weight-conversion memory rewrite (same numerics, lower peak)
  and `not ep_active` guards on the legacy fast paths (OOB-read fix).

Known cosmetic issue shipped as-is: `VLLM_SM70_MTP_DRAFTER_FULL_GRAPH` has a
`TYPE_CHECKING` hint of `True` in `envs.py` but its registered lambda
defaults to `"0"` — the effective default is **False** (which matches the
measured recommendation; the feature is perf-neutral). Similarly, four
C++-side flags (`VLLM_SM70_AWQ_MOE_TUNE_PREFILL_SLOTS`,
`VLLM_SM70_IDX_WMMA_TILE`, `VLLM_SM70_SPARSE_PREFILL_V2`,
`VLLM_SM70_SPARSE_PREFILL_WS`) and two Python debug flags
(`VLLM_SM70_SPARSE_DEBUG`, `VLLM_SM70_PP_DEBUG_INPUTS`) are read via raw
`getenv`/`os.environ` and do not appear in `vllm/envs.py` (they are
consequently NOT forwarded to Ray workers by vLLM itself — the serve script
exports them in the environment Ray inherits).

---

## New files

### csrc/attention/sm70_mla_decode.cu

Single translation unit implementing the full SM70 DeepSeek-V3.2/GLM DSA
sparse MLA-MQA attention + lightning-indexer pipeline on vLLM's paged latent
cache; 17 `__global__` kernels in 6 families, plus fp8-e4m3→fp16
no-hardware-cvt helpers (`sm70_fp8_to_half_div256`,
`sm70_fp8x2_to_half2_div256`), paged loaders (`mla_load4_half`, `dsmla_load4`
for the 656-byte `fp8_ds_mla` row, `idxk_fp8_load4` for the 132-byte
indexer-K row), and `__constant__` tables (`c_idx_accrow`/`c_idx_acccol`)
encoding the empirically-derived Volta m16n16k16 f32 WMMA accumulator layout
(verified at runtime by a probe kernel). Fixed shapes: `D_K=576` (512 NoPE +
64 RoPE), `D_V=512`, indexer `HD=128`, `n_ihead=32`. Python callers:
`vllm/_sm70_ops.py`, `triton_mla.py`, `sparse_attn_indexer.py`.

Kernel families:

- **DSA indexer logits** — `score[q,p] = sum_h relu(q_h·k_p) * w[q,h]` over
  paged fp16/fp8 indexer-K, replacing DeepGEMM `fp8_mqa_logits`; optional
  fused level-1 top-k histogram epilogue.
  - `sm70_indexer_logits_kernel<HD,N_IHEAD,PPB,K_FP8>`: decode/scalar path,
    one warp per KV position, lane == head; Q in registers, depth-2 K
    prefetch, interleaved half2 accumulators.
  - `sm70_indexer_logits_wmma_kernel<HD,N_IHEAD,QT,KT,WARPS,K_FP8>`:
    FlashAttention-style WMMA prefill; K gathered once per tile, reused
    across QT queries and all 32 heads.
  - `sm70_indexer_logits_packed_kernel<K_FP8>` (fixed QT=32/KT=32/WARPS=4):
    fast prefill (~5-6x, 32-44 TFLOPS) consuming caller-pre-packed Q
    `[32, ceil(B/16), 8, 16, 16]`; relu·weight + cross-head accumulation
    fully in registers via the derived accumulator layout; bit-identical to
    the WMMA kernel.
  - `sm70_idx_layout_probe_kernel`: one-warp on-device self-check of the
    accumulator-layout tables at first packed launch. Do not remove.
  - Launchers: `sm70_indexer_logits` (B>=8 → WMMA else scalar;
    `VLLM_SM70_IDX_WMMA_TILE` 0-4 selects tiling; fused `hist` decode-only)
    and `sm70_indexer_logits_packed`.
- **Indexer-K store** — `sm70_indexer_k_store_kernel` (fp16 copy) and
  `sm70_indexer_k_store_fp8_kernel` (132-byte row: 128 e4m3 + fp32 per-token
  scale = amax/448); CUDA-graph-safe, `slot_mapping < 0` skipped on-device.
- **Dense causal MLA decode** — `sm70_mla_decode_kernel<D_K,D_V,HB,TN,WARPS>`:
  gather-once WMMA flash-decode; one block per (token, 32-head group);
  fp16 cache only. The reference/stepping-stone kernel.
- **Sparse (top-k) MLA decode + merge** — attends only indexer-selected
  `idx[B,topk]` positions, O(topk) context-flat; fp16 and `fp8_ds_mla`.
  - `sm70_sparse_mla_decode_kernel<D_K,D_V,TN,WARPS,KV_FP8>`: grid
    `(B, n_head/16, n_splits)` with a kv-split z-axis for occupancy (B=1
    would otherwise put ONE block on 80 SMs); per-split partial
    online-softmax `(P_O, P_ML)` to scratch.
  - `sm70_sparse_mla_merge_kernel<D_V>`: flash-decode reduction of split
    partials.
  - `sm70_sparse_mla_prefill_kernel<KV_FP8>` (512 threads,
    TN=32/WARPS=16/KSPLIT=4): register-accumulator prefill rewrite (~2.5x;
    fp32 PV accumulators, double-buffered 16 B gathers, pre-packed+scaled Q).
  - `sm70_sparse_mla_prefill_ws_kernel<KV_FP8>` (768 threads, 16 producer +
    8 consumer warps): warp-specialized variant using named-barrier
    ping-pong (`sm70_bar_sync`/`sm70_bar_arrive` PTX). +43% isolated.
  - `sm70_sparse_prefill_pack_q`: packs/pre-scales Q into fragment tiles.
  - Launcher `sm70_sparse_mla_decode`: prefill fast path when
    `B>=32 && n_head<=16` (`VLLM_SM70_SPARSE_PREFILL_V2` default 1;
    `VLLM_SM70_SPARSE_PREFILL_WS` default 0 in csrc — the serve script turns
    it on), else split-decode + merge with auto `n_splits` (~2x SM count).
- **Fused decode top-k** — `sm70_decode_topk_kernel` (+
  `topk_find_threshold<NBIN>`): co-resident grid synchronized via global
  arrival counters; reuses the 4096-bin histogram from the indexer epilogue,
  suffix-scan to a threshold bin + compaction, second-level refinement,
  deterministic position tie-break, -1 padding.
- **TP indexer shard-merge** — exact global top-K (K=2048, tp in [2,4]) from
  per-rank shards: `sm70_indexer_shard_merge_kernel<K>` (register-resident
  MSB-first byte-radix over 64-bit `(monotonic-score<<32)|inverted-column`
  keys; NCCL all-gather input) and `sm70_indexer_shard_merge_p2p_kernel<K>`
  (reads peers' IPC-mapped slots over NVLink; double-buffered `slot = seq&1`
  staged/merged flag protocol; helpers `sm70_ipc_flag_set_kernel` /
  `sm70_ipc_slot_wait_kernel`). Host ops: `sm70_indexer_shard_merge`,
  `sm70_indexer_shard_merge_p2p`, `sm70_ipc_alloc/open/flag_set/slot_wait`.

### csrc/attention/sm70_attn_fp8.cu

- Two kernels for fp8-e4m3 weight-only attention/indexer projection GEMMs on
  V100, using the `value/256` no-hardware-cvt bit trick (e4m3 bits shifted
  left 7 read as an fp16 with value/256 exactly; the 256 folds into the block
  scale). Weights: row-major fp8 `[N,K]` + fp32 `[ceil(N/128),ceil(K/128)]`
  block `scale_inv` — the official GLM/DeepSeek FP8 checkpoint layout.
- `sm70_attn_fp8_gemv_kernel<M>` (M=1..8, 256 threads): decode GEMV
  `y[M,N] = x[M,K] @ W^T`; one warp per output row, X staged in smem,
  8-wide weight loads with hfma2, warp-shuffle reduction (~630 GB/s at M=1,
  ~1.94x cuBLAS fp16).
- `sm70_attn_fp8_dequant_kernel`: bulk W→fp16 scratch dequant for the cuBLAS
  prefill/large-M path.
- Host entry points `sm70_attn_fp8_gemv` / `sm70_attn_fp8_dequant`; invoked
  from `vllm/model_executor/layers/linear.py`. Constraints: K % 16 == 0,
  N % 8 == 0; e4m3 NaN weights assumed absent.

### vllm/v1/attention/backends/mla/prefill/sm70_triton.py

- Dependency-free MLA prefill backend for Volta, replacing the
  `vllm_flash_attn`-based FLASH_ATTN backend that SM70 builds cannot ship.
  Registered as `MLAPrefillBackendEnum.SM70_TRITON`, selected for major==7.
- Core `_varlen_attention(...)`: ragged multi-head attention per sequence in
  fp32 with a streaming (flash-style) online softmax over K/V tiles of
  `VLLM_SM70_MLA_PREFILL_KV_BLOCK` length (default 512) — peak scratch
  O(H·q_len·BLOCK) instead of O(H·q_len·k_len). Exact dense output + LSE.
- Interface: `run_prefill_new_tokens(...)` (causal) and
  `run_prefill_context_chunk(...)` (non-causal, returns LSE for merge).
- This is the correctness-first DENSE fallback; the sparse prefill lives in
  `triton_mla.py::_forward_mha_sparse`.

### replay_prompt.py (shipped in this release as tools/replay_prompt.py)

- Standalone stdlib-only CLI that loads a prompt-dump JSON (written under
  `VLLM_DEBUG_DUMP_PROMPTS_DIR` by `_dump_debug_prompt`) and inspects or
  replays it against a running server. Modes: `chat` (verbatim replay through
  the full template + tool-parser pipeline), `tokens` (token-exact raw-model
  replay via `/v1/completions`), `text`. `--set key=val` overrides,
  `--stream`, `--show`, `--show-prompt`.

---

## Modified files

### CMakeLists.txt
- Adds the two new CUDA sources to `VLLM_SM70_TURBOMIND_SRCS` (inside the
  existing `if (SM70_TURBOMIND_ARCHS)` block; non-SM70 builds untouched).

### csrc/libtorch_stable/cache_kernels.cu  *(unconditional bug fix)*
- Adds `kv_elem_to_float<scalar_t>()`: for `uint16_t` (fp16 carried as raw
  bits under `DISPATCH_BY_KV_CACHE_DTYPE`) calls `half_to_float(v)`, else
  `static_cast<float>(v)`.
- Fixes the amax reductions in `concat_and_cache_ds_mla_kernel` and
  `indexer_k_quant_and_cache_kernel`: `fabsf(float(v))` on raw fp16 bits
  interpreted the bit pattern as an integer (fp16 1.0 → 15360.0), corrupting
  scales and zeroing small e4m3 payloads. bf16/Hopper never hit it. **Worth
  upstreaming.**

### csrc/ops.h
- Declares: `sm70_mla_decode`, `sm70_attn_fp8_gemv`, `sm70_attn_fp8_dequant`,
  `sm70_sparse_mla_decode`, `sm70_indexer_shard_merge`,
  `sm70_indexer_shard_merge_p2p`, `sm70_ipc_{alloc,open,flag_set,slot_wait}`,
  `sm70_indexer_logits`, `sm70_indexer_logits_packed`, `sm70_indexer_k_store`,
  `sm70_decode_topk`.
- Extends two existing MoE signatures with defaulted trailing optionals for
  EP: `awq_moe_single_token_indexed_dense_w13_sm70_out(...,
  std::optional<Tensor> expert_map)` and
  `awq_moe_single_token_weighted_reduce_out(..., std::optional<Tensor>
  valid_count)`.

### csrc/sm70_turbomind/ops/awq_sm70_gemm.cu
- Env-gated prefill autotune shape `moe_tune_prefill_slots()`
  (`VLLM_SM70_AWQ_MOE_TUNE_PREFILL_SLOTS`, default 0 = off): the tuner covers
  the exact full-chunk grouped-GEMM shape in addition to the decode window —
  exact-match only, so tail chunks never trigger mid-request ~9 s kMeasure
  sweeps.
- `awq_moe_single_token_prepare_kernel` + reduce kernels extended for EP:
  optional `expert_map` (global→local remap on device, non-local slots →
  sentinel zero-row groups) and `valid_count` (skip never-written rows —
  0·inf=NaN hazard). Routing-metadata emission parallelized.
- CAVEAT: `awq_moe_single_token_indexed_dense_w13_sm70_out` flips
  `vectorized_prepare` from `false` to `true` unconditionally (half2
  compact-input copy; assumes even hidden size).

### csrc/torch_bindings.cpp
- Registers all the new ops (schema + impl) inside the existing
  `#ifdef ENABLE_SM70_TURBOMIND` block; `sm70_ipc_open`/`sm70_ipc_slot_wait`
  on kCPU, the rest kCUDA. MoE schema updates use `Tensor? ...=None`.

### tests/tool_parsers/test_glm47_moe_tool_parser.py
- Strengthens the zero-argument streaming test: asserts the name delta list
  equals `["get_current_date"]` and arguments parse as `{}` (the old test
  passed vacuously).

### vllm/_sm70_ops.py
- Python wrappers + `@register_fake("_C::...")` stubs for the new ops;
  `expert_map`/`valid_count` optionals threaded through the existing MoE
  wrappers. Every fake registration is guarded by
  `hasattr(torch.ops._C, name)` so non-SM70 builds import cleanly.
- Note: the IPC/P2P ops and `sm70_indexer_shard_merge*` have NO wrappers here
  — they are called directly as `torch.ops._C.*` from
  `sparse_attn_indexer.py`.

### vllm/config/speculative.py
- For Eagle/MTP drafts with `pipeline_parallel_size > 1`, draft-model
  verification runs against a temporarily-forced PP=1 view of the draft
  parallel config (restored in `finally`) — Eagle/MTP drafts are never
  pipeline-split (the whole draft runs on the last rank), so the draft need
  not implement `SupportsPP`. TP checks intact. Not SM70-specific.

### vllm/engine/arg_utils.py
- Adds `and not model_config.use_mla` guards to two speculative-config
  auto-defaults: `use_local_argmax_reduction` (requires a drafter
  `get_top_tokens()` that DeepSeekMTP lacks) and forcing
  `attention_backend="TRITON_ATTN"` (cannot run MLA layers). Explicit
  user-provided keys still respected.

### vllm/entrypoints/openai/{chat_completion,completion}/serving.py
- One `self._dump_debug_prompt(...)` call each after `engine_input` is built.
  No-op unless `VLLM_DEBUG_DUMP_PROMPTS_DIR` is set.

### vllm/entrypoints/openai/engine/serving.py
- Adds `OpenAIServing._dump_debug_prompt(...)`: writes one JSON per request
  (original API body, rendered prompt, token ids, resolved params) with an
  atomic tmp+rename, wrapped in try/except so it can never fail a request.
  This JSON schema is the contract consumed by `replay_prompt.py`.

### vllm/envs.py
- 25 new env registrations, no existing variable changed. See
  `docs/ENV_FLAGS.md` for the full annotated table. Remember: a flag must be
  registered here to be forwarded to Ray workers.

### vllm/model_executor/layers/attention/mla_attention.py
- `VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_TOKENS` override of the 64k workspace
  token cap in `MLACommonMetadataBuilder.__init__` (<=0 → upstream constant,
  byte-for-byte). The workspace comes straight out of the KV budget
  (~896 MiB at 64k/TP4) and the SM70 sparse prefill doesn't use it.

### vllm/model_executor/layers/linear.py
- The SM70 fp8-e4m3 weight-only storage path (`VLLM_SM70_ATTN_FP8`,
  `VLLM_SM70_MLP_FP8`): `_maybe_sm70_attn_fp8_prepare` (called first in
  `UnquantizedLinearMethod.process_weights_after_loading`) [128,128]-block
  quantizes allowlisted fp16 weights to uint8 e4m3 + fp32 scale_inv and
  replaces the parameter; `_maybe_sm70_attn_fp8_forward` (checked first in
  `.apply`) routes M<=max_m to `sm70_attn_fp8_gemv`, larger M to dequant into
  a shared per-device fp16 scratch + cuBLAS. Default allowlist:
  fused_qkv_a_proj, q_b_proj, o_proj, wq_b (+ shared/dense gate_up_proj,
  down_proj under MLP_FP8). Guards: op present, fp16/CUDA/2D, capability
  exactly (7,0), N%8==0 && K%16==0.

### vllm/model_executor/layers/mla.py
- `MultiHeadLatentAttentionWrapper._backend_use_sparse()`: on SM70 with
  `VLLM_SM70_GLM_SPARSE_MLA`, backend selection sees `use_sparse=False` so a
  dense TRITON_MLA backend is chosen (the upstream sparse backends are
  Hopper+-only) — but the indexer still runs and fills
  `topk_indices_buffer`; the sparse attention itself happens inside
  `TritonMLAImpl`. Only backend choice is overridden.

### vllm/model_executor/layers/quantization/awq_sm70_moe.py
- EP-safe single-token decode route (`VLLM_SM70_AWQ_MOE_EP_SINGLE_TOKEN`,
  default on): w13 dense (with `expert_map=`) → silu-mul → stage op →
  weighted reduce (with `valid_count=`).
- `not ep_active` guards on ALL legacy single-token fast paths — they index
  the per-expert pointer table with raw GLOBAL topk_ids and OOB-read under EP
  (illegal memory access). *(unconditional correctness fix for EP runs)*
- TurboMind weight-conversion rewrite: preallocated stacked tensors instead
  of per-expert list + `torch.stack`; frees + `empty_cache()` moved before
  the pointer-table build. Reason: the MTP draft layer converts last on ranks
  already holding the full target — the ~2x double-buffer exhausted VRAM and
  an unchecked `cudaMallocAsync` in `MakeStridedPtrs` returned null and
  poisoned the CUDA context. Same numerics.

### vllm/model_executor/layers/sparse_attn_indexer.py
- ~716 added lines; upstream `forward_cuda` untouched. New custom op
  `sm70_sparse_indexer` (via `direct_register_custom_op`,
  `mutates_args=["topk_indices_buffer"]`): the fp16 Volta indexer.
- Decode branch: `sm70_indexer_logits` + fused histogram top-k
  (`VLLM_SM70_FUSED_DECODE_TOPK`; launches chunked below 8 rows because the
  B>=8 WMMA dispatch rejects the histogram arg) or `persistent_topk`
  fallback. Always reserves the radix workspace even on the fused path — a
  capture-time "Workspace is locked" deadlock otherwise.
- Prefill branch (`VLLM_SM70_GLM_SPARSE_PREFILL`): per-token causal top-k;
  true causal length `ke = cu_seqlen_ke - cu_seqlen_ks` (the cross-request
  corruption fix); authoritative request boundaries from
  `chunk.query_start_loc_cpu`; per-request-segment launches; 128-row
  sub-chunking with KT-aligned causal bounds; causal-max memo (~312 → 4 host
  syncs per chunk). Packed-Q path when n>=8 (`VLLM_SM70_IDX_PACKED_WMMA`).
- TP key-sharding (`VLLM_SM70_IDXER_TP_SHARD_PREFILL`, min ctx
  `VLLM_SM70_IDXER_TP_SHARD_MIN_CTX`): each rank scores 1/tp of keys + local
  top-k, exact global merge via NCCL all-gather + torch.topk, the fused merge
  kernel (`VLLM_SM70_IDXER_FUSED_MERGE`), or NVLink P2P
  (`VLLM_SM70_IDXER_P2P_MERGE`; IPC handle exchange via
  `dist.all_gather_object`, permanent NCCL fallback on any init failure).
- Bridge to attention: module global `_SM70_PREFILL_META["chunks"]` stashes
  per-chunk `(tok_lo, n_q, ke)` consumed by
  `TritonMLAImpl._forward_mha_sparse`.

### vllm/model_executor/models/config.py
- SM70 branches in `DeepseekV32ForCausalLM.verify_and_update_config`: dense
  fallback → early return; sparse + `VLLM_SM70_MLA_KV_FP8` → requires
  `VLLM_SM70_GLM_SPARSE_PREFILL` (else ValueError) and sets
  `cache_config.cache_dtype = "fp8_ds_mla"`.
- Registers `MODELS_CONFIG_MAP["GlmMoeDsaForCausalLM"] =
  DeepseekV32ForCausalLM`.

### vllm/model_executor/models/deepseek_mtp.py
- `is_v32` uses `_config_is_dsa(config)` (respects the SM70 dense fallback);
  `load_weights` skips indexer weights when not DSA and loads the draft's
  `embed_tokens` from the target's top-level embedding (GLM MTP checkpoints
  ship none; under PP>1 the target's embedding lives on stage 0 while the
  draft runs on the last stage).

### vllm/model_executor/models/deepseek_v2.py
- SM70 gate helpers: `_sm70_dense_mla_fallback()`, `_sm70_glm_sparse_mla()`,
  `_config_is_dsa(config)` — imported by `models/config.py` and
  `deepseek_mtp.py`.
- `_fp16_exact_stream_scale(config)` (`VLLM_DEEPSEEK_FP16_EXACT_STREAM_SCALE`,
  "auto" = on only for `model_type == "glm_moe_dsa"`): skips BOTH fp16
  1/routed_scaling residual-stream scalings and folds routed scaling into the
  routing weights instead (exact math; the upstream trick breaks RMSNorm
  scale-invariance via eps for GLM's tiny chat-marker embedding rows —
  the fp16 output-quality root cause). DeepSeek-V2 (rsf=16, real overflow
  risk) keeps upstream behavior.
- `Indexer.__init__`/`.forward` SM70 sparse branches: fp16 (or fp8 inline)
  indexer-K cache, rope + k_norm + pre-scaled weights, delegate to
  `sm70_sparse_indexer(...)`.
- All `is_v32 = hasattr(config, "index_topk")` sites → `_config_is_dsa()`;
  `load_weights` skips `.indexer.*` when not DSA.

### vllm/tool_parsers/glm4_moe_tool_parser.py
- `_extract_tool_name_from_region(..., is_complete=False)`: when the region
  is complete and has no `\n`/`<arg_key>` delimiters, the whole inner text is
  the tool name — zero-argument streamed tool calls were silently dropped
  before. *(unconditional correctness fix, tested)*

### vllm/v1/attention/backends/mla/indexer.py
- `DeepseekV32IndexerPrefillChunkMetadata.query_start_loc_cpu`: authoritative
  per-request query offsets for the SM70 sparse prefill (replaces a
  "ke decreases" boundary heuristic that missed short-after-long).
- The `reorder_batch_threshold += num_speculative_tokens` add is now applied
  when `sm70_mla_spec_as_decode(...)` on SM70 (mirroring
  `TritonMLAMetadataBuilder` so both builders classify verify batches
  identically); non-SM70 keeps the upstream unconditional add.
- `next_n > 1` on SM70 forces `use_flattening = True` (the SM70 indexer
  kernels are next_n=1-only).

### vllm/v1/attention/backends/mla/prefill/{registry,selector}.py
- Registers `SM70_TRITON` and selects it (alone) for capability major == 7.

### vllm/v1/attention/backends/mla/triton_mla.py
- `sm70_mla_spec_as_decode(vllm_config)`: SM70 + sparse + spec config +
  `fp8_ds_mla` → verify steps of uniform qlen `1+num_spec` are treated as
  DECODE, so FULL_DECODE_ONLY CUDA graphs cover them. Builder advertises
  `QueryLenSupport.UNIFORM` in that mode.
- `TritonMLABackend`: `fp8_ds_mla` support + its 656 B/token
  `get_kv_cache_shape`.
- `TritonMLAImpl`: `_forward_mqa_sparse` (sparse top-k decode via
  `sm70_sparse_mla_decode`, per-token row expansion at static shapes for
  spec-as-decode), `forward_mha` override + `_forward_mha_sparse` (sparse
  PREFILL: q_nope→latent via cached `W_UK_T`, per-token causal seq_lens from
  the `_SM70_PREFILL_META` bridge, then `W_UV` up-projection; errors fall
  back to dense EXCEPT on a uint8 fp8 cache, where dense would misread bytes
  → re-raise loudly), `_forward_mqa_torch` (CUDA-graph-safe plain-torch
  decode), and the `VLLM_SM70_MLA_DECODE_KV_SPLITS` override of the
  capture-frozen split count.

### vllm/v1/attention/ops/triton_decode_attention.py
- `_is_sm70` + one branch in `_decode_grouped_att_m_fwd`: BLOCK=16 /
  num_stages=2 on SM70+MLA (BLOCK_N=32 overflows Volta's 96 KB smem opt-in).

### vllm/v1/core/sched/scheduler.py
- PP interleaving (`VLLM_SM70_PP_INTERLEAVE_GROUPS`, default 0 = inert):
  decodes scheduled in ~N independent waves (`interleave_cap =
  ceil(num_running_decodes / N)`; in-flight requests — `num_output_placeholders
  > 0` — are never rescheduled), so the PP batch queue keeps N independent
  steps flowing. Prefills unaffected.

### vllm/v1/engine/core.py
- With interleaving + spec decode: backfills `request.spec_token_ids` from
  `model_output.draft_token_ids` after `update_from_output` (a
  `take_draft_token_ids` RPC would block ~57 ms behind the busy worker).

### vllm/v1/outputs.py
- New optional `ModelRunnerOutput.draft_token_ids` field (default None):
  carries draft ids worker→engine without a blocking RPC.

### vllm/v1/spec_decode/llm_base_proposer.py
- Drafter FULL_DECODE_ONLY graph mode when method==mtp + MLA + SM70 +
  `VLLM_SM70_MTP_DRAFTER_FULL_GRAPH` (measured perf-neutral; scaffolding).

### vllm/v1/worker/gpu_model_runner.py
- *(unconditional correctness fix)* The async-spec output-token trim changes
  `elif` → `if` so it runs on **every** PP rank — non-last ranks otherwise
  accumulate +1 token per rejected draft until verify inputs become
  placeholders (visible as repetition collapse).
- Side-stream PP broadcast machinery for async spec: `_get_pp_bcast_stream`,
  `_pp_wait_bcast_event` (lazy, consumer-side), `_pp_broadcast_draft_token_ids`
  (draft values exist only on the last rank's GPU), rewritten
  `_pp_broadcast_prev_sampled_token_ids` broadcasting raw
  `[num_reqs, num_spec+1]` sampler rows with -1 rejects; receivers derive
  resolved tokens locally; `record_stream` lifetime guards. All on a
  dedicated CUDA stream — main-stream collectives deadlock, and so does a
  device-wide `torch.cuda.synchronize()`.
- Robustness: `getattr(self, "drafter", None)` + `is_last_rank` guards
  (drafter exists only on the last rank); empty-batch guards in
  `_propose_draft_token_ids`; `_dummy_run`/`profile_run` branch on
  `isinstance(hidden_states, torch.Tensor)` vs `IntermediateTensors`.
- Gated additions: `output.draft_token_ids = take_draft_token_ids()` under
  interleaving; drafter `CUDAGraphWrapper` under the drafter-graph flag;
  `VLLM_SM70_PP_DEBUG_INPUTS` per-rank input logging (`.tolist()` only).
