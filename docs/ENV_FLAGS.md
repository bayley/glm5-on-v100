# Environment flag reference

Engine defaults are what `vllm/envs.py` ships after applying the patch;
"serve" is what `scripts/serve_glm.sh` sets (the validated production
config). Flags marked *(csrc)* are read by `getenv` inside the CUDA code.

## Sparse / DSA path

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_GLM_DENSE_MLA` | 1 | 1 | Run DSA models as exact dense MLA on SM70 (bypass indexer) |
| `VLLM_SM70_GLM_SPARSE_MLA` | 0 | 1 | Volta-native DSA sparse MLA decode (fp16 indexer + WMMA sparse kernel) |
| `VLLM_SM70_GLM_SPARSE_PREFILL` | 0 | 1 | Sparse (top-k) prefill attention; requires SPARSE_MLA |
| `VLLM_SM70_SPARSE_DECODE_MIN_CTX` | 2048 | 2048 | Min max_seq_len to route decode to the sparse kernel (static at graph capture) |
| `VLLM_SM70_SPARSE_PREFILL_V2` *(csrc)* | 1 | 1 | Register-accumulator sparse prefill kernel for B>=32; 0 = old kernel |
| `VLLM_SM70_SPARSE_PREFILL_WS` *(csrc)* | 0 | 1 | Warp-specialized (16 producer + 8 consumer warps) sparse prefill kernel |
| `VLLM_SM70_FUSED_DECODE_TOPK` | 1 | 1 | Fused histogram decode top-k instead of persistent_topk |
| `VLLM_SM70_IDX_WMMA_TILE` *(csrc)* | 0 | 2 | Staged WMMA prefill-indexer tile variant (2 = QT32/KT64) |
| `VLLM_SM70_IDX_PACKED_WMMA` | 1 | 1 | Packed-Q register-accumulator prefill indexer (0 = staged WMMA) |
| `VLLM_SM70_IDXER_TP_SHARD_PREFILL` | 0 | 1 | TP key-sharded indexer scoring for prefill (exact global top-k at 1/TP cost) |
| `VLLM_SM70_IDXER_TP_SHARD_MIN_CTX` | 16384 | 16384 | Causal-reach threshold for the sharded path |
| `VLLM_SM70_IDXER_FUSED_MERGE` | 1 | 1 | Single-kernel shard-merge epilogue (0 = torch.topk epilogue) |
| `VLLM_SM70_IDXER_P2P_MERGE` | 1 | 1 | IPC/NVLink direct shard exchange instead of NCCL all-gather (NCCL fallback on init failure) |

## Dense MLA path (reference / fallback)

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_MLA_DECODE_KV_SPLITS` | 16 | 16 | SM70 override of the graph-frozen num_kv_splits in Triton MLA decode (0 = upstream heuristic) |
| `VLLM_SM70_MLA_DECODE_TORCH` | 0 | 0 | torch-bmm MQA decode; ~2x faster than the Triton kernel above ~2k ctx on the dense path |
| `VLLM_SM70_MLA_PREFILL_KV_BLOCK` | 512 | 128 | K/V tile length of the streaming-softmax dense prefill (memory vs speed) |

## CUDA graphs

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE` | 0 | 1 | FULL_DECODE_ONLY CUDA graphs without torch.compile/Inductor |
| `VLLM_SM70_FLASH_V100_DECODE_GRAPH_CAPTURE_SIZE` | 1 | max_num_seqs | Largest decode batch captured (must be >= max_num_seqs) |

## Weight / cache quantization (load-time e4m3; checkpoint untouched)

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_ATTN_FP8` | 0 | 1 | fp8-e4m3 weight-only storage of the fp16 attention/indexer projections |
| `VLLM_SM70_ATTN_FP8_ALLOWLIST` | default set | — | Which projections to quantize (fused_qkv_a_proj, q_b_proj, o_proj, wq_b) |
| `VLLM_SM70_ATTN_FP8_GEMV_MAX_M` | 8 | 8 | M threshold: custom fp8 GEMV vs dequant+cuBLAS |
| `VLLM_SM70_MLP_FP8` | 0 | 1 | Extend fp8 storage to shared-expert + dense-MLP weights |
| `VLLM_SM70_MLA_KV_FP8` | 0 | 1 | fp8_ds_mla 656 B/token latent KV cache (requires the sparse path) |
| `VLLM_SM70_INDEXER_K_FP8` | 0 | 0 (1 in MTP mode) | fp8 132 B/token indexer-K cache: +9-15% KV pool, ~-3% decode scan |

## MoE / GEMM

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_AWQ_MOE_EP_SINGLE_TOKEN` | 1 | 1 | EP-safe single-token MoE decode route (device-side expert_map remap) |
| `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` | 0 | 1 | Warmup measure-tuning of M<=8 TurboMind GEMM dispatch (MTP verify shapes) |
| `VLLM_SM70_AWQ_MOE_TUNE_PREFILL_SLOTS` | unset | mnbt*8 | One EXACT extra MoE GEMM tune shape for full prefill chunks (do NOT widen the tune range instead — see GOTCHAS) |
| `VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS` | 128 | 128 | Decode tuning window |
| `VLLM_SM70_MOE_SINGLE_TOKEN_*_FASTPATH` (5 flags) | 0 | 1 iff EP=0 | Legacy TP-only single-token fast paths (NOT EP-safe) |

## MTP / pipeline parallelism

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_SM70_MTP_DRAFTER_FULL_GRAPH` | 0 | 0 | Drafter FULL-graph scaffolding (measured perf-neutral; the MTP tail is ~2.5 ms) |
| `VLLM_SM70_PP_INTERLEAVE_GROUPS` | 0 | 2 | Independent-batch PP decode interleaving wave count (0 = lockstep) |
| `VLLM_PP_LAYER_PARTITION` (upstream env) | unset | 21,19,19,19 / 22,19,19,18 (MTP) | Byte-balanced PP layer split |

## Memory / infra / debug

| Flag | Engine | Serve | Purpose |
| --- | --- | --- | --- |
| `VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_TOKENS` | 0 (= upstream 64k) | 16384 | Cap the chunked-prefill up-projection workspace (~0.65 GiB/GPU reclaimed) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 1 | 1 | Keep ON — disabling reclaims nothing (documented dead end) |
| `VLLM_DEEPSEEK_FP16_EXACT_STREAM_SCALE` | auto | auto | Exact residual-stream scaling (fp16 quality fix; auto-on for GLM DSA) |
| `VLLM_DEBUG_DUMP_PROMPTS_DIR` | "" | via GLM_DUMP_PROMPTS=1 | Per-request prompt capture for tools/replay_prompt.py |
| `VLLM_DEBUG_MTP_LOAD` | 0 | 1 (MTP) | Log draft-layer weight loading |
| `VLLM_SM70_PP_DEBUG_INPUTS` | 0 | 0 | Per-rank input dumps for PP/spec bisects |
| `VLLM_SM70_MTP_DUMP_STEP_DIR` / `_MAX` / `_STEPS` | unset | unset | Last-rank draft/verify step dumps |
| `VLLM_SM70_SPARSE_DEBUG` | unset | unset | Multi-request indexer instrumentation |

`scripts/serve_glm.sh` exposes short convenience knobs (`MTP`, `SPARSE`,
`KV_FP8`, `ATTN_FP8`, `MLP_FP8`, `EP`, `TUNE`, `INTERLEAVE`, `EAGER`, ...)
that map onto these; the script's comments carry the A/B measurement behind
each default.
