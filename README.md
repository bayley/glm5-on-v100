# glm5-on-v100 — GLM-5.1 (DSA sparse-MLA MoE) on 16x Tesla V100

This is a **mini-release** of the custom CUDA kernels, vLLM integration work,
and hard-won operational knowledge from getting **GLM-5.1-AWQ** — a
DeepSeek-V3.2-style **MLA + DSA sparse-attention MoE** model
(`GlmMoeDsaForCausalLM`, ~420 GB AWQ 4-bit, 78 layers, 256 experts) — serving
well on **two 8x V100-32GB nodes** (TP=4 x PP=4 over Ray).

Volta (SM70) has no DeepGEMM, no FlashMLA, no fp8 tensor cores, no
FlashAttention-2 wheels, and a 96 KB shared-memory ceiling. Everything the
modern DSA stack assumes is missing. This release contains the SM70-native
replacements and the plumbing that wires them into vLLM.

It is **not a runnable fork by itself**. The base it applies to is
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) `v1.2.1`
(commit `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`) — an SM70/Volta-focused
vLLM fork that provides the TurboMind-derived AWQ kernels and V100 CUDA-graph
machinery we build on. See [docs/INTEGRATION.md](docs/INTEGRATION.md) for
step-by-step instructions (written so that a coding agent such as Claude Code
can perform the integration).

## Headline results

Measured on 16x V100-32GB (two SXM2 nodes, InfiniBand, TP4xPP4, all release
defaults: sparse MLA + sparse prefill, fp8 weight storage for attention/MLP
projections, fp8 latent KV, decode CUDA graphs, MTP-1 speculative decoding,
PP interleaving):

| Metric | Result |
| --- | --- |
| Decode, single stream @2k ctx | **36-37 tok/s** (26.5 without MTP) |
| Decode @180k ctx | 21.4 tok/s (context-flat sparse attention) |
| Aggregate decode, 4 streams @2k | **103 tok/s** |
| Aggregate decode, 4 streams @24k | 87 tok/s |
| Cold prefill @64k | **~2,500 tok/s** |
| Cold prefill @180k | 1,881 tok/s (~context-flat) |
| Max serving context | 183k tokens (MTP config) / 198k (plain) |
| Long-context retrieval | needle tests PASS at 96k / 180k / 200k |

For reference, the starting point on the same hardware was **~6 tok/s decode
and ~343 tok/s prefill at 2k context**, with 8k+ contexts OOM-ing outright.
The kernel-level wins behind the table: the DSA indexer prefill kernel went
from 6.3 to 44 TFLOPS (packed-Q register-accumulator WMMA), the sparse-attn
prefill kernel from 4.45 to 15.7 TFLOPS (warp specialization), and sparse
decode attention is O(topk)=O(2048) per token instead of O(context).

## What's in this release

```
csrc/attention/sm70_mla_decode.cu   The SM70 DSA kernel suite (~2,900 lines):
                                    WMMA sparse MLA decode (split-topk,
                                    flash-decode merge), fp16 indexer logits
                                    (GEMV decode + staged/packed WMMA prefill
                                    variants), register-accumulator +
                                    warp-specialized sparse prefill, fused
                                    indexer shard-merge, IPC P2P shard
                                    exchange, indexer-K cache scatter.
csrc/attention/sm70_attn_fp8.cu     fp8-e4m3 weight-only GEMV + dequant for
                                    the fp16 attention/MLP projections.
vllm/v1/attention/backends/mla/prefill/sm70_triton.py
                                    Dependency-free dense-MLA varlen prefill
                                    (streaming online-softmax, O(q_len*BLOCK)
                                    scratch) — the exact-reference fallback.
patches/glm5-v100-sm70.patch        THE integration artifact: the complete
                                    code diff vs upstream v1.2.1 (kernels,
                                    Python plumbing, env flags, MTP/PP work,
                                    bug fixes). Applies clean with git apply.
patches/MANIFEST.md                 Per-file description of every hunk in the
                                    patch — read this before applying.
scripts/serve_glm.sh                The tuned production launch script, with
                                    every default annotated with the A/B
                                    measurement that justified it.
scripts/start_ray.sh, stop_glm.sh   2-node Ray cluster bring-up/teardown with
                                    the footguns designed out.
tools/replay_prompt.py              Capture/replay tooling: re-run any client
                                    request verbatim or token-exact against
                                    the server (how we bisect quality/parser
                                    bugs in minutes).
docs/INTEGRATION.md                 How to apply this to upstream 1Cat-vLLM,
                                    build, and validate — agent-oriented.
docs/GOTCHAS.md                     The distilled engineering log: every trap
                                    we hit (driver bugs, CUDA-graph capture
                                    semantics, profiler lies, Ray footguns,
                                    memory ceilings, numerics).
```

The two `.cu` files and `sm70_triton.py` are also inside the patch; they are
additionally provided as plain files because they are self-contained and
useful to read (or port elsewhere) on their own.

## The feature list (short form)

All SM70-gated and env-flag-controlled; upstream/non-Volta behavior is
preserved. Defaults below are the ones `scripts/serve_glm.sh` ships.

**Attention / DSA sparse path**
- Volta-native **DSA sparse MLA decode**: WMMA head-tile kernel over the paged
  latent KV cache, split-topk across SMs with a flash-decode online-softmax
  merge. Decode attention is capped at `index_topk=2048` keys/token —
  context-flat decode where dense MLA collapses (12→7→5 tok/s at 2k/4k/8k).
- **fp16 lightning-indexer kernels** replacing DeepGEMM fp8 MQA logits: a GEMV
  for decode, and for prefill a packed-Q register-accumulator WMMA kernel
  (6.3 → 44 TFLOPS, bit-exact vs reference).
- **Sparse prefill**: per-token causal top-k attention, O(q_len x topk); the
  final kernel is warp-specialized (16 producer + 8 consumer warps, named
  barriers), +43% over the register-accumulator version, bit-identical output.
- **TP key-sharding of indexer scoring** + a fused shard-merge kernel + an
  optional NVLink/IPC P2P shard exchange: exact global top-k at 1/TP the
  per-GPU cost (the indexer was 84% of GPU time at 200k prefill before this).
- **Dense MLA fallback** (exact reference, used below the 2048-token sparse
  crossover and for A/B validation), including a streaming-softmax varlen
  prefill and V100 shared-memory fixes for the Triton MLA decode kernel.

**Memory (weights are the constraint: ~25-28 GiB/GPU of a 32 GiB card)**
- fp8-e4m3 **weight-only storage** for the fp16 attention/indexer projections
  and MLP/shared-expert weights (quantized at load; GLM-5.1's official FP8
  release ships these tensors in this exact format): -1.46 GiB/GPU, +8.6%
  decode, ppl unchanged.
- **fp8 latent KV cache** (656 B/token vs 1152) and optional fp8 indexer-K
  cache: the sparse kernels dequantize inline in their gathers at no cost.
- **Byte-balanced PP partition** (`VLLM_PP_LAYER_PARTITION`): the even
  layer-count split leaves the last stage ~1.9 GiB heavier; rebalancing MoE
  layer counts more than doubled the KV pool in the MTP config.

**Throughput machinery**
- **Decode CUDA graphs without torch.compile**: eager decode was
  CPU-launch-bound (~16x launch overhead, 78 layers x 8 ranks); graphs took
  single-stream decode 6.0 → 19.1 tok/s.
- **MTP-1 speculative decoding** on the checkpoint's own MTP layer, verified
  as-decode inside the CUDA graphs (spec-as-decode), with the PP-correctness
  fixes that made it work (see MANIFEST/GOTCHAS).
- **Independent-batch PP interleaving**: decode scheduled in N independent
  waves so the pipeline holds N steps in flight (+10% at 4 streams).
- **TurboMind AWQ GEMM shape tuning** extended to the MTP verify shapes
  (M=2..8) and the full prefill chunk shape (+21-40% concurrent decode,
  W2 prefill GEMM 2.45x).
- EP-safety and single-token fast-path fixes for the SM70 AWQ MoE kernels
  (EP=0 with TP-sharded experts measured best and is the default).

**Correctness / serving**
- **fp16 residual-stream quality fix**: the upstream 1/2.5 residual scaling
  breaks RMSNorm scale-invariance for GLM's tiny embedding rows (Chinese
  drift, degenerate loops); folding routed scaling into router weights fixed
  it, verified token-for-token against an fp32 golden reference.
- GLM tool-parser streaming fix (zero-argument tool calls were dropped),
  reasoning-parser wiring, prompt capture + replay tooling.

## Provenance and acknowledgements

Upstream chain: [vLLM](https://github.com/vllm-project/vllm) →
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (一猫之下, the SM70 fork whose
TurboMind-derived AWQ kernels and V100 graph machinery this builds on) → this
work.

The sparse-attention WMMA kernel design was originally ported from a
[llama.cpp](https://github.com/ggml-org/llama.cpp) V100 DSA bringup, then
substantially rewritten (register accumulators, warp specialization,
split-topk, paged-KV gather, fp8 dequant-in-gather). Also standing on:
[lmdeploy/TurboMind](https://github.com/InternLM/lmdeploy),
[flash-attention-v100](https://github.com/ai-bond/flash-attention-v100),
[marlin_v100](https://github.com/zhinianqin/marlin_v100).

AI assistance: this work was done with heavy AI assistance (Claude Code); all
changes were human-reviewed and validated against the measurements quoted
above.

## License

Apache-2.0, following the upstream vLLM license model. See [LICENSE](LICENSE).
