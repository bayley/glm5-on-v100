# GOTCHAS — the distilled engineering log

Everything here was learned the hard way on a real 16x V100 cluster. If you
are reproducing this work (human or coding agent), read this before touching
anything. Organized by layer: environment → cluster ops → memory → CUDA
graphs → kernel development → MTP/PP → measurement → serving.

## Environment / driver stack

### cuBLAS 12.8 is broken for fp16 tensor-op GEMM on Volta

The pip cuBLAS 12.8 that ships with torch cu128 fails **any** fp16
`F.linear` / `a@b` with M > ~16 on V100 with `CUBLAS_STATUS_INVALID_VALUE`
(in `cublasGemmEx(... CUBLAS_GEMM_DEFAULT_TENSOR_OP)`). GLM hits this
immediately: its attention projections are unquantized fp16 and go through
raw `F.linear`. **Fix**: symlink the venv's
`site-packages/nvidia/cublas/lib/libcublas.so.12` and `libcublasLt.so.12` to
CUDA **12.9**'s copies (12.9 fixed the regression) — in *every* venv that
workers actually launch from, on *every* node. Verify:

```bash
python -c "import torch;a=torch.randn(256,6144,device='cuda',dtype=torch.float16);\
b=torch.randn(6144,2816,device='cuda',dtype=torch.float16);print((a@b).shape)"
```

Watch for **dangling symlinks** after copying files between nodes — a node
that "has the fix" may be silently resolving to nothing.

### Online-softmax init: -1e30, not -INFINITY

A fully-masked tile gives `expf(-inf - -inf) = NaN`. Every online-softmax
accumulator in these kernels initializes `m_i = -1e30f`. If you port or
re-derive any of them, keep this.

## Cluster / Ray operations

- **raylet hardcodes its python-worker command** to whatever `ray` binary
  started it. Start Ray from the venv whose interpreter has your source tree
  and the cuBLAS fix, on **both** nodes — otherwise workers silently run a
  different vLLM than your driver.
- **NCCL/GLOO socket ifnames and `NCCL_IB_HCA` differ per node** and must be
  exported *before* `ray start` so they propagate into the Ray actors. Set
  the same env on the driver process.
- A crashed run can leave a **stale placement group** reserving all GPUs
  ("0.0 used of 16.0 reserved"). `ray stop` on both nodes and restart.
- **Never `pkill -f python`** or other broad patterns — they match raylet's
  wrapper and kill the Ray worker. Also, `pkill -f '<pattern>'` matches your
  *own shell* if the pattern appears in your own compound command (and a
  `until ! pgrep -f ...` waiter loop then never exits). Kill by exact PID or
  use the `[b]racket` trick in the pattern.
- **New `VLLM_*` env flags must be registered in `vllm/envs.py`** or they are
  never forwarded to Ray workers (`get_env_vars_to_copy`) — the flag silently
  applies to the driver only. Corollary: `/proc/<pid>/environ` is the
  exec-time snapshot and misses Ray `runtime_env` vars applied post-spawn, so
  it falsely suggests the env is missing.
- Worker-side `logger.info` does not reliably surface in the serve log under
  Ray — debug worker code with file-write tracers.
- **Every code change must be synced to the worker node(s)** (and
  checksum-verified) before relaunch. A half-synced A/B silently runs old
  code on half the pipeline stages, and the startup prints can look exactly
  like baseline. This is the single most repeated operational mistake.

## Memory ceiling (weights-bound serving)

Weights land at ~25-28 GiB per 32 GiB V100; CUDA context/driver eat ~0.9 GiB,
so only ~30.8 of 31.7 GiB is allocatable. Consequences:

- `gpu_memory_utilization >= 0.98` is rejected at startup. The practical
  ceiling is **0.95 (PP4) / 0.965 (PP2)** — NOT 0.97: NCCL lazily
  `cudaCalloc`s comm buffers *after* the memory profile, **outside the torch
  allocator** (torch cannot release cached blocks to satisfy them), and the
  binding rank has zero slack by construction. The failure reads
  "Failed to CUDA calloc async ... bytes" deep inside a PP recv.
- The CUDA-graph **memory estimator's ~0.5 GiB reservation is real**. The
  startup log line claiming "0.06 GiB actual vs 0.58 estimated (883%)" is
  misleading — the estimator's throwaway capture leaves persistent workspaces
  resident, so "actual" only shows the marginal cost. Disabling the estimator
  (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) reclaims nothing and OOMs
  later. Documented dead end; keep it on.
- The default even PP split counts **layers, not bytes**: GLM's 3 dense
  layers are ~free vs its MoE layers, and the last stage carries the sampler,
  logits, and (with MTP) the draft layer. Balancing MoE counts per stage
  (`VLLM_PP_LAYER_PARTITION=21,19,19,19`; `22,19,19,18` with MTP) more than
  **doubled** the MTP-mode KV pool (78k → 162k tokens).
- The upstream MLA chunked-prefill workspace is sized for 64k tokens
  (~896 MiB at TP=4) and comes straight out of the KV budget; the SM70 sparse
  prefill doesn't use it. `VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_TOKENS=16384`
  reclaims ~0.65 GiB/GPU.
- **`prompt_logprobs` requests kill the production server**: the per-chunk
  prompt-logits allocation (~300 MiB) OOMs the last PP stage at serving
  utilization and the engine dies (EngineDeadError). Run perplexity tests on
  a dedicated launch at util <= 0.90 with a small max_model_len.

## CUDA-graph semantics (the recurring trap)

FULL decode CUDA graphs replay whatever the capture run computed. Three
distinct bugs came from the same root:

1. **Frozen launch parameters.** Capture sets `max_seq_len = max_model_len`,
   so anything derived from attention metadata at capture time is baked in.
   Upstream's `num_kv_splits = max_seq_len // 512` heuristic froze at a value
   that launched 16 blocks on 80 SMs (`VLLM_SM70_MLA_DECODE_KV_SPLITS=16`
   fixes it on SM70). Rule: in graph-captured paths, shapes and launch
   geometry must be static; all data-dependence must live in device-side
   arithmetic masks (`pos < seq_len`), never in host-computed values.
2. **Frozen branch selection.** The capture batch size decides which kernel
   branch gets captured. The fused decode-topk was only valid below B=8; a
   spec-decode verify batch of exactly 8 rows captured the slow fallback into
   the size-8 graph and 4-stream throughput collapsed by 7x. The dmon
   signature of a wrong-branch cliff: SM ~100%, MEM ~0%, ~80-120 W (busy
   spinning, moving no bytes).
3. **Stale persistent capture buffers.** If capture binds the graph to
   persistent metadata buffers (block_table, seq_lens), the runtime metadata
   build path must refresh those exact buffers every step — a builder that
   works eagerly can still replay capture-time dummy values under graphs.

Also: `.any()` host syncs and boolean-mask indexing (data-dependent shapes)
invalidate capture — replace with device-side scatter/mask kernels.

Eager-mode context, for motivation: eager decode was **CPU-launch-bound**
(~306 ms host launch time vs 19.5 ms GPU work per token across 78 layers x 8
TP ranks — TP all-reduces were spin-waiting on the slowest CPU launcher).
Graphs are not optional on this class of model/hardware; they were a 3.2x.

## Kernel development on Volta

- **Run compute-sanitizer memcheck on edge shapes before shipping.**
  Bit-exact output does not prove no OOB: the packed indexer kernel's tail
  tile read past its buffer for `B mod 32 ∈ (0,16]` (writes were masked, so
  output was correct) and only manifested later as probabilistic Xid 31 MMU
  faults.
- **racecheck false-positives and mis-executes named-barrier kernels** (it
  does not model `bar.sync` with thread counts). For warp-specialized
  kernels use synccheck + memcheck + a soak test instead.
- Real race found that racecheck couldn't see: in the warp-specialized
  prefill kernel, a shared-memory staging buffer aliased K-buffer 0; with an
  odd number of tiles, fast consumer warps overwrote the last tile under
  straggler PV reads (~30% corruption on odd-tile shapes only). The v5
  loop-tail `__syncthreads()` had been load-bearing. Watch for parity-
  dependent corruption as an aliasing signature.
- **WMMA accumulator layout is undocumented.** The packed indexer kernel
  derives Volta's fragment→lane mapping and verifies it with an on-device
  layout probe at init, aborting loudly if the toolchain ever changes it. Do
  not remove the probe.
- The WMMA prefill indexer gathers each K tile via the block table of the
  query tile's **first row** — a 16-row query tile straddling two requests
  mis-gathers. The wiring launches per request segment; preserve that
  invariant if you re-derive the kernel.
- Workspace managers that size during warmup and then **lock** will deadlock
  a PP pipeline if a runtime-reachable fallback path needs a workspace the
  warmup path never touched ("Workspace is locked" on one rank, silent
  cross-node hang; py-spy both nodes to find it). Reserve workspaces for
  every reachable branch during warmup.
- cuBLAS/allocator interplay: unchecked `cudaMallocAsync` in glue code (the
  MTP MoE conversion hit this in `MakeStridedPtrs`) turns an OOM into a
  null-pointer write. Check every allocation in csrc.

## MTP / pipeline-parallel speculative decoding

- **Draft token values exist only on the last PP rank's GPU** under async
  scheduling; the scheduler's CPU-side ids are `-1` placeholders. Any
  non-last rank that "reconstructs" drafts from scheduler output embeds
  garbage. Broadcast the draft ids explicitly (we ship them on
  `ModelRunnerOutput` — an RPC side-channel stalled 57 ms/step).
- **All spec/PP side-band collectives must run on a dedicated CUDA stream**
  with event ordering. Main-stream broadcasts deadlock against eager pageable
  H2D syncs, and a device-wide `torch.cuda.synchronize()` re-couples the side
  stream and deadlocks too. Consumer-side `wait_event` must be lazy or the
  interleaved waves serialize. `record_stream` the side-stream tensors or the
  allocator reuses them under load (illegal access).
- **Per-rejected-draft bookkeeping must run on ALL PP ranks.** The
  output_token_ids trim ran only on the last rank; non-last ranks inflated by
  +1 token per rejected draft until their inputs became `-1`/`0` placeholders
  — the visible symptom was repetition collapse minutes into generation.
- `disable_padded_drafter_batch=true` is a dead end in this fork: it force-
  disables async scheduling, the previous sampled token then never reaches
  the workers, and the verify batch shifts one row (verification is garbage).
- Both attention-metadata builders (main MLA + indexer) must classify verify
  batches identically — mirror any `reorder_batch_threshold` change in both.
- MTP output collapse with all-position acceptance ≈ 1.000 is a **symptom**
  (target and draft in the same degenerate stream), not proof of correctness.
- Debug method that cracked all of these: per-rank input dumps (`.tolist()`
  only — no tensor prints, they sync) compared across stages; the divergence
  "stage0 ids=[641,-1] vs stage3 [641,279]" is what identified the broadcast
  gap.
- Measured perspective before optimizing MTP machinery: the whole MTP tail
  (propose + broadcast) is ~2.5 ms of an ~82 ms step and overlaps the PP
  wait. The drafter-graph work measured neutral; C++-rewrite ceiling ~3%.
  Profile before building.

## Measurement discipline

- **torch.profiler bills NCCL spin-wait as compute.** NCCL kernels spin
  on-GPU waiting for peers; the profiler attributes that to the *waiting*
  rank. An "87% ncclDevKernel_Broadcast" profile of a PP decode is the other
  stage's compute seen as idle-spin, and a "2.85 ms all-reduce" was
  rank-arrival skew (the standalone collective is 0.09 ms). Tiebreakers:
  `nvidia-smi dmon` (pure spin = SM ~95% / MEM ~4% / ~80 W; real compute =
  high MEM% / ~165+ W) and standalone microbenches of the same collective.
- With prefix caching off, **every `generate()` re-prefills the prompt** —
  short-generation "decode tok/s" is contaminated by prefill. Measure decode
  as prefix-cached pure decode (two calls differing only in max_tokens;
  marginal rate), or profile decode-only with caching on.
- Alternate A/B configs **across restarts in one session** (A,B,A,B...).
  Single cross-restart pairs misled twice (restart variance looked like a +1%
  and then a +5% effect; the 6-round alternating A/B settled it at +2.3%).
- Profiler runs on a high-utilization server are sacrificial: trace export
  can OOM a PP worker post-capture, and each stop stalls the engine 60-90 s.
  Capture early, expect the run to wedge.
- Long benchmarks: run the driver under `setsid` — a plain `nohup ... &` dies
  when the wrapping shell times out and reaps the process group, which looks
  exactly like a mysterious mid-prefill abort. And put drivers under
  `if __name__ == "__main__":` (spawn workers re-import the module).
- Needle-in-haystack tests: **disable thinking** for the retrieval question
  (`{"chat_template_kwargs": {"enable_thinking": false}}`) or the model
  enumerates the filler in its reasoning budget forever and a PASS looks like
  a FAIL. Random-letter filler tokenizes at ~3.2 tok/word with the GLM
  tokenizer — size prompts accordingly.

## Model / serving correctness

- **Use the chat template.** Raw prompts ("2 + 2 =") are out-of-distribution
  for an RLHF chat model and produce low-confidence garbage that looks
  exactly like a numerics bug. Check logprobs are finite before blaming
  kernels.
- **The fp16 residual-stream scaling trick corrupts GLM.** Upstream divides
  the residual stream by `routed_scaling_factor` (2.5) to avoid fp16
  overflow, relying on RMSNorm scale-invariance — which the epsilon breaks
  for GLM's tiny embedding rows (`<|user|>`, `<think>` markers have
  mean(x²) ≈ 2.4e-6; they came out 2.3x too small after layer 0). Symptoms:
  Chinese drift, ignored think tags, broken punctuation, degenerate loops.
  Fix (`VLLM_DEEPSEEK_FP16_EXACT_STREAM_SCALE`, auto-on for GLM): fold routed
  scaling into router weights — exact math, verified token-for-token against
  a pure-fp32 golden reference. DeepSeek-V2 (rsf=16, real overflow risk)
  keeps upstream behavior.
- **Upstream fp8 cache-write bug (worth upstreaming):**
  `concat_and_cache_ds_mla_kernel` and `indexer_k_quant_and_cache_kernel` in
  `csrc/libtorch_stable/cache_kernels.cu` compute amax on the raw fp16 **bit
  pattern** when the KV dtype dispatch hands them a `uint16_t` carrier —
  garbage scales, zeroed e4m3 payloads. bf16/Hopper deployments never hit it.
  Fixed here with a `kv_elem_to_float<scalar_t>()` helper.
- GLM-5.1 tool calls need `--tool-call-parser glm47`, **not** glm45: the
  template emits `<tool_call>{name}<arg_key>...` with no newline after the
  name, and glm45's regex requires one. The streaming parser also dropped
  zero-argument calls entirely (fixed in this patch, with tests).
- This fork defaults `enable_thinking=false` in chat-template kwargs
  (Qwen-centric); GLM-5.1's template wants thinking ON by default — the serve
  script restores it.
- Custom one-shot all-reduce is unsafe on SXM2 cube-mesh boxes at TP=8
  (no full 1-hop NVLink mesh; forcing it → illegal memory access). At TP=4 it
  works.
- Concurrent long prefills serialize FCFS at `max_num_batched_tokens=1024`
  (first request takes the whole chunk budget at full solo speed) — this is
  fine. At 512 they interleaved badly and aggregate prefill collapsed ~3x.
- Unresolved (rare): one unreproduced worker CUDA "unknown error" during
  temp>0 sampling wedged the engine — peers block in collectives, the API
  stays up, generations hang; only a full restart recovers. If you see it,
  capture the prompt (prompt-dump tooling) and bisect the MoE fastpath flags.
