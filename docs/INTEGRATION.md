# Integration guide

How to apply this release to upstream 1Cat-vLLM. Written to be directly
executable by a coding agent (Claude Code or similar) as well as a human.
Read [GOTCHAS.md](GOTCHAS.md) first — most integration failures are one of
the traps listed there, not a code problem.

## What you are integrating

A single patch (`patches/glm5-v100-sm70.patch`) against a known upstream
base, containing:

- two new CUDA source files (`csrc/attention/sm70_mla_decode.cu`,
  `csrc/attention/sm70_attn_fp8.cu`) plus their op registrations
  (`csrc/ops.h`, `csrc/torch_bindings.cpp`, `CMakeLists.txt`) and Python
  wrappers (`vllm/_sm70_ops.py`),
- modifications to ~30 Python files wiring the SM70 DSA sparse path, fp8
  weight/cache storage, decode CUDA graphs, MTP speculative decoding, PP
  interleaving, and serving fixes — all gated behind env flags and/or
  `device_capability.major == 7` checks (upstream and non-Volta behavior is
  preserved; the handful of unconditional changes are bug fixes, each
  explained in `patches/MANIFEST.md`),
- fixes to two upstream CUDA cache kernels whose fp16 dispatch path was
  broken (see MANIFEST and GOTCHAS "Upstream fp8 cache-write bug").

## Step 1 — get the base

```bash
git clone https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
git checkout v1.2.1   # == commit 4e9fdbc807178baa3bc98a1a59af7af7d3b63131
```

The patch was generated against exactly this commit and applies clean there.

## Step 2 — apply

```bash
git apply --check /path/to/release/patches/glm5-v100-sm70.patch   # must be silent
git apply --stat  /path/to/release/patches/glm5-v100-sm70.patch   # review scope
git apply         /path/to/release/patches/glm5-v100-sm70.patch
cp -r /path/to/release/scripts/serve_glm.sh scripts_or_root_of_your_choice/
cp /path/to/release/tools/replay_prompt.py .
```

**If `git apply --check` fails** (you are on a newer upstream): do not
force it. Instead:

1. Read `patches/MANIFEST.md` — it describes every file's changes and their
   purpose, so you can re-apply hunks contextually.
2. The three largest new files are provided as plain files in this release
   (`csrc/attention/*.cu`, `vllm/.../prefill/sm70_triton.py`) — copy them
   verbatim; only their *registration* hunks need contextual re-application.
3. Integrate in the dependency order of the "Staged bring-up" section below,
   validating each stage before the next. Every feature is env-gated, so a
   partially-integrated tree still runs.

## Step 3 — build

Requirements: CUDA toolkit **12.9** (12.8's cuBLAS is broken for fp16 GEMM on
Volta — see GOTCHAS; you need 12.9 for nvcc *and* for the runtime cuBLAS
symlink), a torch build with SM70 support (torch 2.10 + cu128 wheels work,
with the cuBLAS symlink fix), Python 3.12.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e . --no-build-isolation   # or the fork's documented install
# For an in-place extension rebuild (what we actually used):
export PATH=/usr/local/cuda-12.9/bin:$PATH CUDACXX=/usr/local/cuda-12.9/bin/nvcc
export TORCH_CUDA_ARCH_LIST="7.0" VLLM_TARGET_DEVICE=cuda MAX_JOBS=$(nproc) CMAKE_BUILD_TYPE=Release
python setup.py build_ext --inplace        # ~15-20 min with ccache
```

Then apply the **cuBLAS 12.9 symlink fix** from GOTCHAS to every venv on
every node, and verify with the one-line fp16 GEMM check there.

Multi-node: the built `.so` files and any Python change must be present at
the same repo path on every node, every time. Checksum-verify after syncing;
this is the most repeated operational mistake (GOTCHAS, "Cluster / Ray").

## Step 4 — staged bring-up (do not enable everything at once)

Each stage has a validation gate. The env flags default such that a fresh
apply starts at stage 1.

**Stage 1 — dense MLA reference (no custom csrc needed at runtime).**
Defaults as-applied: `VLLM_SM70_GLM_DENSE_MLA=1`, everything else off.
Serve with `--enforce-eager`, short `max_model_len` (2048), fp16 KV.
Gate: chat-templated "What is the capital of France?" → coherent "Paris";
finite logprobs. (Raw un-templated prompts produce garbage by design —
GOTCHAS.) This path is the mathematically exact reference used by every
later A/B.

**Stage 2 — decode CUDA graphs.**
`VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1`,
`..._CAPTURE_SIZE >= max_num_seqs`, drop `--enforce-eager`.
Gate: temp-0 output identical to stage 1; single-stream decode ~3x faster.

**Stage 3 — sparse DSA path (needs the built csrc ops).**
`VLLM_SM70_GLM_SPARSE_MLA=1 VLLM_SM70_GLM_SPARSE_PREFILL=1`.
Gate: temp-0 outputs **identical to dense** for any prompt that fits under
2048 tokens of context (top-k selects all keys there, so sparse == dense
exactly); decode tok/s flat as context grows past 2048 while the dense path
degrades. Then raise `max_model_len` and run a needle-in-haystack test
(with thinking disabled — GOTCHAS).

**Stage 4 — memory levers.** `VLLM_SM70_ATTN_FP8=1`, then
`VLLM_SM70_MLP_FP8=1`, then `VLLM_SM70_MLA_KV_FP8=1` (requires sparse), plus
`VLLM_PP_LAYER_PARTITION` and
`VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_TOKENS=16384`.
Gate per lever: startup "GPU KV cache size" grows as documented in
ENV_FLAGS.md; temp-0 answers unchanged; ppl on a dedicated low-util launch
if you want rigor (never send `prompt_logprobs` to a prod-util server).

**Stage 5 — prefill throughput.** `VLLM_SM70_IDX_PACKED_WMMA=1` (default),
`VLLM_SM70_IDXER_TP_SHARD_PREFILL=1`, `VLLM_SM70_IDXER_FUSED_MERGE=1`,
`VLLM_SM70_SPARSE_PREFILL_WS=1`, `VLLM_SM70_AWQ_MOE_TUNE_PREFILL_SLOTS`,
`max_num_batched_tokens=1024`.
Gate: cold-prefill tok/s at 16k/64k; needle PASS at your max context.

**Stage 6 — MTP + concurrency.** `--speculative-config '{"method":"mtp",
"num_speculative_tokens":1}'`, `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=1`,
`VLLM_SM70_PP_INTERLEAVE_GROUPS=2`, MTP PP partition.
Gate: temp-0 identical to non-MTP; acceptance ~50-60%; no repetition in a
1000+ token generation (the repetition-collapse failure mode is described in
GOTCHAS); multi-stream aggregate scales.

`scripts/serve_glm.sh` is the end state of all six stages with every default
annotated; diff your config against it when something regresses.

## Step 5 — validation toolkit

- **Dense/sparse A/B**: the strongest correctness tool in this codebase.
  Sparse must equal dense (temp-0, token-for-token) below 2048 context.
- **Needle tests** at 16k/96k/max, thinking disabled.
- **tools/replay_prompt.py** + `VLLM_DEBUG_DUMP_PROMPTS_DIR`: capture any
  misbehaving client request and replay it through the full pipeline
  (`--stream`) or token-exact against the bare model (`--mode tokens`) to
  separate model bugs from template/parser bugs in one pass.
- **Decode benchmarking**: prefix-cached pure decode (two `/v1/completions`
  calls, max_tokens 1 vs 201, marginal rate) — see GOTCHAS "Measurement
  discipline" for why naive measurements lie here.
- **compute-sanitizer memcheck on edge shapes** for any kernel you touch
  (B mod 32 tails especially); synccheck instead of racecheck for the
  warp-specialized kernel.

## Porting to other hardware / models

- The kernels are written for SM70 (V100): nvcuda::wmma 16x16x16 fp16 tiles,
  96 KB smem opt-in, no async-copy. SM75 (T4, 64 KB smem default but similar
  WMMA) would need smem budget review; SM80+ should use the upstream Hopper
  paths instead — these kernels are a substitute for missing hardware, not a
  general improvement.
- Model-wise the work targets `GlmMoeDsaForCausalLM` but the sparse path is
  generic DeepSeek-V3.2/DSA machinery (uniform full indexer, index_topk from
  config, MLA latent 512+64). DeepSeek-V3.2-class checkpoints should work
  with at most model-config plumbing (`vllm/model_executor/models/config.py`
  registration, residual-scale policy).
- GLM-5.1-AWQ itself needs ~420 GB of weights = 16x 32 GB with the fp8
  levers on. It does not fit 8x V100.

## Notes for coding agents

- Treat `patches/MANIFEST.md` as your map and GOTCHAS.md as your priors.
  When something fails, check GOTCHAS before debugging from scratch — the
  failure signatures there (dmon wattage patterns, "Failed to CUDA calloc",
  parity-dependent corruption, repetition collapse, wedged engine) are
  diagnostic.
- Never skip a stage gate to save time: several bugs here produced *correct
  output* while being wrong (masked OOB reads, wrong-branch graph capture at
  one specific batch size) and only later validation caught them.
- The dense path is not legacy — it is the reference oracle. Keep it working.
- If you improve a kernel, keep its predecessor behind the existing env flag
  and A/B with alternating restarts (single-pair A/Bs on this stack misled
  twice; restart variance is real).
