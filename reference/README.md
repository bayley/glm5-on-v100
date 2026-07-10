# reference/ — read-only snapshots

Full post-patch versions of the most interesting **modified** Python files,
provided for browsing. Unlike the files under `csrc/` and `vllm/` (which are
new files you copy verbatim during integration), these are upstream vLLM
files interwoven with our additions — **do not copy them over an upstream
tree**; they are integrated via the patch. Provenance: identical to what
`patches/glm5-v100-sm70-full.patch` produces on the v1.2.1 base.

| File | What's interesting in it |
| --- | --- |
| `sparse_attn_indexer.py` | The SM70 fp16 DSA indexer machinery (`sm70_sparse_indexer` custom op): decode logits + fused histogram top-k, per-request-segment causal prefill scoring with sub-chunking and the causal-max memo, TP key-sharded scoring, the NCCL / fused-kernel / NVLink-IPC shard-merge trio, and the `_SM70_PREFILL_META` bridge to the attention backend. |
| `triton_mla.py` | The SM70 sparse wiring inside the dense TRITON_MLA backend: `_forward_mqa_sparse` (sparse decode + spec-as-decode row expansion at static shapes), `_forward_mha_sparse` (sparse prefill: q→latent projection, per-token causal seq_lens, W_UV up-projection), `_forward_mqa_torch` (CUDA-graph-safe plain-torch decode), `sm70_mla_spec_as_decode`, fp8_ds_mla cache support, and the graph-frozen `num_kv_splits` override. |
| `_sm70_ops.py` | The Python wrappers + fake-tensor registrations for every custom op in `csrc/attention/*.cu` — the op signatures and shape contracts in one place. |
