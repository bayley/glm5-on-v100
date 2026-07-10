# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)





def sm70_mla_spec_as_decode(vllm_config) -> bool:
    """True when SM70 spec-decode verify steps should route down the sparse
    DECODE kernel (uniform qlen 1+num_spec treated as decode) instead of the
    prefill path — the prerequisite for FULL_DECODE_ONLY CUDA graph capture
    of verify steps. Requires the fp8_ds_mla cache so that ALL decode
    traffic reaches the SM70 sparse kernel (the dense triton/torch decode
    paths are next_n=1-only and cannot read that layout anyway). The
    DeepseekV32 indexer builder mirrors this condition so both builders
    classify a verify batch the same way.
    """
    return (
        current_platform.is_device_capability(70)
        and envs.VLLM_SM70_GLM_SPARSE_MLA
        and vllm_config.speculative_config is not None
        and vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
    )


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        if sm70_mla_spec_as_decode(vllm_config):
            # Instance attribute shadows the SINGLE_ONLY class default; the
            # base __init__ reads it to raise reorder_batch_threshold to
            # 1 + num_speculative_tokens (spec-as-decode).
            self.query_len_support = QueryLenSupport.UNIFORM
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kwargs)


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        # SM70 sparse path only (VLLM_SM70_MLA_KV_FP8): the Volta sparse WMMA
        # kernels dequantize the 656-byte fp8_ds_mla layout in their gathers.
        # The dense Triton/torch paths cannot read it (guarded in forward_mqa).
        "fp8_ds_mla",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # 656-byte custom storage format (512 e4m3 NoPE + 4 fp32 tile
            # scales + 64 fp16 RoPE). Same as flashmla_sparse.py.
            return (num_blocks, block_size, 656)
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

        # SM70-only KV-split override for decode CUDA graphs (see forward_mqa).
        self._sm70_force_kv_splits = 0
        self._sm70_mla_decode_torch = False
        self._sm70_sparse_decode = False
        self._sm70_sparse_min_ctx = 0
        self._sm70_sparse_prefill = False
        try:
            cap = current_platform.get_device_capability()
            if cap is not None and cap.major == 7:
                self._sm70_force_kv_splits = envs.VLLM_SM70_MLA_DECODE_KV_SPLITS
                self._sm70_mla_decode_torch = envs.VLLM_SM70_MLA_DECODE_TORCH
                # DSA sparse decode (Volta WMMA top-k MLA). Only active when the
                # sparse path is enabled; decode routes to sm70_sparse_mla_decode
                # at/above the crossover context (see _forward_mqa_sparse).
                self._sm70_sparse_decode = envs.VLLM_SM70_GLM_SPARSE_MLA
                self._sm70_sparse_min_ctx = envs.VLLM_SM70_SPARSE_DECODE_MIN_CTX
                self._sm70_sparse_prefill = (
                    envs.VLLM_SM70_GLM_SPARSE_MLA
                    and envs.VLLM_SM70_GLM_SPARSE_PREFILL
                )
        except Exception:
            self._sm70_force_kv_splits = 0
            self._sm70_mla_decode_torch = False
            self._sm70_sparse_decode = False
            self._sm70_sparse_min_ctx = 0
            self._sm70_sparse_prefill = False


    def _forward_mqa_sparse(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SM70 DSA sparse (top-k) MLA decode via the Volta WMMA kernel.

        Uses the top-k logical KV positions produced by the DSA indexer (stored
        in ``layer.indexer.topk_indices_buffer``) instead of walking the full
        context, so per-token attention is O(topk) and context-INDEPENDENT past
        ``index_topk`` (2048). Dense MLA is numerically identical out to
        index_topk (the top-k selects every valid position when seq_len<=topk),
        so this is a pure speedup at long context with no quality change up to
        topk.

        idx (per-request logical positions) + the MAIN latent cache decode
        block_table/seq_lens are handed to the kernel, which resolves each
        selected position through the paged block_table. The kernel masks
        positions >= seq_len and idx == -1, so it is correct at any context
        (including seq_len <= topk); the crossover only gates when it is a win.

        q:   [B, H, kv_lora_rank + qk_rope]  (== 576 for GLM)
        out: o [B, H, kv_lora_rank(512)] fp16, lse [B, H] fp32
        """
        import vllm._sm70_ops as sm70_ops

        decode = attn_metadata.decode
        assert decode is not None
        indexer = getattr(layer, "indexer", None)
        assert indexer is not None, "sparse decode requires an indexer on the layer"
        topk_buf = indexer.topk_indices_buffer
        topk_tokens = indexer.topk_tokens

        B, H, Lk = q.shape
        Lv = self.kv_lora_rank

        # idx: per decode token, the top-k logical positions. The buffer is
        # laid out [num_batched_tokens, topk]; decode tokens are at the front
        # (the indexer writes one top-k row per TOKEN).
        idx = topk_buf[:B, :topk_tokens].to(torch.int32)

        block_table = decode.block_table.to(torch.int32)
        seq_lens = decode.seq_lens
        if seq_lens.dim() == 2:
            seq_lens = seq_lens[:, -1]
        seq_lens = seq_lens.contiguous().to(torch.int32)

        # Spec-as-decode (uniform qlen 1+num_spec verify steps): the kernel
        # is one-row-per-token with one block-table row and one causal
        # length per row, but decode metadata is per REQUEST. Expand:
        # token j of request b sits at position seq_lens[b]-next_n+j, so its
        # causal length is seq_lens[b]-(next_n-1)+j; its block-table row is
        # request b's. Static shapes at capture — CUDA-graph safe.
        num_reqs = seq_lens.shape[0]
        if B != num_reqs:
            assert B % num_reqs == 0, (
                f"non-uniform decode batch: {B} tokens, {num_reqs} requests"
            )
            next_n = B // num_reqs
            offs = torch.arange(next_n, device=q.device, dtype=torch.int32)
            seq_lens = (
                (seq_lens.unsqueeze(1) - (next_n - 1) + offs)
                .reshape(-1)
                .contiguous()
            )
            block_table = block_table.repeat_interleave(next_n, dim=0).contiguous()

        o = torch.empty(B, H, Lv, dtype=torch.float16, device=q.device)
        lse = torch.empty(B, H, dtype=torch.float32, device=q.device)

        # fp8_ds_mla cache (VLLM_SM70_MLA_KV_FP8): pass the raw uint8 tensor;
        # the kernel dequantizes in its gather. (A .to(fp16) here would
        # numerically convert the BYTES — never do that.)
        if kv_c_and_k_pe_cache.dtype == torch.uint8:
            kv_cache_arg = kv_c_and_k_pe_cache
        else:
            kv_cache_arg = kv_c_and_k_pe_cache.to(torch.float16)

        sm70_ops.sm70_sparse_mla_decode(
            o,
            lse,
            q.to(torch.float16).contiguous(),
            kv_cache_arg,
            idx,
            block_table,
            seq_lens,
            float(self.scale),
        )
        return o.to(q.dtype), lse

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # SM70 DSA sparse PREFILL: route prefill tokens through the latent sparse
        # top-k attention (same sm70_sparse_mla_decode kernel, fed B=q_len with
        # per-token causal metadata) instead of the dense streaming-softmax
        if self._sm70_sparse_prefill:
            try:
                self._forward_mha_sparse(
                    q, kv_c_and_k_pe_cache, attn_metadata, output
                )
                return
            except Exception as e:
                # Fall back to dense prefill on any error rather than diverging
                # one PP rank (which would deadlock the peer on a gloo timeout).
                # Log once; dense is the exact reference and always correct.
                # EXCEPTION: with an fp8_ds_mla cache the dense prefill would
                # misread raw bytes as values (silent garbage) — re-raise and
                # fail loudly instead.
                if kv_c_and_k_pe_cache.dtype == torch.uint8:
                    logger.error(
                        "SM70 sparse prefill failed with an fp8_ds_mla KV "
                        "cache; the dense fallback cannot read that layout. "
                        "Re-raising."
                    )
                    raise
                if not getattr(self, "_sm70_sparse_prefill_warned", False):
                    import traceback
                    logger.warning(
                        "SM70 sparse prefill failed (%s); falling back to dense "
                        "prefill for this and future steps.\n%s",
                        repr(e), traceback.format_exc()
                    )
                    self._sm70_sparse_prefill_warned = True
                    self._sm70_sparse_prefill = False
        super().forward_mha(
            q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output
        )

    def _forward_mha_sparse(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        output: torch.Tensor,
    ) -> None:
        """SM70 DSA sparse prefill attention over the paged latent cache.

        Each prefill token attends only its indexer-selected top-k causal keys.
        Projects q_nope -> latent via W_UK_T (like the MQA decode path), builds
        per-token causal seq_lens + per-token block_table from the prefill
        metadata, reads the indexer's per-token top-k, then reuses
        sm70_sparse_mla_decode with B = num_prefill_tokens.

        q: [q_len, n_head, qk_nope + qk_rope]; output: [q_len, n_head*v_head_dim].
        The latent cache must already hold this batch's keys (written by the MLA
        layer's cache insert before attention), which the indexer top-k reference.
        """
        import vllm._sm70_ops as sm70_ops

        prefill = attn_metadata.prefill
        assert prefill is not None
        indexer = self.indexer
        assert indexer is not None, "sparse prefill requires an indexer on the impl"
        topk_buf = indexer.topk_indices_buffer
        topk_tokens = indexer.topk_tokens

        q_len, H, _ = q.shape
        Lk = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        Lv = self.kv_lora_rank
        # prefill topk rows follow the decode-token rows in the shared buffer.
        nd0 = attn_metadata.num_decode_tokens or 0

        # project q_nope -> latent: (N,B,P) x (N,P,L) -> (N,B,L) -> (B,N,L).
        # W_UK_T [N, P, L] lives on the MLAAttention LAYER (set in
        # process_weights_after_loading), not the impl; build+cache it here from
        # the impl's kv_b_proj (same derivation) on first use.
        W_UK_T = getattr(self, "_sm70_W_UK_T", None)
        W_UV = getattr(self, "_sm70_W_UV", None)
        if W_UK_T is None or W_UV is None:
            from vllm.model_executor.layers.attention.mla_attention import (
                get_and_maybe_dequant_weights,
            )
            kv_b = get_and_maybe_dequant_weights(self.kv_b_proj, out_dtype=q.dtype).T
            kv_b = kv_b.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            W_UK = kv_b[..., : self.qk_nope_head_dim]  # [L, N, P]
            W_UV_lnv = kv_b[..., self.qk_nope_head_dim :]  # [L, N, V]
            W_UK_T = W_UK.permute(1, 2, 0).contiguous()  # [N, P, L]
            W_UV = W_UV_lnv.transpose(0, 1).contiguous()  # [N, L, V]
            self._sm70_W_UK_T = W_UK_T
            self._sm70_W_UV = W_UV
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_nope_nbp = q_nope.transpose(0, 1)  # [N, q_len, P]
        ql_nope = torch.bmm(q_nope_nbp, W_UK_T)  # [N, q_len, L]
        ql_nope = ql_nope.transpose(0, 1)  # [q_len, N, L]
        q_latent = torch.cat([ql_nope, q_pe], dim=-1)  # [q_len, N, Lk]

        # per-token causal metadata (seq_lens + block_table) was computed by the
        # indexer while selecting top-k (cu_seqlen_ke); reuse it to stay exactly
        # consistent. The indexer ran immediately before this attention in the
        # same layer forward and stashed it in _SM70_PREFILL_META.
        from vllm.model_executor.layers.sparse_attn_indexer import (
            _SM70_PREFILL_META,
        )

        pm_meta = _SM70_PREFILL_META
        # Per-chunk (tok_lo, n_q, ke) entries: the indexer builder may split a
        # step's prefill tokens into SEVERAL chunks (its flat-logits budget is
        # q_len*ctx-bounded — 1024-token engine chunks split beyond ~131k
        # ctx), so validate the entries tile [nd0, nd0+q_len) contiguously
        # and concatenate their per-token causal seq_lens in token order.
        entries = pm_meta.get("chunks")
        ok = bool(entries)
        if ok:
            expect = nd0
            for tl, n, _ in entries:
                if tl != expect:
                    ok = False
                    break
                expect = tl + n
            ok = ok and (expect - nd0) == q_len
        if not ok:
            raise RuntimeError(
                "SM70 sparse prefill: indexer per-token metadata missing/"
                "mismatched (chunks="
                f"{[(e[0], e[1]) for e in entries] if entries else None}, "
                f"need [{nd0}, {nd0 + q_len}))"
            )
        per_tok_seqlen = (
            entries[0][2]
            if len(entries) == 1
            else torch.cat([e[2] for e in entries])
        ).to(torch.int32)
        # per-token request id from the MAIN prefill metadata (query_start_loc
        # gives per-request token ranges over the prefill tokens) — the
        # indexer chunks' request indices are chunk-local so they cannot be
        # used across a multi-chunk step.
        qsl = prefill.query_start_loc
        req_id = (
            torch.searchsorted(
                qsl,
                torch.arange(q_len, device=q.device, dtype=qsl.dtype),
                right=True,
            )
            - 1
        ).clamp_(min=0)
        # per-token block_table from the MAIN latent cache (index_topk positions
        # are logical; the kernel resolves them via this block_table).
        main_bt = prefill.block_table.to(torch.int32)  # [num_reqs, max_blocks]
        per_tok_bt = main_bt.index_select(0, req_id.to(torch.long)).contiguous()

        idx = topk_buf[nd0 : nd0 + q_len, :topk_tokens].to(torch.int32)

        o = torch.empty(q_len, H, Lv, dtype=torch.float16, device=q.device)
        lse = torch.empty(q_len, H, dtype=torch.float32, device=q.device)
        if kv_c_and_k_pe_cache.dtype == torch.uint8:
            kv_cache_arg = kv_c_and_k_pe_cache  # fp8_ds_mla: kernel dequantizes
        else:
            kv_cache_arg = kv_c_and_k_pe_cache.to(torch.float16)
        sm70_ops.sm70_sparse_mla_decode(
            o,
            lse,
            q_latent.to(torch.float16).contiguous(),
            kv_cache_arg,
            idx,
            per_tok_bt,
            per_tok_seqlen,
            float(self.scale),
        )
        # v_up projection: latent [q_len,N,L] -> value [q_len,N,V] via W_UV
        # (mirrors the layer's _v_up_proj after forward_mqa). bmm over heads:
        # (N,q_len,L) x (N,L,V) -> (N,q_len,V) -> (q_len,N,V).
        o_nbl = o.to(q.dtype).transpose(0, 1)  # [N, q_len, L]
        o_nbv = torch.bmm(o_nbl, W_UV.to(q.dtype))  # [N, q_len, V]
        out_bnv = o_nbv.transpose(0, 1).reshape(q_len, H * self.v_head_dim)
        output.copy_(out_bnv)

    def _forward_mqa_torch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Plain-torch MLA decode (bmm + masked softmax), SM70 fast path.

        The Triton flash-decode grouped kernel is crippled on Volta by the
        96 KB smem cap (forced BLOCK_N=16), running ~10-24x slower than a naive
        torch bmm at long context. This gathers the paged latent KV to a
        fixed-width buffer (so it is CUDA-graph safe: block_table width and the
        gather shape are static at capture; the live seq_lens only drive an
        arithmetic mask) and computes MQA attention directly.

        q:   [B, H, kv_lora_rank + qk_rope]  (== 576 for GLM)
        cache: paged [num_blocks, PAGE, kv_lora_rank + qk_rope]
        out: o [B, H, kv_lora_rank], lse [B, H]  (matching the Triton path)
        """
        decode = attn_metadata.decode
        assert decode is not None
        block_table = decode.block_table            # [B, max_blocks] int
        seq_lens = decode.seq_lens                  # [B] int
        B, H, _ = q.shape
        Lk = q.shape[-1]                            # 576
        Lv = self.kv_lora_rank                      # 512
        PAGE = kv_c_and_k_pe_cache.shape[1]
        max_blocks = block_table.shape[1]
        max_kv = max_blocks * PAGE

        # Gather the full (fixed-width) latent KV for each sequence.
        # cache: [num_blocks, PAGE, Lk] -> flat rows [num_blocks*PAGE, Lk].
        cache_flat = kv_c_and_k_pe_cache.reshape(-1, Lk)
        # per-(b, block) physical row base, then expand to per-token rows.
        # row index for (b, blk, p) = block_table[b, blk] * PAGE + p
        blk = block_table.to(torch.long)                       # [B, max_blocks]
        row_base = (blk * PAGE).unsqueeze(-1)                  # [B, max_blocks, 1]
        offs = torch.arange(PAGE, device=q.device).view(1, 1, PAGE)
        rows = (row_base + offs).reshape(B, max_kv)           # [B, max_kv]
        rows = rows.clamp_(0, cache_flat.shape[0] - 1)
        # [B, max_kv, Lk]
        kv = cache_flat.index_select(0, rows.reshape(-1)).reshape(B, max_kv, Lk)

        # scores [B, H, max_kv] = q @ kv^T
        scores = torch.bmm(q.float(), kv.transpose(1, 2).float()) * self.scale

        # mask positions >= seq_len (causal upper bound for a single decode tok)
        pos = torch.arange(max_kv, device=q.device).view(1, max_kv)
        valid = pos < seq_lens.view(B, 1)                     # [B, max_kv]
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

        m = scores.max(dim=-1, keepdim=True).values
        m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        p = torch.exp(scores - m_safe)
        denom = p.sum(dim=-1, keepdim=True)
        # value = first Lv dims of the latent
        o = torch.bmm(p.to(kv.dtype), kv[..., :Lv]) / denom   # [B, H, Lv]
        o = o.to(q.dtype)

        lse = (m_safe.squeeze(-1) + torch.log(denom.squeeze(-1))).to(q.dtype)
        return o, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)

        # SM70 DSA sparse decode: at/above the crossover context, use the Volta
        # WMMA top-k kernel (O(topk), context-flat). The decision is driven by
        # attn_metadata.max_seq_len, which is frozen to max_model_len at CUDA
        # graph capture, so the branch is static across replays (the kernel is
        # correct at any runtime seq_len regardless). Requires the indexer to
        # have populated topk_indices_buffer this step.
        # With an fp8_ds_mla cache (VLLM_SM70_MLA_KV_FP8) the sparse kernel is
        # the ONLY decode path that can read the layout, so it is used at ANY
        # max_seq_len (warmup/profile dummy runs pass small seq lens that
        # would otherwise route to the dense path and fail).
        kv_fp8 = kv_c_and_k_pe_cache.dtype == torch.uint8
        if (
            self._sm70_sparse_decode
            and getattr(layer, "indexer", None) is not None
            and (
                kv_fp8
                or int(attn_metadata.max_seq_len) >= self._sm70_sparse_min_ctx
            )
        ):
            return self._forward_mqa_sparse(
                q, kv_c_and_k_pe_cache, attn_metadata, layer
            )

        # The dense decode paths below (torch gather / Triton flash-decode)
        # read the latent cache as real fp16/bf16 (or per-tensor-scale fp8)
        # values; the 656-byte fp8_ds_mla layout would be misread as garbage.
        if kv_fp8:
            raise RuntimeError(
                "TritonMLA dense decode cannot read the fp8_ds_mla KV cache "
                "layout (VLLM_SM70_MLA_KV_FP8=1 requires the SM70 sparse "
                "decode path; check VLLM_SM70_GLM_SPARSE_MLA / "
                "VLLM_SM70_SPARSE_DECODE_MIN_CTX <= max_model_len)."
            )

        if self._sm70_mla_decode_torch:
            return self._forward_mqa_torch(q, kv_c_and_k_pe_cache, attn_metadata)

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # Minimum work per split
            # hardware dependent
            min_work_per_split = 512

            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)

            # use power of 2 to avoid excessive kernel instantiations
            ideal_splits = triton.next_power_of_2(ideal_splits)

            # Calculate SM-based maximum splits with occupancy multiplier
            # 2-4x allows multiple blocks per SM for latency hiding
            # hardware dependent
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

            # SM70 / V100: under a decode CUDA graph num_kv_splits is frozen at
            # capture time (capture sees max_seq_len == max_model_len). The
            # upstream min_work_per_split=512 heuristic then picks very few
            # splits (e.g. 4 at a 2048 max_model_len), which for single-request
            # / low-batch decode under-occupies the GPU on the sequence axis and
            # makes per-layer attention grow ~linearly with context. Force a
            # higher split count so the (frozen) grid parallelizes the KV
            # dimension across SMs. Capped so partial splits still hold >= one
            # BLOCK_N(=16) of work at the capture-time context, and never fewer
            # than the upstream choice. See VLLM_SM70_MLA_DECODE_KV_SPLITS.
            if self._sm70_force_kv_splits > 0:
                forced = self._sm70_force_kv_splits
                seq_cap = max(1, int(attn_metadata.max_seq_len))
                # keep >= ~16 tokens/split at the (frozen) capture context
                forced = min(forced, max(1, seq_cap // 16))
                forced = min(forced, max_splits)
                num_kv_splits = max(num_kv_splits, forced)

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse
