# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


# P2P shard-merge state (Stage B): per-process IPC exchange for the prefill
# indexer TP shard merge. One double-buffered slot pair per rank; peers read
# shards directly over NVLink instead of an NCCL all-gather. Init is a TP
# collective (all ranks reach the sharded branch together — same scheduler
# output); any failure marks the state failed and the NCCL path is used.
_SM70_P2P_STATE: dict | None = None
_SM70_P2P_FAILED = False


def _sm70_p2p_state(device: torch.device, K: int, shard_tp: int):
    global _SM70_P2P_STATE, _SM70_P2P_FAILED
    if _SM70_P2P_FAILED:
        return None
    if _SM70_P2P_STATE is not None:
        return _SM70_P2P_STATE
    try:
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_tp_group

        grp = get_tp_group()
        rank = grp.rank_in_group
        try:
            from vllm.config import get_current_vllm_config

            rows_cap = int(
                get_current_vllm_config().scheduler_config
                .max_num_batched_tokens
            )
        except Exception:
            rows_cap = 1024
        slot_bytes = rows_cap * 2 * K * 4
        buf, handle = torch.ops._C.sm70_ipc_alloc(2 * slot_bytes)
        fbuf, fhandle = torch.ops._C.sm70_ipc_alloc(64)
        payload = (
            handle.numpy().tobytes(),
            fhandle.numpy().tobytes(),
        )
        objs: list = [None] * shard_tp
        dist.all_gather_object(objs, payload, group=grp.cpu_group)
        data_ptrs, flag_ptrs = [], []
        for i, (h, fh) in enumerate(objs):
            if i == rank:
                data_ptrs.append(buf.data_ptr())
                flag_ptrs.append(fbuf.data_ptr())
            else:
                ht = torch.frombuffer(bytearray(h), dtype=torch.uint8)
                fht = torch.frombuffer(bytearray(fh), dtype=torch.uint8)
                data_ptrs.append(torch.ops._C.sm70_ipc_open(ht))
                flag_ptrs.append(torch.ops._C.sm70_ipc_open(fht))
        _SM70_P2P_STATE = {
            "buf": buf,
            "fbuf": fbuf,
            "rows_cap": rows_cap,
            "slot_bytes": slot_bytes,
            "my_flag_ptr": fbuf.data_ptr(),
            "data_ptrs_t": torch.tensor(data_ptrs, dtype=torch.int64),
            "flag_ptrs_t": torch.tensor(flag_ptrs, dtype=torch.int64),
            "seq": 0,
        }
        logger.info(
            "SM70 indexer P2P shard merge initialized (tp=%d, rows_cap=%d)",
            shard_tp, rows_cap,
        )

    except Exception:
        logger.exception(
            "SM70 indexer P2P shard merge init FAILED; using NCCL path"
        )
        _SM70_P2P_FAILED = True
        return None
    return _SM70_P2P_STATE

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# Bridge for SM70 sparse PREFILL: the indexer computes per-token causal seq_lens
# (cu_seqlen_ke) and per-token block_table while selecting top-k; the sparse
# prefill ATTENTION (_forward_mha_sparse) needs the SAME per-token metadata but
# runs from the (different) main-attention prefill metadata. The indexer runs
# immediately before its layer's attention in the same forward, so we hand the
# metadata across via this module-global (same single-owner-at-a-time pattern as
# the shared topk_indices_buffer). Keyed by global token range for multi-chunk.
_SM70_PREFILL_META: dict = {}


def _sm70_indexer_logits_prefill(score, q_slice, kv_cache, w_slice, seg_bt,
                                 seq_lens):
    """Prefill indexer logits: packed-Q register-accumulator WMMA kernel
    (~5-6x, bit-identical scores) when available, else the legacy op.

    The pack (pad to a 16-row multiple + view/permute/contiguous) replaces the
    q_slice .contiguous() copy the caller was already paying for.
    """
    import vllm._sm70_ops as sm70_ops

    n = q_slice.shape[0]
    if (
        n >= 8
        and envs.VLLM_SM70_IDX_PACKED_WMMA
        and hasattr(torch.ops._C, "sm70_indexer_logits_packed")
    ):
        n16 = (n + 15) // 16 * 16
        if n16 != n:
            q_slice = torch.nn.functional.pad(
                q_slice, (0, 0, 0, 0, 0, n16 - n)
            )
        q_packed = (
            q_slice.view(n16 // 16, 16, 32, 8, 16)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        sm70_ops.sm70_indexer_logits_packed(
            score, q_packed, kv_cache, w_slice, seg_bt, seq_lens
        )
    else:
        sm70_ops.sm70_indexer_logits(
            score, q_slice.contiguous(), kv_cache, w_slice, seg_bt, seq_lens
        )

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


# ---------------------------------------------------------------------------
# SM70 (Volta) fp16 sparse indexer: no DeepGEMM/fp8. Stores fp16 indexer keys,
# computes logits with the Volta fp16 kernel, top-k with persistent_topk.
# ---------------------------------------------------------------------------
def _sm70_sparse_indexer(
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    import vllm._sm70_ops as sm70_ops

    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer  # profiling / dummy run
    md = attn_metadata[k_cache_prefix]
    assert isinstance(md, DeepseekV32IndexerMetadata)

    num_tokens = k.shape[0]
    # store fp16 indexer keys into the cache (flat rows [n_rows, head_dim]).
    # Device scatter that skips slot < 0 (CUDA-graph padding) on-device — no
    # host sync, no boolean-mask dynamic shapes, race-free (one block/token).
    slot = md.slot_mapping[:num_tokens].to(torch.int32)
    sm70_ops.sm70_indexer_k_store(
        kv_cache, k[:num_tokens].to(torch.float16).contiguous(), slot
    )

    topk_indices_buffer[:num_tokens] = -1

    has_decode = md.num_decodes > 0
    if has_decode:
        dm = md.decode
        assert dm is not None
        nd = md.num_decode_tokens
        block_table = dm.block_table
        seq_lens = dm.seq_lens
        if seq_lens.dim() == 2:
            seq_lens = seq_lens[:, -1].contiguous()
        seq_lens = seq_lens.to(torch.int32)
        max_kv = block_table.shape[1] * kv_cache.shape[1]
        score = torch.empty(
            (nd, max_kv), dtype=torch.float32, device=q.device
        )
        import vllm.envs as envs

        # ALWAYS reserve the persistent_topk radix workspace, even on the
        # fused path (which doesn't use it): the workspace manager sizes
        # during warmup/capture then LOCKS, and the PREFILL branch still uses
        # persistent_topk — without this reservation the first real prefill
        # hits "Workspace is locked ... requires 1.00 MB" on a worker rank
        # and deadlocks the peer PP stage.
        (topk_workspace,) = current_workspace_manager().get_simultaneous(
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        if envs.VLLM_SM70_FUSED_DECODE_TOPK:
            # Fused decode path: the indexer kernel also builds a 4096-bin
            # score histogram; sm70_decode_topk consumes it (threshold +
            # ordered compaction) instead of persistent_topk's multi-pass
            # radix select. ~2x the isolated indexer+topk cost at 16k and
            # near-context-flat top-k. All device-side; CUDA-graph-safe.
            #
            # The extension routes B >= 8 scoring to the WMMA prefill-style
            # kernel which cannot build the histogram, so issue the fused
            # kernels in <8-row chunks (rows are independent; row slices of
            # the score/hist/topk buffers are contiguous). Spec-as-decode
            # verify steps reach nd=8 at max_num_seqs=4 x next_n=2. Chunk
            # count is static per CUDA-graph capture size — graph-safe.
            topk_idx = topk_indices_buffer[:nd, :topk_tokens]
            q_nd = q[:nd].contiguous()
            w_nd = weights[:nd].contiguous()
            hist = torch.zeros((nd, 4096), dtype=torch.int32, device=q.device)
            chunk = nd if nd < 8 else 4
            for lo in range(0, nd, chunk):
                hi = min(lo + chunk, nd)
                sm70_ops.sm70_indexer_logits(
                    score[lo:hi],
                    q_nd[lo:hi],
                    kv_cache,
                    w_nd[lo:hi],
                    block_table[lo:hi],
                    seq_lens[lo:hi],
                    hist[lo:hi],
                )
                sm70_ops.sm70_decode_topk(
                    topk_idx[lo:hi],
                    score[lo:hi],
                    seq_lens[lo:hi],
                    hist[lo:hi],
                    topk_tokens,
                )
        else:
            sm70_ops.sm70_indexer_logits(
                score,
                q[:nd].contiguous(),
                kv_cache,
                weights[:nd].contiguous(),
                block_table,
                seq_lens,
            )
            topk_idx = topk_indices_buffer[:nd, :topk_tokens]
            seq_lens2d = seq_lens.view(nd, 1)
            torch.ops._C.persistent_topk(
                score,
                seq_lens2d,
                topk_idx,
                topk_workspace,
                topk_tokens,
                md.max_seq_len,
            )

    # SM70 sparse PREFILL: per-token causal top-k. Each prefill token attends its
    # own causal context [0, pos]; the indexer picks its top-k over that context
    # (identical to dense out to topk). Reuses sm70_indexer_logits with per-token
    # causal seq_lens + per-token block_table (broadcast per request), then
    # persistent_topk per row. Only active with VLLM_SM70_GLM_SPARSE_PREFILL.
    import vllm.envs as envs

    has_prefill = md.num_prefills > 0
    if has_prefill and envs.VLLM_SM70_GLM_SPARSE_PREFILL:
        pm = md.prefill
        assert pm is not None
        nd = md.num_decode_tokens  # prefill tokens follow decode tokens
        # Per-chunk (tok_lo, n_q, ke) entries for the sparse-attention bridge.
        # The upstream builder may split a step's prefill tokens into SEVERAL
        # chunks (its flat-logits budget is q_len*ctx-bounded: 1024-token
        # engine chunks split beyond ~131k ctx) — the old single-slot stash
        # only kept the LAST chunk and the attention hard-errored on the
        # shape mismatch. Accumulate them all.
        _pf_entries = []
        for chunk in pm.chunks:
            # cu_seqlen_ks/ke are CUMULATIVE offsets into the DeepGEMM
            # flattened-KV buffer (all requests' gathered KV concatenated):
            # ks[i] = start of query-token i's request's KV in that buffer,
            # ke[i] = ks[i] + the token's causal length. The SM70 path reads
            # the PAGED cache directly via per-request block tables, so the
            # value it needs is the TRUE causal length ke[i]-ks[i]. Using raw
            # ke was correct ONLY for the first request in a chunk (ks=0);
            # for any later request it over-stated the causal window by the
            # preceding requests' KV (e.g. a short prompt batched after a 78k
            # long prefill got ke~78k), scoring/attending stale block-table
            # entries -> cross-request garbage. Verified live via SM70DBG.
            ks = chunk.cu_seqlen_ks.to(torch.int32)
            ke = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            n_q = ke.shape[0]
            tok_lo = chunk.token_start
            # per-token block_table: all query tokens of a request share its
            # block_table row. Map each query token to its request.
            bt_chunk = chunk.block_table  # [num_reqs, max_blocks] (main-cache rows)
            num_reqs = chunk.num_reqs
            page = kv_cache.shape[1]
            full_kv = bt_chunk.shape[1] * page
            _KT = 64  # WMMA key-tile; keep max_kv a multiple so no tile is wasted

            # AUTHORITATIVE per-request query boundaries (added to the chunk
            # metadata for this path). The old heuristic (req boundary where ke
            # decreases) MISSED boundaries where the next request's causal
            # length is HIGHER than the previous one's last token (e.g. a short
            # prompt batched together with a mid-flight long prefill chunk) ->
            # wrong block_table rows -> garbage top-k -> corrupted output for
            # the misattributed request. Verified live: mixed 96k-long + short
            # concurrent requests corrupted the short ones.
            qsl_cpu = getattr(chunk, "query_start_loc_cpu", None)
            if qsl_cpu is not None:
                bounds = [int(x) for x in qsl_cpu.tolist()]
            elif num_reqs == 1:
                bounds = [0, n_q]
            else:
                # Fallback heuristic (pre-fix behavior), kept only for safety.
                dec = torch.ones(n_q, dtype=torch.int32, device=ke.device)
                dec[1:] = (ke[1:] < ke[:-1]).to(torch.int32)
                rid = (torch.cumsum(dec, dim=0) - 1).clamp_(0, num_reqs - 1)
                bcpu = rid.cpu()
                bounds = [0] + [
                    i + 1
                    for i in range(n_q - 1)
                    if int(bcpu[i + 1]) != int(bcpu[i])
                ] + [n_q]
                if len(bounds) > num_reqs + 1:  # more splits than requests
                    bounds = bounds[:num_reqs] + [n_q]
                while len(bounds) < num_reqs + 1:
                    bounds.append(n_q)

            # per-token request id for the sparse-attention bridge (below).
            seg_lens = torch.tensor(
                [max(bounds[r + 1] - bounds[r], 0) for r in range(num_reqs)],
                dtype=torch.long,
            )
            req_id = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.long), seg_lens
            ).to(ke.device)

            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )

            # TP key-sharding of the indexer scoring (see envs.py): each rank
            # scores 1/tp of the key positions + local top-k; all-gather the
            # candidate sets + merge top-k == exact global top-k at 1/tp cost.
            shard_tp = 1
            shard_rank = 0
            shard_min_ctx = 0
            if envs.VLLM_SM70_IDXER_TP_SHARD_PREFILL:
                from vllm.distributed import (
                    get_tensor_model_parallel_rank,
                    get_tensor_model_parallel_world_size,
                    tensor_model_parallel_all_gather,
                )

                shard_tp = get_tensor_model_parallel_world_size()
                shard_rank = get_tensor_model_parallel_rank()
                shard_min_ctx = envs.VLLM_SM70_IDXER_TP_SHARD_MIN_CTX

            # Deferred shard-merge batch: one fused all-gather + one
            # batched top-k per CHUNK instead of two 1 MB gathers + a
            # torch.topk kernel swarm per 128-row sub-block (the profiler
            # showed ~12k latency-bound AllGather calls/30 s = 8-12% of GPU
            # time). Entries: (buffer_row_lo, n_rows, local_vals, global_idx).
            _shard_pending = []

            # Launch the indexer PER REQUEST SEGMENT:
            #  - the WMMA kernel (dispatched for >=8 queries) gathers each key
            #    tile via the block table of the query TILE's first row; a tile
            #    straddling two requests mis-gathers for the later request's
            #    rows. Per-segment launches make straddling impossible.
            #  - also lets each segment CAUSAL-BOUND its own logits scan
            #    (ke.max() of the segment, rounded to the page and the WMMA
            #    KT=64 key-tile) instead of the whole chunk's max - only the
            #    key-tiles a segment actually needs are launched (the O(ctx^2)
            #    fix from the earlier session, now per request).
            for r in range(num_reqs):
                lo, hi = bounds[r], bounds[r + 1]
                if hi <= lo:
                    continue
                # Sub-chunk the query rows so the fp32 score buffer stays
                # bounded at very long context: a full 512-token chunk at 200k
                # causal reach would be 512 * 200k * 4 B ~= 400 MB — more than
                # the post-KV VRAM slack. 128-row sub-blocks cap it at ~100 MB
                # and (bonus) causal-bound each sub-block tighter (earlier
                # rows scan less). persistent_topk is per-row, so the split is
                # exact.
                _SUBQ = 128
                # The per-sub-block causal reach (ke.max) is IDENTICAL for all
                # 78 layers of a step (ke comes from the shared chunk metadata),
                # but .item() is a host sync that stalls the launch pipeline.
                # Memoize it on the chunk object so only the first layer pays
                # the sync (78 layers x 4 sub-blocks = ~312 syncs/chunk -> 4).
                _cm_cache = getattr(chunk, "_sm70_causal_max_cache", None)
                if _cm_cache is None:
                    _cm_cache = {}
                    chunk._sm70_causal_max_cache = _cm_cache
                for lo2 in range(lo, hi, _SUBQ):
                    hi2 = min(lo2 + _SUBQ, hi)
                    n_seg = hi2 - lo2
                    ke_seg = ke[lo2:hi2].contiguous()
                    causal_max = _cm_cache.get((lo2, hi2))
                    if causal_max is None:
                        causal_max = int(ke_seg.max().item())
                        _cm_cache[(lo2, hi2)] = causal_max
                    max_kv = min(full_kv, ((causal_max + _KT - 1) // _KT) * _KT)
                    max_kv = max(max_kv, page)  # guard tiny/empty segments
                    q_slice = q[tok_lo + lo2 : tok_lo + hi2].contiguous()
                    w_slice = weights[tok_lo + lo2 : tok_lo + hi2].contiguous()
                    topk_idx = topk_indices_buffer[
                        tok_lo + lo2 : tok_lo + hi2, :topk_tokens
                    ]
                    if shard_tp > 1 and causal_max >= shard_min_ctx:
                        # -- sharded: score only keys [off, off+span) --
                        unit = max(_KT, page)
                        per = (
                            ((max_kv + shard_tp - 1) // shard_tp + unit - 1)
                            // unit
                            * unit
                        )
                        off = shard_rank * per
                        span = min(per, max(max_kv - off, 0))
                        span = max(span, page)  # valid launch shape when empty
                        pg_lo = off // page
                        pg_hi = min(
                            bt_chunk.shape[1], (off + span + page - 1) // page
                        )
                        pg_hi = max(pg_hi, pg_lo + 1)
                        seg_bt = (
                            bt_chunk[r : r + 1, pg_lo:pg_hi]
                            .expand(n_seg, pg_hi - pg_lo)
                            .contiguous()
                        )
                        # local causal length within this shard's slice.
                        # Clamped to >=1 so persistent_topk always has a valid
                        # candidate row; rows whose causal reach ends BEFORE
                        # this shard (sl_raw <= 0) would then wrongly score
                        # position `off` as real, so force their single
                        # candidate to -inf after the kernel (it loses the
                        # merge and never enters the selection).
                        sl_raw = ke_seg - off
                        sl_local = sl_raw.clamp(1, span).contiguous()
                        score = torch.empty(
                            (n_seg, span), dtype=torch.float32, device=q.device
                        )
                        _sm70_indexer_logits_prefill(
                            score, q_slice, kv_cache, w_slice, seg_bt, sl_local
                        )
                        if shard_rank > 0:
                            score[:, 0].masked_fill_(
                                sl_raw <= 0, float("-inf")
                            )
                        loc_idx = torch.empty(
                            (n_seg, topk_tokens),
                            dtype=torch.int32,
                            device=q.device,
                        )
                        torch.ops._C.persistent_topk(
                            score,
                            sl_local.view(n_seg, 1),
                            loc_idx,
                            topk_workspace,
                            topk_tokens,
                            md.max_seq_len,
                        )
                        neg = loc_idx < 0
                        loc_val = score.gather(
                            1, loc_idx.clamp(min=0).to(torch.long)
                        )
                        loc_val.masked_fill_(neg, float("-inf"))
                        gidx = torch.where(neg, loc_idx, loc_idx + off)
                        _shard_pending.append(
                            (tok_lo + lo2, n_seg, loc_val, gidx)
                        )
                    else:
                        bt_cols = min(
                            bt_chunk.shape[1], (max_kv + page - 1) // page
                        )
                        seg_bt = (
                            bt_chunk[r : r + 1, :bt_cols]
                            .expand(n_seg, bt_cols)
                            .contiguous()
                        )
                        score = torch.empty(
                            (n_seg, max_kv), dtype=torch.float32, device=q.device
                        )
                        _sm70_indexer_logits_prefill(
                            score, q_slice, kv_cache, w_slice, seg_bt, ke_seg
                        )
                        torch.ops._C.persistent_topk(
                            score,
                            ke_seg.view(n_seg, 1),
                            topk_idx,
                            topk_workspace,
                            topk_tokens,
                            md.max_seq_len,
                        )
            if _shard_pending:
                K = topk_tokens
                R = sum(p[1] for p in _shard_pending)
                _p2p = None
                if (
                    envs.VLLM_SM70_IDXER_P2P_MERGE
                    and K == 2048
                    and 2 <= shard_tp <= 4
                ):
                    _p2p = _sm70_p2p_state(q.device, K, shard_tp)
                    if _p2p is not None and R > _p2p["rows_cap"]:
                        _p2p = None  # oversize chunk: NCCL fallback
                if _p2p is not None:
                    _p2p["seq"] += 1
                    _seq = _p2p["seq"]
                    _slot = _seq & 1
                    _soff = _slot * _p2p["slot_bytes"]
                    if _seq > 2:
                        # gate slot reuse on peers' merges of seq-2
                        torch.ops._C.sm70_ipc_slot_wait(
                            _p2p["flag_ptrs_t"], shard_tp, _seq - 2
                        )
                    slot_view = (
                        _p2p["buf"][_soff : _soff + R * 2 * K * 4]
                        .view(torch.float32)
                        .view(R, 2 * K)
                    )
                    ro = 0
                    for _, n_r, lv, gi in _shard_pending:
                        slot_view[ro : ro + n_r, :K] = lv
                        slot_view[ro : ro + n_r, K:] = gi.view(torch.float32)
                        ro += n_r
                    torch.ops._C.sm70_ipc_flag_set(
                        _p2p["my_flag_ptr"], 0, _seq
                    )
                    contig = True
                    exp_row = _shard_pending[0][0]
                    for row_lo, n_r, _, _ in _shard_pending:
                        if row_lo != exp_row:
                            contig = False
                            break
                        exp_row = row_lo + n_r
                    if contig:
                        dst = torch.empty(
                            0, dtype=torch.int32, device=q.device
                        )
                        base = _shard_pending[0][0]
                    else:
                        dst = torch.cat(
                            [
                                torch.arange(
                                    row_lo,
                                    row_lo + n_r,
                                    dtype=torch.int32,
                                    device=q.device,
                                )
                                for row_lo, n_r, _, _ in _shard_pending
                            ]
                        )
                        base = -1
                    torch.ops._C.sm70_indexer_shard_merge_p2p(
                        topk_indices_buffer, _p2p["data_ptrs_t"],
                        _p2p["flag_ptrs_t"], dst, base, K, shard_tp, R,
                        _seq, _soff
                    )
                    torch.ops._C.sm70_ipc_flag_set(
                        _p2p["my_flag_ptr"], 1, _seq
                    )
                    # bridge metadata + skip the NCCL epilogue (and the
                    # SPARSE_DEBUG dump, which is not wired for this path)
                    _pf_entries.append((tok_lo, n_q, ke))
                    continue
                # fused payload: [R, 2K] fp32 (vals | idx bitcast to fp32)
                fused = torch.empty(
                    (R, 2 * K), dtype=torch.float32, device=q.device
                )
                ro = 0
                for _, n_r, lv, gi in _shard_pending:
                    fused[ro : ro + n_r, :K] = lv
                    fused[ro : ro + n_r, K:] = gi.view(torch.float32)
                    ro += n_r
                allf = tensor_model_parallel_all_gather(fused, dim=0)
                allf = allf.view(shard_tp, R, 2 * K)
                if (
                    envs.VLLM_SM70_IDXER_FUSED_MERGE
                    and K == 2048
                    and 2 <= shard_tp <= 4
                ):
                    # Fused merge kernel: exact top-K straight off the
                    # all-gather payload (no permute/contiguous round-trips,
                    # no torch.topk radix pass, no gather/where epilogue).
                    # Same selected set as the torch.topk path (identical
                    # column-order tie-break); positions emitted ASCENDING
                    # instead of value-descending — same attention result up
                    # to fp reassociation, and the ascending gather is
                    # page-coalesced. ~1.3 ms/layer -> ~0.35 ms at tp=4.
                    contig = True
                    exp_row = _shard_pending[0][0]
                    for row_lo, n_r, _, _ in _shard_pending:
                        if row_lo != exp_row:
                            contig = False
                            break
                        exp_row = row_lo + n_r
                    if contig:
                        # dst_rows unused when dst_base >= 0 (op still needs
                        # a tensor; 0-elem alloc is allocator-cached)
                        torch.ops._C.sm70_indexer_shard_merge(
                            topk_indices_buffer, allf,
                            torch.empty(0, dtype=torch.int32, device=q.device),
                            _shard_pending[0][0], K
                        )
                    else:
                        dst = torch.cat(
                            [
                                torch.arange(
                                    row_lo,
                                    row_lo + n_r,
                                    dtype=torch.int32,
                                    device=q.device,
                                )
                                for row_lo, n_r, _, _ in _shard_pending
                            ]
                        )
                        torch.ops._C.sm70_indexer_shard_merge(
                            topk_indices_buffer, allf, dst, -1, K
                        )
                else:
                    av = (
                        allf[:, :, :K]
                        .permute(1, 0, 2)
                        .reshape(R, shard_tp * K)
                    )
                    ai = (
                        allf[:, :, K:]
                        .permute(1, 0, 2)
                        .reshape(R, shard_tp * K)
                        .contiguous()
                        .view(torch.int32)
                    )
                    mv, mp = torch.topk(av, K, dim=1)
                    mi = ai.gather(1, mp)
                    mi = torch.where(
                        torch.isinf(mv), torch.full_like(mi, -1), mi
                    ).to(torch.int32)
                    ro = 0
                    for row_lo, n_r, _, _ in _shard_pending:
                        topk_indices_buffer[row_lo : row_lo + n_r, :K].copy_(
                            mi[ro : ro + n_r]
                        )
                        ro += n_r

            # Stash this chunk's per-token causal metadata for the sparse
            # prefill ATTENTION (_forward_mha_sparse). Only the causal seq_len
            # is bridged; the attention derives per-token request ids from its
            # OWN prefill.query_start_loc and builds the per-token block_table
            # from the MAIN latent-cache block_table (the indexer's block
            # table is for the separate INDEXER cache).
            _pf_entries.append((tok_lo, n_q, ke))

            import os
            if os.environ.get("VLLM_SM70_SPARSE_DEBUG") and num_reqs > 1:
                global _SM70_DBG_COUNT
                try:
                    _SM70_DBG_COUNT += 1
                except NameError:
                    _SM70_DBG_COUNT = 1
                if _SM70_DBG_COUNT <= 6:
                    import sys
                    ke_cpu = ke.cpu()
                    lines = [
                        f"[SM70DBG] mixed prefill chunk: num_reqs={num_reqs} "
                        f"n_q={n_q} tok_lo={tok_lo} bounds={bounds}"
                    ]
                    ks_cpu = ks.cpu()
                    for r in range(num_reqs):
                        lo, hi = bounds[r], bounds[r + 1]
                        if hi <= lo:
                            continue
                        idx_row = (
                            topk_indices_buffer[tok_lo + hi - 1, :8].cpu().tolist()
                        )
                        lines.append(
                            f"[SM70DBG]  seg{r}: q[{lo}:{hi}] "
                            f"ke={int(ke_cpu[lo])}..{int(ke_cpu[hi-1])} "
                            f"ks={int(ks_cpu[lo])} "
                            f"bt_row0={int(bt_chunk[r,0])} "
                            f"last_tok_top8idx={idx_row}"
                        )
                    print("\n".join(lines), file=sys.stderr, flush=True)
        _SM70_PREFILL_META["chunks"] = _pf_entries
    return topk_indices_buffer



def _sm70_sparse_indexer_fake(
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sm70_sparse_indexer",
    op_func=_sm70_sparse_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=_sm70_sparse_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


def sm70_sparse_indexer(
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.sm70_sparse_indexer(
        k_cache_prefix,
        kv_cache,
        q,
        k,
        weights,
        topk_tokens,
        head_dim,
        max_model_len,
        topk_indices_buffer,
    )


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
