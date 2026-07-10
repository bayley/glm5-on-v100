# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 (Volta / V100) MLA prefill backend.

The default FLASH_ATTN MLA prefill backend requires the vllm_flash_attn
extension (FA2/FA3), which is not built on SM70 / V100 wheels. This backend
provides a dependency-free varlen MLA prefill using plain PyTorch scaled
dot-product attention, run per-sequence over the ragged (cu_seqlens) layout.

It supports:
- ``run_prefill_new_tokens``: causal self-attention over the new tokens of
  each request (q_len == k_len).
- ``run_prefill_context_chunk``: non-causal cross-attention of the new-token
  queries against a gathered context chunk (q_len != k_len), returning the
  log-sum-exp needed to merge chunked-context partials.

MLA uses different q/k head dims (qk_nope + qk_rope) and a smaller v head
dim; both are handled directly (no V padding needed since we compute the
attention ourselves). Correctness-first: this is the mathematically exact
dense-MLA prefill. Performance tuning (Triton flash kernel) is future work.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


def _kv_block_size() -> int:
    """K/V tile length for the streaming (flash-style) softmax.

    Bounds prefill attention scratch to O(H * q_len * BLOCK_K) instead of
    O(H * q_len * k_len). Tunable via VLLM_SM70_MLA_PREFILL_KV_BLOCK.
    """
    try:
        v = int(envs.VLLM_SM70_MLA_PREFILL_KV_BLOCK)
    except (ValueError, TypeError):
        v = 512
    return max(1, v)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )

logger = init_logger(__name__)


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    return_softmax_lse: bool,
):
    """Ragged multi-head attention in fp32, computed one sequence at a time.

    Uses a *streaming* (flash-style) online softmax over K/V tiles of length
    ``_kv_block_size()`` so peak attention scratch is O(H * q_len * BLOCK_K),
    not O(H * q_len * k_len). This is what lets prefill run at long context on
    V100 (32 GB) without materializing the full [H, q_len, k_len] scores/probs
    matrices (which are multiple GiB at 4k+ context with 64 heads). The result
    (out + LSE) is the mathematically exact dense-MLA prefill, identical to the
    non-tiled reference up to fp32 rounding.

    q: [total_q, H, Dq], k: [total_k, H, Dk (== Dq)], v: [total_k, H, Dv].
    cu_seqlens_*: int32 [num_seqs + 1] cumulative token offsets.
    Returns out [total_q, H, Dv] (fp16/bf16 of q.dtype) and optionally
    LSE [H, total_q] in fp32 (matching vllm_flash_attn's layout).
    """
    num_heads = q.shape[1]
    v_head_dim = v.shape[-1]
    total_q = q.shape[0]
    block_k = _kv_block_size()

    out = torch.empty(
        (total_q, num_heads, v_head_dim), dtype=q.dtype, device=q.device
    )
    lse = None
    if return_softmax_lse:
        lse = torch.empty(
            (num_heads, total_q), dtype=torch.float32, device=q.device
        )

    q_starts = cu_seqlens_q.tolist()
    k_starts = cu_seqlens_k.tolist()
    num_seqs = len(q_starts) - 1

    for i in range(num_seqs):
        qs, qe = q_starts[i], q_starts[i + 1]
        ks, ke = k_starts[i], k_starts[i + 1]
        q_len = qe - qs
        k_len = ke - ks
        if q_len == 0:
            continue

        # [H, q_len, D]
        qi = q[qs:qe].transpose(0, 1).float()
        if k_len == 0:
            # No keys for this chunk: attention over nothing. Emit zeros and
            # a -inf LSE so merge_attn_states discards this partial.
            out[qs:qe].zero_()
            if lse is not None:
                lse[:, qs:qe] = float("-inf")
            continue

        # Causal diagonal is aligned to the END of the key sequence (queries
        # are the last q_len positions of a k_len-long sequence).
        offset = k_len - q_len if causal else 0
        qpos = (
            torch.arange(q_len, device=q.device).unsqueeze(1) + offset
            if causal
            else None
        )

        # Online (streaming) softmax accumulators over K tiles.
        # m: running max [H, q_len]; l: running denom [H, q_len];
        # acc: running weighted V sum [H, q_len, Dv].
        m = torch.full(
            (num_heads, q_len), float("-inf"), device=q.device, dtype=torch.float32
        )
        l = torch.zeros(
            (num_heads, q_len), device=q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (num_heads, q_len, v_head_dim), device=q.device, dtype=torch.float32
        )

        for kb in range(ks, ke, block_k):
            kb_end = min(kb + block_k, ke)
            n = kb_end - kb
            # [H, n, D] / [H, n, Dv]
            kj = k[kb:kb_end].transpose(0, 1).float()
            vj = v[kb:kb_end].transpose(0, 1).float()

            # scores tile: [H, q_len, n]
            sj = torch.matmul(qi, kj.transpose(-1, -2)) * softmax_scale

            if causal:
                # absolute key positions for this tile
                kpos = torch.arange(
                    kb - ks, kb_end - ks, device=q.device
                ).unsqueeze(0)
                disallowed = kpos > qpos  # [q_len, n]
                sj = sj.masked_fill(disallowed.unsqueeze(0), float("-inf"))

            # online-softmax update
            tile_max = torch.max(sj, dim=-1).values  # [H, q_len]
            m_new = torch.maximum(m, tile_max)
            # guard rows still fully -inf (no valid keys yet) to avoid NaNs
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)  # rescale prior accs; exp(-inf-0)=0
            alpha = torch.where(torch.isinf(m), torch.zeros_like(alpha), alpha)
            p = torch.exp(sj - m_safe.unsqueeze(-1))  # [H, q_len, n]

            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, vj)
            m = m_new

        # finalize
        denom = torch.where(l == 0, torch.ones_like(l), l)
        oi = acc / denom.unsqueeze(-1)  # [H, q_len, Dv]
        out[qs:qe] = oi.transpose(0, 1).to(q.dtype)

        if lse is not None:
            m_finite = torch.where(torch.isinf(m), torch.zeros_like(m), m)
            seq_lse = m_finite + torch.log(l)
            # rows with no valid keys -> -inf so merge discards them
            fully_masked = l == 0
            seq_lse = torch.where(
                fully_masked,
                torch.full_like(seq_lse, float("-inf")),
                seq_lse,
            )
            lse[:, qs:qe] = seq_lse

    if return_softmax_lse:
        return out, lse
    return out


class SM70TritonMLAPrefillBackend(MLAPrefillBackend):
    """Dependency-free MLA prefill for Volta / V100 (SM70)."""

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "SM70_TRITON"

    @classmethod
    def supports_compute_capability(
        cls, device_capability: "DeviceCapability"
    ) -> bool:
        return device_capability.major == 7

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ):
        return _varlen_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        assert self._prefill_metadata.chunked_context is not None
        return _varlen_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[
                chunk_idx
            ],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )
