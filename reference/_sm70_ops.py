# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def _op(name: str):
    if not hasattr(torch.ops._C, name):
        raise RuntimeError(
            f"SM70 TurboMind op _C::{name} is not available. "
            "Build vLLM with CUDA arch 7.0 to enable it."
        )
    return getattr(torch.ops._C, name)


def silu_and_mul_interleaved(out: torch.Tensor, input: torch.Tensor) -> None:
    _op("silu_and_mul_interleaved")(out, input)


if hasattr(torch.ops._C, "silu_and_mul_interleaved"):

    @register_fake("_C::silu_and_mul_interleaved")
    def _silu_and_mul_interleaved_fake(
        out: torch.Tensor, input: torch.Tensor
    ) -> None:
        del out, input
        return None


def awq_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("awq_sm70_prepare")(
        qweight, scales, qzeros, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "awq_sm70_prepare"):

    @register_fake("_C::awq_sm70_prepare")
    def _awq_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del qzeros, group_size, interleave_gated_silu
        n = qweight.size(1) * 8
        num_groups = scales.size(0)
        tm_weight = torch.empty_like(qweight)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def uint4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("uint4_sm70_prepare")(
        qweight, scales, zeros, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "uint4_sm70_prepare"):

    @register_fake("_C::uint4_sm70_prepare")
    def _uint4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del zeros, group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.int32,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def fp8_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("fp8_sm70_prepare")(
        qweight, scales, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "fp8_sm70_prepare"):

    @register_fake("_C::fp8_sm70_prepare")
    def _fp8_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        n = qweight.size(0)
        k = qweight.size(1)
        num_groups = scales.size(1)
        tm_weight = torch.empty((k, n), dtype=torch.uint8, device=qweight.device)
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def mxfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("mxfp4_sm70_prepare")(
        qweight, scales, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "mxfp4_sm70_prepare"):

    @register_fake("_C::mxfp4_sm70_prepare")
    def _mxfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.uint8,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def nvfp4_sm70_prepare(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    interleave_gated_silu: bool = False,
) -> list[torch.Tensor]:
    return _op("nvfp4_sm70_prepare")(
        qweight, scales, group_size, interleave_gated_silu
    )


if hasattr(torch.ops._C, "nvfp4_sm70_prepare"):

    @register_fake("_C::nvfp4_sm70_prepare")
    def _nvfp4_sm70_prepare_fake(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        interleave_gated_silu: bool,
    ) -> list[torch.Tensor]:
        del group_size, interleave_gated_silu
        k = qweight.size(0)
        n = qweight.size(1)
        num_groups = scales.size(0)
        tm_weight = torch.empty(
            (k, n // 8),
            dtype=torch.int32,
            device=qweight.device,
        )
        tm_scales = torch.empty(
            (num_groups, n),
            dtype=torch.float16,
            device=qweight.device,
        )
        meta = torch.empty((2,), dtype=torch.int64, device=qweight.device)
        return [tm_weight, tm_scales, meta]


def sm70_f16_prepare(weight: torch.Tensor) -> list[torch.Tensor]:
    return _op("sm70_f16_prepare")(weight)


if hasattr(torch.ops._C, "sm70_f16_prepare"):

    @register_fake("_C::sm70_f16_prepare")
    def _sm70_f16_prepare_fake(weight: torch.Tensor) -> list[torch.Tensor]:
        meta = torch.empty((1,), dtype=torch.int64, device=weight.device)
        return [torch.empty_like(weight), meta]


def awq_gemm_sm70(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    return _op("awq_gemm_sm70")(input, qweight, scales, group_size, k_ld, q_ld)


if hasattr(torch.ops._C, "awq_gemm_sm70"):

    @register_fake("_C::awq_gemm_sm70")
    def _awq_gemm_sm70_fake(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
    ) -> torch.Tensor:
        del scales, group_size, k_ld, q_ld
        return torch.empty(
            (input.size(0), qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        )


def awq_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "awq_gemm_sm70_out"):

    @register_fake("_C::awq_gemm_sm70_out")
    def _awq_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def awq_gemm_sm70_out_tile_reduce(
    out: torch.Tensor,
    staging: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    fa_ptr: int,
    tile_numel: int,
    reducer_blocks: int,
    kernel_reducer_blocks: int,
    overlap: bool,
) -> None:
    _op("awq_gemm_sm70_out_tile_reduce")(
        out,
        staging,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        fa_ptr,
        tile_numel,
        reducer_blocks,
        kernel_reducer_blocks,
        overlap,
    )


if hasattr(torch.ops._C, "awq_gemm_sm70_out_tile_reduce"):

    @register_fake("_C::awq_gemm_sm70_out_tile_reduce")
    def _awq_gemm_sm70_out_tile_reduce_fake(
        out: torch.Tensor,
        staging: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        fa_ptr: int,
        tile_numel: int,
        reducer_blocks: int,
        kernel_reducer_blocks: int,
        overlap: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "fp8_gemm_sm70_out"):

    @register_fake("_C::fp8_gemm_sm70_out")
    def _fp8_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def mxfp4_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("mxfp4_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "mxfp4_gemm_sm70_out"):

    @register_fake("_C::mxfp4_gemm_sm70_out")
    def _mxfp4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def nvfp4_gemm_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("nvfp4_gemm_sm70_out")(
        out, input, qweight, scales, group_size, k_ld, q_ld, gated_silu
    )


if hasattr(torch.ops._C, "nvfp4_gemm_sm70_out"):

    @register_fake("_C::nvfp4_gemm_sm70_out")
    def _nvfp4_gemm_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        k_ld: int,
        q_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_gemm_sm70_out_auto(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    _op("fp8_gemm_sm70_out_auto")(out, input, qweight, scales)


def fp8_gemm_sm70_out_meta(
    out: torch.Tensor,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    gated_silu: bool = False,
) -> None:
    _op("fp8_gemm_sm70_out_meta")(out, input, qweight, scales, meta, gated_silu)


def sm70_f16_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _op("sm70_f16_gemm")(input, weight)


if hasattr(torch.ops._C, "sm70_f16_gemm"):

    @register_fake("_C::sm70_f16_gemm")
    def _sm70_f16_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (input.size(0), weight.size(0)),
            dtype=input.dtype,
            device=input.device,
        )


def sm70_f16_gemm_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    gated_silu: bool = False,
) -> None:
    _op("sm70_f16_gemm_out")(out, input, weight, k_ld, gated_silu)


if hasattr(torch.ops._C, "sm70_f16_gemm_out"):

    @register_fake("_C::sm70_f16_gemm_out")
    def _sm70_f16_gemm_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        gated_silu: bool,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_out")
    def _sm70_f16_lm_head_top1_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_lm_head_top1_tc_out(
    values_out: torch.Tensor,
    indices_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    k_ld: int,
    vocab_start_index: int,
    num_vocab_padding: int,
) -> None:
    _op("sm70_f16_lm_head_top1_tc_out")(
        values_out,
        indices_out,
        input,
        weight,
        k_ld,
        vocab_start_index,
        num_vocab_padding,
    )


if hasattr(torch.ops._C, "sm70_f16_lm_head_top1_tc_out"):

    @register_fake("_C::sm70_f16_lm_head_top1_tc_out")
    def _sm70_f16_lm_head_top1_tc_out_fake(
        values_out: torch.Tensor,
        indices_out: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        k_ld: int,
        vocab_start_index: int,
        num_vocab_padding: int,
    ) -> None:
        return None


def sm70_f16_gate_mul_out(
    out: torch.Tensor,
    input: torch.Tensor,
    gate_weight: torch.Tensor,
) -> None:
    _op("sm70_f16_gate_mul_out")(out, input, gate_weight)


if hasattr(torch.ops._C, "sm70_f16_gate_mul_out"):

    @register_fake("_C::sm70_f16_gate_mul_out")
    def _sm70_f16_gate_mul_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> None:
        return None


def sm70_gemm_import_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_import_cache")(device_hint, path)


def sm70_gemm_export_cache(device_hint: torch.Tensor, path: str) -> int:
    return _op("sm70_gemm_export_cache")(device_hint, path)


def awq_moe_build_strided_ptrs(
    tm_weights: torch.Tensor,
    tm_scales: torch.Tensor,
    k_ld: int,
    q_ld: int,
    num_experts: int,
) -> list[torch.Tensor]:
    return _op("awq_moe_build_strided_ptrs")(
        tm_weights, tm_scales, k_ld, q_ld, num_experts
    )


if hasattr(torch.ops._C, "awq_moe_build_strided_ptrs"):

    @register_fake("_C::awq_moe_build_strided_ptrs")
    def _awq_moe_build_strided_ptrs_fake(
        tm_weights: torch.Tensor,
        tm_scales: torch.Tensor,
        k_ld: int,
        q_ld: int,
        num_experts: int,
    ) -> list[torch.Tensor]:
        del tm_scales, k_ld, q_ld
        buf = num_experts * 16
        opts = dict(dtype=torch.uint8, device=tm_weights.device)
        return [torch.empty(buf, **opts), torch.empty(buf, **opts)]


def awq_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


def awq_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("awq_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if hasattr(torch.ops._C, "awq_moe_gemm_sm70_out"):

    @register_fake("_C::awq_moe_gemm_sm70_out")
    def _awq_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C, "awq_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_C::awq_moe_gemm_sm70_per_expert_dispatch_out")
    def _awq_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


def awq_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_dense_stage_sm70_out")
    def _awq_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_active_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    permuted_experts_id: torch.Tensor,
    active_expert_offsets: torch.Tensor,
    active_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    total_slots: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_active_dense_stage_sm70_out")(
        out,
        input,
        permuted_experts_id,
        active_expert_offsets,
        active_expert_ids,
        ptrs_w,
        ptrs_s,
        total_slots,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_active_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_active_dense_stage_sm70_out")
    def _awq_moe_active_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        permuted_experts_id: torch.Tensor,
        active_expert_offsets: torch.Tensor,
        active_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        total_slots: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_single_token_dense_stage_sm70_out")
    def _awq_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("awq_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_C::awq_moe_single_token_indexed_dense_stage_sm70_out")
    def _awq_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def awq_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_dense_w13_sm70_out")
    def _awq_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
    expert_map: torch.Tensor | None = None,
) -> None:
    _op("awq_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
        expert_map,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_indexed_dense_w13_sm70_out")
    def _awq_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        return None


def awq_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_C::awq_moe_single_token_compact_dense_w13_sm70_out")
    def _awq_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def awq_moe_single_token_exact_layout_prepare(
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    compact_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    num_experts: int,
) -> None:
    _op("awq_moe_single_token_exact_layout_prepare")(
        topk_ids,
        x,
        compact_input,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        num_experts,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_exact_layout_prepare"):

    @register_fake("_C::awq_moe_single_token_exact_layout_prepare")
    def _awq_moe_single_token_exact_layout_prepare_fake(
        topk_ids: torch.Tensor,
        x: torch.Tensor,
        compact_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        num_experts: int,
    ) -> None:
        return None


def awq_moe_single_token_weighted_reduce_out(
    sorted_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    hidden_logical_size: int,
    valid_count: torch.Tensor | None = None,
) -> None:
    _op("awq_moe_single_token_weighted_reduce_out")(
        sorted_output,
        topk_weights,
        inv_permuted_idx,
        out,
        top_k,
        hidden_logical_size,
        valid_count,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_weighted_reduce_out"):

    @register_fake("_C::awq_moe_single_token_weighted_reduce_out")
    def _awq_moe_single_token_weighted_reduce_out_fake(
        sorted_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        out: torch.Tensor,
        top_k: int,
        hidden_logical_size: int,
        valid_count: torch.Tensor | None = None,
    ) -> None:
        return None


def awq_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("awq_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "awq_moe_single_token_sm70_out"):

    @register_fake("_C::awq_moe_single_token_sm70_out")
    def _awq_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        del (
            out,
            x,
            topk_weights,
            topk_ids,
            src_w13_ptrs_w_rows,
            src_w13_ptrs_s_rows,
            src_w2_ptrs_w_rows,
            src_w2_ptrs_s_rows,
            compact_input,
            intermediate,
            sorted_output,
            dst_w13_ptrs_w_rows,
            dst_w13_ptrs_s_rows,
            dst_w2_ptrs_w_rows,
            dst_w2_ptrs_s_rows,
            expert_offsets,
            inv_permuted_idx,
            w13_k,
            w13_n,
            w2_k,
            w2_n,
            group_size,
            hidden_logical_size,
        )
        return None


def fp8_moe_gemm_sm70_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


def fp8_moe_gemm_sm70_per_expert_dispatch_out(
    out: torch.Tensor,
    sorted_input: torch.Tensor,
    expert_offsets: torch.Tensor,
    strided_ptrs_w: torch.Tensor,
    strided_ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
    gated_silu: bool = False,
) -> None:
    _op("fp8_moe_gemm_sm70_per_expert_dispatch_out")(
        out,
        sorted_input,
        expert_offsets,
        strided_ptrs_w,
        strided_ptrs_s,
        num_experts,
        k,
        n,
        group_size,
        gated_silu,
    )


if hasattr(torch.ops._C, "fp8_moe_gemm_sm70_out"):

    @register_fake("_C::fp8_moe_gemm_sm70_out")
    def _fp8_moe_gemm_sm70_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


if hasattr(torch.ops._C, "fp8_moe_gemm_sm70_per_expert_dispatch_out"):

    @register_fake("_C::fp8_moe_gemm_sm70_per_expert_dispatch_out")
    def _fp8_moe_gemm_sm70_per_expert_dispatch_out_fake(
        out: torch.Tensor,
        sorted_input: torch.Tensor,
        expert_offsets: torch.Tensor,
        strided_ptrs_w: torch.Tensor,
        strided_ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
        gated_silu: bool,
    ) -> None:
        return None


def fp8_moe_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    dense_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        dense_expert_ids,
        ptrs_w,
        ptrs_s,
        num_experts,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_dense_stage_sm70_out")
    def _fp8_moe_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        dense_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        num_experts: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_dense_stage_sm70_out")
    def _fp8_moe_single_token_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_stage_sm70_out(
    out: torch.Tensor,
    input: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    ptrs_w: torch.Tensor,
    ptrs_s: torch.Tensor,
    top_k: int,
    k: int,
    n: int,
    group_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_stage_sm70_out")(
        out,
        input,
        expert_offsets,
        sorted_expert_ids,
        ptrs_w,
        ptrs_s,
        top_k,
        k,
        n,
        group_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_stage_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_indexed_dense_stage_sm70_out")
    def _fp8_moe_single_token_indexed_dense_stage_sm70_out_fake(
        out: torch.Tensor,
        input: torch.Tensor,
        expert_offsets: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        ptrs_w: torch.Tensor,
        ptrs_s: torch.Tensor,
        top_k: int,
        k: int,
        n: int,
        group_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_dense_w13_sm70_out")
    def _fp8_moe_single_token_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_indexed_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_indexed_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_indexed_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_indexed_dense_w13_sm70_out")
    def _fp8_moe_single_token_indexed_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_compact_dense_w13_sm70_out(
    gate_up: torch.Tensor,
    compact_input: torch.Tensor,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_ptrs_w: torch.Tensor,
    w13_ptrs_s: torch.Tensor,
    compact_w13_ptrs_w: torch.Tensor,
    compact_w13_ptrs_s: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_offsets64: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    w13_k: int,
    w13_n: int,
    group_size: int,
    hidden_logical_size: int,
) -> None:
    _op("fp8_moe_single_token_compact_dense_w13_sm70_out")(
        gate_up,
        compact_input,
        x,
        topk_ids,
        w13_ptrs_w,
        w13_ptrs_s,
        compact_w13_ptrs_w,
        compact_w13_ptrs_s,
        expert_offsets,
        expert_offsets64,
        inv_permuted_idx,
        sorted_expert_ids,
        w13_k,
        w13_n,
        group_size,
        hidden_logical_size,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_compact_dense_w13_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_compact_dense_w13_sm70_out")
    def _fp8_moe_single_token_compact_dense_w13_sm70_out_fake(
        gate_up: torch.Tensor,
        compact_input: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        w13_ptrs_w: torch.Tensor,
        w13_ptrs_s: torch.Tensor,
        compact_w13_ptrs_w: torch.Tensor,
        compact_w13_ptrs_s: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_offsets64: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        w13_k: int,
        w13_n: int,
        group_size: int,
        hidden_logical_size: int,
    ) -> None:
        return None


def fp8_moe_single_token_sm70_out(
    out: torch.Tensor,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    src_w13_ptrs_w_rows: torch.Tensor,
    src_w13_ptrs_s_rows: torch.Tensor,
    src_w2_ptrs_w_rows: torch.Tensor,
    src_w2_ptrs_s_rows: torch.Tensor,
    compact_input: torch.Tensor,
    gate_up: torch.Tensor,
    intermediate: torch.Tensor,
    sorted_output: torch.Tensor,
    sorted_weights: torch.Tensor,
    dst_w13_ptrs_w_rows: torch.Tensor,
    dst_w13_ptrs_s_rows: torch.Tensor,
    dst_w2_ptrs_w_rows: torch.Tensor,
    dst_w2_ptrs_s_rows: torch.Tensor,
    expert_offsets: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    broadcast_input_indices: torch.Tensor,
    w2_raw_weight: torch.Tensor,
    w2_raw_scale_inv: torch.Tensor,
    w13_k: int,
    w13_n: int,
    w2_k: int,
    w2_n: int,
    group_size: int,
    hidden_logical_size: int,
    fused_gated_silu: bool,
    fused_weighted_reduce: bool,
    broadcast_input: bool,
    w2_direct_reduce: bool,
    indexed_expert_ptrs: bool,
    exact_per_route: bool,
) -> None:
    _op("fp8_moe_single_token_sm70_out")(
        out,
        x,
        topk_weights,
        topk_ids,
        src_w13_ptrs_w_rows,
        src_w13_ptrs_s_rows,
        src_w2_ptrs_w_rows,
        src_w2_ptrs_s_rows,
        compact_input,
        gate_up,
        intermediate,
        sorted_output,
        sorted_weights,
        dst_w13_ptrs_w_rows,
        dst_w13_ptrs_s_rows,
        dst_w2_ptrs_w_rows,
        dst_w2_ptrs_s_rows,
        expert_offsets,
        inv_permuted_idx,
        sorted_expert_ids,
        broadcast_input_indices,
        w2_raw_weight,
        w2_raw_scale_inv,
        w13_k,
        w13_n,
        w2_k,
        w2_n,
        group_size,
        hidden_logical_size,
        fused_gated_silu,
        fused_weighted_reduce,
        broadcast_input,
        w2_direct_reduce,
        indexed_expert_ptrs,
        exact_per_route,
    )


if hasattr(torch.ops._C, "fp8_moe_single_token_sm70_out"):

    @register_fake("_C::fp8_moe_single_token_sm70_out")
    def _fp8_moe_single_token_sm70_out_fake(
        out: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        src_w13_ptrs_w_rows: torch.Tensor,
        src_w13_ptrs_s_rows: torch.Tensor,
        src_w2_ptrs_w_rows: torch.Tensor,
        src_w2_ptrs_s_rows: torch.Tensor,
        compact_input: torch.Tensor,
        gate_up: torch.Tensor,
        intermediate: torch.Tensor,
        sorted_output: torch.Tensor,
        sorted_weights: torch.Tensor,
        dst_w13_ptrs_w_rows: torch.Tensor,
        dst_w13_ptrs_s_rows: torch.Tensor,
        dst_w2_ptrs_w_rows: torch.Tensor,
        dst_w2_ptrs_s_rows: torch.Tensor,
        expert_offsets: torch.Tensor,
        inv_permuted_idx: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        broadcast_input_indices: torch.Tensor,
        w2_raw_weight: torch.Tensor,
        w2_raw_scale_inv: torch.Tensor,
        w13_k: int,
        w13_n: int,
        w2_k: int,
        w2_n: int,
        group_size: int,
        hidden_logical_size: int,
        fused_gated_silu: bool,
        fused_weighted_reduce: bool,
        broadcast_input: bool,
        w2_direct_reduce: bool,
        indexed_expert_ptrs: bool,
        exact_per_route: bool,
    ) -> None:
        return None


def sm70_mla_decode(
    o: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> None:
    """Volta WMMA dense MLA decode: writes o [B,H,512] fp16, lse [B,H] fp32."""
    _op("sm70_mla_decode")(o, lse, q, kv_cache, block_table, seq_lens, scale)


if hasattr(torch.ops._C, "sm70_mla_decode"):

    @register_fake("_C::sm70_mla_decode")
    def _sm70_mla_decode_fake(
        o: torch.Tensor,
        lse: torch.Tensor,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> None:
        return None


def sm70_attn_fp8_gemv(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
) -> None:
    """fp8-e4m3 weight-only GEMV y[M,N] = x[M,K] @ W[N,K]^T (M <= 8).

    w: uint8 e4m3 bits [N,K]; s: fp32 [128,128]-block scale_inv
    [ceil(N/128), ceil(K/128)] (official DeepSeek/GLM fp8 ckpt layout).
    """
    _op("sm70_attn_fp8_gemv")(y, x, w, s)


if hasattr(torch.ops._C, "sm70_attn_fp8_gemv"):

    @register_fake("_C::sm70_attn_fp8_gemv")
    def _sm70_attn_fp8_gemv_fake(
        y: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        s: torch.Tensor,
    ) -> None:
        return None


def sm70_attn_fp8_dequant(
    out: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
) -> None:
    """Dequantize fp8-e4m3 block-scaled weights into an fp16 scratch [N,K]."""
    _op("sm70_attn_fp8_dequant")(out, w, s)


if hasattr(torch.ops._C, "sm70_attn_fp8_dequant"):

    @register_fake("_C::sm70_attn_fp8_dequant")
    def _sm70_attn_fp8_dequant_fake(
        out: torch.Tensor,
        w: torch.Tensor,
        s: torch.Tensor,
    ) -> None:
        return None


def sm70_sparse_mla_decode(
    o: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> None:
    """Volta WMMA sparse (top-k) MLA decode: writes o [B,H,512] fp16, lse fp32."""
    _op("sm70_sparse_mla_decode")(
        o, lse, q, kv_cache, idx, block_table, seq_lens, scale
    )


if hasattr(torch.ops._C, "sm70_sparse_mla_decode"):

    @register_fake("_C::sm70_sparse_mla_decode")
    def _sm70_sparse_mla_decode_fake(
        o: torch.Tensor,
        lse: torch.Tensor,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        idx: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> None:
        return None


def sm70_indexer_logits(
    score: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    hist: torch.Tensor | None = None,
) -> None:
    """Volta fp16 DSA indexer logits -> score [B, max_kv] fp32 (pad=-inf).

    hist (optional, decode path): pre-zeroed int32 [B, 4096] — the kernel also
    builds the level-1 top-k histogram for sm70_decode_topk.
    """
    _op("sm70_indexer_logits")(
        score, q, k_cache, weights, block_table, seq_lens, hist
    )


def sm70_indexer_logits_packed(
    score: torch.Tensor,
    q_packed: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Packed-Q register-accumulator WMMA indexer logits (fast prefill path).

    q_packed: [32, ceil(B/16), 8, 16, 16] fp16 contiguous (from q [B, 32, 128]
    via pad-to-16 + view + permute). Scores are bit-identical to
    sm70_indexer_logits' WMMA path at ~5-6x the speed.
    """
    _op("sm70_indexer_logits_packed")(
        score, q_packed, k_cache, weights, block_table, seq_lens
    )


def sm70_decode_topk(
    idx: torch.Tensor,
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    hist: torch.Tensor,
    k: int,
) -> None:
    """Fused decode top-k select: idx [B, k] <- top-k positions of score
    (p < seq_lens), -1 padded. hist is the histogram built by
    sm70_indexer_logits. CUDA-graph-safe."""
    _op("sm70_decode_topk")(idx, score, seq_lens, hist, k)


if hasattr(torch.ops._C, "sm70_decode_topk"):

    @register_fake("_C::sm70_decode_topk")
    def _sm70_decode_topk_fake(
        idx: torch.Tensor,
        score: torch.Tensor,
        seq_lens: torch.Tensor,
        hist: torch.Tensor,
        k: int,
    ) -> None:
        return None


def sm70_indexer_k_store(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """CUDA-graph-safe fp16 indexer-K scatter into the paged cache (skips <0)."""
    _op("sm70_indexer_k_store")(kv_cache, k, slot_mapping)


if hasattr(torch.ops._C, "sm70_indexer_k_store"):

    @register_fake("_C::sm70_indexer_k_store")
    def _sm70_indexer_k_store_fake(
        kv_cache: torch.Tensor,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        return None



if hasattr(torch.ops._C, "sm70_indexer_logits"):

    @register_fake("_C::sm70_indexer_logits")
    def _sm70_indexer_logits_fake(
        score: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        hist: torch.Tensor | None = None,
    ) -> None:
        return None


if hasattr(torch.ops._C, "sm70_indexer_logits_packed"):

    @register_fake("_C::sm70_indexer_logits_packed")
    def _sm70_indexer_logits_packed_fake(
        score: torch.Tensor,
        q_packed: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        return None
