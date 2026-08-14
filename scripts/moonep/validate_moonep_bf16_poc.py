#!/usr/bin/env python3
"""Distributed validation for the current SGLang MoonEP BF16 reference path.

Run with torchrun on a single NVLink/NVSwitch node, for example:

  PYTHONPATH=python \
  SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 \
  torchrun --standalone --nproc-per-node=4 \
    scripts/moonep/validate_moonep_bf16_poc.py --tokens 128 --hidden-size 1024

The script validates the current SGLang MoonEP BF16 reference path:
MoonEPDispatcher.dispatch -> MoonEPBuffer.prefetch_weight -> BF16 segment runner
-> MoonEPDispatcher.combine.  The expert step is an explicit SiLU reference
runner over synthetic unquantized BF16 weights; this is not production Kimi-K3
SiTU or quantized expert validation.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.layers.moe.token_dispatcher.moonep import (
    MoonEPBuffer,
    MoonEPDispatcher,
    MoonEPExpertWeightLayout,
    run_moonep_bf16_expert,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--prefetch-slots", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def setup_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def make_topk(tokens: int, top_k: int, num_experts: int, device: torch.device):
    # Deterministic but rank-local routing.  Keep weights normalized so the
    # reference magnitude remains bounded.
    topk_ids = torch.randint(
        0,
        num_experts,
        (tokens, top_k),
        device=device,
        dtype=torch.int64,
    )
    raw_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    return topk_ids, topk_weights


def expert_mlp(x, expert_id: int, gate, up, down):
    return F.linear(
        F.silu(F.linear(x, gate[expert_id])) * F.linear(x, up[expert_id]),
        down[expert_id],
    )


def reference_output(hidden, topk_ids, topk_weights, gate, up, down):
    out = torch.zeros_like(hidden)
    tokens, top_k = topk_ids.shape
    for token_idx in range(tokens):
        x = hidden[token_idx : token_idx + 1]
        acc = torch.zeros_like(x)
        for k in range(top_k):
            expert_id = int(topk_ids[token_idx, k].item())
            acc += expert_mlp(x, expert_id, gate, up, down) * topk_weights[
                token_idx, k
            ].to(hidden.dtype)
        out[token_idx] = acc[0]
    return out


def physical_reference_output(dispatch_output, gate, up, down):
    """Compute the dispatched rows using their physical VM-group row indices."""

    hidden_states = dispatch_output.hidden_states
    route_weights_nvs = dispatch_output.route_weights_nvs
    cu_seqlens = dispatch_output.cu_seqlens
    output = torch.empty_like(hidden_states)
    prev = 0
    for group_id in range(cu_seqlens.numel()):
        cur = int(cu_seqlens[group_id].item())
        if cur < prev or cur > hidden_states.shape[0]:
            raise ValueError(
                "MoonEP cu_seqlens must be non-decreasing and within "
                f"dispatched rows: previous={prev}, current={cur}, "
                f"rows={hidden_states.shape[0]}"
            )
        if cur == prev:
            continue

        x = hidden_states[prev:cur]
        y = expert_mlp(x, group_id, gate, up, down)
        if route_weights_nvs is not None:
            y = y * route_weights_nvs[prev:cur].to(dtype=y.dtype).unsqueeze(-1)
        output[prev:cur].copy_(y)
        prev = cur

    if prev < hidden_states.shape[0]:
        output[prev:].zero_()
    return output


def local_experts_to_copy(plan, rank: int) -> torch.Tensor:
    experts_to_copy = plan.experts_to_copy
    if experts_to_copy.ndim == 2:
        return experts_to_copy[rank]
    if experts_to_copy.ndim == 1:
        return experts_to_copy
    raise ValueError(
        "MoonEP plan.experts_to_copy must be rank 1 or rank 2 [R, B], "
        f"got {experts_to_copy.shape}"
    )


def active_prefetch_slots(
    cu_seqlens: torch.Tensor,
    num_experts: int,
    experts_to_copy: torch.Tensor,
) -> list[tuple[int, int]]:
    """Return active ``(slot, source_expert)`` pairs for this rank."""

    active = []
    prev = 0
    for group_id in range(cu_seqlens.numel()):
        cur = int(cu_seqlens[group_id].item())
        if cur < prev:
            raise ValueError(
                "MoonEP cu_seqlens must be non-decreasing: "
                f"previous={prev}, current={cur}"
            )
        if group_id >= num_experts and cur > prev:
            slot = group_id - num_experts
            source_expert = int(experts_to_copy[slot].item())
            if source_expert >= 0:
                active.append((slot, source_expert))
        prev = cur
    return active


def main() -> None:
    args = parse_args()
    os.environ["SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(args.tokens)
    if args.prefetch_slots > 0:
        os.environ["SGLANG_MOONEP_NUM_PREFETCH_SLOTS"] = str(args.prefetch_slots)

    rank, world_size, local_rank = setup_dist()
    try:
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(args.seed + rank)

        num_experts = world_size * args.experts_per_rank
        hidden = torch.randn(
            args.tokens,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        topk_ids, topk_weights = make_topk(args.tokens, args.top_k, num_experts, device)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty(0, device=device),
        )

        dispatcher = MoonEPDispatcher(
            group=dist.group.WORLD,
            router_topk=args.top_k,
            num_experts=num_experts,
            num_local_experts=args.experts_per_rank,
            hidden_size=args.hidden_size,
            params_dtype=torch.bfloat16,
        )

        dispatch_output = dispatcher.dispatch(hidden, topk_output)
        num_prefetch_slots = int(dispatch_output.cu_seqlens.numel()) - num_experts

        # Full global rows are deliberately replicated for this communication
        # correctness PoC. The production path should replace this with true
        # symmetric expert-row mappings owned by each expert's home rank.
        torch.manual_seed(args.seed)
        gate = (
            torch.randn(
                num_experts + num_prefetch_slots,
                args.intermediate_size,
                args.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            / 8
        )
        up = torch.randn_like(gate) / 8
        down = (
            torch.randn(
                num_experts + num_prefetch_slots,
                args.hidden_size,
                args.intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            )
            / 8
        )
        gate[num_experts:].zero_()
        up[num_experts:].zero_()
        down[num_experts:].zero_()
        layout = MoonEPExpertWeightLayout(
            gate.contiguous(),
            up.contiguous(),
            down.contiguous(),
            num_prefetch_slots,
        )

        source_gate = gate[:num_experts].clone()
        source_up = up[:num_experts].clone()
        source_down = down[:num_experts].clone()

        dispatcher.prefetch_weight(dispatch_output.plan, layout)

        experts_to_copy = local_experts_to_copy(dispatch_output.plan, rank)
        slot_rows_ok = experts_to_copy.numel() == num_prefetch_slots
        if slot_rows_ok:
            for slot, source_expert in enumerate(experts_to_copy.tolist()):
                if source_expert < 0:
                    continue
                if source_expert >= num_experts:
                    slot_rows_ok = False
                    continue
                slot_rows_ok = slot_rows_ok and torch.equal(
                    gate[num_experts + slot], source_gate[source_expert]
                )
                slot_rows_ok = slot_rows_ok and torch.equal(
                    up[num_experts + slot], source_up[source_expert]
                )
                slot_rows_ok = slot_rows_ok and torch.equal(
                    down[num_experts + slot], source_down[source_expert]
                )

        # Keep the normal end-to-end check on the original prefetched layout,
        # then probe a cloned layout.  Mutating only the clone's source rows
        # makes a runner that bypasses physical slot rows disagree with this
        # independent physical-row reference, while leaving the real combine
        # inputs untouched.
        active_slots = active_prefetch_slots(
            dispatch_output.cu_seqlens,
            num_experts,
            experts_to_copy,
        )
        probe_gate = gate.clone()
        probe_up = up.clone()
        probe_down = down.clone()
        with torch.no_grad():
            for _slot, source_expert in active_slots:
                probe_gate[source_expert].add_(0.5)
                probe_up[source_expert].add_(0.75)
                probe_down[source_expert].add_(1.0)
        probe_layout = MoonEPExpertWeightLayout(
            probe_gate.contiguous(),
            probe_up.contiguous(),
            probe_down.contiguous(),
            num_prefetch_slots,
        )
        probe_combine_input = run_moonep_bf16_expert(
            dispatch_output,
            probe_layout,
        )
        probe_expected = physical_reference_output(
            dispatch_output,
            probe_gate,
            probe_up,
            probe_down,
        )
        probe_max_abs_err = (
            (probe_combine_input.hidden_states.float() - probe_expected.float())
            .abs()
            .max()
        )
        probe_ok = bool(
            torch.allclose(
                probe_combine_input.hidden_states.float(),
                probe_expected.float(),
                atol=args.atol,
                rtol=args.rtol,
            )
        )

        combine_input = run_moonep_bf16_expert(dispatch_output, layout)
        output = dispatcher.combine(combine_input)

        expected = reference_output(hidden, topk_ids, topk_weights, gate, up, down)
        max_abs_err = (output.float() - expected.float()).abs().max()
        rel_err = max_abs_err / expected.float().abs().max().clamp_min(1e-6)
        local_ok = bool(
            torch.allclose(
                output.float(), expected.float(), atol=args.atol, rtol=args.rtol
            )
        )
        ok_tensor = torch.tensor(
            [1 if local_ok else 0], device=device, dtype=torch.int32
        )
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        slot_rows_ok_tensor = torch.tensor(
            [1 if slot_rows_ok else 0], device=device, dtype=torch.int32
        )
        dist.all_reduce(slot_rows_ok_tensor, op=dist.ReduceOp.MIN)
        active_slot_count_tensor = torch.tensor(
            [len(active_slots)], device=device, dtype=torch.int32
        )
        dist.all_reduce(active_slot_count_tensor, op=dist.ReduceOp.SUM)
        probe_failure_tensor = torch.tensor(
            [1 if active_slots and not probe_ok else 0],
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(probe_failure_tensor, op=dist.ReduceOp.SUM)
        probe_max_abs_err_tensor = probe_max_abs_err.detach().clone().float()
        dist.all_reduce(probe_max_abs_err_tensor, op=dist.ReduceOp.MAX)
        prefetch_slot_compute_validated = bool(
            active_slot_count_tensor.item() > 0 and probe_failure_tensor.item() == 0
        )
        global_ok = bool(
            ok_tensor.item()
            and slot_rows_ok_tensor.item()
            and prefetch_slot_compute_validated
        )

        result = {
            "expert_compute": "reference_silu",
            "validation_scope": "moonep_dispatch_prefetch_reference_combine",
            "rank": rank,
            "world_size": world_size,
            "tokens": args.tokens,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": args.top_k,
            "num_experts": num_experts,
            "num_prefetch_slots": num_prefetch_slots,
            "max_abs_err": float(max_abs_err.item()),
            "relative_err": float(rel_err.item()),
            "local_ok": local_ok,
            "prefetch_slot_rows_verified": bool(slot_rows_ok_tensor.item()),
            "active_prefetch_slot_groups": int(active_slot_count_tensor.item()),
            "prefetch_slot_compute_validated": prefetch_slot_compute_validated,
            "prefetch_probe_max_abs_err": float(probe_max_abs_err_tensor.item()),
            "global_ok": global_ok,
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        dist.barrier(device_ids=[local_rank])
        if rank == 0 and not global_ok:
            raise SystemExit(1)
    finally:
        # MoonEP owns VMM/NVLink resources and must be destroyed while its
        # process group is still alive.
        MoonEPBuffer.destroy_all_buffers()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
