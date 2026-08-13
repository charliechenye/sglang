from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, NoReturn, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import DeepEPMode

_MOONEP_SPLIT_PHASE_UNSUPPORTED_MESSAGE = (
    "The current SGLang MoonEP BF16 reference path supports synchronous/eager "
    "dispatch, weight prefetch, and combine only; split-phase/overlap methods "
    "are not implemented. MoonEP uses a distinct dispatch format and is not "
    "wire-compatible with DeepEP."
)
_MOONEP_CANONICAL_LAYOUT_UNSUPPORTED_MESSAGE = (
    "The current SGLang MoonEP BF16 reference path requires canonical [Gate, Up] "
    "unquantized expert storage; the selected MoE weight method uses a transformed "
    "expert layout."
)
_MOONEP_HETEROGENEOUS_EXPERT_OWNERSHIP_UNSUPPORTED_MESSAGE = (
    "The current SGLang MoonEP BF16 reference path requires all routed experts in "
    "global GPU expert storage and does not support heterogeneous CPU/GPU expert "
    "ownership."
)


def _unwrap_moe_weight_method(quant_method: Any) -> Any:
    """Return the concrete weight method behind optional execution wrappers."""

    seen = set()
    while hasattr(quant_method, "gpu_method"):
        if id(quant_method) in seen:
            raise ValueError("MoE weight method wrapper cycle detected.")
        seen.add(id(quant_method))
        quant_method = quant_method.gpu_method
    return quant_method


def _moe_method_changes_expert_ownership(quant_method: Any) -> bool:
    """Detect wrappers that make GPU expert storage only a partial ownership."""

    seen = set()
    while True:
        if getattr(quant_method, "override_num_local_experts", False):
            return True
        if not hasattr(quant_method, "gpu_method"):
            return False
        if id(quant_method) in seen:
            raise ValueError("MoE weight method wrapper cycle detected.")
        seen.add(id(quant_method))
        quant_method = quant_method.gpu_method


def validate_moonep_reference_bf16_weight_layout(
    *,
    quant_method: Any,
    layer: torch.nn.Module | None = None,
) -> None:
    """Reject weight methods whose post-load storage is not canonical [Gate, Up]."""

    if _moe_method_changes_expert_ownership(quant_method):
        raise NotImplementedError(
            _MOONEP_HETEROGENEOUS_EXPERT_OWNERSHIP_UNSUPPORTED_MESSAGE
        )

    method = _unwrap_moe_weight_method(quant_method)
    if getattr(method, "load_up_proj_weight_first", False):
        raise NotImplementedError(_MOONEP_CANONICAL_LAYOUT_UNSUPPORTED_MESSAGE)

    has_transformed_layout = getattr(
        method, "has_transformed_expert_weight_layout", None
    )
    if callable(has_transformed_layout) and has_transformed_layout(layer):
        raise NotImplementedError(_MOONEP_CANONICAL_LAYOUT_UNSUPPORTED_MESSAGE)


def validate_moonep_reference_bf16_config(
    *,
    quant_config: Any,
    params_dtype: torch.dtype | None,
    num_fused_shared_experts: int,
    with_bias: bool,
    activation: str,
    quant_method: Any | None = None,
    layer: torch.nn.Module | None = None,
) -> None:
    """Validate the current SGLang MoonEP BF16 reference-path contract."""

    if quant_config is not None:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference path supports unquantized "
            "expert weights only; "
            "quant_config must be None."
        )
    if params_dtype is not torch.bfloat16:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference path requires "
            "params_dtype=torch.bfloat16, "
            f"got {params_dtype!r}."
        )
    if num_fused_shared_experts != 0:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference path does not support "
            "fused shared experts yet."
        )
    if with_bias:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference path does not support "
            "expert bias; use with_bias=False."
        )
    if activation != "silu":
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference expert runner supports "
            "SiLU only; production "
            "Kimi-K3 SiTU compute is not wired through this PoC."
        )
    if quant_method is not None:
        validate_moonep_reference_bf16_weight_layout(
            quant_method=quant_method,
            layer=layer,
        )


class MoonEPDispatchOutput(NamedTuple):
    """MoonEP dispatch output.

    ``plan`` is intentionally typed as ``Any`` so this module can define the
    SGLang-side contract without importing the optional ``moonep`` package at
    module import time.
    """

    hidden_states: torch.Tensor
    route_weights_nvs: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    plan: Any
    expert_ids: torch.Tensor
    num_tokens: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MOONEP


assert isinstance(MoonEPDispatchOutput, DispatchOutput)


class MoonEPCombineInput(NamedTuple):
    """MoonEP combine input."""

    hidden_states: torch.Tensor
    route_weights_nvs: Optional[torch.Tensor]
    plan: Any
    num_tokens: int

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MOONEP


assert isinstance(MoonEPCombineInput, CombineInput)


class MoonEPExpertWeightLayout(NamedTuple):
    """Contiguous BF16 expert weights in MoonEP prefetch layout."""

    full_gate_weight: torch.Tensor
    full_up_weight: torch.Tensor
    full_down_weight: torch.Tensor
    num_prefetch_slots: int


@dataclass(frozen=True)
class MoonEPBufferKey:
    """Static MoonEP buffer dimensions.

    MoonEP allocates its communication buffers from static shape parameters,
    unlike DeepEP's normal-dispatch path.  Keep the dimensions explicit so the
    process-wide facade never reuses a buffer with incompatible token capacity,
    model shape, EP topology, or prefetch-slot layout.
    """

    num_max_dispatch_tokens_per_rank: int
    hidden_size: int
    router_topk: int
    num_experts: int
    num_ep_ranks: int
    group_id: int
    num_prefetch_slots: int
    token_padding: int
    num_sms: int


class MoonEPBuffer:
    """Process-wide facade for MoonEP communication buffers.

    The underlying ``moonep.Buffer`` owns NVLink/VMM allocations and is keyed by
    MoonEP's static allocation dimensions.  The state lives on
    ``ctx.resources.buffers`` so tests can reset it with ``reset_context()`` and
    future runtime code has one lifecycle hook per process.
    """

    @classmethod
    def _state(cls):
        from types import SimpleNamespace

        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        state = buffers.get("moonep_ep_state")
        if state is None:
            state = SimpleNamespace(
                buffers={},
                active_key=None,
            )
            buffers["moonep_ep_state"] = state
        return state

    @staticmethod
    def _require_positive_int(name: str, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _resolve_num_ep_ranks(group: dist.ProcessGroup) -> int:
        try:
            num_ep_ranks = dist.get_world_size(group=group)
        except (AssertionError, RuntimeError, TypeError, ValueError):
            group_size = getattr(group, "size", None)
            if not callable(group_size):
                raise
            num_ep_ranks = group_size()
        return MoonEPBuffer._require_positive_int("num_ep_ranks", int(num_ep_ranks))

    @staticmethod
    def _resolve_num_prefetch_slots(
        num_prefetch_slots: int | None,
        num_experts: int,
        num_ep_ranks: int,
    ) -> int:
        if num_experts % num_ep_ranks != 0:
            raise ValueError(
                "MoonEP requires num_experts to be divisible by the EP group size: "
                f"num_experts={num_experts}, num_ep_ranks={num_ep_ranks}"
            )

        if num_prefetch_slots is None:
            num_prefetch_slots = envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.get()
        num_prefetch_slots = int(num_prefetch_slots)
        if num_prefetch_slots <= 0:
            return num_experts // num_ep_ranks
        return MoonEPBuffer._require_positive_int(
            "num_prefetch_slots", num_prefetch_slots
        )

    @classmethod
    def build_key(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_prefetch_slots: int | None = None,
        token_padding: int | None = None,
        num_sms: int | None = None,
    ) -> MoonEPBufferKey:
        if num_max_dispatch_tokens_per_rank is None:
            num_max_dispatch_tokens_per_rank = (
                envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
            )
        if token_padding is None:
            token_padding = envs.SGLANG_MOONEP_TOKEN_PADDING.get()
        if num_sms is None:
            num_sms = envs.SGLANG_MOONEP_NUM_SMS.get()

        num_ep_ranks = cls._resolve_num_ep_ranks(group)
        num_experts = cls._require_positive_int("num_experts", int(num_experts))
        num_prefetch_slots = cls._resolve_num_prefetch_slots(
            num_prefetch_slots,
            num_experts,
            num_ep_ranks,
        )

        return MoonEPBufferKey(
            num_max_dispatch_tokens_per_rank=cls._require_positive_int(
                "num_max_dispatch_tokens_per_rank",
                int(num_max_dispatch_tokens_per_rank),
            ),
            hidden_size=cls._require_positive_int("hidden_size", int(hidden_size)),
            router_topk=cls._require_positive_int("router_topk", int(router_topk)),
            num_experts=num_experts,
            num_ep_ranks=num_ep_ranks,
            group_id=id(group),
            num_prefetch_slots=num_prefetch_slots,
            token_padding=cls._require_positive_int(
                "token_padding", int(token_padding)
            ),
            num_sms=cls._require_positive_int("num_sms", int(num_sms)),
        )

    @classmethod
    def get_existing_buffer(
        cls,
        key: MoonEPBufferKey | None = None,
    ):
        """Return an already-created buffer, if any.

        Without a key this returns the most recently requested MoonEP buffer.
        Future runtime code should pass an explicit key when multiple static
        capacities are in play.
        """

        state = cls._state()
        if key is None:
            key = state.active_key
        if key is None:
            return None
        return state.buffers.get(key)

    @classmethod
    def get_moonep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int | None = None,
        num_prefetch_slots: int | None = None,
        token_padding: int | None = None,
        num_sms: int | None = None,
    ):
        key = cls.build_key(
            group=group,
            hidden_size=hidden_size,
            router_topk=router_topk,
            num_experts=num_experts,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            num_prefetch_slots=num_prefetch_slots,
            token_padding=token_padding,
            num_sms=num_sms,
        )

        state = cls._state()
        buffer = state.buffers.get(key)
        if buffer is not None:
            state.active_key = key
            return buffer

        try:
            from moonep import Buffer
        except ImportError as exc:
            raise ImportError(
                "MoonEP is not installed. Install MoonEP before running SGLang "
                "with --moe-a2a-backend moonep."
            ) from exc

        buffer = Buffer(
            S=key.num_max_dispatch_tokens_per_rank,
            H=key.hidden_size,
            K=key.router_topk,
            E=key.num_experts,
            num_ep_ranks=key.num_ep_ranks,
            num_sms=key.num_sms,
            token_padding=key.token_padding,
            B=key.num_prefetch_slots,
            group=group,
        )
        state.buffers[key] = buffer
        state.active_key = key
        return buffer

    @classmethod
    def destroy_buffer(cls, key: MoonEPBufferKey | None = None) -> None:
        state = cls._state()
        if key is None:
            key = state.active_key
        if key is None:
            return

        buffer = state.buffers.pop(key, None)
        destroy = getattr(buffer, "destroy", None)
        if callable(destroy):
            destroy()
        if state.active_key == key:
            state.active_key = next(reversed(state.buffers), None)

    @classmethod
    def destroy_all_buffers(cls) -> None:
        state = cls._state()
        for key in list(state.buffers):
            cls.destroy_buffer(key)
        state.active_key = None


def get_moonep_num_prefetch_slots(num_experts: int, num_ep_ranks: int) -> int:
    return MoonEPBuffer._resolve_num_prefetch_slots(
        num_prefetch_slots=None,
        num_experts=num_experts,
        num_ep_ranks=num_ep_ranks,
    )


def get_moonep_expert_weight_layout(
    layer: torch.nn.Module,
    num_prefetch_slots: int,
) -> MoonEPExpertWeightLayout:
    """Return cached contiguous BF16 gate/up/down tensors for MoonEP.

    The current SGLang MoonEP BF16 reference path uses unquantized BF16 weights
    stored in global expert-id order. Rows ``[E, E+B)`` are mutable prefetch
    slots and are intentionally preserved across calls so ``buffer.prefetch_weight``
    can fill them before the MoonEP expert runner consumes the layout.
    """

    if num_prefetch_slots <= 0:
        raise ValueError(
            f"num_prefetch_slots must be positive, got {num_prefetch_slots}"
        )

    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight
    moe_runner_config = layer.moe_runner_config
    validate_moonep_reference_bf16_config(
        quant_config=getattr(layer, "quant_config", None),
        params_dtype=getattr(layer, "params_dtype", w13_weight.dtype),
        num_fused_shared_experts=getattr(
            moe_runner_config, "num_fused_shared_experts", 0
        ),
        with_bias=getattr(layer, "with_bias", False),
        activation=getattr(moe_runner_config, "activation", "silu"),
        quant_method=getattr(layer, "quant_method", None),
        layer=layer,
    )
    if not getattr(layer.moe_runner_config, "is_gated", True):
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference path requires gated "
            "w13 experts."
        )
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference expert runner supports "
            "BF16 weights only."
        )
    if not w13_weight.is_contiguous() or not w2_weight.is_contiguous():
        raise ValueError("MoonEP expert source weights must be contiguous.")

    num_experts = int(layer.num_experts)
    intermediate_size = int(layer.intermediate_size_per_partition)
    hidden_size = int(layer.hidden_size)
    expected_w13_shape = (num_experts, 2 * intermediate_size, hidden_size)
    expected_w2_shape = (num_experts, hidden_size, intermediate_size)
    if tuple(w13_weight.shape) != expected_w13_shape:
        raise ValueError(
            "The current SGLang MoonEP BF16 reference path requires global "
            "w13_weight shape "
            f"{expected_w13_shape}, got {tuple(w13_weight.shape)}."
        )
    if tuple(w2_weight.shape) != expected_w2_shape:
        raise ValueError(
            "The current SGLang MoonEP BF16 reference path requires global "
            "w2_weight shape "
            f"{expected_w2_shape}, got {tuple(w2_weight.shape)}."
        )

    cache_key = (
        num_prefetch_slots,
        w13_weight.data_ptr(),
        w2_weight.data_ptr(),
        tuple(w13_weight.shape),
        tuple(w2_weight.shape),
    )
    cache = getattr(layer, "_moonep_weight_layout_cache", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]

    full_gate_weight = torch.empty(
        num_experts + num_prefetch_slots,
        intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=w13_weight.device,
    )
    full_up_weight = torch.empty_like(full_gate_weight)
    full_down_weight = torch.empty(
        num_experts + num_prefetch_slots,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=w2_weight.device,
    )

    full_gate_weight[:num_experts].copy_(w13_weight[:, :intermediate_size, :])
    full_up_weight[:num_experts].copy_(
        w13_weight[:, intermediate_size : 2 * intermediate_size, :]
    )
    full_down_weight[:num_experts].copy_(w2_weight)
    full_gate_weight[num_experts:].zero_()
    full_up_weight[num_experts:].zero_()
    full_down_weight[num_experts:].zero_()

    layout = MoonEPExpertWeightLayout(
        full_gate_weight=full_gate_weight.contiguous(),
        full_up_weight=full_up_weight.contiguous(),
        full_down_weight=full_down_weight.contiguous(),
        num_prefetch_slots=num_prefetch_slots,
    )
    layer._moonep_weight_layout_cache = (cache_key, layout)
    return layout


def run_moonep_bf16_expert(
    dispatch_output: MoonEPDispatchOutput,
    weight_layout: MoonEPExpertWeightLayout,
    *,
    activation: str = "silu",
) -> MoonEPCombineInput:
    """Run a simple BF16 MoonEP expert core over ``cu_seqlens`` segments.

    This is a correctness-first reference runner. It consumes MoonEP's already
    expert-grouped ``[NvS, H]`` token layout, applies gate/up/down expert
    weights for each non-empty `cu_seqlens` segment, multiplies each dispatched
    row by its route weight, and returns a `MoonEPCombineInput` for
    `MoonEPDispatcher.combine`.
    """

    if activation != "silu":
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference expert runner supports "
            "SiLU only; production "
            f"Kimi-K3 SiTU compute is not wired through this PoC (got {activation!r})."
        )

    hidden_states = dispatch_output.hidden_states
    route_weights_nvs = dispatch_output.route_weights_nvs
    cu_seqlens = dispatch_output.cu_seqlens
    expert_ids = dispatch_output.expert_ids

    if hidden_states.ndim != 2:
        raise ValueError(
            f"MoonEP hidden states must be [NvS, H], got {hidden_states.shape}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference expert runner requires "
            "BF16 hidden states, "
            f"got {hidden_states.dtype}."
        )
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {cu_seqlens.shape}")
    if (
        cu_seqlens.dtype == torch.bool
        or cu_seqlens.is_floating_point()
        or cu_seqlens.is_complex()
    ):
        raise TypeError(
            "MoonEP cu_seqlens must use an integer dtype, " f"got {cu_seqlens.dtype}."
        )
    if (
        expert_ids.dtype == torch.bool
        or expert_ids.is_floating_point()
        or expert_ids.is_complex()
    ):
        raise TypeError(
            "MoonEP expert_ids must use an integer dtype, " f"got {expert_ids.dtype}."
        )
    if expert_ids.shape != cu_seqlens.shape:
        raise ValueError(
            f"expert_ids shape {expert_ids.shape} must match cu_seqlens "
            f"shape {cu_seqlens.shape}"
        )
    if route_weights_nvs is not None and route_weights_nvs.ndim != 1:
        raise ValueError(f"route_weights_nvs must be 1D, got {route_weights_nvs.shape}")
    if (
        route_weights_nvs is not None
        and route_weights_nvs.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError(
            "route_weights_nvs length must match dispatched hidden rows: "
            f"weights={route_weights_nvs.shape[0]}, "
            f"hidden_rows={hidden_states.shape[0]}"
        )
    if dispatch_output.num_tokens < 0:
        raise ValueError(
            f"MoonEP num_tokens must be nonnegative, got {dispatch_output.num_tokens}"
        )
    if dispatch_output.num_tokens > hidden_states.shape[0]:
        raise ValueError(
            "MoonEP num_tokens cannot exceed dispatched hidden rows: "
            f"num_tokens={dispatch_output.num_tokens}, "
            f"hidden_rows={hidden_states.shape[0]}"
        )

    gate_weight = weight_layout.full_gate_weight
    up_weight = weight_layout.full_up_weight
    down_weight = weight_layout.full_down_weight
    if gate_weight.ndim != 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError(
            "MoonEP expert weights must all be 3D [experts, output, input] "
            f"tensors, got gate={gate_weight.shape}, up={up_weight.shape}, "
            f"down={down_weight.shape}"
        )
    if any(
        weight.dtype != torch.bfloat16
        for weight in (gate_weight, up_weight, down_weight)
    ):
        raise NotImplementedError(
            "The current SGLang MoonEP BF16 reference expert runner requires "
            "BF16 gate/up/down "
            f"weights, got gate={gate_weight.dtype}, up={up_weight.dtype}, "
            f"down={down_weight.dtype}."
        )
    if not (gate_weight.shape[0] == up_weight.shape[0] == down_weight.shape[0]):
        raise ValueError(
            "MoonEP expert weight row counts must match: "
            f"gate={gate_weight.shape[0]}, up={up_weight.shape[0]}, "
            f"down={down_weight.shape[0]}"
        )
    if (
        gate_weight.shape[2] != hidden_states.shape[1]
        or up_weight.shape[2] != hidden_states.shape[1]
    ):
        raise ValueError(
            "MoonEP gate/up input width must match hidden states: "
            f"hidden={hidden_states.shape[1]}, gate={gate_weight.shape[2]}, "
            f"up={up_weight.shape[2]}"
        )
    if (
        gate_weight.shape[1] != up_weight.shape[1]
        or down_weight.shape[2] != gate_weight.shape[1]
    ):
        raise ValueError(
            "MoonEP gate/up/down intermediate dimensions must match: "
            f"gate={gate_weight.shape[1]}, up={up_weight.shape[1]}, "
            f"down_input={down_weight.shape[2]}"
        )
    if down_weight.shape[1] != hidden_states.shape[1]:
        raise ValueError(
            "MoonEP down weight output width must match hidden states: "
            f"down={down_weight.shape[1]}, hidden={hidden_states.shape[1]}"
        )

    output = torch.empty_like(hidden_states)
    prev = 0
    for group_id in range(cu_seqlens.numel()):
        cur = int(cu_seqlens[group_id].item())
        if cur < 0:
            raise ValueError(
                f"MoonEP cu_seqlens[{group_id}] must be nonnegative, got {cur}"
            )
        if cur < prev:
            raise ValueError(
                "MoonEP cu_seqlens must be non-decreasing: "
                f"cu_seqlens[{group_id - 1}]={prev}, "
                f"cu_seqlens[{group_id}]={cur}"
            )
        if cur > hidden_states.shape[0]:
            raise ValueError(
                f"MoonEP cu_seqlens[{group_id}]={cur} exceeds dispatched "
                f"hidden rows={hidden_states.shape[0]}"
            )
        if cur == prev:
            continue

        expert_id = int(expert_ids[group_id].item())
        if expert_id < 0:
            raise ValueError(
                f"MoonEP expert_ids[{group_id}]={expert_id} is invalid for a "
                "non-empty segment"
            )
        if expert_id >= weight_layout.full_gate_weight.shape[0]:
            raise ValueError(
                f"expert_id {expert_id} exceeds MoonEP weight rows "
                f"{weight_layout.full_gate_weight.shape[0]}"
            )

        x = hidden_states[prev:cur]
        gate = F.linear(x, weight_layout.full_gate_weight[expert_id])
        up = F.linear(x, weight_layout.full_up_weight[expert_id])
        activated = F.silu(gate) * up
        y = F.linear(activated, weight_layout.full_down_weight[expert_id])
        if route_weights_nvs is not None:
            y = y * route_weights_nvs[prev:cur].to(dtype=y.dtype).unsqueeze(-1)
        output[prev:cur].copy_(y)
        prev = cur

    if prev < hidden_states.shape[0]:
        output[prev:].zero_()

    return MoonEPCombineInput(
        hidden_states=output,
        route_weights_nvs=route_weights_nvs,
        plan=dispatch_output.plan,
        num_tokens=dispatch_output.num_tokens,
    )


class MoonEPDispatcher(BaseDispatcher):
    """Dispatcher for the current SGLang MoonEP BF16 reference path."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.async_finish = async_finish
        self.return_recv_hook = return_recv_hook
        self.expert_mask_gpu = None
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        if self.num_max_dispatch_tokens_per_rank <= 0:
            raise ValueError(
                "SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK must be positive, "
                f"got {self.num_max_dispatch_tokens_per_rank}."
            )
        self.num_prefetch_slots = None

    @staticmethod
    def _raise_unimplemented(operation: str) -> NoReturn:
        raise NotImplementedError(
            f"The current SGLang MoonEP BF16 reference path does not implement "
            f"{operation}. "
            f"{_MOONEP_SPLIT_PHASE_UNSUPPORTED_MESSAGE}"
        )

    def _get_buffer(self):
        if self.hidden_size is None or self.num_experts is None:
            raise ValueError(
                "MoonEPDispatcher requires hidden_size and num_experts to "
                "create a MoonEP buffer."
            )
        return MoonEPBuffer.get_moonep_buffer(
            group=self.group,
            hidden_size=self.hidden_size,
            router_topk=self.router_topk,
            num_experts=self.num_experts,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            num_prefetch_slots=self.num_prefetch_slots,
        )

    def _get_rank(self) -> int:
        return dist.get_rank(group=self.group)

    def _validate_dispatch_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> None:
        if self.hidden_size is None:
            raise ValueError("MoonEPDispatcher requires hidden_size.")
        if hidden_states.ndim != 2:
            raise ValueError(
                "MoonEP hidden_states must be rank 2 [num_tokens, hidden_size], "
                f"got {hidden_states.shape}"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                "MoonEP hidden_states width does not match configured hidden_size: "
                f"width={hidden_states.shape[1]}, hidden_size={self.hidden_size}"
            )
        if hidden_states.dtype != torch.bfloat16:
            raise NotImplementedError(
                "The current SGLang MoonEP BF16 reference dispatcher supports "
                "BF16 hidden states only, "
                f"got {hidden_states.dtype}."
            )

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        if not isinstance(topk_ids, torch.Tensor) or topk_ids.ndim != 2:
            shape = getattr(topk_ids, "shape", None)
            raise ValueError(
                "MoonEP topk_ids must be rank 2 [num_tokens, router_topk], "
                f"got {shape}"
            )
        if not isinstance(topk_weights, torch.Tensor) or topk_weights.ndim != 2:
            shape = getattr(topk_weights, "shape", None)
            raise ValueError(
                "MoonEP topk_weights must be rank 2 [num_tokens, router_topk], "
                f"got {shape}"
            )
        if topk_ids.shape != topk_weights.shape:
            raise ValueError(
                "MoonEP topk_ids and topk_weights must have the same shape: "
                f"ids={topk_ids.shape}, weights={topk_weights.shape}"
            )
        if topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "MoonEP routing row count must match hidden_states: "
                f"ids={topk_ids.shape[0]}, hidden={hidden_states.shape[0]}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                "MoonEP routing width must match router_topk: "
                f"width={topk_ids.shape[1]}, router_topk={self.router_topk}"
            )
        if (
            topk_ids.device != hidden_states.device
            or topk_weights.device != hidden_states.device
        ):
            raise ValueError(
                "MoonEP hidden_states, topk_ids, and topk_weights must be on the "
                "same device."
            )
        if (
            topk_ids.dtype == torch.bool
            or topk_ids.is_floating_point()
            or topk_ids.is_complex()
        ):
            raise TypeError(
                "MoonEP topk_ids must use an integer dtype before conversion to "
                f"int32, got {topk_ids.dtype}."
            )
        if not topk_weights.is_floating_point():
            raise TypeError(
                "MoonEP topk_weights must use a floating-point dtype before "
                f"conversion to float32, got {topk_weights.dtype}."
            )

    def _pad_to_capacity(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_tokens = int(hidden_states.shape[0])
        capacity = int(self.num_max_dispatch_tokens_per_rank)
        if capacity <= 0:
            raise ValueError(
                "MoonEP static buffer capacity must be positive, " f"got {capacity}."
            )
        if num_tokens > capacity:
            raise ValueError(
                "MoonEP runtime batch has more tokens than its static buffer "
                f"capacity: num_tokens={num_tokens}, capacity={capacity}. "
                "Increase SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
            )

        hidden_states = hidden_states.contiguous()
        topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        if num_tokens == capacity:
            return hidden_states, topk_ids, topk_weights, num_tokens

        pad_tokens = capacity - num_tokens
        # Known limitation of the current reference path: zero-padded rows use
        # expert id 0, so planning counts include the padding under expert 0.
        # Preserve this behavior until MoonEP exposes padding metadata.
        hidden_pad = hidden_states.new_zeros(pad_tokens, hidden_states.shape[1])
        id_pad = topk_ids.new_zeros(pad_tokens, topk_ids.shape[1])
        weight_pad = topk_weights.new_zeros(pad_tokens, topk_weights.shape[1])
        return (
            torch.cat((hidden_states, hidden_pad), dim=0).contiguous(),
            torch.cat((topk_ids, id_pad), dim=0).contiguous(),
            torch.cat((topk_weights, weight_pad), dim=0).contiguous(),
            num_tokens,
        )

    def _tokens_per_expert(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.num_experts is None or self.num_experts <= 0:
            raise ValueError(
                "MoonEPDispatcher requires a positive num_experts, "
                f"got {self.num_experts}."
            )
        tokens_per_expert = torch.bincount(
            topk_ids.reshape(-1).to(dtype=torch.int64),
            minlength=self.num_experts,
        )
        if tokens_per_expert.numel() != self.num_experts:
            raise ValueError(
                "MoonEP topk_ids contains an expert ID outside the configured "
                f"range [0, {self.num_experts}): tokens_per_expert has "
                f"{tokens_per_expert.numel()} bins."
            )
        return tokens_per_expert.to(dtype=torch.int32)

    def _expert_ids_from_plan(
        self,
        cu_seqlens: torch.Tensor,
        plan: Any,
    ) -> torch.Tensor:
        if self.num_experts is None or self.num_experts <= 0:
            raise ValueError(
                "MoonEPDispatcher requires a positive num_experts, "
                f"got {self.num_experts}."
            )
        if cu_seqlens.ndim != 1:
            raise ValueError(
                f"MoonEP cu_seqlens must be rank 1, got {cu_seqlens.shape}"
            )
        num_groups = int(cu_seqlens.numel())
        if num_groups < self.num_experts:
            raise ValueError(
                "MoonEP cu_seqlens must contain one group per expert plus "
                f"prefetch groups: groups={num_groups}, num_experts={self.num_experts}"
            )
        expert_ids = torch.full_like(cu_seqlens, -1)
        experts_to_copy = getattr(plan, "experts_to_copy", None)
        if not isinstance(experts_to_copy, torch.Tensor):
            raise ValueError("MoonEP plan.experts_to_copy must be a tensor.")
        if experts_to_copy.ndim == 2:
            experts_to_copy = experts_to_copy[self._get_rank()]
        elif experts_to_copy.ndim != 1:
            raise ValueError(
                "MoonEP plan.experts_to_copy must be rank 1 or rank 2 [R, B], "
                f"got {experts_to_copy.shape}"
            )

        num_prefetch_slots = num_groups - self.num_experts
        if experts_to_copy.numel() != num_prefetch_slots:
            raise ValueError(
                "MoonEP plan prefetch slots do not match cu_seqlens: "
                f"plan_slots={experts_to_copy.numel()}, "
                f"cu_seqlens_slots={num_prefetch_slots}"
            )

        prev = 0
        for group_id in range(num_groups):
            cur = int(cu_seqlens[group_id].item())
            if cur > prev:
                if group_id < self.num_experts:
                    expert_ids[group_id] = group_id
                else:
                    expert_ids[group_id] = experts_to_copy[group_id - self.num_experts]
            prev = cur
        return expert_ids

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise NotImplementedError(
                "The current SGLang MoonEP BF16 reference path requires standard "
                "top-k output before dispatch."
            )
        if self.num_experts is None:
            raise ValueError("MoonEPDispatcher requires num_experts.")
        self._validate_dispatch_inputs(hidden_states, topk_output)

        hidden_states, topk_ids, topk_weights, num_tokens = self._pad_to_capacity(
            hidden_states,
            topk_output.topk_ids,
            topk_output.topk_weights,
        )
        tokens_per_expert = self._tokens_per_expert(topk_ids)
        buffer = self._get_buffer()
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            tokens_per_expert,
            async_finish=False,
        )
        return MoonEPDispatchOutput(
            hidden_states=hidden_nvsh,
            route_weights_nvs=route_weights_nvs,
            cu_seqlens=cu_seqlens,
            plan=plan,
            expert_ids=self._expert_ids_from_plan(cu_seqlens, plan),
            num_tokens=num_tokens,
        )

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._raise_unimplemented("dispatch_a")

    def dispatch_b(self):
        self._raise_unimplemented("dispatch_b")

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.MOONEP:
            raise TypeError(
                f"MoonEPDispatcher.combine expected MOONEP input, got "
                f"{combine_input.format}"
            )
        hidden_states, _route_weights_sk, _event = self._get_buffer().combine(
            plan=combine_input.plan,
            hidden_nvsh=combine_input.hidden_states,
            route_weights_nvs=None,
            async_finish=False,
        )
        if combine_input.num_tokens < 0:
            raise ValueError(
                "MoonEP combine num_tokens must be nonnegative, "
                f"got {combine_input.num_tokens}"
            )
        if hidden_states.shape[0] < combine_input.num_tokens:
            raise ValueError(
                "MoonEP combine returned fewer rows than the original input: "
                f"rows={hidden_states.shape[0]}, "
                f"num_tokens={combine_input.num_tokens}"
            )
        return hidden_states[: combine_input.num_tokens].contiguous()

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        self._raise_unimplemented("combine_a")

    def combine_b(self):
        self._raise_unimplemented("combine_b")

    def prefetch_weight(
        self,
        plan: Any,
        weight_layout: MoonEPExpertWeightLayout,
    ) -> None:
        self._get_buffer().prefetch_weight(
            plan=plan,
            async_finish=False,
            full_gate_weight=weight_layout.full_gate_weight,
            full_up_weight=weight_layout.full_up_weight,
            full_down_weight=weight_layout.full_down_weight,
        )

    def register_deepep_dispatch_hook(self, hook):
        self._raise_unimplemented("register_deepep_dispatch_hook")
