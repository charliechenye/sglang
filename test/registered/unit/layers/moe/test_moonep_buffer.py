import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.moonep import (
    MoonEPBuffer,
    MoonEPDispatchOutput,
    MoonEPExpertWeightLayout,
    get_moonep_expert_weight_layout,
    run_moonep_bf16_expert,
    validate_moonep_reference_bf16_config,
    validate_moonep_reference_bf16_weight_layout,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import (
    cleanup_distributed_resources,
    reset_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeMoonEPBuffer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.destroy_calls = 0
        self.__class__.instances.append(self)

    def destroy(self):
        self.destroy_calls += 1


class _FailOnceMoonEPBuffer(_FakeMoonEPBuffer):
    def destroy(self):
        self.destroy_calls += 1
        if self.destroy_calls == 1:
            raise RuntimeError("transient MoonEP destroy failure")


def _fake_moonep_module():
    module = types.ModuleType("moonep")
    module.Buffer = _FakeMoonEPBuffer
    return module


class TestMoonEPBuffer(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeMoonEPBuffer.instances.clear()

    def tearDown(self):
        try:
            MoonEPBuffer.destroy_all_buffers()
        finally:
            reset_context()
            _FakeMoonEPBuffer.instances.clear()

    def test_lazily_constructs_and_reuses_buffer_for_static_key(self):
        group = object()

        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            same_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            larger_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=512,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )

        self.assertIs(buffer, same_buffer)
        self.assertIsNot(buffer, larger_buffer)
        self.assertEqual(len(_FakeMoonEPBuffer.instances), 2)
        self.assertEqual(
            buffer.kwargs,
            {
                "S": 256,
                "H": 1024,
                "K": 8,
                "E": 64,
                "num_ep_ranks": 4,
                "num_sms": 20,
                "token_padding": 64,
                "B": 16,
                "group": group,
            },
        )
        self.assertIs(MoonEPBuffer.get_existing_buffer(), larger_buffer)

    def test_resolves_env_defaults_and_training_safe_prefetch_slots(self):
        group = object()

        with (
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(384),
            envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.override(-1),
            envs.SGLANG_MOONEP_TOKEN_PADDING.override(32),
            envs.SGLANG_MOONEP_NUM_SMS.override(18),
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=8,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=2048,
                router_topk=6,
                num_experts=128,
            )

        self.assertEqual(buffer.kwargs["S"], 384)
        self.assertEqual(buffer.kwargs["token_padding"], 32)
        self.assertEqual(buffer.kwargs["num_sms"], 18)
        self.assertEqual(buffer.kwargs["B"], 16)

    def test_rejects_non_divisible_experts_before_allocating(self):
        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=6,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "divisible"):
                MoonEPBuffer.get_moonep_buffer(
                    group=object(),
                    hidden_size=1024,
                    router_topk=8,
                    num_experts=64,
                )

        self.assertEqual(_FakeMoonEPBuffer.instances, [])

    def test_destroy_failure_keeps_buffer_owned_for_retry(self):
        group = object()

        with (
            patch.dict(
                sys.modules,
                {
                    "moonep": types.SimpleNamespace(
                        Buffer=_FailOnceMoonEPBuffer,
                    )
                },
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
            )

            with self.assertRaisesRegex(RuntimeError, "transient MoonEP"):
                cleanup_distributed_resources()

            self.assertIs(MoonEPBuffer.get_existing_buffer(), buffer)
            self.assertEqual(buffer.destroy_calls, 1)

            cleanup_distributed_resources()

        self.assertEqual(buffer.destroy_calls, 2)
        self.assertIsNone(MoonEPBuffer.get_existing_buffer())


class TestMoonEPExpertWeightLayout(unittest.TestCase):
    def _fake_layer(self):
        num_experts, hidden_size, intermediate_size = 3, 4, 5
        w13_weight = torch.arange(
            num_experts * 2 * intermediate_size * hidden_size,
            dtype=torch.bfloat16,
        ).reshape(num_experts, 2 * intermediate_size, hidden_size)
        w2_weight = torch.arange(
            num_experts * hidden_size * intermediate_size,
            dtype=torch.bfloat16,
        ).reshape(num_experts, hidden_size, intermediate_size)

        return SimpleNamespace(
            quant_config=None,
            params_dtype=torch.bfloat16,
            with_bias=False,
            moe_runner_config=SimpleNamespace(
                num_fused_shared_experts=0,
                is_gated=True,
                activation="silu",
            ),
            use_triton_kernels=False,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            num_experts=num_experts,
            intermediate_size_per_partition=intermediate_size,
            hidden_size=hidden_size,
        )

    def test_layout_splits_gate_up_down_and_adds_prefetch_slots(self):
        layer = self._fake_layer()

        layout = get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)

        self.assertEqual(tuple(layout.full_gate_weight.shape), (5, 5, 4))
        self.assertEqual(tuple(layout.full_up_weight.shape), (5, 5, 4))
        self.assertEqual(tuple(layout.full_down_weight.shape), (5, 4, 5))
        torch.testing.assert_close(
            layout.full_gate_weight[:3],
            layer.w13_weight[:, :5, :],
        )
        torch.testing.assert_close(
            layout.full_up_weight[:3],
            layer.w13_weight[:, 5:10, :],
        )
        torch.testing.assert_close(layout.full_down_weight[:3], layer.w2_weight)
        self.assertTrue(torch.all(layout.full_gate_weight[3:] == 0))
        self.assertTrue(torch.all(layout.full_up_weight[3:] == 0))
        self.assertTrue(torch.all(layout.full_down_weight[3:] == 0))

    def test_layout_rejects_local_expert_storage(self):
        layer = self._fake_layer()
        layer.w13_weight = layer.w13_weight[:2].contiguous()

        with self.assertRaisesRegex(ValueError, "global w13_weight"):
            get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)

    def test_rejects_transformed_expert_layouts(self):
        layer = self._fake_layer()
        methods = (
            UnquantizedFusedMoEMethod(use_triton_kernels=True),
            UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True),
        )
        for method in methods:
            with self.subTest(method=type(method).__name__):
                layer.quant_method = method
                with self.assertRaisesRegex(NotImplementedError, "canonical.*Gate, Up"):
                    get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)


class TestMoonEPRealWeightLoaderLayout(unittest.TestCase):
    @staticmethod
    def _make_layer():
        layer = object.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.quant_config = None
        layer.params_dtype = torch.bfloat16
        layer.with_bias = False
        layer.use_presharded_weights = False
        layer.use_triton_kernels = False
        layer.use_flashinfer_trtllm_moe = False
        layer.use_padded_loading = False
        layer.moe_tp_size = 1
        layer.moe_tp_rank = 0
        layer.num_experts = 1
        layer.hidden_size = 3
        layer.intermediate_size_per_partition = 2
        layer.moe_runner_config = SimpleNamespace(
            is_gated=True,
            num_fused_shared_experts=0,
            activation="silu",
        )
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(1, 4, 3, dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(1, 3, 2, dtype=torch.bfloat16), requires_grad=False
        )
        layer.w13_weight._sglang_require_global_experts = True
        layer.w2_weight._sglang_require_global_experts = True
        layer.quant_method = UnquantizedFusedMoEMethod()
        layer.quant_method.use_flashinfer_cutlass = False
        return layer

    def test_real_loader_preserves_checkpoint_gate_up_semantics(self):
        layer = self._make_layer()
        checkpoint_gate = torch.full((2, 3), 3, dtype=torch.bfloat16)
        checkpoint_up = torch.full((2, 3), 7, dtype=torch.bfloat16)
        checkpoint_down = torch.ones((3, 2), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.get_global_expert_location_metadata",
            return_value=None,
        ):
            layer.weight_loader(
                layer.w13_weight,
                checkpoint_gate,
                "experts.0.gate_proj.weight",
                "w1",
                0,
            )
            layer.weight_loader(
                layer.w13_weight,
                checkpoint_up,
                "experts.0.up_proj.weight",
                "w3",
                0,
            )
            layer.weight_loader(
                layer.w2_weight,
                checkpoint_down,
                "experts.0.down_proj.weight",
                "w2",
                0,
            )

        layout = get_moonep_expert_weight_layout(layer, num_prefetch_slots=1)
        torch.testing.assert_close(layout.full_gate_weight[0], checkpoint_gate)
        torch.testing.assert_close(layout.full_up_weight[0], checkpoint_up)
        torch.testing.assert_close(layout.full_down_weight[0], checkpoint_down)

    def test_incompatible_loader_layout_is_rejected_before_copying(self):
        layer = self._make_layer()
        layer.quant_method.use_flashinfer_cutlass = True

        with self.assertRaisesRegex(NotImplementedError, "canonical.*Gate, Up"):
            get_moonep_expert_weight_layout(layer, num_prefetch_slots=1)


class TestMoonEPBf16ExpertRunner(unittest.TestCase):
    def test_segment_runner_uses_physical_vm_group_weight_rows(self):
        hidden_states = torch.tensor(
            [[1.0, 2.0], [5.0, 6.0]],
            dtype=torch.bfloat16,
        )
        route_weights = torch.tensor([1.0, 1.5], dtype=torch.float32)
        # E=2 source rows plus B=1 physical prefetch slot.  This is the
        # planner-realistic shape for one ordinary source group, one empty
        # source group, and one non-empty prefetch slot group.  Group 2 must
        # use weight row 2 directly; no plan mapping is involved in the
        # runner.
        cu_seqlens = torch.tensor([1, 1, 2], dtype=torch.int32)
        gate = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                # Empty source row 1 is deliberately different from the
                # controlled prefetch-slot payload in physical row 2.
                [[0.5, 0.0], [0.0, 0.5]],
                [[3.0, 0.0], [0.0, 3.0]],
            ],
            dtype=torch.bfloat16,
        )
        up = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.5, 0.0], [0.0, 1.5]],
                [[4.0, 0.0], [0.0, 4.0]],
            ],
            dtype=torch.bfloat16,
        )
        down = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[5.0, 0.0], [0.0, 5.0]],
            ],
            dtype=torch.bfloat16,
        )
        layout = MoonEPExpertWeightLayout(
            full_gate_weight=gate,
            full_up_weight=up,
            full_down_weight=down,
            num_prefetch_slots=1,
        )
        dispatch_output = MoonEPDispatchOutput(
            hidden_states=hidden_states,
            route_weights_nvs=route_weights,
            cu_seqlens=cu_seqlens,
            plan=object(),
            num_tokens=2,
        )

        combine_input = run_moonep_bf16_expert(dispatch_output, layout)

        expected = torch.empty_like(hidden_states)
        for start, end, physical_row in [(0, 1, 0), (1, 2, 2)]:
            x = hidden_states[start:end]
            y = torch.nn.functional.linear(
                torch.nn.functional.silu(
                    torch.nn.functional.linear(x, gate[physical_row])
                )
                * torch.nn.functional.linear(x, up[physical_row]),
                down[physical_row],
            )
            expected[start:end] = y * route_weights[start:end, None]

        torch.testing.assert_close(combine_input.hidden_states, expected)
        self.assertIs(combine_input.plan, dispatch_output.plan)
        self.assertEqual(combine_input.num_tokens, 2)


class TestMoonEPConfigContract(unittest.TestCase):
    def test_rejects_unsupported_layer_configurations(self):
        cases = [
            ({"quant_config": object()}, "quant_config must be None"),
            ({"params_dtype": torch.float16}, "params_dtype"),
            ({"num_fused_shared_experts": 1}, "fused shared experts"),
            ({"with_bias": True}, "expert bias"),
            ({"activation": "situ"}, "SiLU only"),
        ]
        for overrides, message in cases:
            kwargs = dict(
                quant_config=None,
                params_dtype=torch.bfloat16,
                num_fused_shared_experts=0,
                with_bias=False,
                activation="silu",
            )
            kwargs.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(NotImplementedError, message):
                    validate_moonep_reference_bf16_config(**kwargs)

    def test_rejects_heterogeneous_expert_ownership(self):
        with self.assertRaisesRegex(NotImplementedError, "heterogeneous CPU/GPU"):
            validate_moonep_reference_bf16_weight_layout(
                quant_method=SimpleNamespace(override_num_local_experts=True)
            )


if __name__ == "__main__":
    unittest.main()
