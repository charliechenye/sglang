"""CPU-testable boundaries between MoonEP and DeepEP/K3 integration paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    create_moe_dispatcher,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.moonep import MoonEPDispatcher
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoonEPFactory(unittest.TestCase):
    def test_factory_constructs_moonep_directly(self):
        group = object()
        config = MoeRunnerConfig(
            num_experts=16,
            num_local_experts=16,
            hidden_size=32,
            top_k=2,
        )
        with (
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MOONEP,
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group",
                return_value=SimpleNamespace(device_group=group),
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.MoonEPDispatcher",
                wraps=MoonEPDispatcher,
            ) as dispatcher_cls,
        ):
            dispatcher = create_moe_dispatcher(config)

        self.assertIsInstance(dispatcher, MoonEPDispatcher)
        dispatcher_cls.assert_called_once_with(
            group=group,
            router_topk=2,
            num_experts=16,
            num_local_experts=16,
            hidden_size=32,
            params_dtype=None,
        )

    def test_tbo_wrapper_rejects_moonep(self):
        with patch(
            "sglang.srt.batch_overlap.two_batch_overlap.get_moe_a2a_backend",
            return_value=MoeA2ABackend.MOONEP,
        ):
            with self.assertRaisesRegex(ValueError, "synchronous MoonEPDispatcher"):
                MaybeTboDeepEPDispatcher()


class TestMoonEPDeepEPPolicyIsolation(unittest.TestCase):
    def test_moonep_does_not_use_deep_ep_deprecate_policy(self):
        owner = object.__new__(DeepEPMoE)
        with (
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.FusedMoE.__init__",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MOONEP,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_deepep_mode",
                return_value=SimpleNamespace(enable_low_latency=lambda: False),
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_runner_backend"
            ) as runner_backend,
        ):
            DeepEPMoE.__init__(
                owner,
                num_experts=16,
                top_k=2,
                hidden_size=32,
                intermediate_size=64,
                layer_id=0,
                quant_config=None,
            )

        self.assertTrue(owner._moonep_reference_path)
        self.assertFalse(owner.deprecate_flag)
        runner_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
