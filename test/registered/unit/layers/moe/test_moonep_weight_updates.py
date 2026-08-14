"""Unit tests for MoonEP's online weight-update restriction."""

import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.ep_moe.layer import (
    _should_use_deepep_bf16_dispatch_fallback,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMoonEPWeightUpdateContract(unittest.TestCase):
    def test_reference_layout_cache_rejects_online_updates(self):
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.MOONEP,
        ):
            error = _unsupported_derived_weight_cache_error()

        self.assertIsNotNone(error)
        self.assertIn("MoonEP", error)
        self.assertIn("copied expert", error)

    def test_deepep_bf16_flag_does_not_divert_moonep(self):
        with (
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(
                envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MOONEP,
            ),
        ):
            self.assertFalse(_should_use_deepep_bf16_dispatch_fallback())


if __name__ == "__main__":
    unittest.main()
