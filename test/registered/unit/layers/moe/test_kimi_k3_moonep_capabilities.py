"""CPU-testable Kimi-K3 MoE backend capability decisions."""

import unittest

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.models.kimi_k3 import _get_kimi_k3_moe_backend_capabilities
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKimiK3MoonepCapabilities(unittest.TestCase):
    def test_backend_capability_split(self):
        expected = {
            MoeA2ABackend.NONE: (False, False),
            MoeA2ABackend.DEEPEP: (True, True),
            MoeA2ABackend.MEGAMOE: (True, True),
            MoeA2ABackend.MOONEP: (True, False),
        }
        for backend, (token_a2a, shared_expert_overlap) in expected.items():
            with self.subTest(backend=backend.value):
                capabilities = _get_kimi_k3_moe_backend_capabilities(backend)
                self.assertEqual(capabilities.token_a2a, token_a2a)
                self.assertEqual(
                    capabilities.shared_expert_overlap,
                    shared_expert_overlap,
                )

    def test_moonep_is_token_a2a_but_not_shared_expert_overlap(self):
        capabilities = _get_kimi_k3_moe_backend_capabilities(MoeA2ABackend.MOONEP)
        self.assertTrue(capabilities.token_a2a)
        self.assertFalse(capabilities.shared_expert_overlap)


if __name__ == "__main__":
    unittest.main()
