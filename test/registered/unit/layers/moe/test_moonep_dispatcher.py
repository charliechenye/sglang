import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.moonep import (
    MoonEPCombineInput,
    MoonEPDispatcher,
    MoonEPExpertWeightLayout,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeMoonEPBuffer:
    def __init__(self, num_experts: int, num_prefetch_slots: int = 2):
        self.num_experts = num_experts
        self.num_prefetch_slots = num_prefetch_slots
        self.dispatch_calls = []
        self.prefetch_calls = []
        self.combine_calls = []
        self.combine_output = None

    def dispatch(
        self,
        hidden_states,
        route_weights_sk,
        topk_experts_sk,
        tokens_per_expert,
        *,
        async_finish,
    ):
        self.dispatch_calls.append(
            {
                "hidden_states": hidden_states,
                "route_weights_sk": route_weights_sk,
                "topk_experts_sk": topk_experts_sk,
                "tokens_per_expert": tokens_per_expert,
                "async_finish": async_finish,
            }
        )
        num_groups = self.num_experts + self.num_prefetch_slots
        cu_seqlens = torch.arange(
            1,
            num_groups + 1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        plan = SimpleNamespace(
            experts_to_copy=torch.tensor(
                [
                    list(range(10, 10 + self.num_prefetch_slots)),
                    list(range(20, 20 + self.num_prefetch_slots)),
                ],
                dtype=torch.int32,
                device=hidden_states.device,
            )
        )
        route_weights_nvs = torch.ones(
            hidden_states.shape[0],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return hidden_states.clone(), route_weights_nvs, cu_seqlens, plan

    def prefetch_weight(
        self,
        *,
        plan,
        async_finish,
        full_gate_weight,
        full_up_weight,
        full_down_weight,
    ):
        self.prefetch_calls.append(
            {
                "plan": plan,
                "async_finish": async_finish,
                "full_gate_weight": full_gate_weight,
                "full_up_weight": full_up_weight,
                "full_down_weight": full_down_weight,
            }
        )

    def combine(self, *, plan, hidden_nvsh, route_weights_nvs, async_finish):
        self.combine_calls.append(
            {
                "plan": plan,
                "hidden_nvsh": hidden_nvsh,
                "route_weights_nvs": route_weights_nvs,
                "async_finish": async_finish,
            }
        )
        output = hidden_nvsh if self.combine_output is None else self.combine_output
        return output, None, None


class TestMoonEPDispatcher(unittest.TestCase):
    def setUp(self):
        self.rank_patcher = patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_rank",
            return_value=0,
        )
        self.rank_patcher.start()
        self.addCleanup(self.rank_patcher.stop)

    def _make_dispatcher(self, *, capacity=4, num_experts=2, prefetch_slots=2):
        with envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(capacity):
            dispatcher = MoonEPDispatcher(
                group=object(),
                router_topk=2,
                num_experts=num_experts,
                hidden_size=2,
            )
        fake_buffer = _FakeMoonEPBuffer(num_experts, prefetch_slots)
        return dispatcher, fake_buffer

    @staticmethod
    def _topk_output(num_tokens, *, num_experts=2, dtype=torch.int64):
        topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=dtype,
        )[:num_tokens]
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2]],
            dtype=torch.float16,
        )[:num_tokens]
        if num_experts != 2:
            topk_ids = topk_ids.remainder(num_experts)
        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.empty(0),
        )

    def test_dispatch_converts_inputs_and_preserves_moonep_sync_contract(self):
        dispatcher, fake_buffer = self._make_dispatcher()
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)

        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ), patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_rank",
            return_value=1,
        ):
            dispatch_output = dispatcher.dispatch(
                hidden_states,
                self._topk_output(4),
            )

        call = fake_buffer.dispatch_calls[0]
        self.assertEqual(call["hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(call["topk_experts_sk"].dtype, torch.int32)
        self.assertEqual(call["route_weights_sk"].dtype, torch.float32)
        torch.testing.assert_close(
            call["tokens_per_expert"],
            torch.tensor([4, 4], dtype=torch.int32),
        )
        self.assertFalse(call["async_finish"])
        torch.testing.assert_close(
            dispatch_output.expert_ids,
            torch.tensor([0, 1, 20, 21], dtype=torch.int32),
        )

    def test_dispatch_prefetch_and_combine_share_plan_and_use_expected_weights(self):
        dispatcher, fake_buffer = self._make_dispatcher()
        hidden_states = torch.ones(4, 2, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            dispatch_output = dispatcher.dispatch(
                hidden_states,
                self._topk_output(4),
            )

            layout = MoonEPExpertWeightLayout(
                full_gate_weight=torch.ones(4, 3, 2, dtype=torch.bfloat16),
                full_up_weight=torch.ones(4, 3, 2, dtype=torch.bfloat16),
                full_down_weight=torch.ones(4, 2, 3, dtype=torch.bfloat16),
                num_prefetch_slots=2,
            )
            dispatcher.prefetch_weight(dispatch_output.plan, layout)

            combine_hidden = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
            fake_buffer.combine_output = combine_hidden
            combine_input = MoonEPCombineInput(
                hidden_states=combine_hidden,
                route_weights_nvs=torch.ones(4),
                plan=dispatch_output.plan,
                num_tokens=2,
            )
            output = dispatcher.combine(combine_input)

        prefetch_call = fake_buffer.prefetch_calls[0]
        self.assertIs(prefetch_call["plan"], dispatch_output.plan)
        self.assertIs(prefetch_call["full_gate_weight"], layout.full_gate_weight)
        self.assertIs(prefetch_call["full_up_weight"], layout.full_up_weight)
        self.assertIs(prefetch_call["full_down_weight"], layout.full_down_weight)
        self.assertFalse(prefetch_call["async_finish"])

        combine_call = fake_buffer.combine_calls[0]
        self.assertIs(combine_call["plan"], dispatch_output.plan)
        self.assertIs(combine_call["hidden_nvsh"], combine_hidden)
        self.assertIsNone(combine_call["route_weights_nvs"])
        self.assertFalse(combine_call["async_finish"])
        torch.testing.assert_close(output, combine_hidden[:2])

    def test_combine_rejects_undersized_underlying_output(self):
        dispatcher, fake_buffer = self._make_dispatcher()
        fake_buffer.combine_output = torch.ones(1, 2, dtype=torch.bfloat16)
        combine_input = MoonEPCombineInput(
            hidden_states=torch.ones(4, 2, dtype=torch.bfloat16),
            route_weights_nvs=None,
            plan=object(),
            num_tokens=2,
        )

        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            with self.assertRaisesRegex(ValueError, "fewer rows"):
                dispatcher.combine(combine_input)

    def test_dispatch_enforces_static_capacity(self):
        dispatcher, fake_buffer = self._make_dispatcher(capacity=3)
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            for num_tokens in (2, 3):
                with self.subTest(num_tokens=num_tokens):
                    hidden_states = torch.ones(num_tokens, 2, dtype=torch.bfloat16)
                    dispatcher.dispatch(
                        hidden_states,
                        self._topk_output(num_tokens),
                    )
                    self.assertEqual(
                        fake_buffer.dispatch_calls[-1]["hidden_states"].shape[0],
                        3,
                    )

            with self.assertRaisesRegex(ValueError, "more tokens than"):
                dispatcher.dispatch(
                    torch.ones(4, 2, dtype=torch.bfloat16),
                    self._topk_output(4),
                )

    def test_dispatch_allows_empty_local_input(self):
        dispatcher, fake_buffer = self._make_dispatcher(capacity=3)
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            dispatch_output = dispatcher.dispatch(
                torch.empty(0, 2, dtype=torch.bfloat16),
                self._topk_output(0),
            )

        self.assertEqual(dispatch_output.num_tokens, 0)
        self.assertEqual(fake_buffer.dispatch_calls[0]["hidden_states"].shape, (3, 2))
        torch.testing.assert_close(
            fake_buffer.dispatch_calls[0]["tokens_per_expert"],
            torch.tensor([6, 0], dtype=torch.int32),
        )

    def test_dispatch_rejects_malformed_shapes_and_dtype(self):
        dispatcher, fake_buffer = self._make_dispatcher()
        valid_topk = self._topk_output(4)
        cases = [
            (
                torch.ones(2, dtype=torch.bfloat16),
                valid_topk,
                "rank 2",
            ),
            (
                torch.ones(4, 3, dtype=torch.bfloat16),
                valid_topk,
                "width does not match",
            ),
            (
                torch.ones(4, 2, dtype=torch.float16),
                valid_topk,
                "BF16",
            ),
            (
                torch.ones(4, 2, dtype=torch.bfloat16),
                StandardTopKOutput(
                    valid_topk.topk_weights,
                    valid_topk.topk_ids[:, 0],
                    valid_topk.router_logits,
                ),
                "topk_ids must be rank 2",
            ),
            (
                torch.ones(4, 2, dtype=torch.bfloat16),
                StandardTopKOutput(
                    valid_topk.topk_weights[:3],
                    valid_topk.topk_ids,
                    valid_topk.router_logits,
                ),
                "same shape",
            ),
            (
                torch.ones(4, 2, dtype=torch.bfloat16),
                StandardTopKOutput(
                    valid_topk.topk_weights[:, :1],
                    valid_topk.topk_ids[:, :1],
                    valid_topk.router_logits,
                ),
                "router_topk",
            ),
        ]
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            for hidden_states, topk_output, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(
                        (ValueError, TypeError, NotImplementedError), message
                    ):
                        dispatcher.dispatch(hidden_states, topk_output)

    def test_dispatch_rejects_out_of_range_and_negative_expert_ids(self):
        dispatcher, fake_buffer = self._make_dispatcher()
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            for ids, message in (
                (torch.tensor([[0, 2], [1, 0], [0, 1], [1, 0]]), "outside"),
                (torch.tensor([[0, -1], [1, 0], [0, 1], [1, 0]]), "negative"),
            ):
                with self.subTest(ids=ids.tolist()):
                    topk_output = StandardTopKOutput(
                        topk_weights=self._topk_output(4).topk_weights,
                        topk_ids=ids,
                        router_logits=torch.empty(0),
                    )
                    expected_exception = (
                        RuntimeError if message == "negative" else ValueError
                    )
                    with self.assertRaisesRegex(expected_exception, message):
                        dispatcher.dispatch(
                            torch.ones(4, 2, dtype=torch.bfloat16),
                            topk_output,
                        )

    def test_split_phase_methods_fail_with_accurate_scope(self):
        dispatcher, _fake_buffer = self._make_dispatcher()

        with self.assertRaisesRegex(
            NotImplementedError,
            "dispatch_a.*synchronous/eager.*split-phase/overlap",
        ):
            dispatcher.dispatch_a(None, None)
        with self.assertRaisesRegex(
            NotImplementedError,
            "dispatch_b.*synchronous/eager.*split-phase/overlap",
        ):
            dispatcher.dispatch_b()
        with self.assertRaisesRegex(
            NotImplementedError,
            "combine_a.*synchronous/eager.*split-phase/overlap",
        ):
            dispatcher.combine_a(None)
        with self.assertRaisesRegex(
            NotImplementedError,
            "combine_b.*synchronous/eager.*split-phase/overlap",
        ):
            dispatcher.combine_b()

    def test_padding_currently_skews_planning_counts_toward_expert_zero(self):
        dispatcher, fake_buffer = self._make_dispatcher(
            capacity=4,
            num_experts=3,
            prefetch_slots=1,
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            topk_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            router_logits=torch.empty(0),
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.moonep.MoonEPBuffer.get_moonep_buffer",
            return_value=fake_buffer,
        ):
            dispatcher.dispatch(
                torch.ones(1, 2, dtype=torch.bfloat16),
                topk_output,
            )

        torch.testing.assert_close(
            fake_buffer.dispatch_calls[0]["tokens_per_expert"],
            torch.tensor([6, 1, 1], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
