from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import core.core.communication as communication
import logicpipe.orchestrator as orchestrator
from core.core.schedules import PipelineRuntime


class _CaptureCommunicationHandler:
    def __init__(self):
        self.tensor_tag = {"forward": "forward"}
        self.sent = []

    def send(self, tensor, tag):
        self.sent.append((tensor, tag))


class PipelineRuntimeTests(unittest.TestCase):
    def test_send_activation_forward_pads_to_fixed_forward_shape(self):
        runtime = object.__new__(PipelineRuntime)
        runtime.stage = 0
        runtime.total_stage = 2
        runtime.config = SimpleNamespace(max_sub_sequence_len=5, hidden_size=2)
        runtime.comm_handler = _CaptureCommunicationHandler()
        activation = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

        runtime.send_activation_forward(activation)

        sent_activation, tag = runtime.comm_handler.sent[0]
        self.assertEqual(tag, "forward")
        self.assertEqual(sent_activation.shape, (1, 5, 2))
        self.assertTrue(torch.equal(sent_activation[:, :3, :], activation))
        self.assertTrue(torch.equal(sent_activation[:, 3:, :], torch.zeros(1, 2, 2)))


class PrefillCommunicationHandlerTests(unittest.TestCase):
    def test_send_dispatches_forward_tensor_synchronously(self):
        handler = object.__new__(communication.CommunicationHandler)
        handler.next_rank = 1
        handler.tensor_tag = {"forward": "forward", "seq_len": "seq_len"}
        tensor = torch.ones(1, 5, 2)
        calls = []
        original_send = communication._send
        communication._send = lambda sent_tensor, dst_rank, tag: calls.append(
            (sent_tensor, dst_rank, tag)
        )
        try:
            handler.send(tensor, "forward")
        finally:
            communication._send = original_send

        self.assertEqual(calls, [(tensor, 1, "forward")])

    def test_recv_dispatches_forward_tensor_synchronously(self):
        handler = object.__new__(communication.CommunicationHandler)
        handler.pre_rank = 0
        handler.tensor_shape = {"forward": (1, 5, 2), "seq_len": (1, 1)}
        handler.tensor_type = {"forward": torch.float32, "seq_len": torch.int64}
        handler.tensor_tag = {"forward": "forward", "seq_len": "seq_len"}
        handler.device = "cpu"
        expected = torch.ones(1, 5, 2)
        calls = []
        original_recv = communication._recv
        communication._recv = lambda shape, src_rank, tag, dtype: (
            calls.append((shape, src_rank, tag, dtype)) or expected
        )
        try:
            received = handler.recv("forward")
        finally:
            communication._recv = original_recv

        self.assertIs(received, expected)
        self.assertEqual(calls, [((1, 5, 2), 0, "forward", torch.float32)])


class _StopAfterPrefill(Exception):
    pass


class LogicPipeOrchestratorTests(unittest.TestCase):
    def test_decoding_pipeline_is_not_created_before_prefill(self):
        flags = {"decode_constructed": False, "prefill_saw_decode": None}

        class FakePlanner:
            def run(self, **_kwargs):
                return None, SimpleNamespace(
                    stage_num_hidden_layers_list=[1, 1, 1, 1],
                    bottleneck_ms=1.0,
                    selected_devices=[],
                )

        class FakePrefillEngine:
            def __init__(self, _runtime):
                pass

            def prefill_prompt(self, _prompt):
                flags["prefill_saw_decode"] = flags["decode_constructed"]
                raise _StopAfterPrefill

        class FakeController:
            def __init__(self, *_args, **_kwargs):
                pass

        def fake_decoding_pipeline(*_args, **_kwargs):
            flags["decode_constructed"] = True
            return object()

        fake_config = SimpleNamespace(num_hidden_layers=4)
        fake_args = SimpleNamespace(
            rank=0,
            world=4,
            num_stages=4,
            config_file="tasks/medusa_llama/config/vicuna_7b_config.json",
        )

        with patch.object(orchestrator.LlamaConfig, "from_pretrained", return_value=fake_config), \
            patch.object(orchestrator, "OfflinePipelinePlanner", return_value=FakePlanner()), \
            patch.object(
                orchestrator,
                "build_runtime",
                return_value=(fake_config, SimpleNamespace(), SimpleNamespace()),
            ), \
            patch.object(orchestrator, "LogicPipeController", FakeController), \
            patch.object(orchestrator, "set_controller", lambda _controller: None), \
            patch.object(orchestrator, "IntraSequencePrefillEngine", FakePrefillEngine), \
            patch.object(orchestrator, "DecodingPipeline", fake_decoding_pipeline):
            with self.assertRaises(_StopAfterPrefill):
                orchestrator.LogicPipeOrchestrator(fake_args, "2+2?").run()

        self.assertFalse(flags["prefill_saw_decode"])


if __name__ == "__main__":
    unittest.main()
