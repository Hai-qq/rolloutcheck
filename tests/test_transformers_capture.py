from types import SimpleNamespace

import pytest

from rolloutcheck.case import CaseError
from rolloutcheck.trace import TraceRecorder, inspect_trace
from rolloutcheck.transformers_capture import generate_recorded

torch = pytest.importorskip("torch", reason="Optional generation extra")
transformers = pytest.importorskip("transformers", reason="Optional generation extra")


class Model:
    config = SimpleNamespace(is_encoder_decoder=False)

    def __init__(self):
        self.calls = 0

    def generate(self, *, input_ids, attention_mask, generation_config, use_model_defaults):
        assert use_model_defaults is False
        self.calls += 1
        assert attention_mask.tolist() == [[1] * input_ids.shape[1]]
        return torch.cat([input_ids, torch.tensor([[8, 9]], device=input_ids.device)], dim=1)


def call(model, writer, turn="1", parent=None, input_ids=None, config=None):
    return generate_recorded(
        model,
        input_ids if input_ids is not None else torch.tensor([[1, 2]]),
        recorder=writer,
        session_id="s",
        branch_id="main",
        turn_id=turn,
        parent_turn_id=parent,
        token_space="fake-model-test",
        contract={
            "version": "history-prefix/v1",
            "mode": "append_only",
            "history_policy": "preserved",
        },
        generation_config=config
        or transformers.GenerationConfig(
            max_new_tokens=4,
            eos_token_id=9,
            pad_token_id=0,
        ),
    )


def test_adapter_slices_returned_sequence_retaining_eos(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, trace_id="test", evidence_kind="synthetic") as writer:
        output, stats = call(Model(), writer)
        assert output == [8, 9] and stats["stop_reason"] == "eos"
        assert stats["capture_seconds"] >= 0
        call(Model(), writer, "2", "1", torch.tensor([[1, 2, *output, 5]]))
    report, cases = inspect_trace(path)
    assert report["status"] == "PASS"
    assert cases[0]["previous"]["output_ids"] == [8, 9]


@pytest.mark.parametrize(
    "options",
    [
        {"num_beams": 2},
        {"num_return_sequences": 2},
        {"token_healing": True},
        {"stop_strings": ["stop"]},
        {"return_dict_in_generate": True},
        {"forced_eos_token_id": 9},
    ],
)
def test_unsupported_generation_mode_rejected_before_execution(tmp_path, options):
    model = Model()
    config = transformers.GenerationConfig(max_new_tokens=4, do_sample=True, **options)
    with TraceRecorder(
        tmp_path / "test.jsonl", trace_id="test", evidence_kind="synthetic"
    ) as writer:
        with pytest.raises(CaseError):
            call(model, writer, config=config)
    assert model.calls == 0


@pytest.mark.parametrize(
    "inputs",
    [torch.tensor([[1], [2]]), torch.tensor([1, 2]), torch.tensor([[1.0]]), torch.tensor([[0, 1]])],
)
def test_invalid_input_is_rejected_before_execution(tmp_path, inputs):
    model = Model()
    with TraceRecorder(
        tmp_path / "test.jsonl", trace_id="test", evidence_kind="synthetic"
    ) as writer:
        with pytest.raises(CaseError):
            call(model, writer, input_ids=inputs)
    assert model.calls == 0


def test_changed_returned_prefix_is_not_recorded_as_real(tmp_path):
    class BadModel(Model):
        def generate(self, **kwargs):
            return torch.tensor([[99, 2, 8, 9]])

    path = tmp_path / "bad.jsonl"
    with TraceRecorder(path, trace_id="test", evidence_kind="synthetic") as writer:
        with pytest.raises(CaseError, match="changed"):
            call(BadModel(), writer)
    assert path.read_bytes() == b""


def test_eos_alias_in_history_is_not_mistaken_for_padding(tmp_path):
    config = transformers.GenerationConfig(max_new_tokens=4, eos_token_id=9, pad_token_id=9)
    with TraceRecorder(
        tmp_path / "alias.jsonl", trace_id="test", evidence_kind="synthetic"
    ) as writer:
        output, stats = call(Model(), writer, input_ids=torch.tensor([[1, 9, 2]]), config=config)
    assert output == [8, 9] and stats["stop_reason"] == "eos"


def test_actual_tiny_decoder_does_not_inherit_model_beam_defaults(tmp_path):
    # No downloaded model: exercise the real generate implementation with random weights.
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=0,
        )
    ).eval()
    model.generation_config.num_beams = 2
    model.generation_config.num_return_sequences = 2
    model.generation_config.transformers_version = transformers.__version__
    path = tmp_path / "real-api.jsonl"
    with TraceRecorder(path, trace_id="test", evidence_kind="synthetic") as writer:
        output, stats = call(model, writer)
    assert 1 <= len(output) <= 4
    assert stats["stop_reason"] in ("eos", "length")
