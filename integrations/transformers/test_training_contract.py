"""Framework contract tests; small random CPU model, no pretrained weight download."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_handoff import checkpoint_manifest, save, validate_resume  # noqa: E402

from rolloutcheck.case import CaseError  # noqa: E402
from rolloutcheck.training_batch import audit_batch, pad_features  # noqa: E402


def test_real_lm_collator_overwrites_masks_and_guard_rejects_it():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(10)}, unk_token="0")),
        pad_token="9",
        eos_token="9",
        unk_token="0",
    )
    features = [{"input_ids": [1, 2, 3, 9], "labels": [-100, -100, 3, 9]}]
    wrong = DataCollatorForLanguageModeling(tokenizer, mlm=False)(features)
    report = audit_batch(features, {k: v.tolist() for k, v in wrong.items()}, pad_token_id=9)
    assert report["status"] == "MISMATCH" and report["mismatched_fields"] == ["labels"]
    assert wrong["labels"].tolist() == [[1, 2, 3, -100]]
    assert (
        audit_batch(features, pad_features(features, pad_token_id=9), pad_token_id=9)["status"]
        == "MATCHED"
    )


def test_actual_qwen_loss_matches_masked_shifted_cross_entropy():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_dropout=0.0,
            use_cache=False,
        )
    ).eval()
    rows = [
        {"input_ids": [1, 2, 3, 9], "labels": [-100, -100, 3, 9]},
        {"input_ids": [1, 9], "labels": [-100, 9]},
    ]
    batch = {k: torch.tensor(v) for k, v in pad_features(rows, pad_token_id=9).items()}
    output = model(**batch)
    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, 16),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_checkpoint_guards_reject_changed_input_and_incomplete_files(tmp_path):
    # These are inert placeholder bytes: this test never deserializes checkpoint files.
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    ):
        (tmp_path / name).write_bytes(b"fixture")
    contract = {"handoff_sha256": "a"}
    save(tmp_path / "rolloutcheck-checkpoint.json", checkpoint_manifest(tmp_path, contract))
    validate_resume(tmp_path, contract)
    with pytest.raises(CaseError, match="identity mismatch"):
        validate_resume(tmp_path, {"handoff_sha256": "b"})
    (tmp_path / "optimizer.pt").write_bytes(b"modified")
    with pytest.raises(CaseError, match="identity mismatch"):
        validate_resume(tmp_path, contract)
    (tmp_path / "optimizer.pt").unlink()
    with pytest.raises(CaseError, match="Incomplete"):
        validate_resume(tmp_path, contract)
    assert (
        json.loads((tmp_path / "rolloutcheck-checkpoint.json").read_text())["contract"] == contract
    )
