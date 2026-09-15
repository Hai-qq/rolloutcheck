"""Check actual Trainer/PEFT token weighting across unequal microbatches."""

import copy

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments

from rolloutcheck.training_batch import pad_features


def test_trainer_accumulation_matches_token_weighted_batch(tmp_path):
    torch.manual_seed(7)
    model = get_peft_model(
        Qwen3ForCausalLM(
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
        ),
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        ),
    )
    rows = [
        {"input_ids": [1, 2, 3, 9], "labels": [-100, -100, 3, 9]},
        {"input_ids": [1, 9], "labels": [-100, 9]},
    ]

    def collate(features):
        return {k: torch.tensor(v) for k, v in pad_features(features, pad_token_id=9).items()}

    states = []
    for batch, accumulation in ((2, 1), (1, 2)):
        instance = copy.deepcopy(model)
        Trainer(
            model=instance,
            train_dataset=rows,
            data_collator=collate,
            args=TrainingArguments(
                output_dir=str(tmp_path / str(batch)),
                use_cpu=True,
                max_steps=1,
                per_device_train_batch_size=batch,
                gradient_accumulation_steps=accumulation,
                learning_rate=0.01,
                optim="sgd",
                max_grad_norm=0.0,
                lr_scheduler_type="constant",
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                label_names=["labels"],
                seed=7,
                data_seed=7,
            ),
        ).train()
        states.append(
            {name: p.detach().clone() for name, p in instance.named_parameters() if p.requires_grad}
        )
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], atol=1e-7, rtol=1e-5)
