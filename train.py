"""Phase 3 — LoRA fine-tune on the 4.5k training split using Unsloth.

Run after `python prep_data.py`. Saves the LoRA adapter to outputs/adapter/.
"""

import json
import sys
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    cfg = load_config()

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    train_path = cfg["paths"]["train_jsonl"]
    if not Path(train_path).exists():
        print(f"ERROR: {train_path} not found. Run `python prep_data.py` first.", file=sys.stderr)
        return 1

    print(f"Loading base model: {cfg['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["training"]["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    print("Attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        random_state=cfg["seed"],
        use_gradient_checkpointing="unsloth",
    )

    train_examples = load_jsonl(train_path)
    print(f"Loaded {len(train_examples)} training examples")

    def to_chat_text(ex: dict) -> dict:
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    train_ds = Dataset.from_list(train_examples).map(
        to_chat_text, remove_columns=["prompt", "completion", "question", "dataset_name"]
    )

    Path(cfg["paths"]["adapter_dir"]).mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=cfg["paths"]["adapter_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        warmup_steps=cfg["training"]["warmup_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        bf16=cfg["training"]["bf16"],
        optim=cfg["training"]["optim"],
        seed=cfg["seed"],
        report_to="none",
        save_strategy="steps",
        save_total_limit=2,
        max_seq_length=cfg["training"]["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_args,
    )

    print("Starting training")
    trainer.train()

    print(f"Saving adapter to {cfg['paths']['adapter_dir']}")
    model.save_pretrained(cfg["paths"]["adapter_dir"])
    tokenizer.save_pretrained(cfg["paths"]["adapter_dir"])

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
