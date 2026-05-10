# LoRA Cybersecurity Judge

Fine-tune Llama-3.2-3B-Instruct (LoRA, via Unsloth) to act as a cybersecurity-answer judge.
Distills GPT-OSS 140B's ranking behavior over (question, 3 candidate answers) → ranking string.

## Phases implemented here (1-4)

1. **Data prep** — load the 5k labelled CSV, split into 4500 train / 500 test (deterministic, `SEED=42`).
2. **Prompt formatting** — Geetanjali's exact 3-candidate judge template.
3. **LoRA training** — Unsloth + SFTTrainer on Llama-3.2-3B-Instruct.
4. **Test-split validation** — sanity metrics on the 500 held-out (GPT-OSS labels as GT).

Phase 5 (final 529 human-eval) is intentionally out of scope for this round.

## Setup

```bash
pip install -r requirements.txt
```

Unsloth installation may need to be adjusted for your CUDA version — see https://github.com/unslothai/unsloth.

## Run order

```bash
# 1. Generate train.jsonl + test.jsonl (runs anywhere, no GPU needed)
python prep_data.py

# --- 3B variant (default config.yaml) ---
# 2a. Fine-tune Llama-3.2-3B
python train.py

# 3a. Evaluate the fine-tuned 3B adapter
python eval_test.py

# 4a. Evaluate the un-fine-tuned 3B base model (lift comparison)
python eval_test.py --model unsloth/Llama-3.2-3B-Instruct

# --- 8B variant (config_8b.yaml) ---
# 2b. Fine-tune Llama-3.1-8B
python train.py --config config_8b.yaml

# 3b. Evaluate the fine-tuned 8B adapter
python eval_test.py --config config_8b.yaml

# 4b. Evaluate the un-fine-tuned 8B base model (lift comparison)
python eval_test.py --config config_8b.yaml --model unsloth/Meta-Llama-3.1-8B-Instruct
```

Artifacts land in `outputs/` with per-model subfolders so the runs don't collide:

```
outputs/
├── train.jsonl                  # formatted SFT training data (model-agnostic)
├── test.jsonl                   # formatted SFT test data
├── finetuned/                   # 3B run (config.yaml) — flat layout
│   ├── adapter/                 # LoRA adapter weights + tokenizer
│   └── metrics.json             # fine-tuned 3B metrics
├── base/                        # 3B base baseline
│   └── metrics.json
└── 8b/                          # 8B run (config_8b.yaml) — fully nested
    ├── finetuned/
    │   ├── adapter/
    │   └── metrics.json
    └── base/
        └── metrics.json
```

`eval_test.py` derives the right output path automatically from the chosen
config's `paths.adapter_dir` (replacing the `finetuned` segment with `base` for
the base eval). You don't need to pass `--out` unless overriding the convention.

## Configuration

All knobs live in `config.yaml`. Key fields:
- `base_model` — defaults to `unsloth/Llama-3.2-3B-Instruct`. Change to `unsloth/Phi-3.5-mini-instruct` if Llama doesn't work.
- `paths.csv` — relative path from the script's CWD to the labelled CSV.
- `split.train_size` / `split.test_size` — 4500 / 500 per Geetanjali (2026-05-09).
- `lora.*` — LoRA hyperparameters (r=16, alpha=16, dropout=0).
- `training.*` — SFT hyperparameters.

`train.py` and `eval_test.py` require a CUDA GPU (Varshini's A100 on Unity).

## Notes for Varshini (A100 on Unity)

- `bf16: true` in the config — A100 supports it natively.
- `load_in_4bit=True` in `train.py` — should comfortably fit Llama-3.2-3B in 4-bit on a single A100.
- See `scripts/slurm_train.sh` for a starting SLURM batch script (adjust partition / account names for Unity).
- The CSV path defaults to `../../Final/train_5k_candidates_labelled.csv`. If the file is elsewhere on Unity, override via `paths.csv` in `config.yaml`.

## Open coordination items (not blocking these phases)

- Whether `paths.csv` works for Varshini's environment — she may need to update `config.yaml`.
- Final base model choice (Llama-3.2-3B vs Phi-3.5-mini) — start with Llama; swap one config value if needed.
