# Local base-model eval (Ollama on Windows laptop)

Runs the same Phase 4 base-model evaluation as the cluster path, but on a
laptop via Ollama (no A100 / Unsloth / Linux required). Useful when waiting
for cluster GPU access.

The Ollama `llama3.2:3b` weights are the same Meta Llama-3.2-3B-Instruct
weights Unsloth uses, just in GGUF (Q4_K_M) format instead of BNB-4bit.
For the base-vs-fine-tuned comparison this is apples-to-apples.

## What it measures

Identical to the cluster's base eval — the script reads the *same*
`outputs/test.jsonl` produced by `prep_data.py`, sends each prompt to
Ollama, parses the predicted ranking, and computes:

- Parse rate
- Exact-match accuracy
- Top-1 accuracy (best candidate identified)
- Pairwise agreement
- Per-dataset breakdown

Output is written to `outputs/base/metrics.json` with `"backend": "ollama"`
in the JSON so it's clear which inference engine produced the numbers.

## Setup (one-time)

1. Install Ollama for Windows: https://ollama.com/download/windows
2. After install, in PowerShell:
   ```powershell
   ollama pull llama3.2:3b
   ```
3. Install the Python client and YAML library:
   ```powershell
   pip install ollama pyyaml
   ```

## Run

From the repo root (`code/LoRA/`):

```powershell
python local_eval/eval_local.py
```

The script auto-chdirs to its parent directory at startup so the relative
paths in `config.yaml` (`outputs/test.jsonl`, etc.) resolve correctly
regardless of where you invoked it from.

### Quick sanity check first (10 examples)

```powershell
python local_eval/eval_local.py --limit 10
```

### Run against the 8B config

```powershell
python local_eval/eval_local.py --config config_8b.yaml --ollama-model llama3.1:8b
```

## Expected runtime

| Hardware | Per example | 500 examples |
|---|---|---|
| RTX 5070 (CUDA via Ollama) | ~1-2 s | **5-15 min** |
| Modern CPU only | ~10-30 s | 1-4 hours |

Ollama auto-detects an NVIDIA GPU on Windows — no extra config needed.
You can verify GPU is in use by watching `nvidia-smi` while the eval runs,
or by checking Ollama's startup logs.

## After it finishes

Result lands at `outputs/base/metrics.json` (or `outputs/8b/base/metrics.json`
if you used the 8B config). Compare side-by-side with the fine-tuned
`outputs/finetuned/metrics.json` to see the lift from LoRA fine-tuning —
exactly what Geetanjali asked for.
