"""Phase 1 — load the 5k labelled CSV and split deterministically into train/test JSONL.

Each output JSONL row contains:
  - prompt:       the formatted judge prompt (no completion appended)
  - completion:   the GPT-OSS ranking label (one of 6 permutation strings)
  - question:     original question text (for inspection)
  - dataset_name: source dataset (for per-dataset breakdown later)

Mapping (per Geetanjali's GPT-OSS prompt):
  R1 = answer (reference)
  R2 = candidate_llama_3_2_1b_instruct
  R3 = candidate_gemma3_27b_it
"""

import csv
import json
import random
import sys
from pathlib import Path

import yaml

from prompt import format_prompt, VALID_RANKINGS


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_example(row: dict) -> dict:
    return {
        "prompt": format_prompt(
            question=row["question"],
            r1=row["answer"],
            r2=row["candidate_llama_3_2_1b_instruct"],
            r3=row["candidate_gemma3_27b_it"],
        ),
        "completion": row["gpt_oss_judge_ranking"].strip(),
        "question": row["question"],
        "dataset_name": row["dataset_name"],
    }


def write_jsonl(examples: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> int:
    cfg = load_config()
    random.seed(cfg["seed"])

    csv_path = cfg["paths"]["csv"]
    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        return 1

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    valid_rows = [r for r in rows if r["gpt_oss_judge_ranking"].strip() in VALID_RANKINGS]
    dropped = len(rows) - len(valid_rows)
    if dropped:
        print(f"Dropped {dropped} rows with invalid ranking labels")

    train_n = cfg["split"]["train_size"]
    test_n = cfg["split"]["test_size"]
    needed = train_n + test_n
    if len(valid_rows) < needed:
        print(
            f"ERROR: have {len(valid_rows)} valid rows, need {needed} "
            f"({train_n} train + {test_n} test)",
            file=sys.stderr,
        )
        return 1

    random.shuffle(valid_rows)
    train_rows = valid_rows[:train_n]
    test_rows = valid_rows[train_n : train_n + test_n]

    train_examples = [make_example(r) for r in train_rows]
    test_examples = [make_example(r) for r in test_rows]

    write_jsonl(train_examples, cfg["paths"]["train_jsonl"])
    write_jsonl(test_examples, cfg["paths"]["test_jsonl"])

    print(f"Wrote {len(train_examples)} train -> {cfg['paths']['train_jsonl']}")
    print(f"Wrote {len(test_examples)} test  -> {cfg['paths']['test_jsonl']}")

    from collections import Counter
    train_dist = Counter(ex["completion"] for ex in train_examples)
    test_dist = Counter(ex["completion"] for ex in test_examples)
    print("\nLabel distribution:")
    print(f"  {'label':<12} {'train':>6} {'test':>6}")
    for label in sorted(VALID_RANKINGS):
        print(f"  {label:<12} {train_dist.get(label, 0):>6} {test_dist.get(label, 0):>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
