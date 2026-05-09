"""Phase 4 — sanity eval on the 500-row held-out test split.

Compares the fine-tuned model's predicted ranking against the GPT-OSS label for each
test row, and reports parse rate, exact-match accuracy, top-1 accuracy, and
per-position pairwise agreement.

Run after `python train.py`. Writes outputs/metrics_test.json.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from prompt import VALID_RANKINGS


RANKING_RE = re.compile(r"R[123]>R[123]>R[123]")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_ranking(text: str) -> str | None:
    """Extract the first valid ranking string from model output, or None."""
    match = RANKING_RE.search(text)
    if match and match.group(0) in VALID_RANKINGS:
        return match.group(0)
    return None


def ranking_to_position_map(ranking: str) -> dict:
    """Map candidate id -> rank (1=best, 3=worst). E.g. 'R3>R2>R1' -> {R3:1, R2:2, R1:3}."""
    parts = ranking.split(">")
    return {cand: i + 1 for i, cand in enumerate(parts)}


def pairwise_agreement(pred: str, gt: str) -> float:
    """Fraction of the 3 candidate pairs whose relative order matches."""
    p = ranking_to_position_map(pred)
    g = ranking_to_position_map(gt)
    cands = ["R1", "R2", "R3"]
    pairs = [(cands[i], cands[j]) for i in range(len(cands)) for j in range(i + 1, len(cands))]
    agree = sum(1 for a, b in pairs if (p[a] < p[b]) == (g[a] < g[b]))
    return agree / len(pairs)


def main() -> int:
    cfg = load_config()

    adapter_dir = cfg["paths"]["adapter_dir"]
    test_path = cfg["paths"]["test_jsonl"]
    if not Path(adapter_dir).exists():
        print(f"ERROR: adapter dir {adapter_dir} not found. Run `python train.py` first.", file=sys.stderr)
        return 1
    if not Path(test_path).exists():
        print(f"ERROR: {test_path} not found. Run `python prep_data.py` first.", file=sys.stderr)
        return 1

    from unsloth import FastLanguageModel

    print(f"Loading fine-tuned model from {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=cfg["training"]["max_seq_length"],
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    test = load_jsonl(test_path)
    print(f"Evaluating on {len(test)} test examples")

    n_total = 0
    n_parsed = 0
    n_exact = 0
    n_top1 = 0
    pairwise_sum = 0.0
    pred_dist = Counter()
    gt_dist = Counter()
    per_dataset = {}
    failures = []

    for ex in test:
        gt = ex["completion"]
        gt_dist[gt] += 1

        messages = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg["eval"]["max_new_tokens"],
            do_sample=cfg["eval"]["do_sample"],
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred = parse_ranking(gen_text)
        n_total += 1

        ds = ex.get("dataset_name", "unknown")
        per_dataset.setdefault(ds, {"n": 0, "exact": 0, "top1": 0, "parsed": 0})
        per_dataset[ds]["n"] += 1

        if pred is None:
            failures.append({"question": ex["question"][:120], "raw_output": gen_text[:120], "gt": gt})
            continue

        n_parsed += 1
        per_dataset[ds]["parsed"] += 1
        pred_dist[pred] += 1

        if pred == gt:
            n_exact += 1
            per_dataset[ds]["exact"] += 1

        pred_top = pred.split(">")[0]
        gt_top = gt.split(">")[0]
        if pred_top == gt_top:
            n_top1 += 1
            per_dataset[ds]["top1"] += 1

        pairwise_sum += pairwise_agreement(pred, gt)

    metrics = {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "exact_match_acc": n_exact / n_total if n_total else 0.0,
        "top1_acc": n_top1 / n_total if n_total else 0.0,
        "pairwise_agreement_avg": pairwise_sum / n_parsed if n_parsed else 0.0,
        "pred_distribution": dict(pred_dist),
        "gt_distribution": dict(gt_dist),
        "per_dataset": per_dataset,
        "n_parse_failures": len(failures),
        "first_5_failures": failures[:5],
    }

    out_path = Path(cfg["paths"]["output_dir"]) / "metrics_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nWrote metrics to {out_path}")
    print(f"  Parse rate           : {metrics['parse_rate']:.2%}")
    print(f"  Exact-match accuracy : {metrics['exact_match_acc']:.2%}")
    print(f"  Top-1 accuracy       : {metrics['top1_acc']:.2%}")
    print(f"  Pairwise agreement   : {metrics['pairwise_agreement_avg']:.2%}  (avg over parsed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
