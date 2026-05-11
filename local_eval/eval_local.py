"""Local base-model eval via Ollama — runs on Pramith's Windows laptop.

Self-contained alternative to running the cluster eval on the A100. Sends each
test prompt to a locally-running Ollama (http://localhost:11434), which uses
llama.cpp under the hood. With an NVIDIA GPU (e.g. RTX 5070) Ollama auto-uses
CUDA and the full 500-row eval completes in 5-15 min.

This script reads the *same* outputs/test.jsonl that the fine-tuned eval used,
so the comparison is on the identical 500 examples.

Setup (one-time):
  1. Install Ollama for Windows: https://ollama.com/download/windows
  2. In PowerShell:
       ollama pull llama3.2:3b
  3. pip install ollama pyyaml

Run from anywhere — the script chdirs to its parent (code/LoRA/) at startup
so paths in config.yaml resolve naturally:

  python local_eval/eval_local.py

Output lands in outputs/base/metrics.json (with backend="ollama" so it's
clear what produced it). Same path the cluster version writes to.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml


# Switch CWD to code/LoRA/ so the relative paths in config.yaml resolve
# regardless of where the user invoked the script from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
os.chdir(PROJECT_DIR)


# ---------------------------------------------------------------------------
# Inlined helpers — kept self-contained so this folder doesn't depend on the
# parent-folder modules. The logic must stay identical to eval_test.py;
# update both together if the parsing rules ever change.
# ---------------------------------------------------------------------------

VALID_RANKINGS = frozenset({
    "R1>R2>R3", "R1>R3>R2", "R2>R1>R3", "R2>R3>R1", "R3>R1>R2", "R3>R2>R1",
})

RANKING_RE = re.compile(r"R[123]>R[123]>R[123]")


def parse_ranking(text: str):
    match = RANKING_RE.search(text)
    if match and match.group(0) in VALID_RANKINGS:
        return match.group(0)
    return None


def pairwise_agreement(pred: str, gt: str) -> float:
    p = {c: i for i, c in enumerate(pred.split(">"))}
    g = {c: i for i, c in enumerate(gt.split(">"))}
    cands = ["R1", "R2", "R3"]
    pairs = [(cands[i], cands[j]) for i in range(len(cands)) for j in range(i + 1, len(cands))]
    agree = sum(1 for a, b in pairs if (p[a] < p[b]) == (g[a] < g[b]))
    return agree / len(pairs)


def derive_default_out(adapter_dir: str) -> str:
    """Sibling 'base' folder of adapter_dir's parent. e.g.
       outputs/finetuned/adapter    -> outputs/base/metrics.json
       outputs/8b/finetuned/adapter -> outputs/8b/base/metrics.json
    """
    finetuned_dir = Path(adapter_dir).parent
    base_dir = finetuned_dir.parent / finetuned_dir.name.replace("finetuned", "base", 1)
    return str(base_dir / "metrics.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="YAML config (default: config.yaml).")
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:3b",
        help="Ollama model tag (default: llama3.2:3b — Llama-3.2-3B-Instruct in Q4_K_M).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap on test examples (sanity check).")
    parser.add_argument("--out", default=None, help="Override the output path inside outputs/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: config {args.config} not found (cwd={Path.cwd()}).", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    test_path = cfg["paths"]["test_jsonl"]
    if not Path(test_path).exists():
        print(f"ERROR: {test_path} not found. Run `python prep_data.py` first.", file=sys.stderr)
        return 1

    try:
        import ollama
    except ImportError:
        print("ERROR: ollama Python library not installed. Run: pip install ollama", file=sys.stderr)
        return 1

    try:
        ollama.list()
    except Exception as e:
        print(
            "ERROR: Can't reach Ollama at http://localhost:11434.\n"
            "Is Ollama installed and running? See https://ollama.com\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        return 1

    test = [json.loads(line) for line in open(test_path, "r", encoding="utf-8")]
    if args.limit:
        test = test[: args.limit]
    print(f"Evaluating {len(test)} test examples via Ollama '{args.ollama_model}'")

    max_new = cfg["eval"]["max_new_tokens"]

    n_total = n_parsed = n_exact = n_top1 = 0
    pairwise_sum = 0.0
    pred_dist: Counter = Counter()
    gt_dist: Counter = Counter()
    per_dataset: dict = {}
    failures: list = []

    start = time.time()
    for i, ex in enumerate(test):
        gt = ex["completion"]
        gt_dist[gt] += 1
        ds = ex.get("dataset_name", "unknown")
        per_dataset.setdefault(ds, {"n": 0, "exact": 0, "top1": 0, "parsed": 0})
        per_dataset[ds]["n"] += 1
        n_total += 1

        try:
            response = ollama.chat(
                model=args.ollama_model,
                messages=[{"role": "user", "content": ex["prompt"]}],
                options={"num_predict": max_new, "temperature": 0.0},
            )
            gen_text = response["message"]["content"]
        except Exception as e:
            print(f"\nERROR on example {i}: {e}", file=sys.stderr)
            failures.append({"question": ex["question"][:120], "raw_output": f"[ollama error] {e}", "gt": gt})
            continue

        pred = parse_ranking(gen_text)
        if pred is None:
            failures.append({"question": ex["question"][:120], "raw_output": gen_text[:200], "gt": gt})
        else:
            n_parsed += 1
            per_dataset[ds]["parsed"] += 1
            pred_dist[pred] += 1
            if pred == gt:
                n_exact += 1
                per_dataset[ds]["exact"] += 1
            if pred.split(">")[0] == gt.split(">")[0]:
                n_top1 += 1
                per_dataset[ds]["top1"] += 1
            pairwise_sum += pairwise_agreement(pred, gt)

        if (i + 1) % 10 == 0 or (i + 1) == len(test):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(test) - (i + 1)) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>4}/{len(test)}] elapsed {elapsed:6.0f}s  "
                f"({rate:.2f} ex/s, ETA {eta/60:.1f} min)  "
                f"parsed {n_parsed}/{n_total}  top1 {n_top1}/{n_total}  exact {n_exact}/{n_total}"
            )

    out_relpath = args.out if args.out else derive_default_out(cfg["paths"]["adapter_dir"])
    out_path = Path(cfg["paths"]["output_dir"]) / out_relpath if args.out else Path(out_relpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": f"ollama:{args.ollama_model}",
        "model_label": f"base / Ollama: {args.ollama_model}",
        "backend": "ollama",
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
        "elapsed_seconds": round(time.time() - start, 1),
    }

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
