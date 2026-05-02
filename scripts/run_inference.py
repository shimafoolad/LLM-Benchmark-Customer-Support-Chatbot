"""
scripts/run_inference.py
─────────────────────────────────────────────────────────────
Run inference for one or all models and produce per-model CSVs.

Usage:
  python scripts/run_inference.py --model all
  python scripts/run_inference.py --model deepseek_transformers
  python scripts/run_inference.py --model deepseek_unsloth_4bit  --limit 5
  python scripts/run_inference.py --model deepseek_vllm
  python scripts/run_inference.py --model phi4_unsloth_4bit

After running all models, use scripts/run_evaluation.py to score and visualize.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import INPUT_CSV

REGISTRY = {
    "deepseek_transformers":  ("src.models.deepseek_transformers",  "run", "results/output_DeepSeek-R1-Distill-Llama-8B.csv"),
    "deepseek_unsloth_4bit":  ("src.models.deepseek_unsloth",       "run", "results/output_DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit.csv"),
    "deepseek_vllm":          ("src.models.deepseek_vllm",          "run", "results/output_DeepSeek-R1-Distill-Llama-8B-vllm.csv"),
    "phi4_unsloth_4bit":      ("src.models.phi4_unsloth",           "run", "results/output_phi-4-unsloth-bnb-4bit.csv"),
}


def run_model(name: str, limit: int | None):
    if name not in REGISTRY:
        print(f"❌ Unknown model '{name}'. Choose from: {list(REGISTRY)}")
        sys.exit(1)

    module_path, fn_name, output_csv = REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, fn_name)
    print(f"\n{'='*60}")
    print(f"  Model : {name}")
    print(f"  Input : {INPUT_CSV}")
    print(f"  Output: {output_csv}")
    if limit:
        print(f"  Limit : {limit} rows")
    print(f"{'='*60}")
    fn(INPUT_CSV, output_csv, limit=limit)


def merge_all():
    """Merge all per-model result CSVs into a single benchmark.csv."""
    from src.utils.csv_utils import merge_results
    result_files = [v[2] for v in REGISTRY.values() if os.path.exists(v[2])]
    if not result_files:
        print("❌ No result files found. Run inference first.")
        return
    merge_results(result_files, "results/benchmark.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM inference benchmark")
    parser.add_argument(
        "--model", required=True,
        choices=list(REGISTRY.keys()) + ["all"],
        help="Model to run, or 'all' to run every model sequentially.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of rows to process (useful for quick tests).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="After inference, merge all available result CSVs into benchmark.csv.",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in REGISTRY:
            run_model(name, args.limit)
        merge_all()
    else:
        run_model(args.model, args.limit)
        if args.merge:
            merge_all()
