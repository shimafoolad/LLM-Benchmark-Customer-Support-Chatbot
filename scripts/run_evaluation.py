"""
scripts/run_evaluation.py
─────────────────────────────────────────────────────────────
Score model outputs with an LLM judge and visualize results.

Usage:
  # Score all models in results/benchmark.csv
  python scripts/run_evaluation.py

  # Score and then plot
  python scripts/run_evaluation.py --visualize

  # Only visualize (skip scoring — use existing evaluated_responses.csv)
  python scripts/run_evaluation.py --only-visualize

Requires:
  export ANTHROPIC_API_KEY=sk-ant-...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import EVALUATED_CSV, SYSTEM_PROMPT_FILE
from src.utils.csv_utils import load_prompt
from src.evaluation.evaluator import evaluate_all
from src.evaluation.visualize import run_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize LLM benchmark")
    parser.add_argument(
        "--benchmark-csv", default="results/benchmark.csv",
        help="Path to merged benchmark CSV (default: results/benchmark.csv)",
    )
    parser.add_argument(
        "--output-csv", default=EVALUATED_CSV,
        help=f"Path for evaluated output CSV (default: {EVALUATED_CSV})",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate plots after evaluation.",
    )
    parser.add_argument(
        "--only-visualize", action="store_true",
        help="Skip evaluation; only run visualizations on existing evaluated CSV.",
    )
    args = parser.parse_args()

    if args.only_visualize:
        if not os.path.exists(args.output_csv):
            print(f"❌ Evaluated CSV not found: {args.output_csv}")
            sys.exit(1)
        print(f"📊 Visualizing {args.output_csv} …")
        run_all(args.output_csv)
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.benchmark_csv):
        print(f"❌ Benchmark CSV not found: {args.benchmark_csv}")
        print("   Run  scripts/run_inference.py --model all --merge  first.")
        sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("❌ ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
    print(f"🔍 Evaluating {args.benchmark_csv} …\n")
    evaluate_all(args.benchmark_csv, system_prompt, output_csv=args.output_csv)

    if args.visualize:
        print("\n📊 Generating visualizations …")
        run_all(args.output_csv)


if __name__ == "__main__":
    main()
