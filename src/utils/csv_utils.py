"""
Utility functions for reading and writing benchmark CSVs.
"""
import csv
import ast
import os
import pandas as pd


def load_prompt(prompt_file: str) -> str:
    """Load system prompt from a text file."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_dataset(input_file: str) -> list[dict]:
    """Read the input CSV and return a list of row dicts."""
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_prompt_column(prompt_str: str) -> list[dict] | None:
    """Parse the 'prompt' column (stored as a Python literal) into a list of messages."""
    if not prompt_str:
        return None
    try:
        return ast.literal_eval(prompt_str)
    except (ValueError, SyntaxError):
        return None


def clean_user_input(text: str) -> str:
    """Strip problematic characters from user input."""
    return text.replace('"', "").replace(",", "").strip()


def clean_model_output(text: str) -> str:
    """Strip thinking tags and artifacts from model output."""
    return (
        text.split("</think>")[-1]
        .replace("<｜end▁of▁sentence｜>", "")
        .split("response")[-1]
        .replace(":", "")
        .replace("\n", " ")
        .replace("'", "")
        .replace('"', "")
        .replace("}", "")
        .strip()
    )


def write_results(
    input_file: str,
    output_file: str,
    model_col: str,
    time_col: str,
    inference_fn,
    limit: int | None = None,
) -> None:
    """
    Generic CSV runner:
    - Reads input_file row by row.
    - Calls inference_fn(user_input, prompt_messages) → str.
    - Writes results to output_file with two new columns: model_col and time_col.

    Args:
        input_file:   Path to input CSV (must have 'USER INPUT', 'prompt' columns).
        output_file:  Path for the output CSV.
        model_col:    Column name for model responses.
        time_col:     Column name for inference time (seconds).
        inference_fn: Callable(user_input: str, prompt: list[dict]) → str.
        limit:        Max rows to process (None = all rows).
    """
    import time

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    try:
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", newline="", encoding="utf-8") as outfile,
        ):
            reader = csv.DictReader(infile)
            fieldnames = list(reader.fieldnames) + [model_col, time_col]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break

                raw_input = row.get("USER INPUT", "")
                user_input = clean_user_input(raw_input)
                prompt = parse_prompt_column(row.get("prompt", ""))

                if user_input and prompt:
                    t0 = time.time()
                    raw_output = inference_fn(user_input, prompt)
                    elapsed = time.time() - t0

                    row[model_col] = clean_model_output(raw_output)
                    row[time_col] = round(elapsed, 2)
                else:
                    row[model_col] = ""
                    row[time_col] = 0

                writer.writerow(row)
                print(f"[{i+1}] ✓  ({row[time_col]}s) {row[model_col][:60]}")

        print(f"\n✅ Done → {output_file}")

    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def merge_results(result_files: list[str], output_file: str) -> pd.DataFrame:
    """
    Merge multiple per-model result CSVs into a single benchmark CSV.
    All files must share 'USER INPUT' and 'TRUE OUTPUT' columns.
    """
    base = pd.read_csv(result_files[0])
    for path in result_files[1:]:
        df = pd.read_csv(path)
        new_cols = [c for c in df.columns if c not in base.columns]
        base = base.join(df[new_cols])

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    base.to_csv(output_file, index=False)
    print(f"✅ Merged → {output_file}  ({len(base)} rows, {len(base.columns)} cols)")
    return base
