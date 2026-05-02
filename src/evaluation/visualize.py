"""
Visualization: plots and summary tables for the benchmark results.
Reads the evaluated_responses.csv and produces charts saved to results/plots/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _score_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("score_")]


def _model_label(col: str) -> str:
    return col.replace("score_", "")


# ── 1. Bar chart — mean score per model ──────────────────────────────────────
def plot_mean_scores(df: pd.DataFrame, save: bool = True) -> pd.Series:
    cols   = _score_cols(df)
    labels = [_model_label(c) for c in cols]

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    means   = numeric.mean()
    stds    = numeric.std()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, color=PALETTE[:len(cols)], width=0.55,
                  yerr=stds, capsize=5, edgecolor="white")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean Score (0-100)", fontsize=12)
    ax.set_title("Mean LLM Score per Model (higher = better)", fontsize=14, pad=14)
    ax.bar_label(bars, fmt="%.1f", padding=6, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()

    if save:
        path = f"{PLOT_DIR}/mean_scores.png"
        plt.savefig(path, dpi=150)
        print(f"✅ Saved → {path}")
    plt.show()
    return means


# ── 2. Box plot — score distribution per model ────────────────────────────────
def plot_score_distribution(df: pd.DataFrame, save: bool = True):
    cols   = _score_cols(df)
    labels = [_model_label(c) for c in cols]
    data   = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE[:len(cols)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Score (0-100)", fontsize=12)
    ax.set_title("Score Distribution per Model", fontsize=14, pad=14)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()

    if save:
        path = f"{PLOT_DIR}/score_distribution.png"
        plt.savefig(path, dpi=150)
        print(f"✅ Saved → {path}")
    plt.show()


# ── 3. Heatmap — score per row × model ───────────────────────────────────────
def plot_score_heatmap(df: pd.DataFrame, save: bool = True):
    cols = _score_cols(df)
    heat = df[cols].apply(pd.to_numeric, errors="coerce")
    heat.columns = [_model_label(c) for c in cols]
    heat.index   = [f"Q{i+1}" for i in range(len(heat))]

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 2), max(5, len(heat) * 0.5)))
    sns.heatmap(
        heat, annot=True, fmt=".0f", cmap="RdYlGn",
        vmin=0, vmax=100, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Score"},
    )
    ax.set_title("Per-Question Score Heatmap (rows = questions)", fontsize=13, pad=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Question", fontsize=11)
    plt.tight_layout()

    if save:
        path = f"{PLOT_DIR}/score_heatmap.png"
        plt.savefig(path, dpi=150)
        print(f"✅ Saved → {path}")
    plt.show()


# ── 4. Inference time comparison ─────────────────────────────────────────────
def plot_inference_time(df: pd.DataFrame, save: bool = True):
    time_cols = [c for c in df.columns if c.startswith("inference-time-")]
    if not time_cols:
        print("No inference-time columns found — skipping.")
        return

    labels = [c.replace("inference-time-", "") for c in time_cols]
    means  = [pd.to_numeric(df[c], errors="coerce").mean() for c in time_cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, means, color=PALETTE[:len(time_cols)], edgecolor="white")
    ax.set_xlabel("Mean Inference Time (seconds)", fontsize=12)
    ax.set_title("Mean Inference Time per Model (lower = faster)", fontsize=14, pad=12)
    ax.bar_label(bars, fmt="%.1fs", padding=5, fontsize=10)
    plt.tight_layout()

    if save:
        path = f"{PLOT_DIR}/inference_time.png"
        plt.savefig(path, dpi=150)
        print(f"✅ Saved → {path}")
    plt.show()


# ── 5. Summary table ──────────────────────────────────────────────────────────
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = _score_cols(df)
    rows = []
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        time_col = "inference-time-" + _model_label(col)
        time_vals = pd.to_numeric(df.get(time_col, pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "Model":        _model_label(col),
            "Mean Score":   round(numeric.mean(), 2),
            "Std Dev":      round(numeric.std(), 2),
            "Min Score":    int(numeric.min()),
            "Max Score":    int(numeric.max()),
            "Median Score": round(numeric.median(), 2),
            "Avg Time (s)": round(time_vals.mean(), 2) if not time_vals.isna().all() else "-",
            "No Response":  (df[col] == "No Response").sum(),
        })
    summary = pd.DataFrame(rows).sort_values("Mean Score", ascending=False).reset_index(drop=True)
    print("\n📊 Benchmark Summary")
    print(summary.to_string(index=False))
    summary.to_csv(f"{PLOT_DIR}/summary_table.csv", index=False)
    return summary


# ── Run all visualizations ────────────────────────────────────────────────────
def run_all(evaluated_csv: str):
    df = pd.read_csv(evaluated_csv)
    print(f"Loaded {len(df)} rows from {evaluated_csv}\n")

    plot_mean_scores(df)
    plot_score_distribution(df)
    plot_score_heatmap(df)
    plot_inference_time(df)
    summary_table(df)


if __name__ == "__main__":
    from config import EVALUATED_CSV
    run_all(EVALUATED_CSV)
