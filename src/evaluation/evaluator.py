"""
Evaluator: scores each model's response against the TRUE OUTPUT
using an LLM-as-judge approach (Anthropic Claude via API).

The judge returns a single integer 0-100.
"""
import os
import asyncio
import anthropic
import pandas as pd
from config import EVAL_MODEL_COLS, EVALUATED_CSV


# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Scoring prompt ────────────────────────────────────────────────────────────
EVAL_PROMPT_TEMPLATE = """\
### LLM Response Evaluation

**Task:** Score the quality of a Model Response for a customer support chatbot.
Consider the System Prompt, User Input, and a True Output (a reference good response).
Return a single integer from 0 to 100.

**Evaluation criteria:**
1. Adherence to instructions in the system prompt (tone, length, constraints)
2. Relevance and accuracy relative to the user's question
3. Completeness — does it fully answer the question?
4. Clarity and naturalness of language
5. Similarity in quality to the True Output (used as a reference, not a strict target)

**Scoring guide:**
- 90-100 Excellent: flawlessly answers the query, matches or exceeds the reference
- 80-89  Good: minor gaps, mostly accurate and clear
- 70-79  Fair: partially answers, some inaccuracies or verbosity
- 60-69  Needs work: noticeable issues in accuracy or clarity
- 50-59  Poor: significant problems
- 0-49   Very poor: fails to address the query

---
**System Prompt:** {system_prompt}

**User Input:** {user_input}

**True Output (reference):** {true_output}

**Model Response:** {model_response}

---
Return ONLY a single integer between 0 and 100. No explanation.
"""


# ── Single evaluation call ────────────────────────────────────────────────────
def evaluate_response(
    user_input: str,
    true_output: str,
    model_response: str,
    system_prompt: str,
) -> int | str:
    """Call Claude to score one model response. Returns int or error string."""
    prompt = EVAL_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        user_input=user_input,
        true_output=true_output,
        model_response=model_response,
    )
    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = message.content[0].text.strip()
        score = int(score_str)
        if 0 <= score <= 100:
            return score
        return "Invalid Score (Out of Range)"
    except ValueError:
        return "Invalid Score (Not an Integer)"
    except Exception as e:
        print(f"  ⚠ Evaluation error: {e}")
        return "Error"


# ── Batch evaluation ──────────────────────────────────────────────────────────
def evaluate_all(
    benchmark_csv: str,
    system_prompt: str,
    output_csv: str = EVALUATED_CSV,
    model_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read benchmark_csv, evaluate every model column, write evaluated CSV.

    Args:
        benchmark_csv: merged CSV with all model outputs and a TRUE OUTPUT column.
        system_prompt: the system prompt text (used in the judge prompt).
        output_csv:    destination file.
        model_cols:    list of model column names to evaluate (default: config.EVAL_MODEL_COLS).
    """
    if model_cols is None:
        model_cols = EVAL_MODEL_COLS

    df = pd.read_csv(benchmark_csv)
    total = len(df) * len(model_cols)
    done  = 0

    for idx, row in df.iterrows():
        user_input  = str(row.get("USER INPUT", "")).replace(",", "").replace('"', "")
        true_output = str(row.get("TRUE OUTPUT", ""))

        for col in model_cols:
            model_response = row.get(col)
            eval_col = f"score_{col}"

            if isinstance(model_response, str) and model_response.strip():
                score = evaluate_response(user_input, true_output, model_response, system_prompt)
            else:
                score = "No Response"

            df.loc[idx, eval_col] = score
            done += 1
            print(f"  [{done}/{total}] row={idx+1} | {col[:40]} → {score}")

    import os
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Evaluations saved → {output_csv}")
    return df


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.csv_utils import load_prompt
    from config import SYSTEM_PROMPT_FILE

    system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
    evaluate_all("results/benchmark.csv", system_prompt)
