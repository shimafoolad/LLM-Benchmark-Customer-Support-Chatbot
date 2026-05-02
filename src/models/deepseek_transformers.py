"""
Inference with deepseek-ai/DeepSeek-R1-Distill-Llama-8B
using the standard HuggingFace Transformers pipeline.

Requirements: pip install -r requirements/transformers.txt
"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import MODELS, MAX_NEW_TOKENS
from src.utils.csv_utils import write_results

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID  = MODELS["deepseek_transformers"]
COL_NAME  = "DeepSeek-R1-Distill-Llama-8B"
TIME_COL  = f"inference-time-{COL_NAME}"


# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    pipe      = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────
def build_inference_fn(pipe):
    """Return a callable (user_input, prompt_messages) → raw_str."""
    def _infer(user_input: str, prompt: list[dict]) -> str:
        result = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, truncation=True)
        return result[-1]["generated_text"][-1]["content"]
    return _infer


# ── Entry Point ───────────────────────────────────────────────────────────────
def run(input_csv: str, output_csv: str, limit: int | None = None):
    print(f"Loading {MODEL_ID} …")
    pipe = load_model()
    inference_fn = build_inference_fn(pipe)
    write_results(input_csv, output_csv, COL_NAME, TIME_COL, inference_fn, limit=limit)


if __name__ == "__main__":
    from config import INPUT_CSV
    run(INPUT_CSV, f"results/output_{COL_NAME}.csv")
