"""
Inference with unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit
using the Unsloth library for fast 4-bit quantized inference.

Requirements: pip install -r requirements/unsloth.txt
"""
from unsloth import FastLanguageModel
import torch
from config import MODELS, MAX_NEW_TOKENS, MAX_SEQ_LENGTH, LOAD_IN_4BIT, DTYPE
from src.utils.csv_utils import write_results

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID = MODELS["deepseek_unsloth_4bit"]
COL_NAME = "DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
TIME_COL = f"inference-time-{COL_NAME}"

ALPACA_PROMPT = """\
Below is an instruction that describes a task, paired with an input that \
provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
def build_inference_fn(model, tokenizer):
    """Return a callable (user_input, prompt_messages) → raw_str."""

    def _infer(user_input: str, prompt: list[dict]) -> str:
        # Extract system message from the prompt list (first system role entry)
        system_content = next(
            (m["content"] for m in prompt if m["role"] == "system"), ""
        )
        inputs = tokenizer(
            [ALPACA_PROMPT.format(system_content, user_input, "")],
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        return tokenizer.batch_decode(outputs)[0]

    return _infer


# ── Entry Point ───────────────────────────────────────────────────────────────
def run(input_csv: str, output_csv: str, limit: int | None = None):
    print(f"Loading {MODEL_ID} …")
    model, tokenizer = load_model()
    inference_fn = build_inference_fn(model, tokenizer)
    write_results(input_csv, output_csv, COL_NAME, TIME_COL, inference_fn, limit=limit)


if __name__ == "__main__":
    from config import INPUT_CSV
    run(INPUT_CSV, f"results/output_{COL_NAME}.csv")
