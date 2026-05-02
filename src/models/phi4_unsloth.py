"""
Inference with unsloth/phi-4-unsloth-bnb-4bit
using the Unsloth library for fast 4-bit quantized inference.

Requirements: pip install -r requirements/unsloth.txt
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import MODELS, MAX_NEW_TOKENS, MAX_SEQ_LENGTH, LOAD_IN_4BIT, DTYPE
from src.utils.csv_utils import write_results

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID = MODELS["phi4_unsloth_4bit"]
COL_NAME = "phi-4-unsloth-bnb-4bit"
TIME_COL = f"inference-time-{COL_NAME}"


# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
def build_inference_fn(model, tokenizer):
    """Return a callable (user_input, prompt_messages) → raw_str."""

    def _infer(user_input: str, prompt: list[dict]) -> str:
        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
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
