"""
Inference with deepseek-ai/DeepSeek-R1-Distill-Llama-8B
using vLLM for high-throughput batched inference.

Requirements: pip install -r requirements/vllm.txt
"""
from vllm import LLM, SamplingParams
from config import MODELS, MAX_NEW_TOKENS
from src.utils.csv_utils import write_results

# ── Model ID ──────────────────────────────────────────────────────────────────
MODEL_ID = MODELS["deepseek_vllm"]
COL_NAME = "DeepSeek-R1-Distill-Llama-8B-vllm"
TIME_COL = f"inference-time-{COL_NAME}"


# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    llm             = LLM(model=MODEL_ID)
    sampling_params = SamplingParams(temperature=0.5, max_tokens=MAX_NEW_TOKENS)
    return llm, sampling_params


# ── Inference ─────────────────────────────────────────────────────────────────
def build_inference_fn(llm, sampling_params):
    """Return a callable (user_input, prompt_messages) → raw_str."""

    def _infer(user_input: str, prompt: list[dict]) -> str:
        outputs = llm.chat(prompt, sampling_params)
        return outputs[0].outputs[0].text

    return _infer


# ── Entry Point ───────────────────────────────────────────────────────────────
def run(input_csv: str, output_csv: str, limit: int | None = None):
    print(f"Loading {MODEL_ID} via vLLM …")
    llm, sampling_params = load_model()
    inference_fn = build_inference_fn(llm, sampling_params)
    write_results(input_csv, output_csv, COL_NAME, TIME_COL, inference_fn, limit=limit)


if __name__ == "__main__":
    from config import INPUT_CSV
    run(INPUT_CSV, f"results/output_{COL_NAME}.csv")
