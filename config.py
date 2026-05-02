# ─────────────────────────────────────────────
#  LLM Benchmark Configuration
# ─────────────────────────────────────────────

# ── Paths ──────────────────────────────────────
INPUT_CSV        = "data/dataset.csv"
OUTPUT_DIR       = "results/"
EVALUATED_CSV    = "results/evaluated_responses.csv"
SYSTEM_PROMPT_FILE = "prompts/system_prompt.txt"

# ── Model Names ────────────────────────────────
MODELS = {
    "deepseek_transformers":   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_unsloth_4bit":   "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "deepseek_vllm":           "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "phi4_unsloth_4bit":       "unsloth/phi-4-unsloth-bnb-4bit",
}

# ── Generation Settings ────────────────────────
MAX_NEW_TOKENS  = 256
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
DTYPE           = None        # None = auto-detect

# ── Evaluation ─────────────────────────────────
EVAL_MODEL_COLS = [
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "DeepSeek-R1-Distill-Llama-8B-vllm",
    "phi-4-unsloth-bnb-4bit",
]
