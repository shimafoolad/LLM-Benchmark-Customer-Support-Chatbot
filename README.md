# LLM Benchmark — Customer Support Chatbot

A structured pipeline for benchmarking multiple LLM backends on a **customer support chatbot** task.
Each model is tested against the same input dataset, outputs are scored by an LLM judge (Claude),
and results are visualized with charts and summary tables.

---

## 📐 Architecture

```
data/dataset.csv
      │
      ├──► src/models/deepseek_transformers.py  ──► results/output_DeepSeek-R1-Distill-Llama-8B.csv
      ├──► src/models/deepseek_unsloth.py        ──► results/output_DeepSeek-...-unsloth-bnb-4bit.csv
      ├──► src/models/deepseek_vllm.py           ──► results/output_DeepSeek-...-vllm.csv
      └──► src/models/phi4_unsloth.py            ──► results/output_phi-4-unsloth-bnb-4bit.csv
                                                              │
                                              src/utils/csv_utils.merge_results()
                                                              │
                                              results/benchmark.csv
                                                              │
                                        src/evaluation/evaluator.py  (Claude judge)
                                                              │
                                        results/evaluated_responses.csv
                                                              │
                                        src/evaluation/visualize.py
                                                              │
                                        results/plots/{mean_scores, distribution,
                                                       heatmap, inference_time}.png
```

---

## 🗂️ Project Structure

```
llm-benchmark/
├── config.py                        # Central config (paths, model IDs, settings)
├── prompts/
│   └── system_prompt.txt            # System prompt for the chatbot
├── data/
│   └── dataset.csv                  # Input CSV (USER INPUT, TRUE OUTPUT, prompt)
├── src/
│   ├── models/
│   │   ├── deepseek_transformers.py # HuggingFace Transformers backend
│   │   ├── deepseek_unsloth.py      # Unsloth 4-bit backend
│   │   ├── deepseek_vllm.py         # vLLM backend
│   │   └── phi4_unsloth.py          # Phi-4 via Unsloth
│   ├── evaluation/
│   │   ├── evaluator.py             # LLM-as-judge scorer (Claude API)
│   │   └── visualize.py             # Charts & summary tables
│   └── utils/
│       └── csv_utils.py             # Shared CSV read/write helpers
├── scripts/
│   ├── run_inference.py             # CLI: run one or all models
│   └── run_evaluation.py            # CLI: score + visualize
├── notebooks/
│   ├── 01_deepseek_transformers.ipynb
│   └── 05_evaluation_and_visualization.ipynb
├── requirements/
│   ├── transformers.txt
│   ├── unsloth.txt
│   └── vllm.txt
└── results/                         # Auto-created; gitignored except sample data
```

---

## 📊 Dataset Format (`data/dataset.csv`)

| Column | Description |
|---|---|
| `USER INPUT` | The user's question to the chatbot |
| `TRUE OUTPUT` | A reference good response (ground truth) |
| `prompt` | Full chat template as a Python list of `{role, content}` dicts |

Example row:
```
USER INPUT: "How do I reset my password?"
TRUE OUTPUT: "Go to the login page, click 'Forgot Password', and follow the email link."
prompt: [{"role": "system", "content": "..."}, {"role": "user", "content": "How do I reset my password?"}]
```

---

## 🚀 Quickstart

### 1. Install dependencies

Pick the backend(s) you need:
```bash
pip install -r requirements/transformers.txt   # HuggingFace pipeline
pip install -r requirements/unsloth.txt        # Unsloth 4-bit (GPU required)
pip install -r requirements/vllm.txt           # vLLM (GPU required)
```

### 2. Set your API key (for evaluation)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run inference

```bash
# Run a single model (add --limit N for a quick test)
python scripts/run_inference.py --model deepseek_transformers --limit 5

# Run all models and merge results
python scripts/run_inference.py --model all --merge
```

Available model keys: `deepseek_transformers`, `deepseek_unsloth_4bit`, `deepseek_vllm`, `phi4_unsloth_4bit`

### 4. Evaluate and visualize

```bash
# Score all model outputs and generate plots
python scripts/run_evaluation.py --visualize

# Only plot (if you already have evaluated_responses.csv)
python scripts/run_evaluation.py --only-visualize
```

---

## 📈 Output Visualizations

| Chart | Description |
|---|---|
| `mean_scores.png` | Bar chart of mean score ± std dev per model |
| `score_distribution.png` | Box plot of score spread per model |
| `score_heatmap.png` | Heatmap: score for each question × model |
| `inference_time.png` | Horizontal bar chart of average inference time |
| `summary_table.csv` | Mean, median, min, max, avg time per model |

---

## ⚙️ Configuration (`config.py`)

| Key | Default | Description |
|---|---|---|
| `INPUT_CSV` | `data/dataset.csv` | Input dataset path |
| `OUTPUT_DIR` | `results/` | Where CSVs are written |
| `MAX_NEW_TOKENS` | `256` | Token budget per response |
| `MAX_SEQ_LENGTH` | `2048` | Max context for Unsloth models |
| `LOAD_IN_4BIT` | `True` | 4-bit quantization for Unsloth |

---

## 🔧 Adding a New Model

1. Create `src/models/my_model.py` following the pattern of existing modules:
   - `load_model()` → returns the model/tokenizer/pipeline
   - `build_inference_fn(...)` → returns `fn(user_input, prompt) → str`
   - `run(input_csv, output_csv, limit)` → calls `write_results(...)`
2. Add an entry in `REGISTRY` in `scripts/run_inference.py`.
3. Add the column name to `EVAL_MODEL_COLS` in `config.py`.

---

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU (required for Unsloth and vLLM backends)
- `ANTHROPIC_API_KEY` environment variable (for evaluation scoring)

---

## 📄 License

MIT
