# RAG Evaluation Pipeline Usage

This guide shows how to run the evaluation pipeline for your RAG Knowledge Assistant.

## Quick Start

### 1. Run Evaluation with Sample Dataset

The repository includes a sample golden dataset with 5 questions for quick testing:

```bash
cd /Users/mohit/rag-knowledge-assistant
source venv/bin/activate
python -m evals.run_eval
```

**Expected Output**:
```
======================================================================
RAG Evaluation Pipeline
======================================================================

Loading golden dataset from evals/golden_dataset.json...
✅ Loaded 5 evaluation entries

Initializing chat engine...
✅ Chat engine initialized

======================================================================
Running Evaluation
======================================================================
[1/5] What is Retrieval-Augmented Generation (RAG)?...
[2/5] What is the purpose of chunking in RAG systems?...
...

✅ Results saved to evals/results.csv

======================================================================
Evaluation Summary
======================================================================
Total Questions: 5

Metric Scores:
  Faithfulness:      0.875 (±0.123)
  Answer Relevancy:  0.912 (±0.089)
  Context Precision: 0.850 (±0.105)
  Context Recall:    0.890 (±0.095)
======================================================================
```

### 2. View Results

Results are saved to `evals/results.csv`:

```bash
# View results in terminal
column -t -s, evals/results.csv | head -20

# Or open in Excel/Numbers
open evals/results.csv
```

### 3. Generate Full Golden Dataset (Optional)

To create a comprehensive dataset from your PDFs:

```bash
# This will generate 20 QA pairs from your PDFs using GLM
python generate_eval_data.py

# Validate the generated dataset
python -m evals.validate_dataset
```

**Note**: Dataset generation makes API calls to GLM for each document chunk, so it may take several minutes.

## Understanding Metrics

The evaluation uses four ragas metrics:

### Faithfulness
- **What it measures**: Whether the answer is grounded in the retrieved context
- **Score range**: 0.0 to 1.0 (higher is better)
- **Good score**: > 0.8
- **What it detects**: Hallucinations where the model makes up facts not in the context

### Answer Relevancy
- **What it measures**: Whether the answer addresses the question
- **Score range**: 0.0 to 1.0 (higher is better)
- **Good score**: > 0.8
- **What it detects**: Irrelevant or off-topic answers

### Context Precision
- **What it measures**: Quality of retrieved context ranking
- **Score range**: 0.0 to 1.0 (higher is better)
- **Good score**: > 0.7
- **What it detects**: Whether the most relevant chunks are retrieved first

### Context Recall
- **What it measures**: Completeness of context retrieval
- **Score range**: 0.0 to 1.0 (higher is better)
- **Good score**: > 0.8
- **What it detects**: Whether all necessary context was retrieved

## LLM Configuration

The evaluation script attempts to use your GLM model for scoring. If GLM integration fails, it falls back to the default ragas LLM (which requires OpenAI API key).

To force using GLM:
1. Ensure `GLM_API_KEY` is set in your `.env` file
2. The script automatically uses your configured GLM model from `config.py`

## Troubleshooting

### Import Errors

If you see import errors, install missing dependencies:

```bash
pip install -r requirements.txt
```

### Chat Engine Initialization Fails

Make sure:
1. Qdrant is running: `docker compose up -d`
2. Vector index exists: Run `python -m app.ingest` first

### Low Metric Scores

If you see consistently low scores (< 0.5):
- **Low Faithfulness**: Model is hallucinating - consider adjusting prompts
- **Low Answer Relevancy**: Retrieval may be pulling wrong context - check chunking strategy
- **Low Context Precision**: Retrieved chunks aren't ranked well - consider hybrid search tuning
- **Low Context Recall**: Not retrieving enough relevant chunks - increase `top_k` parameter

### GLM Integration Fails

If GLM integration fails, the script will automatically fall back to default ragas LLM. To debug:

```python
# Check if GLM is working
python -c "from app.engine import GLMLLM; import config; llm = GLMLLM(model_name=config.LLM_MODEL, api_key=config.GLM_API_KEY); print('✅ GLM working')"
```

## Advanced Usage

### Custom Dataset Format

To create a custom golden dataset, follow this format:

```json
[
  {
    "id": "eval_001",
    "question": "Your question here?",
    "ground_truth_answer": "The expected answer",
    "context_chunk": "The relevant context from documents",
    "metadata": {
      "source_node_id": "optional_node_id"
    }
  }
]
```

Save as `evals/golden_dataset.json` and run `python -m evals.validate_dataset` to verify format.

### Batch Evaluation

To run evaluation in batches:

```python
from pathlib import Path
import pandas as pd
from evals.run_eval import run_evaluation, load_golden_dataset
from app.engine import get_chat_engine

# Load dataset
dataset = load_golden_dataset(Path("evals/golden_dataset.json"))

# Split into batches
batch_size = 5
batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

# Run each batch
chat_engine = get_chat_engine()
all_results = []

for i, batch in enumerate(batches, 1):
    print(f"Running batch {i}/{len(batches)}")
    results_df = run_evaluation(chat_engine, batch)
    all_results.append(results_df)

# Combine results
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv("evals/results_batched.csv", index=False)
```

## Next Steps

1. **Run evaluation regularly** to track RAG performance over time
2. **Compare results** when making changes to chunking, retrieval, or prompts
3. **Expand golden dataset** by generating more questions from your PDFs
4. **Set up CI/CD** to run evaluation automatically on code changes

## Files

- `evals/run_eval.py` - Main evaluation script
- `evals/golden_dataset.json` - Evaluation questions and expected answers
- `evals/results.csv` - Evaluation results (generated)
- `evals/validate_dataset.py` - Dataset format validator
- `generate_eval_data.py` - Golden dataset generator
