# Evaluation Module

This module contains evaluation tools and datasets for assessing RAG system quality using the Ragas framework.

## Golden Dataset

The golden dataset (`golden_dataset.json`) contains manually verified or synthetically generated question-answer-context triples for evaluation.

### Dataset Format

```json
[
  {
    "id": "eval_001",
    "question": "What is the main purpose of...",
    "ground_truth_answer": "The main purpose is...",
    "context_chunk": "The relevant context from the document...",
    "metadata": {
      "source_node_id": "node_id_here"
    }
  }
]
```

### Fields

- **id**: Unique identifier for the evaluation entry
- **question**: The question to be answered by the RAG system
- **ground_truth_answer**: The expected correct answer
- **context_chunk**: The relevant document context that should be retrieved
- **metadata**: Additional metadata including source node ID

## Generating the Golden Dataset

To generate a synthetic evaluation dataset from your PDF documents:

```bash
# Ensure PDFs are in the data/ directory
cp your_document.pdf data/

# Activate virtual environment
source venv/bin/activate

# Set OpenAI API key in .env file
echo "OPENAI_API_KEY=your_key_here" >> .env

# Run the generator
python generate_eval_data.py
```

This will:
1. Load all PDFs from `data/`
2. Use GPT-4 to generate 20 question-answer-context triples
3. Save results to `evals/golden_dataset.json`

## Evaluation Metrics

The following Ragas metrics will be tracked:

- **Context Precision**: Relevance of retrieved context
- **Context Recall**: Coverage of ground truth in retrieved context
- **Faithfulness**: Factual consistency with source documents
- **Answer Relevancy**: Relevance of generated answer to question
- **Answer Correctness**: Semantic similarity to ground truth answer

## Usage

```python
from evals.metrics import evaluate_rag_system
from evals.runner import run_evaluation

# Run evaluation on golden dataset
results = run_evaluation("evals/golden_dataset.json")
```

## Files

- `golden_dataset.json` - Golden evaluation dataset
- `metrics.py` - Ragas metric implementations
- `runner.py` - Evaluation orchestration
- `README.md` - This file
