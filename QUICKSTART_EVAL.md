# Quick Start: Golden Dataset Generation

This guide walks you through generating a golden evaluation dataset for your RAG system.

## Prerequisites

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Install dependencies (if not done yet):
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Option 1: Use Your Own PDFs

```bash
# Add your PDF documents to data/
cp /path/to/your/documents/*.pdf data/

# Generate golden dataset (creates 20 QA pairs)
python generate_eval_data.py

# Validate the generated dataset
python evals/validate_dataset.py
```

## Option 2: Use Sample PDF (for testing)

```bash
# Install reportlab (optional, for creating sample PDF)
pip install reportlab

# Create a sample PDF about RAG systems
python create_sample_pdf.py

# Generate golden dataset from the sample
python generate_eval_data.py

# Validate the dataset
python evals/validate_dataset.py
```

## What Gets Generated

The script `generate_eval_data.py` will:

1. ✅ Load all PDF documents from `data/`
2. ✅ Use GPT-4 to analyze document content
3. ✅ Generate 20 synthetic question-answer pairs
4. ✅ Extract relevant context chunks for each pair
5. ✅ Save results to `evals/golden_dataset.json`

## Dataset Format

```json
[
  {
    "id": "eval_001",
    "question": "What is RAG?",
    "ground_truth_answer": "Retrieval-Augmented Generation...",
    "context_chunk": "RAG is a technique that combines...",
    "metadata": {
      "source_node_id": "abc123"
    }
  }
]
```

## Validation

Run the validation script to check dataset quality:

```bash
python evals/validate_dataset.py
```

This checks for:
- ✅ Required fields present
- ✅ Minimum content lengths
- ✅ No duplicate IDs
- ✅ Proper formatting

## Customization

Edit `generate_eval_data.py` to customize:

- `NUM_QUESTIONS = 20` - Number of QA pairs to generate
- `LLM_MODEL = "gpt-4"` - Model used for generation
- `num_questions_per_chunk=2` - Questions per document chunk

## Next Steps

Once you have a golden dataset:

1. Implement RAG system (ingestion, retrieval, generation)
2. Run evaluation using Ragas metrics (see `evals/metrics.py`)
3. Iterate on system design based on evaluation results

## Troubleshooting

**"No PDF files found in data/"**
- Add PDF files to the `data/` directory
- Or run `python create_sample_pdf.py` to create a test PDF

**"OPENAI_API_KEY not found"**
- Set your API key in `.env` file
- Or export it: `export OPENAI_API_KEY=your_key_here`

**"Module not found: reportlab"**
- Optional for sample PDF creation only
- Install with: `pip install reportlab`
- Or use your own PDFs instead

## Cost Estimate

Generating 20 QA pairs with GPT-4:
- Approximately 5,000-10,000 tokens per run
- Estimated cost: $0.10-0.30 USD
- Depends on document length and complexity

For cheaper testing, change to GPT-3.5:
```bash
# In .env file
LLM_MODEL=gpt-3.5-turbo
```
