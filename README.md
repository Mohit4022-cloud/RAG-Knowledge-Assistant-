# RAG Knowledge Assistant

A Production-Ready Retrieval-Augmented Generation (RAG) system for accurate, cited answers from document knowledge bases.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete architecture specifications and stack constraints.

## Project Structure

```
rag-knowledge-assistant/
├── ARCHITECTURE.md          # Architecture specifications (source of truth)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── ingestion/             # Document ingestion pipeline
│   └── __init__.py
├── app/                   # Main application and UI
│   └── __init__.py
├── evals/                 # Evaluation framework
│   └── __init__.py
├── data/                  # Document storage (gitignored)
└── docs/                  # Additional documentation
```

## Setup

### 1. Prerequisites
- Python 3.10+
- Docker (for local Qdrant)

### 2. Installation

```bash
# Clone or navigate to project directory
cd rag-knowledge-assistant

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys and settings
```

### 4. Start Qdrant (Local Development)

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 5. Start Arize Phoenix (Observability)

```bash
python -m phoenix.server.main serve
```

## Usage

### Generating Golden Evaluation Dataset

Before evaluating your RAG system, generate a golden dataset from your PDF documents:

```bash
# 1. Add PDF documents to the data/ directory
cp your_documents/*.pdf data/

# 2. Ensure your .env file has OPENAI_API_KEY set
# 3. Run the golden dataset generator
python generate_eval_data.py

# 4. Validate the generated dataset
python evals/validate_dataset.py
```

This generates 20 question-answer-context triples in `evals/golden_dataset.json` for RAG evaluation.

See [evals/README.md](evals/README.md) for more details on the evaluation framework.

### Running the Application

(To be implemented)

## Technology Stack

- **Orchestration**: LlamaIndex v0.10+
- **Vector DB**: Qdrant
- **Observability**: Arize Phoenix
- **UI**: Streamlit
- **Evaluation**: Ragas
- **Language**: Python 3.10+

## Development Status

⚠️ **Initial Setup Complete** - Application logic to be implemented.

## License

(To be determined)
