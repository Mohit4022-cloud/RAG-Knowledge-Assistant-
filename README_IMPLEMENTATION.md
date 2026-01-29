# RAG Pipeline Implementation - Quick Start Guide

This guide covers the newly implemented core RAG pipeline with Phoenix observability.

## What's Implemented

✅ **Core Files Created:**
1. `ingestion/pipeline.py` - Complete document ingestion pipeline
2. `app/engine.py` - Query engine and chat engine with hybrid search
3. `Makefile` - Phoenix UI launcher
4. `test_pipeline.py` - Automated test suite
5. `create_sample_pdf.py` - Sample document generator

✅ **Features:**
- PDF document loading with automatic metadata extraction
- Text chunking (512 tokens, 50 overlap)
- OpenAI embeddings (text-embedding-3-small)
- Qdrant vector store with **hybrid search** (BM25 + dense vectors)
- Query engine for one-off questions
- Chat engine for conversation with history
- **Phoenix tracing** for all operations
- Source attribution with file names and page numbers

---

## Quick Start

### 1. Setup Environment

```bash
cd /Users/mohit/rag-knowledge-assistant

# Activate virtual environment
source venv/bin/activate

# Verify dependencies installed
pip install -r requirements.txt

# Ensure .env has OPENAI_API_KEY
cat .env
```

### 2. Create Sample Data

```bash
# Generate a 5-page sample PDF about RAG
python create_sample_pdf.py
```

**Output:**
- Creates `data/sample_rag_overview.pdf` with content about RAG systems, vector databases, implementation, observability, and best practices

### 3. Start Phoenix UI (Optional but Recommended)

Open a **separate terminal**:

```bash
cd /Users/mohit/rag-knowledge-assistant
source venv/bin/activate

# Launch Phoenix
make run_phoenix

# Alternative: python3 -m phoenix.server.main serve
```

**Access UI:** http://localhost:6006

Phoenix will capture traces from both ingestion and query operations.

### 4. Run Ingestion Pipeline

```bash
# Run the test suite (includes ingestion)
python test_pipeline.py
```

**Or run ingestion directly in Python:**

```python
from ingestion.pipeline import build_pipeline

# Build the index
index = build_pipeline()

# Check results
print(f"Nodes created: {len(index.docstore.docs)}")
```

**What happens:**
- Loads all PDFs from `data/` directory
- Chunks documents into 512-token segments with 50-token overlap
- Generates embeddings using OpenAI
- Stores in Qdrant with hybrid search enabled (BM25 + dense)
- Persists to `qdrant_storage/` directory
- Sends traces to Phoenix (if running)

**Expected output:**
```
============================================================
Starting RAG Ingestion Pipeline
============================================================
Loading PDF documents from: /Users/mohit/rag-knowledge-assistant/data
✅ Loaded 1 documents
Configuring chunker (size=512, overlap=50)
Configuring embedding model: text-embedding-3-small
Creating Qdrant vector store: rag_knowledge_base
Storage path: /Users/mohit/rag-knowledge-assistant/qdrant_storage
Hybrid search: ENABLED (BM25 + dense)
Building vector index...
[Progress bar shows embedding and indexing progress]
Persisting storage to disk...
============================================================
✅ Ingestion Pipeline Complete
   Documents processed: 1
   Nodes created: 15
   Collection: rag_knowledge_base
   Storage: /Users/mohit/rag-knowledge-assistant/qdrant_storage
   Phoenix traces: http://localhost:6006
============================================================
```

### 5. Query the System

**Option A: Test Script**
```bash
python test_pipeline.py
```

**Option B: Python REPL**
```python
from app.engine import get_query_engine

# Load query engine
query_engine = get_query_engine()

# Ask a question
response = query_engine.query("What is RAG?")

# View response
print(f"Response: {response.response}")

# View sources with metadata
print("\nSources:")
for i, node in enumerate(response.source_nodes, 1):
    file_name = node.metadata.get('file_name', 'unknown')
    page = node.metadata.get('page_label', 'unknown')
    score = node.score if hasattr(node, 'score') else 'N/A'
    print(f"  [{i}] {file_name}, page {page} (score: {score})")
```

**Option C: Chat Engine (with conversation history)**
```python
from app.engine import get_chat_engine

# Load chat engine
chat_engine = get_chat_engine()

# First question
response1 = chat_engine.chat("What is RAG?")
print(response1.response)

# Follow-up question (uses conversation history)
response2 = chat_engine.chat("What are the main components?")
print(response2.response)

# Another follow-up
response3 = chat_engine.chat("How does hybrid search work?")
print(response3.response)
```

### 6. View Phoenix Traces

1. Go to http://localhost:6006
2. You should see traces for:
   - **Ingestion**: Document loading, chunking, embedding generation, indexing
   - **Query**: Query embedding, hybrid retrieval (BM25 + dense), LLM response generation
   - **Chat**: Same as query, plus conversation history management

3. Click on any trace to see:
   - Latency for each operation
   - Token usage and costs
   - Retrieved documents and scores
   - LLM input prompts and outputs
   - Metadata propagation

---

## File Structure

```
rag-knowledge-assistant/
├── ingestion/
│   └── pipeline.py          # NEW: Complete ingestion pipeline
├── app/
│   └── engine.py            # NEW: Query and chat engines
├── config.py                # Configuration (already exists)
├── requirements.txt         # Dependencies (already exists)
├── Makefile                 # NEW: Phoenix launcher
├── test_pipeline.py         # NEW: Test suite
├── create_sample_pdf.py     # NEW: Sample data generator
├── data/                    # PDF documents (add your PDFs here)
│   └── sample_rag_overview.pdf  # Generated by create_sample_pdf.py
└── qdrant_storage/          # Created by ingestion (vector store data)
    └── collection/
```

---

## Configuration

All settings are in `config.py`:

```python
# Data and storage
DATA_DIR = "./data"                          # Where PDFs are stored
QDRANT_STORAGE_DIR = "./qdrant_storage"      # Where vector index is stored
COLLECTION_NAME = "rag_knowledge_base"       # Qdrant collection name

# Chunking
CHUNK_SIZE = 512                             # Tokens per chunk
CHUNK_OVERLAP = 50                           # Overlap between chunks

# Models
EMBEDDING_MODEL = "text-embedding-3-small"   # OpenAI embedding model
LLM_MODEL = "gpt-4o-mini"                    # OpenAI LLM model

# Retrieval
TOP_K = 5                                    # Number of chunks to retrieve

# Phoenix
PHOENIX_PORT = 6006                          # Phoenix UI port
```

---

## Hybrid Search Explained

The implementation uses **hybrid search** which combines:

1. **BM25 (Sparse)**: Traditional keyword-based search
   - Good for exact term matches
   - Works well with technical terms, names, acronyms

2. **Dense Vectors (Semantic)**: Neural embedding similarity
   - Good for semantic similarity
   - Works well with paraphrased queries, conceptual questions

**Configuration:**
```python
vector_store = QdrantVectorStore(
    enable_hybrid=True,              # Enables both BM25 and dense
    fastembed_sparse_model="Qdrant/bm25"
)

query_engine = index.as_query_engine(
    similarity_top_k=5,              # Final number of chunks
    sparse_top_k=5,                  # BM25 candidates
)
```

**Testing hybrid search:**
```python
# Keyword-heavy query (BM25 helps)
response1 = query_engine.query("Qdrant vector database BM25")

# Semantic query (dense helps)
response2 = query_engine.query("systems that combine retrieval with generation")

# Compare source nodes and scores
```

---

## Metadata and Citations

Every retrieved chunk includes metadata:
- `file_name`: Source PDF filename
- `page_label`: Page number in PDF

**Access metadata:**
```python
response = query_engine.query("What is RAG?")

for node in response.source_nodes:
    print(f"File: {node.metadata['file_name']}")
    print(f"Page: {node.metadata['page_label']}")
    print(f"Score: {node.score}")
    print(f"Text: {node.text[:100]}...")
```

This metadata enables citation formatting (future implementation in `app/citations.py`).

---

## Troubleshooting

### No documents found
```
ValueError: No PDF documents found in /Users/mohit/rag-knowledge-assistant/data
```
**Solution:** Add PDF files to `data/` or run `python create_sample_pdf.py`

### OpenAI API key missing
```
openai.error.AuthenticationError: No API key provided
```
**Solution:** Add `OPENAI_API_KEY=sk-...` to `.env` file

### Qdrant storage not found
```
ValueError: Qdrant storage not found at .../qdrant_storage. Run ingestion pipeline first.
```
**Solution:** Run ingestion first: `python -c "from ingestion.pipeline import build_pipeline; build_pipeline()"`

### Phoenix not showing traces
- Ensure Phoenix is running: `make run_phoenix`
- Check endpoint: http://localhost:6006
- Phoenix instrumentation must be at top of pipeline.py and engine.py (already implemented)

### Module import errors
**Solution:** Activate virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Next Steps

After verifying the core pipeline works:

1. **Add your own PDFs**:
   - Drop PDF files into `data/` directory
   - Re-run ingestion: `python -c "from ingestion.pipeline import build_pipeline; build_pipeline()"`

2. **Experiment with queries**:
   - Try different question types
   - Compare query_engine vs chat_engine
   - Check Phoenix traces for performance

3. **Future enhancements** (not yet implemented):
   - `app/citations.py` - Format citations as `[1], [2]`
   - `app/ui.py` - Streamlit UI
   - `evals/metrics.py` + `evals/runner.py` - Ragas evaluation
   - `app/retriever.py` - Custom retriever with metadata filtering

---

## Testing Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file has `OPENAI_API_KEY`
- [ ] Sample PDF created (`python create_sample_pdf.py`)
- [ ] Phoenix UI running (`make run_phoenix`, open http://localhost:6006)
- [ ] Ingestion successful (`python test_pipeline.py` or manual)
- [ ] `qdrant_storage/` directory exists
- [ ] Query engine returns responses
- [ ] Source nodes include metadata (file_name, page_label)
- [ ] Phoenix traces visible for ingestion and queries

---

## Key Implementation Details

### Phoenix Tracing
- **CRITICAL**: Instrumentation must be at the very top of both files
- Must be called BEFORE any `llama_index.*` imports
- Captures all LLM calls, embeddings, retrievals, transformations

### Storage
- Using **file-based Qdrant**: `path=str(config.QDRANT_STORAGE_DIR)`
- No Docker required for development
- Data persisted to `./qdrant_storage/` directory
- For production: Switch to `url=config.QDRANT_URL` with Qdrant Cloud

### Hybrid Search
- **Cannot be added after index creation** - must be enabled from start
- Uses both BM25 (keyword) and dense (semantic) retrieval
- Fallback: Set `enable_hybrid=False` for dense-only if issues occur

---

## Questions?

Check the plan file for detailed implementation rationale and architecture decisions.

Run the test suite for automated verification:
```bash
python test_pipeline.py
```

View Phoenix traces at http://localhost:6006 for detailed observability.
