"""
Document ingestion pipeline for RAG system.

This module handles:
- PDF document loading from data directory
- Text chunking with configurable overlap
- Embedding generation using OpenAI
- Vector indexing in Qdrant with hybrid search (BM25 + dense)
- Phoenix tracing for all operations
"""

# CRITICAL: Phoenix instrumentation must be first, before any LlamaIndex imports
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces")
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

import logging
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_pipeline() -> VectorStoreIndex:
    """
    Build the complete ingestion pipeline.

    Steps:
    1. Load PDF documents from data directory
    2. Chunk documents with overlap
    3. Generate embeddings
    4. Index in Qdrant with hybrid search enabled
    5. Persist storage to disk

    Returns:
        VectorStoreIndex: The created vector store index

    Raises:
        ValueError: If no documents found in data directory
        Exception: If ingestion pipeline fails
    """
    logger.info("=" * 60)
    logger.info("Starting RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Load documents
    logger.info(f"Loading PDF documents from: {config.DATA_DIR}")

    if not config.DATA_DIR.exists():
        raise ValueError(f"Data directory not found: {config.DATA_DIR}")

    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR),
        required_exts=[".pdf"],
        recursive=True
    ).load_data()

    if not documents:
        raise ValueError(f"No PDF documents found in {config.DATA_DIR}")

    logger.info(f"✅ Loaded {len(documents)} documents")

    # Step 2: Configure text chunking
    logger.info(f"Configuring chunker (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    text_splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    # Step 3: Configure embedding model
    logger.info(f"Configuring embedding model: {config.EMBEDDING_MODEL}")

    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL
    )

    # Step 4: Create Qdrant vector store with hybrid search
    logger.info(f"Creating Qdrant vector store: {config.COLLECTION_NAME}")
    logger.info(f"Storage path: {config.QDRANT_STORAGE_DIR}")
    logger.info("Hybrid search: ENABLED (BM25 + dense)")

    # Ensure storage directory exists
    config.QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Create Qdrant client for file-based storage
    client = QdrantClient(path=str(config.QDRANT_STORAGE_DIR))

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        enable_hybrid=True,                    # Enable BM25 + dense
        fastembed_sparse_model="Qdrant/bm25"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Step 5: Create index with transformations
    logger.info("Building vector index...")
    logger.info("(This may take a few minutes for large document sets)")

    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[text_splitter],
            embed_model=embed_model,
            show_progress=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to create index: {e}")
        raise

    # Step 6: Persist storage
    logger.info("Persisting storage to disk...")
    storage_context.persist(persist_dir=str(config.QDRANT_STORAGE_DIR))

    # Summary
    num_nodes = len(index.docstore.docs)
    logger.info("=" * 60)
    logger.info("✅ Ingestion Pipeline Complete")
    logger.info(f"   Documents processed: {len(documents)}")
    logger.info(f"   Nodes created: {num_nodes}")
    logger.info(f"   Collection: {config.COLLECTION_NAME}")
    logger.info(f"   Storage: {config.QDRANT_STORAGE_DIR}")
    logger.info(f"   Phoenix traces: http://localhost:{config.PHOENIX_PORT}")
    logger.info("=" * 60)

    return index


if __name__ == "__main__":
    try:
        index = build_pipeline()
        logger.info("Pipeline execution successful!")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
