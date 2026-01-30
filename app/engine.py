"""
Query engine for RAG system.

This module handles:
- Loading persisted vector index from Qdrant
- Creating query engine with hybrid search
- Creating chat engine with conversation history (optional)
- Phoenix tracing for all query operations
"""

# CRITICAL: Phoenix instrumentation must be first, before any LlamaIndex imports
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register(endpoint="http://127.0.0.1:6006/v1/traces")
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import requests
from typing import Any

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GLMLLM(CustomLLM):
    """Custom LLM for Zhipu AI's GLM models - NO OpenAI dependency."""

    model_name: str = "glm-4.7"
    api_key: str = ""
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    context_window: int = 128000
    num_output: int = 4096

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt using GLM API."""
        response = self._call_glm_api([{"role": "user", "content": prompt}])
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion using GLM API with fallback."""
        accumulated_text = ""

        try:
            for delta in self._call_glm_api_stream([{"role": "user", "content": prompt}]):
                accumulated_text += delta
                yield CompletionResponse(text=accumulated_text, delta=delta)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            # Fallback to non-streaming
            logger.info("Falling back to non-streaming completion")
            response_text = self._call_glm_api([{"role": "user", "content": prompt}])
            yield CompletionResponse(text=response_text, delta=response_text)

    def chat(self, messages, **kwargs):
        """Chat completion using GLM API."""
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

        # Convert LlamaIndex messages to GLM format
        glm_messages = []
        for msg in messages:
            glm_messages.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            })

        response_text = self._call_glm_api(glm_messages)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
            raw={"content": response_text}
        )

    def stream_chat(self, messages, **kwargs):
        """Stream chat completion using GLM API with fallback."""
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

        # Convert LlamaIndex messages to GLM format
        glm_messages = []
        for msg in messages:
            glm_messages.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            })

        accumulated_text = ""

        try:
            for delta in self._call_glm_api_stream(glm_messages):
                accumulated_text += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated_text),
                    delta=delta
                )
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            # Fallback to non-streaming
            logger.info("Falling back to non-streaming chat")
            response_text = self._call_glm_api(glm_messages)
            yield ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
                delta=response_text
            )

    def _call_glm_api(self, messages: list) -> str:
        """Make direct HTTP call to GLM API."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _parse_sse_stream(self, response_stream):
        """
        Parse Server-Sent Events (SSE) stream from GLM API.

        Yields:
            str: Individual content deltas from the stream
        """
        import json

        for line in response_stream.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # Check for stream end marker
            if line.strip() == 'data: [DONE]':
                break

            # Parse SSE format: "data: {json}"
            if line.startswith('data: '):
                try:
                    json_str = line[6:]  # Remove "data: " prefix
                    chunk = json.loads(json_str)

                    # Extract delta content from GLM API response
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content

                except json.JSONDecodeError:
                    # Skip malformed JSON chunks
                    continue

    def _call_glm_api_stream(self, messages: list):
        """
        Make streaming HTTP call to GLM API.

        Yields:
            str: Content deltas from the streaming response
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True  # Enable streaming
        }

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        # Parse SSE stream
        for content_delta in self._parse_sse_stream(response):
            yield content_delta


def get_query_engine() -> BaseQueryEngine:
    """
    Load the persisted vector index and return a configured query engine.

    The query engine uses:
    - Hybrid search (BM25 + dense vectors)
    - Top-k retrieval for both sparse and dense methods
    - Compact response mode for efficient generation

    Returns:
        BaseQueryEngine: Configured query engine for RAG queries

    Raises:
        ValueError: If Qdrant storage directory not found
        Exception: If index loading fails
    """
    logger.info("Loading query engine...")

    # Verify storage exists
    if not config.QDRANT_STORAGE_DIR.exists():
        raise ValueError(
            f"Qdrant storage not found at {config.QDRANT_STORAGE_DIR}. "
            "Run ingestion pipeline first."
        )

    # Recreate vector store with same config as ingestion
    logger.info(f"Connecting to Qdrant collection: {config.COLLECTION_NAME}")

    # Create Qdrant client for file-based storage
    client = QdrantClient(path=str(config.QDRANT_STORAGE_DIR))

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )

    # Configure embedding model (must match ingestion)
    logger.info(f"Configuring embedding model: {config.EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)

    # Load index from vector store
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        logger.info("✅ Index loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load index: {e}")
        raise

    # Configure LLM (GLM - NO OpenAI dependency)
    logger.info(f"Configuring LLM: {config.LLM_MODEL} (Zhipu AI GLM)")

    llm = GLMLLM(
        model_name=config.LLM_MODEL,
        api_key=config.GLM_API_KEY,
        api_base=config.GLM_API_BASE
    )

    # Create query engine with hybrid search
    logger.info("Creating query engine with hybrid search")
    logger.info(f"  Dense top-k: {config.TOP_K}")
    logger.info(f"  Sparse top-k: {config.TOP_K}")

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.TOP_K,      # Dense vector retrieval
        sparse_top_k=config.TOP_K,          # BM25 retrieval
        response_mode="compact"
    )

    logger.info("✅ Query engine ready")
    logger.info(f"   Phoenix traces: http://localhost:{config.PHOENIX_PORT}")

    return query_engine


def get_chat_engine():
    """
    Load the persisted vector index and return a configured chat engine.

    The chat engine maintains conversation history and provides context-aware
    responses using the same hybrid search as the query engine.

    Returns:
        Chat engine configured for conversational RAG

    Raises:
        ValueError: If Qdrant storage directory not found
        Exception: If index loading fails
    """
    logger.info("Loading chat engine...")

    # Verify storage exists
    if not config.QDRANT_STORAGE_DIR.exists():
        raise ValueError(
            f"Qdrant storage not found at {config.QDRANT_STORAGE_DIR}. "
            "Run ingestion pipeline first."
        )

    # Recreate vector store with same config as ingestion
    logger.info(f"Connecting to Qdrant collection: {config.COLLECTION_NAME}")

    # Create Qdrant client for file-based storage
    client = QdrantClient(path=str(config.QDRANT_STORAGE_DIR))

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )

    # Configure embedding model (must match ingestion)
    logger.info(f"Configuring embedding model: {config.EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)

    # Load index from vector store
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        logger.info("✅ Index loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load index: {e}")
        raise

    # Configure LLM (GLM - NO OpenAI dependency)
    logger.info(f"Configuring LLM: {config.LLM_MODEL} (Zhipu AI GLM)")

    llm = GLMLLM(
        model_name=config.LLM_MODEL,
        api_key=config.GLM_API_KEY,
        api_base=config.GLM_API_BASE
    )

    # Create chat engine with context mode
    logger.info("Creating chat engine with conversation history")
    logger.info(f"  Dense top-k: {config.TOP_K}")
    logger.info(f"  Sparse top-k: {config.TOP_K}")
    logger.info(f"System prompt configured: {config.SYSTEM_PROMPT[:100]}...")

    chat_engine = index.as_chat_engine(
        llm=llm,
        similarity_top_k=config.TOP_K,
        sparse_top_k=config.TOP_K,
        chat_mode="context",  # Context-aware chat with history
        system_prompt=config.SYSTEM_PROMPT
    )

    logger.info("✅ Chat engine ready")
    logger.info(f"   Phoenix traces: http://localhost:{config.PHOENIX_PORT}")

    return chat_engine


if __name__ == "__main__":
    # Test query engine
    try:
        print("\n" + "=" * 60)
        print("Testing Query Engine")
        print("=" * 60)

        query_engine = get_query_engine()

        test_query = "What is RAG?"
        print(f"\nQuery: {test_query}")

        response = query_engine.query(test_query)

        print(f"\nResponse: {response}")
        print("\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            file_name = node.metadata.get('file_name', 'unknown')
            page = node.metadata.get('page_label', 'unknown')
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"  [{i}] {file_name}, page {page} (score: {score})")

        print("\n" + "=" * 60)
        print("✅ Query engine test complete")
        print(f"View traces at: http://localhost:{config.PHOENIX_PORT}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Query engine test failed: {e}")
        raise
