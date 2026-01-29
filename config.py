"""
Configuration Module

Centralizes all configuration settings for the RAG Knowledge Assistant.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "qdrant_storage"

# GLM Configuration (Zhipu AI)
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_API_BASE = os.getenv("GLM_API_BASE", "https://open.bigmodel.cn")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # For Qdrant Cloud

# Collection Settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Phoenix Configuration
PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))

# Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")  # GLM-4.7-Flash
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # Free HuggingFace embeddings

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))

# Application Settings
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# System Prompt Configuration
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """You are an expert AI Systems Engineer and RAG Tutor. Your goal is to explain complex Retrieval-Augmented Generation concepts clearly to users.

Rules:
- Always answer based strictly on the provided context.
- If the context contains academic papers, cite the specific paper title and section.
- Use analogies (e.g., comparing Vector DBs to 'Libraries' or 'Salesforce') when explaining technical terms.
- Be concise but technical."""
)
