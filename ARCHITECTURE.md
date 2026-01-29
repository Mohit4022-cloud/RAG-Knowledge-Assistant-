# RAG Knowledge Assistant

## Project Description
A Production-Ready Retrieval-Augmented Generation (RAG) system designed to provide accurate, cited answers from a knowledge base of documents. The assistant combines semantic search with conversational AI to deliver contextually relevant information while maintaining strict source attribution.

## Stack Constraints

This project strictly adheres to the following technology stack:

1. **Language**: Python 3.10+
2. **Orchestration Framework**: LlamaIndex (v0.10+ syntax - latest API)
3. **Vector Database**: Qdrant
   - Local Docker deployment for development
   - Qdrant Cloud for production
4. **Observability**: Arize Phoenix
   - Trace ALL retrieval operations
   - Trace ALL generation operations
5. **User Interface**: Streamlit
6. **Evaluation Framework**: Ragas

**⚠️ These constraints are non-negotiable. Any deviation must be documented and justified.**

## Core Requirements

### 1. Ingestion System
- **Supported Formats**: PDF and Markdown files
- **Metadata Capture**: Must capture and store:
  - `filename`: Original document filename
  - `page_num`: Page number (for PDFs) or section identifier (for Markdown)
- **Output**: Chunked documents with embedded metadata for retrieval

### 2. Retrieval System
- **Search Strategy**: Hybrid search combining:
  - **Primary**: Keyword + Vector (BM25 + Dense Embeddings)
  - **Fallback**: Dense-only vector search if hybrid unavailable
- **Performance**: Must optimize for relevance and speed
- **Metadata Filtering**: Support filtering by document metadata

### 3. Chat Engine
- **Conversation History**: Must maintain and utilize conversation context
- **Multi-Turn Support**: Handle follow-up questions with context awareness
- **State Management**: Persist conversation state across interactions

### 4. Citation System
- **Format**: All answers must cite sources using `[Source ID]` format
- **Traceability**: Each claim must be traceable to source document
- **Metadata Display**: Show filename and page number for each citation
- **Strict Attribution**: No unsupported claims without citations

## Architecture Principles

1. **Observability First**: All operations must be traced via Arize Phoenix
2. **Modularity**: Clear separation between ingestion, retrieval, and generation
3. **Testability**: Components must be independently testable
4. **Scalability**: Design for growth from local to production deployment
5. **Maintainability**: Follow Python best practices and type hints

## Future Considerations

- Multi-modal support (images, tables)
- Advanced chunking strategies
- Query rewriting and expansion
- Evaluation metrics dashboard
- Production deployment configuration

---

**Last Updated**: 2026-01-28
**Status**: Initial Architecture Definition
