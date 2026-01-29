"""
Create a sample PDF document for testing the RAG pipeline.

This script generates a multi-page PDF with information about RAG systems
to verify the ingestion and query pipeline works correctly.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

import config


def create_sample_pdf():
    """Create a sample PDF about RAG systems."""

    # Ensure data directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = config.DATA_DIR / "sample_rag_overview.pdf"

    # Create PDF
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for content
    story = []

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = styles['Heading2']
    body_style = styles['Justify']

    # Title
    title = Paragraph("Retrieval-Augmented Generation (RAG) Systems", title_style)
    story.append(title)
    story.append(Spacer(1, 12))

    # Page 1: Introduction
    intro_heading = Paragraph("1. What is RAG?", heading_style)
    story.append(intro_heading)
    story.append(Spacer(1, 12))

    intro_text = """
    Retrieval-Augmented Generation (RAG) is an AI framework that combines information
    retrieval with text generation. RAG systems enhance large language models (LLMs) by
    providing them with relevant context from external knowledge sources. This approach
    addresses the limitations of LLMs, such as outdated information and hallucinations,
    by grounding responses in retrieved factual data.
    """
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 12))

    components_text = """
    A typical RAG system consists of three main components: a retrieval system that
    searches for relevant documents, an embedding model that converts text into vector
    representations, and a generation model that produces responses based on the
    retrieved context.
    """
    story.append(Paragraph(components_text, body_style))
    story.append(PageBreak())

    # Page 2: Vector Databases
    vector_heading = Paragraph("2. Vector Databases in RAG", heading_style)
    story.append(vector_heading)
    story.append(Spacer(1, 12))

    vector_text = """
    Vector databases like Qdrant, Pinecone, and Weaviate are essential components of RAG
    systems. They store document embeddings and enable efficient similarity search. When
    a user asks a question, the system converts the query into a vector and retrieves
    the most similar document chunks from the database.
    """
    story.append(Paragraph(vector_text, body_style))
    story.append(Spacer(1, 12))

    hybrid_text = """
    Modern vector databases support hybrid search, combining traditional keyword-based
    search (BM25) with semantic vector search. This dual approach ensures that both
    exact keyword matches and semantically similar content are retrieved, improving
    recall and precision.
    """
    story.append(Paragraph(hybrid_text, body_style))
    story.append(PageBreak())

    # Page 3: Implementation Details
    impl_heading = Paragraph("3. RAG Implementation", heading_style)
    story.append(impl_heading)
    story.append(Spacer(1, 12))

    impl_text = """
    Implementing a RAG system involves several steps. First, documents are loaded and
    split into manageable chunks. These chunks are then converted into embeddings using
    models like OpenAI's text-embedding-3-small. The embeddings are stored in a vector
    database with metadata such as source file names and page numbers.
    """
    story.append(Paragraph(impl_text, body_style))
    story.append(Spacer(1, 12))

    query_text = """
    During query time, the user's question is embedded and used to retrieve the most
    relevant chunks. These chunks, along with their metadata, are passed to an LLM
    like GPT-4 to generate a contextual response. The system can cite sources by
    referencing the metadata from retrieved chunks.
    """
    story.append(Paragraph(query_text, body_style))
    story.append(PageBreak())

    # Page 4: Observability
    obs_heading = Paragraph("4. Observability with Phoenix", heading_style)
    story.append(obs_heading)
    story.append(Spacer(1, 12))

    obs_text = """
    Arize Phoenix is an observability platform for LLM applications. It provides
    real-time tracing of RAG operations, including document ingestion, embedding
    generation, retrieval, and LLM calls. Phoenix captures latency, token usage,
    costs, and quality metrics.
    """
    story.append(Paragraph(obs_text, body_style))
    story.append(Spacer(1, 12))

    tracing_text = """
    Phoenix tracing helps debug RAG systems by visualizing the flow of data through
    the pipeline. Developers can inspect which documents were retrieved for each query,
    examine relevance scores, and identify performance bottlenecks. This visibility is
    crucial for optimizing retrieval quality and generation accuracy.
    """
    story.append(Paragraph(tracing_text, body_style))
    story.append(PageBreak())

    # Page 5: Best Practices
    best_heading = Paragraph("5. RAG Best Practices", heading_style)
    story.append(best_heading)
    story.append(Spacer(1, 12))

    best_text = """
    Effective RAG systems require careful tuning. Chunk size and overlap should be
    optimized based on the document type and query patterns. A chunk size of 512 tokens
    with 50 tokens overlap is a good starting point. Retrieval top-k values typically
    range from 3 to 10, balancing context relevance with token limits.
    """
    story.append(Paragraph(best_text, body_style))
    story.append(Spacer(1, 12))

    eval_text = """
    Evaluation is critical for RAG quality. Metrics like context relevance, answer
    faithfulness, and answer relevance help measure system performance. Tools like
    Ragas provide automated evaluation frameworks. Regular testing with diverse queries
    ensures the system handles edge cases and maintains accuracy.
    """
    story.append(Paragraph(eval_text, body_style))

    # Build PDF
    doc.build(story)

    print(f"✅ Sample PDF created: {output_file}")
    print(f"   Pages: 5")
    print(f"   Topics: RAG overview, vector databases, implementation, observability, best practices")
    print(f"\nYou can now run the ingestion pipeline:")
    print(f"   python -c 'from ingestion.pipeline import build_pipeline; build_pipeline()'")


if __name__ == "__main__":
    try:
        create_sample_pdf()
    except ImportError:
        print("❌ Error: reportlab not installed")
        print("   Install it with: pip install reportlab")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        raise
