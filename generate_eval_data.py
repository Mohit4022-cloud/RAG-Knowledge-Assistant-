"""
Golden Dataset Generator

Generates synthetic evaluation data from PDF documents using LlamaIndex.
Creates question-answer-context triples for RAG system evaluation.

Usage:
    python generate_eval_data.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import GLM model from engine
import sys
sys.path.insert(0, str(Path(__file__).parent))
from app.engine import GLMLLM
import config

# Configuration
DATA_DIR = Path("data")
OUTPUT_FILE = Path("evals/golden_dataset.json")
NUM_QUESTIONS = 20
LLM_MODEL = config.LLM_MODEL


def load_documents() -> List[Document]:
    """
    Load PDF documents from the data directory.

    Returns:
        List of Document objects with metadata
    """
    print(f"Loading documents from {DATA_DIR}...")

    # Check if data directory has PDFs
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}/. "
            f"Please add PDF documents before generating evaluation data."
        )

    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Load documents with metadata
    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".pdf"],
        filename_as_id=True
    )
    documents = reader.load_data()

    print(f"Loaded {len(documents)} document chunks")
    return documents


def generate_golden_dataset(documents: List[Document], num_questions: int = 20) -> List[Dict]:
    """
    Generate synthetic question-answer-context triples using LlamaIndex.

    Args:
        documents: List of documents to generate questions from
        num_questions: Number of QA pairs to generate

    Returns:
        List of dictionaries containing question, answer, and context
    """
    print(f"\nGenerating {num_questions} synthetic QA pairs using {LLM_MODEL}...")

    # Initialize GLM LLM
    llm = GLMLLM(
        model_name=config.LLM_MODEL,
        api_key=config.GLM_API_KEY
    )

    # Initialize dataset generator
    dataset_generator = RagDatasetGenerator.from_documents(
        documents,
        llm=llm,
        num_questions_per_chunk=2,  # Generate multiple questions per chunk
        show_progress=True
    )

    # Generate the dataset
    rag_dataset = dataset_generator.generate_questions_from_nodes(
        num=num_questions
    )

    # Convert to our format
    golden_dataset = []
    for i, example in enumerate(rag_dataset.examples[:num_questions], 1):
        entry = {
            "id": f"eval_{i:03d}",
            "question": example.query,
            "ground_truth_answer": example.reference_answer,
            "context_chunk": example.reference_contexts[0] if example.reference_contexts else "",
            "metadata": {
                "source_node_id": example.reference_node_id if hasattr(example, 'reference_node_id') else None
            }
        }
        golden_dataset.append(entry)
        print(f"  Generated {i}/{num_questions}: {example.query[:60]}...")

    return golden_dataset


def save_golden_dataset(dataset: List[Dict], output_path: Path):
    """
    Save the golden dataset to JSON file.

    Args:
        dataset: List of QA entries
        output_path: Path to save JSON file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON with pretty formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Golden dataset saved to {output_path}")
    print(f"   Total entries: {len(dataset)}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Golden Dataset Generator for RAG Evaluation")
    print("=" * 70)

    # Check for GLM API key
    if not config.GLM_API_KEY:
        raise ValueError(
            "GLM_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )

    try:
        # Step 1: Load documents
        documents = load_documents()

        # Step 2: Generate synthetic QA pairs
        golden_dataset = generate_golden_dataset(documents, num_questions=NUM_QUESTIONS)

        # Step 3: Save to JSON
        save_golden_dataset(golden_dataset, OUTPUT_FILE)

        # Summary
        print("\n" + "=" * 70)
        print("Summary:")
        print(f"  Documents processed: {len(documents)}")
        print(f"  QA pairs generated: {len(golden_dataset)}")
        print(f"  Output file: {OUTPUT_FILE}")
        print("=" * 70)

        # Show sample entry
        if golden_dataset:
            print("\nSample Entry:")
            print(f"  Question: {golden_dataset[0]['question']}")
            print(f"  Answer: {golden_dataset[0]['ground_truth_answer'][:100]}...")
            print(f"  Context: {golden_dataset[0]['context_chunk'][:100]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
