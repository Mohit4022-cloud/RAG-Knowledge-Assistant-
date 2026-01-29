"""
RAG Evaluation Runner

Evaluates the chat engine against the golden dataset using ragas metrics.

Usage:
    python -m evals.run_eval

Metrics:
    - Faithfulness: Checks if answer is grounded in retrieved context
    - Answer Relevancy: Checks if answer addresses the question
    - Context Precision: Quality of retrieved context ranking
    - Context Recall: Completeness of context retrieval
"""

import json
import sys
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.engine import get_chat_engine
from datasets import Dataset
from ragas import evaluate

# Import ragas metrics (using new API to avoid deprecation warnings)
try:
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
except ImportError:
    # Fallback to old API for older ragas versions
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )

# Import ragas LlamaIndex integration for custom LLM support
try:
    from ragas.llms import LlamaIndexLLMWrapper
    from app.engine import GLMLLM
    import config
    USE_GLM = True
except ImportError:
    USE_GLM = False
    print("⚠️  LlamaIndex ragas integration not available, using default LLM")


def load_golden_dataset(dataset_path: Path) -> List[Dict]:
    """Load golden dataset from JSON file."""
    print(f"Loading golden dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} evaluation entries")
    return dataset


def run_evaluation(chat_engine, golden_dataset: List[Dict]) -> pd.DataFrame:
    """
    Run evaluation on golden dataset.

    Args:
        chat_engine: The RAG chat engine to evaluate
        golden_dataset: List of evaluation entries

    Returns:
        DataFrame with results including ragas scores
    """
    print("\n" + "=" * 70)
    print("Querying Chat Engine")
    print("=" * 70)

    results = []

    for i, entry in enumerate(golden_dataset, 1):
        question = entry['question']
        print(f"[{i}/{len(golden_dataset)}] {question[:60]}...")

        try:
            # Query the chat engine
            response = chat_engine.query(question)

            # Extract generated answer
            generated_answer = str(response.response)

            # Get retrieved context from source_nodes
            contexts = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                contexts = [node.get_content() for node in response.source_nodes]

            # If no contexts retrieved, use empty list (ragas can handle this)
            if not contexts:
                contexts = ["No context retrieved"]
                print(f"    ⚠️  No context retrieved for this question")

            # Store result
            results.append({
                'question': question,
                'ground_truth': entry['ground_truth_answer'],
                'generated_answer': generated_answer,
                'contexts': contexts,
                'expected_context': entry['context_chunk']
            })

        except Exception as e:
            print(f"    ❌ Error querying: {e}")
            # Store error result
            results.append({
                'question': question,
                'ground_truth': entry['ground_truth_answer'],
                'generated_answer': f"Error: {str(e)}",
                'contexts': ["Error retrieving context"],
                'expected_context': entry['context_chunk']
            })

    print(f"\n✅ Completed {len(results)} queries")

    # Convert to ragas dataset format
    print("\nPreparing dataset for ragas evaluation...")
    dataset_dict = {
        'question': [r['question'] for r in results],
        'answer': [r['generated_answer'] for r in results],
        'contexts': [r['contexts'] for r in results],
        'ground_truth': [r['ground_truth'] for r in results]
    }

    # Create HuggingFace dataset
    eval_dataset = Dataset.from_dict(dataset_dict)
    print(f"✅ Created evaluation dataset with {len(eval_dataset)} entries")

    # Configure ragas to use GLM if available
    print("\nConfiguring ragas metrics...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    if USE_GLM:
        try:
            print("Attempting to use GLM model for evaluation...")
            # Initialize GLM LLM
            glm_llm = GLMLLM(
                model_name=config.LLM_MODEL,
                api_key=config.GLM_API_KEY
            )
            # Wrap for ragas (if wrapper is available)
            wrapped_llm = LlamaIndexLLMWrapper(glm_llm)

            # Run ragas evaluation with custom LLM
            print("\n" + "=" * 70)
            print("Running Ragas Evaluation with GLM Model")
            print("=" * 70)
            ragas_results = evaluate(
                eval_dataset,
                metrics=metrics,
                llm=wrapped_llm
            )
        except Exception as e:
            print(f"⚠️  GLM integration failed: {e}")
            print("Falling back to default ragas LLM...")
            ragas_results = evaluate(
                eval_dataset,
                metrics=metrics
            )
    else:
        # Use default ragas LLM (OpenAI)
        print("\n" + "=" * 70)
        print("Running Ragas Evaluation with Default LLM")
        print("=" * 70)
        ragas_results = evaluate(
            eval_dataset,
            metrics=metrics
        )

    # Convert to DataFrame
    results_df = ragas_results.to_pandas()

    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print evaluation summary statistics."""
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Total Questions: {len(results_df)}")
    print(f"\nMetric Scores:")

    # Calculate metrics that exist
    metric_cols = [col for col in results_df.columns if col in [
        'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'
    ]]

    for metric in metric_cols:
        mean_score = results_df[metric].mean()
        std_score = results_df[metric].std()
        print(f"  {metric.replace('_', ' ').title():20s}: {mean_score:.3f} (±{std_score:.3f})")

    print("=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("RAG Evaluation Pipeline")
    print("=" * 70)

    # Paths
    dataset_path = project_root / "evals" / "golden_dataset.json"
    output_path = project_root / "evals" / "results.csv"

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Error: {dataset_path} not found!")
        print("\nTo generate the golden dataset, run:")
        print("  python generate_eval_data.py")
        print("\nMake sure you have:")
        print("  1. PDF files in the data/ directory")
        print("  2. OPENAI_API_KEY set in .env file")
        sys.exit(1)

    # Load golden dataset
    golden_dataset = load_golden_dataset(dataset_path)

    # Initialize chat engine
    print("\nInitializing chat engine...")
    try:
        chat_engine = get_chat_engine()
        print("✅ Chat engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize chat engine: {e}")
        sys.exit(1)

    # Run evaluation
    try:
        results_df = run_evaluation(chat_engine, golden_dataset)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to {output_path}")

        # Print summary
        print_summary(results_df)

        # Show sample results
        print("\nSample Results (first 3 entries):")
        print("-" * 70)
        for i in range(min(3, len(results_df))):
            print(f"\nQuestion {i+1}: {results_df.iloc[i]['question'][:60]}...")
            print(f"Answer: {results_df.iloc[i]['answer'][:80]}...")
            if 'faithfulness' in results_df.columns:
                print(f"Faithfulness: {results_df.iloc[i]['faithfulness']:.3f}")
            if 'answer_relevancy' in results_df.columns:
                print(f"Answer Relevancy: {results_df.iloc[i]['answer_relevancy']:.3f}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
