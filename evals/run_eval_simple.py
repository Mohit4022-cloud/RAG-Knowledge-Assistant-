"""
Simple RAG Evaluation Runner (No OpenAI Required)

Uses GLM model to evaluate RAG responses on a simple 1-5 scale.

Usage:
    python -m evals.run_eval_simple
"""

import json
import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.engine import get_chat_engine, GLMLLM
import config


def load_golden_dataset(dataset_path: Path) -> List[Dict]:
    """Load golden dataset from JSON file."""
    print(f"Loading golden dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"✅ Loaded {len(dataset)} evaluation entries")
    return dataset


def evaluate_answer(question: str, ground_truth: str, generated_answer: str,
                    context: str, llm: GLMLLM) -> Dict:
    """
    Evaluate a single answer using GLM.

    Returns scores for:
    - Accuracy (1-5): How well does the answer match the ground truth?
    - Relevance (1-5): Does the answer address the question?
    - Faithfulness (1-5): Is the answer grounded in the provided context?
    """

    prompt = f"""You are an expert evaluator for RAG systems. Evaluate the following answer on three criteria.

Question: {question}

Expected Answer: {ground_truth}

Generated Answer: {generated_answer}

Retrieved Context: {context}

Evaluate the generated answer on these three criteria (rate each 1-5, where 5 is best):

1. ACCURACY: How well does the generated answer match the expected answer?
   - Consider semantic similarity, not just exact wording
   - Score: 1 (completely wrong) to 5 (perfectly accurate)

2. RELEVANCE: Does the generated answer directly address the question?
   - Is it on-topic and helpful?
   - Score: 1 (off-topic) to 5 (highly relevant)

3. FAITHFULNESS: Is the generated answer grounded in the retrieved context?
   - Does it only use facts from the context?
   - Score: 1 (hallucinated/unsupported) to 5 (fully grounded)

Respond in EXACTLY this format:
ACCURACY: <score>
RELEVANCE: <score>
FAITHFULNESS: <score>
REASONING: <brief explanation>"""

    try:
        response = llm.complete(prompt)
        response_text = response.text

        # Parse scores
        scores = {'accuracy': 0, 'relevance': 0, 'faithfulness': 0, 'reasoning': ''}

        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('ACCURACY:'):
                try:
                    scores['accuracy'] = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('RELEVANCE:'):
                try:
                    scores['relevance'] = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('FAITHFULNESS:'):
                try:
                    scores['faithfulness'] = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif line.startswith('REASONING:'):
                scores['reasoning'] = line.split(':', 1)[1].strip()

        return scores

    except Exception as e:
        print(f"    ⚠️  Evaluation error: {e}")
        return {'accuracy': 0, 'relevance': 0, 'faithfulness': 0, 'reasoning': f'Error: {str(e)}'}


def run_evaluation(chat_engine, golden_dataset: List[Dict], evaluator_llm: GLMLLM) -> pd.DataFrame:
    """
    Run evaluation on golden dataset.

    Returns DataFrame with results including scores.
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
            response = chat_engine.chat(question)

            # Extract generated answer
            generated_answer = str(response.response)

            # Get retrieved context from source_nodes
            contexts = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                contexts = [node.get_content() for node in response.source_nodes]

            context_text = "\n\n".join(contexts) if contexts else "No context retrieved"

            # Evaluate the answer
            print(f"    Evaluating response...")
            scores = evaluate_answer(
                question=question,
                ground_truth=entry['ground_truth_answer'],
                generated_answer=generated_answer,
                context=context_text,
                llm=evaluator_llm
            )

            # Store result
            results.append({
                'question': question,
                'ground_truth': entry['ground_truth_answer'],
                'generated_answer': generated_answer,
                'num_contexts': len(contexts),
                'accuracy': scores['accuracy'],
                'relevance': scores['relevance'],
                'faithfulness': scores['faithfulness'],
                'reasoning': scores['reasoning']
            })

            print(f"    Scores - Accuracy: {scores['accuracy']}/5, Relevance: {scores['relevance']}/5, Faithfulness: {scores['faithfulness']}/5")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                'question': question,
                'ground_truth': entry['ground_truth_answer'],
                'generated_answer': f"Error: {str(e)}",
                'num_contexts': 0,
                'accuracy': 0,
                'relevance': 0,
                'faithfulness': 0,
                'reasoning': f'Error: {str(e)}'
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print evaluation summary statistics."""
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Total Questions: {len(results_df)}")
    print(f"\nAverage Scores (out of 5):")

    for metric in ['accuracy', 'relevance', 'faithfulness']:
        mean_score = results_df[metric].mean()
        std_score = results_df[metric].std()
        percentage = (mean_score / 5.0) * 100
        print(f"  {metric.capitalize():15s}: {mean_score:.2f}/5 ({percentage:.1f}%) (±{std_score:.2f})")

    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("Simple RAG Evaluation (GLM-based)")
    print("=" * 70)

    # Paths
    dataset_path = project_root / "evals" / "golden_dataset.json"
    output_path = project_root / "evals" / "results_simple.csv"

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ Error: {dataset_path} not found!")
        print("\nGenerate the golden dataset first:")
        print("  python generate_eval_data.py")
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

    # Initialize evaluator LLM
    print("\nInitializing GLM evaluator...")
    evaluator_llm = GLMLLM(
        model_name=config.LLM_MODEL,
        api_key=config.GLM_API_KEY
    )
    print("✅ Evaluator initialized")

    # Run evaluation
    try:
        results_df = run_evaluation(chat_engine, golden_dataset, evaluator_llm)

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
            print(f"\nQ{i+1}: {results_df.iloc[i]['question'][:60]}...")
            print(f"A: {results_df.iloc[i]['generated_answer'][:80]}...")
            print(f"Scores: Acc={results_df.iloc[i]['accuracy']}/5, Rel={results_df.iloc[i]['relevance']}/5, Faith={results_df.iloc[i]['faithfulness']}/5")
            if results_df.iloc[i]['reasoning']:
                print(f"Reasoning: {results_df.iloc[i]['reasoning'][:100]}...")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
