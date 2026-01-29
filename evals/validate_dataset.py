"""
Golden Dataset Validator

Validates the format and quality of the golden evaluation dataset.

Usage:
    python evals/validate_dataset.py
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_dataset(file_path: Path) -> List[Dict]:
    """Load the golden dataset from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    return dataset


def validate_entry(entry: Dict, index: int) -> Tuple[bool, List[str]]:
    """
    Validate a single dataset entry.

    Args:
        entry: Dataset entry to validate
        index: Index of the entry

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ["id", "question", "ground_truth_answer", "context_chunk"]

    # Check required fields
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
        elif not entry[field] or (isinstance(entry[field], str) and not entry[field].strip()):
            errors.append(f"Empty value for field: {field}")

    # Check field types
    if "id" in entry and not isinstance(entry["id"], str):
        errors.append(f"Field 'id' must be a string, got {type(entry['id'])}")

    # Check minimum lengths
    if "question" in entry and len(entry["question"].strip()) < 10:
        errors.append(f"Question too short: {len(entry['question'])} chars")

    if "ground_truth_answer" in entry and len(entry["ground_truth_answer"].strip()) < 10:
        errors.append(f"Answer too short: {len(entry['ground_truth_answer'])} chars")

    if "context_chunk" in entry and len(entry["context_chunk"].strip()) < 20:
        errors.append(f"Context too short: {len(entry['context_chunk'])} chars")

    return len(errors) == 0, errors


def validate_dataset(dataset: List[Dict]) -> Tuple[bool, Dict]:
    """
    Validate the entire dataset.

    Args:
        dataset: List of dataset entries

    Returns:
        Tuple of (is_valid, validation_report)
    """
    if not dataset:
        return False, {"error": "Dataset is empty"}

    total_entries = len(dataset)
    valid_entries = 0
    invalid_entries = []
    duplicate_ids = []

    # Check for duplicate IDs
    ids_seen = set()
    for i, entry in enumerate(dataset):
        entry_id = entry.get("id", f"entry_{i}")
        if entry_id in ids_seen:
            duplicate_ids.append(entry_id)
        ids_seen.add(entry_id)

        # Validate entry
        is_valid, errors = validate_entry(entry, i)
        if is_valid:
            valid_entries += 1
        else:
            invalid_entries.append({
                "index": i,
                "id": entry_id,
                "errors": errors
            })

    # Calculate statistics
    avg_question_len = sum(len(e.get("question", "")) for e in dataset) / total_entries
    avg_answer_len = sum(len(e.get("ground_truth_answer", "")) for e in dataset) / total_entries
    avg_context_len = sum(len(e.get("context_chunk", "")) for e in dataset) / total_entries

    report = {
        "is_valid": len(invalid_entries) == 0 and len(duplicate_ids) == 0,
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "invalid_entries": len(invalid_entries),
        "duplicate_ids": duplicate_ids,
        "invalid_entry_details": invalid_entries[:5],  # Show first 5
        "statistics": {
            "avg_question_length": round(avg_question_len, 1),
            "avg_answer_length": round(avg_answer_len, 1),
            "avg_context_length": round(avg_context_len, 1),
        }
    }

    return report["is_valid"], report


def print_validation_report(report: Dict):
    """Print validation report in a readable format."""
    print("\n" + "=" * 70)
    print("Golden Dataset Validation Report")
    print("=" * 70)

    if report.get("error"):
        print(f"\n❌ Error: {report['error']}")
        return

    # Overall status
    if report["is_valid"]:
        print("\n✅ Dataset is VALID")
    else:
        print("\n⚠️  Dataset has ISSUES")

    # Statistics
    print(f"\nTotal Entries: {report['total_entries']}")
    print(f"Valid Entries: {report['valid_entries']}")
    print(f"Invalid Entries: {report['invalid_entries']}")

    if report["duplicate_ids"]:
        print(f"\n⚠️  Duplicate IDs found: {', '.join(report['duplicate_ids'])}")

    # Statistics
    stats = report["statistics"]
    print(f"\nAverage Lengths:")
    print(f"  Question: {stats['avg_question_length']} chars")
    print(f"  Answer: {stats['avg_answer_length']} chars")
    print(f"  Context: {stats['avg_context_length']} chars")

    # Show invalid entries
    if report["invalid_entry_details"]:
        print(f"\nInvalid Entries (showing first 5):")
        for entry in report["invalid_entry_details"]:
            print(f"\n  Entry {entry['index']} (ID: {entry['id']}):")
            for error in entry["errors"]:
                print(f"    - {error}")

    print("=" * 70)


def main():
    """Main validation function."""
    dataset_path = Path("evals/golden_dataset.json")

    try:
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset(dataset_path)

        # Validate
        is_valid, report = validate_dataset(dataset)

        # Print report
        print_validation_report(report)

        # Exit with appropriate code
        exit(0 if is_valid else 1)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo generate a golden dataset, run:")
        print("  python generate_eval_data.py")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
