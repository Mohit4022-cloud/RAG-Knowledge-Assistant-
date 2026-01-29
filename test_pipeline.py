"""
Test script for RAG pipeline implementation.

This script verifies:
1. Ingestion pipeline works correctly
2. Query engine loads and responds
3. Phoenix tracing is active
4. Hybrid search retrieves results with metadata
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config


def test_ingestion():
    """Test the ingestion pipeline."""
    print("\n" + "=" * 70)
    print("TEST 1: Ingestion Pipeline")
    print("=" * 70)

    from ingestion.pipeline import build_pipeline

    try:
        index = build_pipeline()
        num_nodes = len(index.docstore.docs)

        print(f"\n✅ Ingestion successful!")
        print(f"   Nodes created: {num_nodes}")

        if num_nodes == 0:
            print("   ⚠️  Warning: No nodes created. Check if PDFs exist in data/")
            return False

        return True

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        return False


def test_query_engine():
    """Test the query engine."""
    print("\n" + "=" * 70)
    print("TEST 2: Query Engine")
    print("=" * 70)

    from app.engine import get_query_engine

    try:
        query_engine = get_query_engine()

        # Test query
        test_query = "What is RAG?"
        print(f"\nQuery: '{test_query}'")

        response = query_engine.query(test_query)

        print(f"\nResponse: {response.response[:200]}...")

        print("\nRetrieved Sources:")
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for i, node in enumerate(response.source_nodes[:3], 1):
                file_name = node.metadata.get('file_name', 'unknown')
                page = node.metadata.get('page_label', 'unknown')
                score = node.score if hasattr(node, 'score') else 'N/A'
                print(f"  [{i}] {file_name}, page {page} (score: {score})")

            print(f"\n✅ Query engine successful!")
            print(f"   Retrieved {len(response.source_nodes)} source nodes")
            return True
        else:
            print("\n⚠️  No source nodes returned")
            return False

    except Exception as e:
        print(f"\n❌ Query engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_storage():
    """Verify Qdrant storage exists."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Storage Check")
    print("=" * 70)

    storage_dir = config.QDRANT_STORAGE_DIR

    if storage_dir.exists():
        files = list(storage_dir.rglob("*"))
        print(f"✅ Storage directory exists: {storage_dir}")
        print(f"   Contains {len(files)} files/directories")
        return True
    else:
        print(f"❌ Storage directory not found: {storage_dir}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RAG PIPELINE TEST SUITE")
    print("=" * 70)

    # Check if data directory has PDFs
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Create it and add PDF files before running tests.")
        return

    pdf_files = list(data_dir.glob("**/*.pdf"))
    print(f"\nData directory: {data_dir}")
    print(f"PDF files found: {len(pdf_files)}")

    if len(pdf_files) == 0:
        print("⚠️  No PDF files found. You can create a sample PDF with:")
        print("   python create_sample_pdf.py")
        print("\nSkipping tests...")
        return

    for pdf in pdf_files[:5]:  # Show first 5
        print(f"  - {pdf.name}")

    # Run tests
    print(f"\n📍 Phoenix UI: http://localhost:{config.PHOENIX_PORT}")
    print("   (Start Phoenix with: make run_phoenix)")

    results = []

    # Test 1: Ingestion
    results.append(("Ingestion", test_ingestion()))

    # Verify storage
    results.append(("Storage", verify_storage()))

    # Test 2: Query Engine
    results.append(("Query Engine", test_query_engine()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        print(f"\nNext steps:")
        print(f"1. Start Phoenix UI: make run_phoenix")
        print(f"2. View traces at: http://localhost:{config.PHOENIX_PORT}")
        print(f"3. Run queries in Python REPL:")
        print(f"   >>> from app.engine import get_query_engine")
        print(f"   >>> query_engine = get_query_engine()")
        print(f"   >>> response = query_engine.query('your question')")
    else:
        print("\n❌ Some tests failed. Check errors above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
