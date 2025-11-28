"""Setup script to create the visa rules knowledge base."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.rag_retriever import create_visa_knowledge_base

if __name__ == "__main__":
    print("=" * 70)
    print("Visa Rules Knowledge Base Setup")
    print("=" * 70)
    print("\nThis will:")
    print("1. Load passport-index dataset")
    print("2. Create embeddings for all visa rules")
    print("3. Store in vector database for fast retrieval")
    print("=" * 70)
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    try:
        vectorstore, retriever = create_visa_knowledge_base(force_recreate=False)
        print("\n✅ Knowledge base created successfully!")
        
        test_query = "What visa do I need from USA to India?"
        results = retriever.invoke(test_query)
        print(f"\nTest Query: {test_query}")
        print(f"Found {len(results)} relevant results")
        if results:
            print(f"Top Result: {results[0].page_content[:150]}...")
        
        print("\nSetup complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

