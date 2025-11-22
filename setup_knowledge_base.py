"""
Setup script to create the visa rules knowledge base.
Run this once to initialize the vector store.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from travel_agent.knowledge_base import create_visa_knowledge_base

if __name__ == "__main__":
    print("=" * 70)
    print("Visa Rules Knowledge Base Setup")
    print("=" * 70)
    print("\nThis will:")
    print("1. Load passport-index dataset")
    print("2. Create embeddings for all visa rules")
    print("3. Store in vector database for fast retrieval")
    print("\nNote: Using open source embeddings (no API key needed)")
    print("      For LLM responses, set GROQ_API_KEY in your .env file")
    print("=" * 70)
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    try:
        vectorstore, retriever = create_visa_knowledge_base(force_recreate=False)
        
        print("\n" + "=" * 70)
        print("✅ Knowledge base created successfully!")
        print("=" * 70)
        
        # Quick test
        print("\nRunning quick test...")
        test_query = "What visa do I need from USA to India?"
        results = retriever.invoke(test_query)
        
        print(f"\nTest Query: {test_query}")
        print(f"Found {len(results)} relevant results")
        if results:
            print(f"\nTop Result:")
            print(f"  {results[0].page_content[:200]}...")
        
        print("\n" + "=" * 70)
        print("Setup complete! You can now use the knowledge base in your travel agent.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. All dependencies are installed: pip install -r requirements_latest.txt")
        print("2. passport-index-dataset folder exists in the project root")
        print("3. For LLM responses, set GROQ_API_KEY in your .env file (optional for testing)")
        sys.exit(1)

