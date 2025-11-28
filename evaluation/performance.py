"""
Evaluation: Knowledge Graph vs RAG Performance Comparison

This script measures:
1. Query latency (how fast each method responds)
2. Accuracy (does it return the correct answer?)
3. Comparison table

This is what you'll show your teacher!
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.knowledge_graph import TravelKnowledgeGraph, get_country_name
from retrieval.rag_retriever import create_visa_knowledge_base


# ============================================================================
# TEST DATASET
# ============================================================================
# These are queries with known ground truth answers.
# We'll test both KG and RAG on these.

TEST_QUERIES = [
    {
        "query": "What visa do I need from Pakistan to Singapore?",
        "origin": "PAK",
        "destination": "SGP",
        "expected_type": "e_visa"
    },
    {
        "query": "Can UK citizens travel visa-free to France?",
        "origin": "GBR",
        "destination": "FRA",
        "expected_type": "visa_free"
    },
    {
        "query": "What about Pakistani traveling to UAE?",
        "origin": "PAK",
        "destination": "ARE",
        "expected_type": "e_visa"
    },
    {
        "query": "Do Americans need visa for Japan?",
        "origin": "USA",
        "destination": "JPN",
        "expected_type": "visa_free"
    },
    {
        "query": "Indian passport to Thailand visa requirement",
        "origin": "IND",
        "destination": "THA",
        "expected_type": "visa_free"  # 60 days visa-free
    },
    {
        "query": "Can Chinese citizens visit Malaysia?",
        "origin": "CHN",
        "destination": "MYS",
        "expected_type": "visa_free"
    },
    {
        "query": "German passport to Canada",
        "origin": "DEU",
        "destination": "CAN",
        "expected_type": "eta"
    },
    {
        "query": "What visa do I need from Bangladesh to Saudi Arabia?",
        "origin": "BGD",
        "destination": "SAU",
        "expected_type": "visa_required"
    },
    {
        "query": "Can Australians travel to UK?",
        "origin": "AUS",
        "destination": "GBR",
        "expected_type": "visa_free"
    },
    {
        "query": "Nigerian passport to South Africa",
        "origin": "NGA",
        "destination": "ZAF",
        "expected_type": "visa_required"
    },
]


class PerformanceEvaluator:
    """
    Evaluates KG vs RAG performance.
    
    Metrics:
    - Latency: How fast is each query?
    - Accuracy: Does it return correct info?
    """
    
    def __init__(self):
        print("Initializing evaluator...")
        
        # Load Knowledge Graph
        print("  Loading Knowledge Graph...")
        self.kg = TravelKnowledgeGraph()
        csv_path = Path(__file__).parent.parent / "data" / "dataset" / "passport-index-tidy-iso3.csv"
        self.kg.build_from_csv(str(csv_path))
        
        # Load RAG
        print("  Loading RAG retriever (this may take a moment)...")
        _, self.rag_retriever = create_visa_knowledge_base(force_recreate=False)
        
        print("✅ Evaluator ready!\n")
    
    def evaluate_kg(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate Knowledge Graph performance.
        
        Returns:
            {
                'latencies_ms': [0.01, 0.02, ...],
                'avg_latency_ms': 0.015,
                'correct': 8,
                'total': 10,
                'accuracy': 0.8
            }
        """
        latencies = []
        correct = 0
        results = []
        
        for query in test_queries:
            # Time the query
            start = time.time()
            result = self.kg.query(query['origin'], query['destination'])
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            # Check accuracy
            is_correct = False
            if result['found']:
                is_correct = result['requirement_type'] == query['expected_type']
                if is_correct:
                    correct += 1
            
            results.append({
                'query': query['query'],
                'expected': query['expected_type'],
                'got': result.get('requirement_type', 'NOT_FOUND'),
                'correct': is_correct,
                'latency_ms': latency_ms
            })
        
        return {
            'method': 'Knowledge Graph',
            'latencies_ms': latencies,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'correct': correct,
            'total': len(test_queries),
            'accuracy': correct / len(test_queries),
            'results': results
        }
    
    def evaluate_rag(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate RAG performance.
        
        Note: RAG returns documents, so we need to check if the
        correct information is in the retrieved documents.
        """
        latencies = []
        correct = 0
        results = []
        
        for query in test_queries:
            # Time the query
            start = time.time()
            docs = self.rag_retriever.invoke(query['query'])
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            # Check if correct info is in retrieved docs
            is_correct = False
            found_type = None
            
            # Get expected country names
            origin_name = get_country_name(query['origin']).lower()
            dest_name = get_country_name(query['destination']).lower()
            
            for doc in docs:
                content_lower = doc.page_content.lower()
                
                # Check if this doc is about the right countries
                if origin_name in content_lower and dest_name in content_lower:
                    # Extract requirement type from doc
                    if 'visa-free' in content_lower or 'visa free' in content_lower:
                        found_type = 'visa_free'
                    elif 'visa on arrival' in content_lower:
                        found_type = 'visa_on_arrival'
                    elif 'e-visa' in content_lower:
                        found_type = 'e_visa'
                    elif 'eta' in content_lower or 'electronic travel' in content_lower:
                        found_type = 'eta'
                    elif 'visa required' in content_lower:
                        found_type = 'visa_required'
                    
                    # Check if it matches expected
                    # Be lenient: visa_free matches both visa_free and days-based visa-free
                    if found_type:
                        if query['expected_type'] == 'visa_free' and found_type in ['visa_free']:
                            is_correct = True
                        elif found_type == query['expected_type']:
                            is_correct = True
                        break
            
            if is_correct:
                correct += 1
            
            results.append({
                'query': query['query'],
                'expected': query['expected_type'],
                'got': found_type or 'NOT_FOUND',
                'correct': is_correct,
                'latency_ms': latency_ms,
                'num_docs': len(docs)
            })
        
        return {
            'method': 'RAG (Vector Search)',
            'latencies_ms': latencies,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'correct': correct,
            'total': len(test_queries),
            'accuracy': correct / len(test_queries),
            'results': results
        }
    
    def run_comparison(self, test_queries: List[Dict] = None) -> Dict:
        """
        Run full comparison between KG and RAG.
        """
        if test_queries is None:
            test_queries = TEST_QUERIES
        
        print("=" * 70)
        print("PERFORMANCE COMPARISON: Knowledge Graph vs RAG")
        print("=" * 70)
        
        # Evaluate KG
        print("\n📊 Evaluating Knowledge Graph...")
        kg_results = self.evaluate_kg(test_queries)
        
        # Evaluate RAG
        print("📊 Evaluating RAG...")
        rag_results = self.evaluate_rag(test_queries)
        
        # Calculate speedup
        speedup = rag_results['avg_latency_ms'] / kg_results['avg_latency_ms']
        
        # Print results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│                    PERFORMANCE COMPARISON                        │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Metric              │ Knowledge Graph │ RAG                     │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ Avg Latency         │ {kg_results['avg_latency_ms']:>10.3f}ms   │ {rag_results['avg_latency_ms']:>10.1f}ms           │")
        print(f"│ Min Latency         │ {kg_results['min_latency_ms']:>10.3f}ms   │ {rag_results['min_latency_ms']:>10.1f}ms           │")
        print(f"│ Max Latency         │ {kg_results['max_latency_ms']:>10.3f}ms   │ {rag_results['max_latency_ms']:>10.1f}ms           │")
        print(f"│ Accuracy            │ {kg_results['accuracy']*100:>10.0f}%     │ {rag_results['accuracy']*100:>10.0f}%             │")
        print(f"│ Correct/Total       │ {kg_results['correct']:>5}/{kg_results['total']:<5}      │ {rag_results['correct']:>5}/{rag_results['total']:<5}          │")
        print("├─────────────────────────────────────────────────────────────────┤")
        print(f"│ SPEEDUP             │ KG is {speedup:,.0f}x faster than RAG              │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        # Print detailed results
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)
        
        print("\nKnowledge Graph Results:")
        for r in kg_results['results']:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['query'][:50]}")
            print(f"     Expected: {r['expected']}, Got: {r['got']}, Time: {r['latency_ms']:.3f}ms")
        
        print("\nRAG Results:")
        for r in rag_results['results']:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['query'][:50]}")
            print(f"     Expected: {r['expected']}, Got: {r['got']}, Time: {r['latency_ms']:.1f}ms")
        
        return {
            'kg': kg_results,
            'rag': rag_results,
            'speedup': speedup
        }


if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    results = evaluator.run_comparison()
    
    print("\n" + "=" * 70)
    print("SUMMARY FOR TEACHER")
    print("=" * 70)
    print(f"""
Key Findings:

1. SPEED: Knowledge Graph is {results['speedup']:,.0f}x faster than RAG
   - KG: {results['kg']['avg_latency_ms']:.3f}ms average
   - RAG: {results['rag']['avg_latency_ms']:.1f}ms average

2. ACCURACY: 
   - KG: {results['kg']['accuracy']*100:.0f}% (exact matches)
   - RAG: {results['rag']['accuracy']*100:.0f}% (semantic search)

3. TRADE-OFFS:
   - KG: Fast, accurate, but only for structured queries
   - RAG: Slower, but handles semantic/paraphrased queries

4. NEXT STEP: Hybrid system combining both methods
   - Use KG for structured queries (fast)
   - Fall back to RAG for semantic queries
""")

