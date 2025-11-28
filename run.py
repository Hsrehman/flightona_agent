#!/usr/bin/env python3
"""
Travel Agent Chatbot - Main Entry Point

Usage:
    python run.py kg              # Run KG chatbot (blocking)
    python run.py kg --stream     # Run KG chatbot (streaming)
    python run.py rag             # Run RAG chatbot (blocking)
    python run.py rag --stream    # Run RAG chatbot (streaming)
    python run.py eval            # Run performance evaluation
    python run.py                 # Show help
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Travel Agent Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py kg              Run KG chatbot (blocking mode)
  python run.py kg --stream     Run KG chatbot (streaming mode)
  python run.py rag             Run RAG chatbot (blocking mode)
  python run.py rag --stream    Run RAG chatbot (streaming mode)
  python run.py eval            Run performance comparison
        """
    )
    
    parser.add_argument(
        "command", 
        nargs="?",
        choices=["kg", "rag", "eval"],
        help="Command to run: kg (Knowledge Graph), rag (Vector Search), or eval (Performance Test)"
    )
    parser.add_argument(
        "--stream", 
        action="store_true",
        help="Enable streaming mode (text appears as it's generated)"
    )
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Hide timing information"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n" + "=" * 50)
        print("Quick Start:")
        print("  python run.py kg          # Fast KG chatbot")
        print("  python run.py kg --stream # With streaming")
        print("=" * 50)
        return
    
    show_timing = not args.no_timing
    
    if args.command == "kg":
        from chatbots.kg_chatbot import run_kg_chatbot_interactive
        run_kg_chatbot_interactive(show_timing=show_timing, stream=args.stream)
        
    elif args.command == "rag":
        from chatbots.rag_chatbot import run_rag_chatbot_interactive
        run_rag_chatbot_interactive(show_timing=show_timing, stream=args.stream)
        
    elif args.command == "eval":
        from evaluation.performance import PerformanceEvaluator
        evaluator = PerformanceEvaluator()
        evaluator.run_comparison()


if __name__ == "__main__":
    main()
