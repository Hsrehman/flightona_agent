"""
Step 3: RAG Integration
Creates a RAG tool from the visa knowledge base for the chatbot to use.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import create_retriever_tool
from travel_agent.knowledge_base import create_visa_knowledge_base


def create_visa_rag_tool():
    """
    Create a RAG tool from the visa knowledge base retriever.
    
    Returns:
        LangChain tool that can be used by the LLM to retrieve visa information
    """
    # Load the knowledge base and get retriever
    print("Loading visa knowledge base...")
    _, retriever = create_visa_knowledge_base(force_recreate=False)
    
    # Create a retriever tool
    visa_tool = create_retriever_tool(
        retriever,
        "visa_requirements_search",
        (
            "Search for visa requirements and travel information. "
            "Use this tool to find specific visa rules for travel between countries. "
            "The tool searches a comprehensive database of visa requirements for 199 countries. "
            "Input should be a question about visa requirements, such as: "
            "'What visa do I need from USA to India?' or "
            "'Can UK citizens travel visa-free to France?'"
        )
    )
    
    print("✅ Visa RAG tool created!")
    return visa_tool


def get_visa_tools():
    """
    Get all tools for the travel agent (currently just visa tool).
    
    Returns:
        List of tools for the chatbot
    """
    visa_tool = create_visa_rag_tool()
    return [visa_tool]

