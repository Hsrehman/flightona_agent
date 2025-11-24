"""RAG integration - creates visa requirements search tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.tools import create_retriever_tool
from travel_agent.knowledge_base import create_visa_knowledge_base


def create_visa_rag_tool():
    """Create a RAG tool from the visa knowledge base retriever."""
    _, retriever = create_visa_knowledge_base(force_recreate=False)
    
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
    
    return visa_tool


def get_visa_tools():
    """Get all tools for the travel agent."""
    return [create_visa_rag_tool()]

