"""
Step 3: Chatbot with RAG Integration
Travel agent chatbot with visa knowledge base integration.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
import sqlite3
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from travel_agent.rag_integration import get_visa_tools

load_dotenv()

# Simple system prompt - let the LLM understand naturally
TRAVEL_AGENT_SYSTEM_PROMPT = """You are James, a human travel agent at Rehman Travels. You're friendly, professional, and natural - just like a real person would be in a travel agency.

Keep responses short and conversational. Match the user's tone.

You have access to a visa requirements database. When users ask about visa requirements, use the visa_requirements_search tool to find accurate information.

For visa questions, you need the user's nationality and destination country. Use your natural language understanding - if someone says "travelling to Dubai", you understand Dubai is the destination. Only ask for what's missing."""


class ChatState(TypedDict):
    """State for the chatbot - stores conversation messages"""
    messages: Annotated[list, add_messages]


def create_rag_chatbot_app(
    checkpoint_db: str = None,
    model: str = "gemini-2.5-flash-lite"
):
    """
    Create a chatbot application with RAG integration for visa information.
    
    Args:
        checkpoint_db: Path to SQLite database for persistent memory. If None, uses data/chatbot_checkpoint.db relative to this file
        model: Gemini model to use
        
    Returns:
        Compiled LangGraph app and checkpointer
    """
    import os
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables.\n"
            "Please add GOOGLE_API_KEY=your_key to your .env file"
        )
    
    # Default checkpoint path - relative to this file's location (not working directory)
    if checkpoint_db is None:
        checkpoint_path = Path(__file__).parent / "data" / "chatbot_checkpoint.db"
    else:
        checkpoint_path = Path(checkpoint_db)
        # If relative path provided, resolve it relative to this file
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).parent / checkpoint_path
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create SQLite connection for persistent memory
    sqlite_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    checkpointer = SqliteSaver(sqlite_conn)
    
    # Get tools (visa RAG tool)
    tools = get_visa_tools()
    
    # Initialize LLM with tools - let it naturally decide when to use them
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: ChatState):
        """Chatbot node - LLM naturally decides when to use tools"""
        messages = state["messages"]
        
        # Check if system message is already in the conversation
        has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
        
        # Add system message if not present (first message or new conversation)
        if not has_system_message:
            system_msg = SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT)
            messages = [system_msg] + messages
        
        # Get response from LLM (with tool calling capability)
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response]
        }
    
    def should_continue(state: ChatState) -> Literal["tools", END]:
        """Route to tools if LLM wants to use them, otherwise end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    # Create tool node - handle retriever tools specially
    try:
        tool_node = ToolNode(tools)
    except TypeError:
        # Fallback: create custom tool node for retriever tools
        from langchain_core.messages import ToolMessage
        
        def tool_node(state: ChatState):
            """Custom tool node that executes tools"""
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find the tool
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
            
            return {"messages": tool_messages}
    
    # Build the graph - simple pattern like 7_chatbot/2_chatbot_with_tools.py
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)
    
    # Set entry point - go straight to chatbot
    graph.set_entry_point("chatbot")
    
    # Add conditional edge: chatbot -> tools (if tool calls) or END
    graph.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # After tools execute, go back to chatbot
    graph.add_edge("tools", "chatbot")
    
    # Compile with checkpointer for memory
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


def run_rag_chatbot_interactive(thread_id: str = "default"):
    """
    Run the RAG-enabled chatbot in interactive mode.
    
    Args:
        thread_id: Unique ID for this conversation thread
    """
    print("=" * 70)
    print("Travel Agent Chatbot with Visa Knowledge Base")
    print("=" * 70)
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)
    print()
    
    app, _ = create_rag_chatbot_app()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "end"]:
                print("\nGoodbye! Safe travels! ✈️")
                break
            
            if not user_input:
                continue
            
            # Invoke the chatbot - simple, let LLM decide naturally
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Get the AI response
            ai_response = result["messages"][-1].content
            print(f"Travel Agent: {ai_response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Safe travels! ✈️")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again.\n")


if __name__ == "__main__":
    # Run interactive chatbot with RAG
    run_rag_chatbot_interactive()
