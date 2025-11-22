"""
Step 2: Basic Chatbot with Conversation Memory
Creates a chatbot foundation with persistent memory to prevent repetition.
"""

from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path

load_dotenv()

# Travel agent system prompt
TRAVEL_AGENT_SYSTEM_PROMPT = """You are a helpful and knowledgeable travel assistant. 
Your role is to help users with travel-related questions, especially about visa requirements, 
baggage rules, and travel information.

Guidelines:
- Be friendly, professional, and conversational
- Provide clear and accurate information
- If you don't know something, admit it and suggest how to find out
- Remember the conversation context
- Keep responses concise but informative
- Use natural, human-like language

You will soon have access to a knowledge base about visa rules. For now, answer general travel questions 
based on your training data, but be ready to integrate specific visa information."""


class ChatState(TypedDict):
    """State for the chatbot - stores conversation messages"""
    messages: Annotated[list, add_messages]


def create_chatbot_app(
    checkpoint_db: str = None,
    model: str = "llama-3.1-8b-instant"
):
    """
    Create a chatbot application with conversation memory.
    
    Args:
        checkpoint_db: Path to SQLite database for persistent memory. If None, uses data/chatbot_checkpoint.db relative to this file
        model: Groq model to use (default: llama-3.1-8b-instant)
        
    Returns:
        Compiled LangGraph app and checkpointer
        
    Note:
        Requires GROQ_API_KEY in your .env file
    """
    import os
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not found in environment variables.\n"
            "Please add GROQ_API_KEY=your_key to your .env file"
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
    
    # Initialize LLM
    llm = ChatGroq(model=model, temperature=0.7)
    
    def chatbot_node(state: ChatState):
        """Chatbot node that processes user messages"""
        messages = state["messages"]
        
        # Add system message if this is the first message in conversation
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            # First message - add system prompt
            system_msg = SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT)
            messages = [system_msg] + messages
        
        # Get response from LLM
        response = llm.invoke(messages)
        
        return {
            "messages": [response]
        }
    
    # Build the graph
    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    
    # Compile with checkpointer for memory
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


def run_chatbot_interactive(thread_id: str = "default"):
    """
    Run the chatbot in interactive mode.
    
    Args:
        thread_id: Unique ID for this conversation thread
    """
    print("=" * 70)
    print("Travel Agent Chatbot")
    print("=" * 70)
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)
    print()
    
    app, _ = create_chatbot_app()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "end"]:
                print("\nGoodbye! Safe travels! ✈️")
                break
            
            if not user_input:
                continue
            
            # Invoke the chatbot
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
            print("Please try again.\n")


if __name__ == "__main__":
    # Run interactive chatbot
    run_chatbot_interactive()

