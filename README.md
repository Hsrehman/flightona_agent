# Travel Agent AI Assistant

A LangGraph-based travel assistant that answers questions about visa requirements using RAG (Retrieval-Augmented Generation).

## Project Structure

```
travel_agent/
├── __init__.py              # Package initialization
├── knowledge_base.py        # Visa rules knowledge base (vector store)
├── rag_integration.py       # RAG tool creation
├── chatbot_with_rag.py      # Main chatbot with RAG integration
├── setup_knowledge_base.py # Setup script for knowledge base
├── env.template             # Environment variables template
├── README.md                # This file
├── data/                    # Generated data
│   ├── dataset/            # Passport index CSV
│   ├── visa_vectorstore/   # Chroma vector database
│   └── chatbot_checkpoint.db # SQLite conversation memory
└── venv/                    # Virtual environment
```

## Setup

### 1. Activate Virtual Environment
```bash
cd travel_agent
source venv/bin/activate
```

### 2. Environment Variables
Create a `.env` file in the `travel_agent` directory:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Initialize Knowledge Base
```bash
python setup_knowledge_base.py
```

This will:
- Load passport-index dataset (39,603 visa rules)
- Create embeddings using BAAI/bge-base-en-v1.5
- Save to `data/visa_vectorstore/`

## Usage

### Run Interactive Chatbot
```bash
python chatbot_with_rag.py
```

### Use Programmatically
```python
from travel_agent.chatbot_with_rag import create_rag_chatbot_app
from langchain_core.messages import HumanMessage

app, _ = create_rag_chatbot_app()
config = {"configurable": {"thread_id": "user123"}}

result = app.invoke(
    {"messages": [HumanMessage(content="What visa do I need from UK to India?")]},
    config=config
)

print(result["messages"][-1].content)
```

## Features

- **Persistent Memory**: SQLite-based conversation history
- **Knowledge Base**: 39,603 visa rules from passport-index dataset
- **High-Quality Embeddings**: BAAI/bge-base-en-v1.5 model
- **Natural Conversations**: Human-like travel agent personality
- **Thread Support**: Multiple conversation threads

## Technology Stack

- **LangGraph**: Graph-based agent framework
- **LangChain**: LLM orchestration
- **Chroma**: Vector database for visa rules
- **BAAI/bge-base-en-v1.5**: Open-source embedding model
- **Gemini 2.5 Flash Lite**: LLM for natural conversations
- **SQLite**: Persistent conversation memory
