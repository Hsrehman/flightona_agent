# Travel Agent AI Assistant

A LangGraph-based travel assistant that answers questions about visa requirements, baggage rules, and travel information.

## Project Structure

```
travel_agent/
├── __init__.py              # Package initialization
├── knowledge_base.py        # Step 1: Visa rules knowledge base (RAG)
├── chatbot_base.py          # Step 2: Basic chatbot with memory
├── rag_integration.py       # Step 3: RAG tool creation
├── chatbot_with_rag.py      # Step 3: Chatbot with RAG integration
├── setup_knowledge_base.py  # Setup script for knowledge base
├── venv/                    # Virtual environment (local)
├── data/                    # Generated data (vectorstore, checkpoints)
│   ├── visa_vectorstore/    # Chroma vector database
│   └── chatbot_checkpoint.db # SQLite conversation memory
├── STEP1_COMPLETE.md        # Step 1 documentation
├── STEP2_COMPLETE.md        # Step 2 documentation
└── README.md                # This file
```

## Setup

### 1. Activate Virtual Environment
```bash
cd travel_agent
source venv/bin/activate
```

**Note**: The virtual environment is already created and dependencies are installed.
If you need to recreate it:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements_latest.txt
pip install langchain-huggingface langchain-chroma
```

### 3. Environment Variables
Create a `.env` file in the `travel_agent` directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Initialize Knowledge Base
```bash
# From travel_agent directory with venv activated
python setup_knowledge_base.py
```

This will:
- Load passport-index dataset
- Create embeddings (takes a few minutes)
- Save to `travel_agent/data/visa_vectorstore/`

## Usage

### Run Interactive Chatbot

**Basic Chatbot (Step 2):**
```bash
# From travel_agent directory
source venv/bin/activate
python chatbot_base.py
```

**RAG-Enabled Chatbot (Step 3) - Recommended:**
```bash
# From travel_agent directory
source venv/bin/activate
python chatbot_with_rag.py
```

The RAG-enabled chatbot can answer visa questions using the knowledge base!

### Use Programmatically
```python
# Run from parent directory or set PYTHONPATH
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from travel_agent.chatbot_base import create_chatbot_app
from langchain_core.messages import HumanMessage

app, _ = create_chatbot_app()
config = {"configurable": {"thread_id": "user123"}}

result = app.invoke(
    {"messages": [HumanMessage(content="Hello!")]},
    config=config
)
```

## Progress

- ✅ **Step 1**: Visa rules knowledge base (complete)
- ✅ **Step 2**: Basic chatbot with memory (complete)
- ✅ **Step 3**: Integrate RAG retriever (complete)
- ⏳ **Step 4**: Intent classification
- ⏳ **Step 5**: Loop prevention
- ⏳ **Step 6**: Quality checks
- ⏳ **Step 7**: Main travel agent graph
- ⏳ **Step 8**: Testing

## Features

- **Persistent Memory**: SQLite-based conversation history
- **Knowledge Base**: 39,603 visa rules from passport-index dataset
- **High-Quality Embeddings**: BAAI/bge-base-en-v1.5 model
- **Travel Agent Personality**: Specialized system prompt
- **Thread Support**: Multiple conversation threads

## Next Steps

See `TRAVEL_AGENT_PLAN.md` for the complete implementation plan.

