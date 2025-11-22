# Step 1 Complete: Visa Rules Knowledge Base ✅

## What We Built

### 1. Knowledge Base Module (`knowledge_base.py`)
- **Purpose**: Converts passport-index CSV data into a searchable vector store
- **Features**:
  - Loads passport-index-tidy-iso3.csv (39,603 visa rules)
  - Converts ISO-3 codes to readable country names
  - Formats requirements into natural language
  - Creates embeddings using OpenAI
  - Stores in Chroma vector database
  - Creates retriever with MMR (Maximum Marginal Relevance) for diverse results

### 2. Key Functions

#### `create_visa_documents(csv_path)`
- Loads CSV file
- Filters out same-country entries (-1)
- Creates Document objects with:
  - Natural language content: "Citizens of [Country] can travel to [Destination] with [Requirement]"
  - Metadata: passport/destination ISO codes, names, requirement type

#### `create_visa_knowledge_base()`
- Creates or loads vector store
- Persists to disk for reuse
- Returns vectorstore and retriever
- Uses MMR search for better diversity in results

### 3. Setup Script (`setup_knowledge_base.py`)
- Interactive script to initialize the knowledge base
- Tests retrieval after creation
- Provides helpful error messages

## Data Structure

### Input: CSV Format
```
Passport,Destination,Requirement
AFG,ALB,e-visa
AFG,DZA,visa required
USA,IND,visa required
...
```

### Output: Document Format
```
Content: "Citizens of Afghanistan (passport code: AFG) can travel to Albania (destination code: ALB) with e-visa required."

Metadata: {
    "passport_iso": "AFG",
    "passport_name": "Afghanistan",
    "destination_iso": "ALB",
    "destination_name": "Albania",
    "requirement": "e-visa",
    "source": "passport-index-2025"
}
```

## Requirement Types Handled

- **Numbers (7-360)**: "visa-free travel for up to X days"
- **"visa free"**: "visa-free travel (no time limit specified)"
- **"visa on arrival"**: "visa on arrival"
- **"eta"**: "Electronic Travel Authorization (ETA) required"
- **"e-visa"**: "e-visa required"
- **"visa required"**: "visa required (must be obtained before travel)"
- **"no admission"**: "no admission allowed"
- **"-1"**: Filtered out (same country)

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create `.env` file with:
```
OPENAI_API_KEY=your_key_here
```

### 3. Run Setup Script
```bash
python travel_agent/setup_knowledge_base.py
```

This will:
- Load the passport-index dataset
- Create embeddings (takes a few minutes)
- Save vector store to `travel_agent/data/visa_vectorstore/`
- Test retrieval

### 4. Use in Code
```python
from travel_agent.knowledge_base import create_visa_knowledge_base

# Load existing or create new
vectorstore, retriever = create_visa_knowledge_base()

# Query
results = retriever.invoke("What visa do I need from USA to India?")
for doc in results:
    print(doc.page_content)
```

## Files Created

1. `travel_agent/__init__.py` - Package initialization
2. `travel_agent/knowledge_base.py` - Main knowledge base module
3. `travel_agent/setup_knowledge_base.py` - Setup script
4. `travel_agent/data/` - Directory for vector store (created on first run)

## Next Steps (Step 2)

Now that we have the knowledge base, we can:
1. Build basic chatbot with conversation memory
2. Integrate RAG retriever into chatbot
3. Add intent classification
4. Implement loop prevention

## Testing

You can test the knowledge base by running:
```python
python travel_agent/knowledge_base.py
```

Or use the setup script which includes a test.

## Notes

- Vector store is persisted, so you only need to create it once
- Set `force_recreate=True` to rebuild from scratch
- The retriever uses MMR with k=5, fetch_k=20, lambda_mult=0.7
- All 199 countries are supported with ISO-3 codes
- Country name mapping included for 199 countries

