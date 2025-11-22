"""
Step 1: Visa Rules Knowledge Base
Creates a vector store from passport-index dataset for RAG queries about visa requirements.
"""

import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# ISO-3 to Country Name mapping (common countries)
ISO3_TO_COUNTRY = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AND": "Andorra",
    "AGO": "Angola", "ATG": "Antigua and Barbuda", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan", "BHS": "Bahamas",
    "BHR": "Bahrain", "BGD": "Bangladesh", "BRB": "Barbados", "BLR": "Belarus",
    "BEL": "Belgium", "BLZ": "Belize", "BEN": "Benin", "BTN": "Bhutan",
    "BOL": "Bolivia", "BIH": "Bosnia and Herzegovina", "BWA": "Botswana", "BRA": "Brazil",
    "BRN": "Brunei", "BGR": "Bulgaria", "BFA": "Burkina Faso", "BDI": "Burundi",
    "KHM": "Cambodia", "CMR": "Cameroon", "CAN": "Canada", "CPV": "Cape Verde",
    "CAF": "Central African Republic", "TCD": "Chad", "CHL": "Chile", "CHN": "China",
    "COL": "Colombia", "COM": "Comoros", "COG": "Congo", "COD": "DR Congo",
    "CRI": "Costa Rica", "CIV": "Ivory Coast", "HRV": "Croatia", "CUB": "Cuba",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DNK": "Denmark", "DJI": "Djibouti",
    "DMA": "Dominica", "DOM": "Dominican Republic", "ECU": "Ecuador", "EGY": "Egypt",
    "SLV": "El Salvador", "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia", "GEO": "Georgia",
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "GRD": "Grenada",
    "GTM": "Guatemala", "GIN": "Guinea", "GNB": "Guinea-Bissau", "GUY": "Guyana",
    "HTI": "Haiti", "HND": "Honduras", "HKG": "Hong Kong", "HUN": "Hungary",
    "ISL": "Iceland", "IND": "India", "IDN": "Indonesia", "IRN": "Iran",
    "IRQ": "Iraq", "IRL": "Ireland", "ISR": "Israel", "ITA": "Italy",
    "JAM": "Jamaica", "JPN": "Japan", "JOR": "Jordan", "KAZ": "Kazakhstan",
    "KEN": "Kenya", "KIR": "Kiribati", "XKX": "Kosovo", "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan", "LAO": "Laos", "LVA": "Latvia", "LBN": "Lebanon",
    "LSO": "Lesotho", "LBR": "Liberia", "LBY": "Libya", "LIE": "Liechtenstein",
    "LTU": "Lithuania", "LUX": "Luxembourg", "MAC": "Macau", "MDG": "Madagascar",
    "MWI": "Malawi", "MYS": "Malaysia", "MDV": "Maldives", "MLI": "Mali",
    "MLT": "Malta", "MHL": "Marshall Islands", "MRT": "Mauritania", "MUS": "Mauritius",
    "MEX": "Mexico", "FSM": "Micronesia", "MDA": "Moldova", "MCO": "Monaco",
    "MNG": "Mongolia", "MNE": "Montenegro", "MAR": "Morocco", "MOZ": "Mozambique",
    "MMR": "Myanmar", "NAM": "Namibia", "NRU": "Nauru", "NPL": "Nepal",
    "NLD": "Netherlands", "NZL": "New Zealand", "NIC": "Nicaragua", "NER": "Niger",
    "NGA": "Nigeria", "PRK": "North Korea", "MKD": "North Macedonia", "NOR": "Norway",
    "OMN": "Oman", "PAK": "Pakistan", "PLW": "Palau", "PSE": "Palestine",
    "PAN": "Panama", "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "RWA": "Rwanda", "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia", "WSM": "Samoa", "SMR": "San Marino", "STP": "São Tomé and Príncipe",
    "SAU": "Saudi Arabia", "SEN": "Senegal", "SRB": "Serbia", "SYC": "Seychelles",
    "SLE": "Sierra Leone", "SGP": "Singapore", "SVK": "Slovakia", "SVN": "Slovenia",
    "SLB": "Solomon Islands", "SOM": "Somalia", "ZAF": "South Africa", "KOR": "South Korea",
    "SSD": "South Sudan", "ESP": "Spain", "LKA": "Sri Lanka", "VCT": "Saint Vincent and the Grenadines",
    "SDN": "Sudan", "SUR": "Suriname", "SWE": "Sweden", "CHE": "Switzerland",
    "SYR": "Syria", "TWN": "Taiwan", "TJK": "Tajikistan", "TZA": "Tanzania",
    "THA": "Thailand", "TLS": "Timor-Leste", "TGO": "Togo", "TON": "Tonga",
    "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TKM": "Turkmenistan", "TUV": "Tuvalu",
    "TUR": "Turkey", "UGA": "Uganda", "UKR": "Ukraine", "ARE": "United Arab Emirates",
    "GBR": "United Kingdom", "USA": "United States", "URY": "Uruguay", "UZB": "Uzbekistan",
    "VUT": "Vanuatu", "VAT": "Vatican City", "VEN": "Venezuela", "VNM": "Vietnam",
    "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe"
}


def get_country_name(iso3_code: str) -> str:
    """Get country name from ISO-3 code, fallback to code if not found."""
    return ISO3_TO_COUNTRY.get(iso3_code, iso3_code)


def format_requirement(requirement: str) -> str:
    """Format requirement into human-readable text."""
    if requirement == "-1":
        return "same country (no visa needed)"
    
    if requirement.isdigit():
        days = int(requirement)
        return f"visa-free travel for up to {days} days"
    
    requirement_map = {
        "visa free": "visa-free travel (no time limit specified)",
        "visa on arrival": "visa on arrival",
        "eta": "Electronic Travel Authorization (ETA) required",
        "e-visa": "e-visa required",
        "visa required": "visa required (must be obtained before travel)",
        "no admission": "no admission allowed"
    }
    
    return requirement_map.get(requirement.lower(), requirement)


def create_visa_documents(csv_path: str) -> List[Document]:
    """
    Load passport-index CSV and convert to Document objects for vector store.
    
    Args:
        csv_path: Path to passport-index-tidy-iso3.csv
        
    Returns:
        List of Document objects with visa information
    """
    print(f"Loading visa data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out same country entries (-1)
    df = df[df['Requirement'] != '-1']
    
    documents = []
    
    for _, row in df.iterrows():
        passport_iso = row['Passport']
        destination_iso = row['Destination']
        requirement = str(row['Requirement'])
        
        passport_name = get_country_name(passport_iso)
        destination_name = get_country_name(destination_iso)
        requirement_text = format_requirement(requirement)
        
        # Create document content
        content = (
            f"Citizens of {passport_name} (passport code: {passport_iso}) can travel to "
            f"{destination_name} (destination code: {destination_iso}) with {requirement_text}."
        )
        
        # Create metadata for filtering and reference
        metadata = {
            "passport_iso": passport_iso,
            "passport_name": passport_name,
            "destination_iso": destination_iso,
            "destination_name": destination_name,
            "requirement": requirement,
            "source": "passport-index-2025"
        }
        
        documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    print(f"Created {len(documents)} visa rule documents")
    return documents


def create_visa_knowledge_base(
    csv_path: str = None,
    persist_directory: str = None,
    force_recreate: bool = False,
    batch_size: int = 1000
) -> tuple[Chroma, any]:
    """
    Create or load visa rules vector store.
    
    Args:
        csv_path: Path to passport-index CSV. If None, uses default location
        persist_directory: Directory to persist vector store. If None, uses data/visa_vectorstore relative to this file
        force_recreate: If True, recreate vector store even if it exists
        batch_size: Number of documents to process in each batch (default: 1000)
                    Larger batches = faster but more memory. Smaller = more progress updates.
        
    Returns:
        Tuple of (vector_store, retriever)
    """
    # Default CSV path - relative to this file's location
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "passport-index-dataset" / "passport-index-tidy-iso3.csv"
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Default persist directory - relative to this file's location (not working directory)
    if persist_directory is None:
        persist_path = Path(__file__).parent / "data" / "visa_vectorstore"
    else:
        persist_path = Path(persist_directory)
        # If relative path provided, resolve it relative to this file
        if not persist_path.is_absolute():
            persist_path = Path(__file__).parent / persist_path
    
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Use BAAI/bge-base-en-v1.5 - one of the best open source embedding models
    # This model consistently ranks at the top for embedding quality
    print("Initializing BAAI/bge-base-en-v1.5 embedding model...")
    print("(This is one of the best open source embedding models available)")
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32  # Process 32 texts at once for efficiency
        }
    )
    print("✅ Embedding model loaded!")
    
    # Check if vector store already exists
    if not force_recreate and (persist_path / "chroma.sqlite3").exists():
        print(f"Loading existing vector store from {persist_path}...")
        vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_function
        )
        print("Vector store loaded successfully!")
    else:
        print("Creating new vector store...")
        # Create documents
        documents = create_visa_documents(str(csv_path))
        
        # Create embeddings and vector store with batching for progress tracking
        print(f"Creating embeddings for {len(documents)} documents...")
        print(f"Processing in batches of {batch_size} documents...")
        
        from tqdm import tqdm
        
        vectorstore = None
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                     desc="Embedding documents", 
                     total=total_batches,
                     unit="batch"):
            batch = documents[i:i + batch_size]
            
            if vectorstore is None:
                # Create vectorstore with first batch
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    persist_directory=str(persist_path)
                )
            else:
                # Add subsequent batches
                vectorstore.add_documents(batch)
        
        print(f"✅ Vector store created and saved to {persist_path}")
        print(f"Total documents: {len(documents)}")
    
    # Create retriever with MMR (Maximum Marginal Relevance) for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,  # Return top 5 most relevant documents
            "fetch_k": 20,  # Fetch 20 candidates for MMR
            "lambda_mult": 0.7  # Balance between relevance and diversity (0.7 = more relevance)
        }
    )
    
    print("Retriever created successfully!")
    return vectorstore, retriever


if __name__ == "__main__":
    # Test the knowledge base creation
    print("=" * 60)
    print("Creating Visa Rules Knowledge Base")
    print("=" * 60)
    
    vectorstore, retriever = create_visa_knowledge_base(force_recreate=False)
    
    # Test retrieval
    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)
    
    test_queries = [
        "What visa do I need to travel from USA to India?",
        "Can I travel visa-free from UK to France?",
        "What are the visa requirements for Australian passport holders going to Japan?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.invoke(query)
        for i, doc in enumerate(results[:2], 1):  # Show top 2 results
            print(f"\nResult {i}:")
            print(f"  {doc.page_content}")
            print(f"  Metadata: {doc.metadata}")

