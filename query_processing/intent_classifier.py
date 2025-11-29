"""
Intent Classifier - SetFit Model

This is YOUR FYP component that classifies user intent using:
1. Fast exact matches for obvious cases (hi, bye, thanks) - <5ms
2. Trained SetFit model for everything else - ~10-20ms

Intent types:
- 'casual': Greetings, thanks, complaints, questions, general chat
- 'visa_query': Questions about visa requirements
- 'follow_up': Providing info for an existing query (country/nationality)
- 'booking': Flight/hotel booking requests (COMING SOON)
- 'ticket_change': Ticket modifications/refunds (COMING SOON)
- 'flight_info': General flight/airline questions (COMING SOON)
- 'clarification_origin': User indicating a country is their nationality/origin
- 'clarification_destination': User indicating a country is their destination

This approach:
- NO hardcoded patterns (trained model)
- Generalizes to unseen phrasings
- Measurable (accuracy, precision, recall)
- Fast (~10-20ms inference)
- FYP-worthy (you trained your own model!)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warning

import re
from typing import Literal
from pathlib import Path

# ============================================================================
# LABELS (must match training data)
# ============================================================================

LABELS = ["casual", "visa_query", "follow_up", "booking", "ticket_change", "flight_info", "clarification_origin", "clarification_destination"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# "Coming soon" intents - these get template responses
COMING_SOON_INTENTS = {"booking", "ticket_change", "flight_info"}

# Clarification intents - used when pending_clarification is active
CLARIFICATION_INTENTS = {"clarification_origin", "clarification_destination"}

# Template responses for coming soon features
COMING_SOON_RESPONSES = {
    "booking": "Flight and hotel booking is coming soon! For now, I can help you with visa requirements. Would you like to check if you need a visa for your destination?",
    "ticket_change": "Ticket changes and refunds are coming soon! For now, I can help you with visa requirements. Is there anything about visas I can assist with?",
    "flight_info": "Flight information features are coming soon! For now, I specialize in visa requirements. Would you like to know if you need a visa somewhere?",
}

# ============================================================================
# FAST EXACT MATCHES (obvious cases only)
# ============================================================================

# These are the ONLY hardcoded patterns - for very obvious, short inputs
# Kept for speed optimization on trivial cases
EXACT_CASUAL = {
    'hi', 'hello', 'hey', 'yo', 'sup',
    'bye', 'goodbye', 'cya', 'later',
    'thanks', 'thank you', 'thx', 'ty',
    'yes', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure',
    'no', 'nope', 'nah',
    'idk', 'dunno',
}

# ============================================================================
# SETFIT MODEL (lazy loaded)
# ============================================================================

_setfit_model = None


def _get_setfit_model():
    """Lazy load the trained SetFit model."""
    global _setfit_model
    if _setfit_model is None:
        from setfit import SetFitModel
        
        # Model path relative to this file
        model_path = Path(__file__).parent.parent / "models" / "intent_classifier_setfit"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"SetFit model not found at {model_path}. "
                "Please train the model first using Text Classification.ipynb"
            )
        
        _setfit_model = SetFitModel.from_pretrained(str(model_path))
    
    return _setfit_model


def _classify_with_setfit(text: str) -> tuple:
    """
    Classify intent using the trained SetFit model.
    
    Returns:
        (intent, confidence) tuple
    """
    model = _get_setfit_model()
    
    # Get prediction (returns label index)
    prediction = model.predict([text])[0]
    
    # Convert to label name
    intent = LABELS[prediction]
    
    # SetFit doesn't give confidence by default, but we can get probabilities
    try:
        probs = model.predict_proba([text])[0]
        confidence = float(max(probs))
    except:
        confidence = 1.0  # If predict_proba not available
    
    return intent, confidence


# ============================================================================
# COUNTRY DETECTION (still needed for context-aware follow_up)
# ============================================================================

from retrieval.knowledge_graph import ISO3_TO_COUNTRY


def _contains_country(text: str) -> bool:
    """Quick check if text contains a country name (exact match)."""
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Check against country names
    for country in ISO3_TO_COUNTRY.values():
        if country.lower() in text_clean:
            return True
    
    # Check common aliases
    aliases = ['uk', 'usa', 'us', 'uae', 'dubai', 'america', 'britain']
    for alias in aliases:
        if re.search(r'\b' + alias + r'\b', text_clean):
            return True
    
    return False


def _fuzzy_contains_country(text: str) -> bool:
    """
    Check if text contains a country/nationality even with typos.
    Uses the entity extractor's fuzzy matching.
    """
    try:
        from query_processing.entity_extractor import extract_countries_from_text
        entities = extract_countries_from_text(text)
        return entities.get('origin') is not None or entities.get('destination') is not None
    except Exception:
        return False


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_intent(
    text: str,
    conversation_has_visa_context: bool = False
) -> str:
    """
    Classify the intent of a user message.
    
    Uses a hybrid approach:
    1. Fast exact match for obvious inputs (hi, bye, thanks) - <5ms
    2. Context-aware country check for short follow-ups - <10ms
    3. Trained SetFit model for everything else - ~10-20ms
    
    Args:
        text: The user's message
        conversation_has_visa_context: Whether we're in a visa conversation
    
    Returns:
        One of: 'casual', 'visa_query', 'follow_up', 'booking', 'ticket_change', 'flight_info'
    """
    text_lower = text.lower().strip()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Empty or very short
    if len(text_clean) < 2:
        return 'casual'
    
    # =========================================================================
    # FAST PATH: Exact matches for obvious cases (<5ms)
    # =========================================================================
    if text_clean in EXACT_CASUAL:
        return 'casual'
    
    # =========================================================================
    # CONTEXT-AWARE: If in visa context and just a country name → follow_up
    # This handles short answers like "pakistani" or "dubai" during a visa query
    # Also catches typos like "pakisatni" via fuzzy matching
    # =========================================================================
    words = text_clean.split()
    if conversation_has_visa_context and len(words) <= 3:
        if _contains_country(text_clean) or _fuzzy_contains_country(text_clean):
            return 'follow_up'
    
    # =========================================================================
    # SETFIT CLASSIFICATION (~10-20ms)
    # =========================================================================
    intent, confidence = _classify_with_setfit(text_lower)
    
    # =========================================================================
    # LOW CONFIDENCE → LET LLM HANDLE IT
    # If classifier isn't confident, default to casual so LLM can respond
    # =========================================================================
    CONFIDENCE_THRESHOLD = 0.5
    if confidence < CONFIDENCE_THRESHOLD:
        return 'casual'
    
    return intent


def is_coming_soon_intent(intent: str) -> bool:
    """Check if an intent is a coming soon feature."""
    return intent in COMING_SOON_INTENTS


def is_clarification_intent(intent: str) -> bool:
    """Check if an intent is a clarification response (origin or destination)."""
    return intent in CLARIFICATION_INTENTS


def get_coming_soon_response(intent: str) -> str:
    """Get the template response for a coming soon intent."""
    return COMING_SOON_RESPONSES.get(intent, "This feature is coming soon!")


def get_intent_confidence(text: str) -> dict:
    """
    Get detailed classification info with confidence scores.
    Useful for evaluation and debugging.
    """
    text_lower = text.lower().strip()
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Check fast path
    if text_clean in EXACT_CASUAL:
        return {
            'intent': 'casual',
            'confidence': 1.0,
            'method': 'exact_match',
        }
    
    # SetFit classification
    intent, confidence = _classify_with_setfit(text_lower)
    
    return {
        'intent': intent,
        'confidence': confidence,
        'method': 'setfit',
    }


# ============================================================================
# INITIALIZATION (pre-load model)
# ============================================================================

def init_classifier():
    """Pre-load the SetFit model for faster first inference."""
    print("Loading intent classifier (SetFit)...")
    _get_setfit_model()
    print("Intent classifier ready!")


# ============================================================================
# Tests moved to tests/test_comprehensive.py
# ============================================================================
