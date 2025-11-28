"""
Intent Classifier - Fast keyword + semantic classification

This is YOUR component (not LLM) that decides the intent of user messages.
Fast for clear cases, semantic similarity for ambiguous cases.

Intent types:
- 'casual': Greetings, thanks, general chat
- 'visa_query': Questions about visa requirements
- 'follow_up': Continuing a previous topic (providing missing info)
"""

import re
from typing import Literal

# ============================================================================
# KEYWORD PATTERNS (Fast - <5ms)
# ============================================================================

# Casual conversation indicators
CASUAL_PATTERNS = {
    'greetings': [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'^(hi|hello|hey)[\s!.,]*$',
    ],
    'thanks': [
        r'\b(thanks|thank you|thx|appreciate|cheers)\b',
    ],
    'farewell': [
        r'\b(bye|goodbye|see you|take care|later)\b',
    ],
    'how_are_you': [
        r'\bhow are you\b',
        r'\bhow\'s it going\b',
        r'\bwhat\'s up\b',
    ],
    'affirmative': [
        r'^(yes|yeah|yep|sure|ok|okay|yup|alright)[\s!.,]*$',
    ],
    'negative': [
        r'^(no|nope|nah)[\s!.,]*$',
    ],
    'uncertainty': [
        # User expressing they don't know - LLM should handle contextually
        r'^idk[\s!.,]*$',
        r'^i\s*don\'?t\s*know[\s!.,]*$',
        r'^not\s*sure[\s!.,]*$',
        r'^no\s*idea[\s!.,]*$',
        r'^unsure[\s!.,]*$',
        r'^no\s*clue[\s!.,]*$',
        r'^haven\'?t\s*decided[\s!.,]*$',
        r'^i\'?m\s*not\s*sure[\s!.,]*$',
        r'^dunno[\s!.,]*$',
    ],
    'dispute': [
        # User challenging/disputing info - LLM should handle gracefully
        r'\bbut\s+i\s+heard\b',
        r'\bbut\s+someone\s+(?:told|said)\b',
        r'\banother\s+(?:agent|person|friend)\b',
        r'\bthat\'?s\s+(?:not\s+)?(?:right|correct|true|wrong)\b',
        r'\bare\s+you\s+sure\b',
        r'\bi\s+(?:think|thought)\s+(?:that\s+)?(?:i\s+)?(?:don\'?t|do\s+not)\b',
        r'\bi\s+don\'?t\s+(?:think|believe)\b',
        r'\breally\?\b',
        r'\bactually\b.*\bheard\b',
    ],
    'followup_question': [
        # User asking for more details - LLM should elaborate
        r'\b(?:tell|give)\s+(?:me\s+)?more\b',
        r'\bmore\s+(?:info|information|details)\b',
        r'\bhow\s+(?:do\s+i|can\s+i|to)\s+apply\b',
        r'\bwhat\s+(?:are\s+the|is\s+the)\s+(?:requirements?|process|steps?)\b',
        r'\bexplain\b',
        r'\bwhy\b',
    ],
}

# Visa/travel query indicators
VISA_PATTERNS = {
    'visa_keywords': [
        r'\bvisa\b',
        r'\bpassport\b',
        r'\btravel(?:ing|ling)?\b',
        r'\bvisit(?:ing)?\b',
        r'\bentry\b',
        r'\bimmigration\b',
    ],
    'travel_verbs': [
        r'\b(go|going|went|fly|flying|flew|travel|visit|trip)\s+to\b',
    ],
    'requirement_words': [
        r'\b(need|require|necessary|must|have to)\b',
        r'\bdo i need\b',
        r'\bcan i\b',
        r'\bam i allowed\b',
    ],
}

# Nationality/country indicators (suggesting follow-up to visa query)
NATIONALITY_PATTERNS = [
    r'\bi(?:\'?m| am)\s+(?:a\s+)?(\w+)\b',  # "I'm Pakistani" or "I am a Pakistani"
    r'\b(\w+)\s+(?:citizen|national|passport)\b',  # "Pakistani citizen"
    r'^(\w+)[\s!.,]*$',  # Just a country/nationality name
]

# Country names that might indicate follow-up
from retrieval.knowledge_graph import ISO3_TO_COUNTRY, COUNTRY_TO_ISO3


def _matches_any_pattern(text: str, patterns: list) -> bool:
    """Check if text matches any pattern in the list."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _is_country_or_nationality(text: str) -> bool:
    """Check if the text is a country name or nationality."""
    text_lower = text.lower().strip()
    
    # Remove punctuation for matching
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Check against known countries
    for country in ISO3_TO_COUNTRY.values():
        if country.lower() in text_clean or text_clean in country.lower():
            return True
    
    # Check nationality forms (e.g., "pakistani", "american")
    nationality_suffixes = ['ian', 'an', 'ish', 'ese', 'i']
    for country in ISO3_TO_COUNTRY.values():
        country_lower = country.lower()
        for suffix in nationality_suffixes:
            # Check if text could be a nationality form
            if text_clean.endswith(suffix):
                # Remove suffix and check if it's close to a country
                base = text_clean[:-len(suffix)] if len(suffix) > 0 else text_clean
                if base in country_lower or country_lower.startswith(base):
                    return True
    
    # Also check ISO3 codes
    if text_clean.upper() in ISO3_TO_COUNTRY:
        return True
    
    return False


def classify_intent(
    text: str,
    conversation_has_visa_context: bool = False
) -> Literal['casual', 'visa_query', 'follow_up']:
    """
    Classify the intent of a user message.
    
    Args:
        text: The user's message
        conversation_has_visa_context: Whether we're already in a visa conversation
    
    Returns:
        'casual': General conversation (hi, thanks, etc.)
        'visa_query': New visa-related question
        'follow_up': Providing more info for an existing query
    
    Speed: <10ms for keyword matching
    """
    text_lower = text.lower().strip()
    
    # Empty or very short text
    if len(text_lower) < 2:
        return 'casual'
    
    # =========================================================================
    # STEP 1: Check for DISPUTES/CHALLENGES first (always casual, even with visa words)
    # These need LLM handling, not re-querying the knowledge base
    # =========================================================================
    if _matches_any_pattern(text_lower, CASUAL_PATTERNS.get('dispute', [])):
        return 'casual'
    
    if _matches_any_pattern(text_lower, CASUAL_PATTERNS.get('followup_question', [])):
        return 'casual'
    
    # =========================================================================
    # STEP 2: Fast keyword check for other CASUAL patterns
    # =========================================================================
    for category, patterns in CASUAL_PATTERNS.items():
        if category in ['dispute', 'followup_question']:
            continue  # Already checked above
        if _matches_any_pattern(text_lower, patterns):
            # But check if it's combined with visa keywords
            has_visa_words = False
            for visa_category, visa_patterns in VISA_PATTERNS.items():
                if _matches_any_pattern(text_lower, visa_patterns):
                    has_visa_words = True
                    break
            
            if not has_visa_words:
                return 'casual'
    
    # =========================================================================
    # STEP 2: Check for VISA_QUERY keywords
    # =========================================================================
    visa_score = 0
    
    for category, patterns in VISA_PATTERNS.items():
        if _matches_any_pattern(text_lower, patterns):
            visa_score += 1
    
    # Strong visa indicators
    if visa_score >= 2:
        return 'visa_query'
    
    # Contains visa-related keywords
    if visa_score >= 1:
        return 'visa_query'
    
    # =========================================================================
    # STEP 3: Check for FOLLOW_UP (providing country/nationality info)
    # =========================================================================
    if conversation_has_visa_context:
        # If we're in a visa conversation and user provides country/nationality
        if _is_country_or_nationality(text_lower):
            return 'follow_up'
        
        # Check nationality patterns
        for pattern in NATIONALITY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return 'follow_up'
    
    # =========================================================================
    # STEP 4: Check if it contains any country (might be visa-related)
    # =========================================================================
    if _is_country_or_nationality(text_lower):
        # If just a country name, probably follow-up if we have context
        if conversation_has_visa_context:
            return 'follow_up'
        # Otherwise might be starting a visa query
        return 'visa_query'
    
    # =========================================================================
    # DEFAULT: Treat as casual
    # =========================================================================
    return 'casual'


def get_intent_confidence(text: str) -> dict:
    """
    Get confidence scores for each intent type.
    Useful for evaluation and debugging.
    
    Returns:
        {
            'intent': str,
            'confidence': float,
            'scores': {'casual': float, 'visa_query': float, 'follow_up': float}
        }
    """
    text_lower = text.lower().strip()
    
    scores = {
        'casual': 0.0,
        'visa_query': 0.0,
        'follow_up': 0.0,
    }
    
    # Count casual matches
    for category, patterns in CASUAL_PATTERNS.items():
        if _matches_any_pattern(text_lower, patterns):
            scores['casual'] += 0.3
    
    # Count visa matches
    for category, patterns in VISA_PATTERNS.items():
        if _matches_any_pattern(text_lower, patterns):
            scores['visa_query'] += 0.25
    
    # Check for country/nationality (could be follow-up)
    if _is_country_or_nationality(text_lower):
        scores['follow_up'] += 0.4
        scores['visa_query'] += 0.2
    
    # Normalize
    total = sum(scores.values())
    if total > 0:
        for key in scores:
            scores[key] /= total
    else:
        scores['casual'] = 1.0  # Default
    
    # Get highest
    intent = max(scores, key=scores.get)
    confidence = scores[intent]
    
    return {
        'intent': intent,
        'confidence': confidence,
        'scores': scores,
    }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        # Casual
        ("hi", False),
        ("hello", False),
        ("how are you", False),
        ("thanks", False),
        ("bye", False),
        
        # Visa query
        ("do i need a visa", False),
        ("visa requirements for dubai", False),
        ("can i travel to singapore", False),
        ("i want to go to france", False),
        
        # Follow-up (with context)
        ("pakistani", True),
        ("i'm from pakistan", True),
        ("dubai", True),
        ("united states", True),
        
        # Ambiguous
        ("what about uk", True),
        ("and for france?", True),
    ]
    
    print("Intent Classification Test")
    print("=" * 60)
    
    for text, has_context in test_cases:
        intent = classify_intent(text, has_context)
        details = get_intent_confidence(text)
        print(f"'{text}' (context={has_context})")
        print(f"  → Intent: {intent}")
        print(f"  → Confidence: {details['confidence']:.2f}")
        print()

