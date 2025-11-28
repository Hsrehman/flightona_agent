"""
Template Responses - Fast responses without LLM

These templates skip the LLM entirely for predictable responses.
Speed: <1ms (vs 200-500ms for LLM)

Used for:
- Asking for missing information
- Welcome/goodbye messages
- Error messages
- Formatting visa results

The LLM is only used for:
- Casual conversation
- Complex/nuanced responses
"""

import random
from typing import Optional, Dict, List

# ============================================================================
# TEMPLATE DEFINITIONS
# ============================================================================

TEMPLATES = {
    # ----- Missing information templates -----
    "need_origin": [
        "Which country's passport do you hold?",
        "What's your nationality?",
        "Which passport will you be traveling with?",
        "And what country are you a citizen of?",
    ],
    
    "need_destination": [
        "Where are you planning to travel?",
        "Which country would you like to visit?",
        "And where do you want to go?",
        "What's your destination?",
    ],
    
    "need_both": [
        "I'd be happy to help! Which passport do you hold, and where are you planning to travel?",
        "Sure thing! Just need to know your nationality and destination.",
        "I can check that for you! What's your passport country and where are you headed?",
    ],
    
    # ----- Clarification templates (when country is ambiguous) -----
    "clarify_country": [
        "Is {country} where you're from, or where you want to travel?",
        "Just to clarify - is {country} your nationality or your destination?",
        "Are you traveling from {country}, or to {country}?",
    ],
    
    # ----- Welcome templates -----
    "welcome": [
        "Hello, welcome to Rehman Travels. How can I help you today?",
    ],
    
    # ----- Goodbye templates -----
    "goodbye": [
        "Goodbye! Safe travels! ✈️",
        "Take care! Have a great trip! ✈️",
        "Bye! Feel free to come back if you have more questions!",
    ],
    
    # ----- Acknowledgment templates -----
    "acknowledge_casual": [
        "I'm doing great, thanks for asking! How can I help you with your travel plans?",
        "Doing well, thank you! What can I help you with today?",
        "Great, thanks! Ready to help with any travel questions you have.",
    ],
    
    # ----- Error templates -----
    "error_not_found": [
        "I couldn't find visa information for that route. Please check the country names and try again.",
        "Sorry, I don't have data for that specific route. Could you verify the countries?",
    ],
    
    "error_invalid_country": [
        "I don't recognize that country. Could you try spelling it differently?",
        "Hmm, I'm not sure about that country. Could you double-check the name?",
    ],
    
    # ----- Filler templates (for streaming/waiting) -----
    "filler": [
        "Let me check that for you...",
        "Looking that up now...",
        "One moment...",
        "Checking the visa requirements...",
    ],
}

# ============================================================================
# VISA RESULT FORMATTING
# ============================================================================

VISA_RESULT_TEMPLATES = {
    "visa_free": [
        "{origin_name} passport holders can visit {destination_name} visa-free for up to {days} days. No visa application needed!",
        "Great news! With a {origin_name} passport, you can travel to {destination_name} without a visa for up to {days} days.",
        "You're in luck! {origin_name} citizens get visa-free entry to {destination_name} for {days} days.",
    ],
    
    "visa_free_no_days": [
        "{origin_name} passport holders can visit {destination_name} visa-free. No visa application needed!",
        "Great news! {origin_name} citizens don't need a visa for {destination_name}.",
    ],
    
    "e_visa": [
        "{origin_name} passport holders need an e-visa for {destination_name}. You can apply online before your trip.",
        "You'll need an e-visa for {destination_name}. It's available online - pretty straightforward to get!",
        "For {destination_name}, {origin_name} citizens need to apply for an e-visa online before traveling.",
    ],
    
    "visa_required": [
        "{origin_name} passport holders need a visa for {destination_name}. You'll need to apply at an embassy before traveling.",
        "A visa is required for {origin_name} citizens traveling to {destination_name}. Best to apply well in advance.",
        "You'll need to get a visa for {destination_name}. Apply at the nearest embassy or consulate.",
    ],
    
    "visa_on_arrival": [
        "{origin_name} passport holders can get a visa on arrival in {destination_name}.",
        "Good news! You can get your visa on arrival in {destination_name}. No need to apply in advance.",
        "For {destination_name}, {origin_name} citizens can obtain a visa upon arrival.",
    ],
    
    "eta": [
        "{origin_name} passport holders need an ETA (Electronic Travel Authorization) for {destination_name}. Quick online application!",
        "You'll need to get an ETA online before traveling to {destination_name}. It's a simple online process.",
    ],
    
    "no_admission": [
        "Unfortunately, travel from {origin_name} to {destination_name} is not currently permitted.",
        "I'm sorry, but {origin_name} passport holders cannot currently enter {destination_name}.",
    ],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_template_response(
    template_key: str,
    rotate: bool = True
) -> str:
    """
    Get a template response.
    
    Args:
        template_key: Key in TEMPLATES dict
        rotate: If True, randomly select from options; if False, use first
    
    Returns:
        Template string
    """
    templates = TEMPLATES.get(template_key, ["I'm not sure how to respond to that."])
    
    if rotate and len(templates) > 1:
        return random.choice(templates)
    return templates[0]


def get_welcome_message() -> str:
    """Get the welcome message."""
    return TEMPLATES["welcome"][0]


def get_goodbye_message() -> str:
    """Get a goodbye message."""
    return random.choice(TEMPLATES["goodbye"])


def get_clarification_question(country_name: str) -> str:
    """
    Get a clarification question when a country is ambiguous.
    
    Args:
        country_name: The country name to ask about
    
    Returns:
        Formatted clarification question
    """
    template = random.choice(TEMPLATES["clarify_country"])
    return template.format(country=country_name)


def format_visa_result(
    origin_name: str,
    destination_name: str,
    requirement_type: str,
    days: Optional[int] = None,
    rotate: bool = True
) -> str:
    """
    Format a visa result into a natural response.
    
    Args:
        origin_name: Human-readable origin country name
        destination_name: Human-readable destination country name
        requirement_type: Type of requirement (visa_free, e_visa, visa_required, etc.)
        days: Number of days allowed (if applicable)
        rotate: Randomly select from template options
    
    Returns:
        Formatted response string
    """
    # Normalize requirement type
    type_mapping = {
        'visa free': 'visa_free',
        'visa-free': 'visa_free',
        'visa_free': 'visa_free',
        'e-visa': 'e_visa',
        'evisa': 'e_visa',
        'e_visa': 'e_visa',
        'visa required': 'visa_required',
        'visa_required': 'visa_required',
        'visa on arrival': 'visa_on_arrival',
        'voa': 'visa_on_arrival',
        'visa_on_arrival': 'visa_on_arrival',
        'eta': 'eta',
        'no admission': 'no_admission',
        'no_admission': 'no_admission',
        '-1': 'no_admission',
    }
    
    normalized_type = type_mapping.get(requirement_type.lower(), 'visa_required')
    
    # Handle visa-free with/without days
    if normalized_type == 'visa_free':
        if days and days > 0:
            templates = VISA_RESULT_TEMPLATES['visa_free']
        else:
            templates = VISA_RESULT_TEMPLATES['visa_free_no_days']
    else:
        templates = VISA_RESULT_TEMPLATES.get(normalized_type, VISA_RESULT_TEMPLATES['visa_required'])
    
    # Select template
    if rotate and len(templates) > 1:
        template = random.choice(templates)
    else:
        template = templates[0]
    
    # Format with values
    return template.format(
        origin_name=origin_name,
        destination_name=destination_name,
        days=days or "",
    )


def get_filler_message() -> str:
    """Get a filler message for while waiting."""
    return random.choice(TEMPLATES["filler"])


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Template Responses Test")
    print("=" * 60)
    
    # Test missing info templates
    print("\n1. Missing info templates:")
    print(f"   need_origin: {get_template_response('need_origin')}")
    print(f"   need_destination: {get_template_response('need_destination')}")
    print(f"   need_both: {get_template_response('need_both')}")
    
    # Test visa formatting
    print("\n2. Visa result formatting:")
    print(f"   Visa-free: {format_visa_result('Pakistan', 'Singapore', 'visa_free', 30)}")
    print(f"   E-visa: {format_visa_result('Pakistan', 'UAE', 'e_visa')}")
    print(f"   Required: {format_visa_result('Pakistan', 'UK', 'visa_required')}")
    
    # Test welcome/goodbye
    print("\n3. Welcome/goodbye:")
    print(f"   Welcome: {get_welcome_message()}")
    print(f"   Goodbye: {get_goodbye_message()}")

