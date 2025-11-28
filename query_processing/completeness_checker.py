"""
Query Completeness Checker

Checks if we have enough information to perform a visa query.
Integrates with ConversationState to determine what's missing.

This is YOUR logic - not LLM deciding.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from memory.conversation_state import ConversationState


@dataclass
class CompletenessResult:
    """Result of completeness check."""
    complete: bool
    missing: list
    can_query: bool
    suggestion: str
    clarification_country: str = None  # Country name if clarification needed
    
    def __repr__(self):
        if self.clarification_country:
            return f"CompletenessResult(complete={self.complete}, clarification_needed={self.clarification_country})"
        return f"CompletenessResult(complete={self.complete}, missing={self.missing})"


def check_completeness(state: "ConversationState") -> CompletenessResult:
    """
    Check if we have enough info to query visa requirements.
    
    Args:
        state: Current conversation state
    
    Returns:
        CompletenessResult with:
        - complete: bool (do we have everything?)
        - missing: list of what's missing
        - can_query: bool (can we perform the query?)
        - suggestion: str (what to ask the user, or 'clarify_country' for ambiguous case)
        - clarification_country: str (country name if clarification needed)
    """
    # First check if we're waiting for clarification
    if state.needs_clarification():
        return CompletenessResult(
            complete=False,
            missing=['clarification'],
            can_query=False,
            suggestion="clarify_country",
            clarification_country=state.pending_clarification_name,
        )
    
    missing = state.get_missing()
    
    if not missing:
        # We have everything!
        return CompletenessResult(
            complete=True,
            missing=[],
            can_query=True,
            suggestion=None,
        )
    
    # Determine what to ask
    if 'origin' in missing and 'destination' in missing:
        suggestion = "need_both"
    elif 'origin' in missing:
        suggestion = "need_origin"
    else:  # destination missing
        suggestion = "need_destination"
    
    return CompletenessResult(
        complete=False,
        missing=missing,
        can_query=False,
        suggestion=suggestion,
    )


def check_query_validity(
    origin: str,
    destination: str
) -> Dict:
    """
    Check if a specific query is valid (both countries exist in our data).
    
    Args:
        origin: ISO3 code of origin country
        destination: ISO3 code of destination country
    
    Returns:
        Dict with validity info
    """
    from retrieval.knowledge_graph import ISO3_TO_COUNTRY
    
    origin_valid = origin in ISO3_TO_COUNTRY
    destination_valid = destination in ISO3_TO_COUNTRY
    
    return {
        'valid': origin_valid and destination_valid,
        'origin_valid': origin_valid,
        'destination_valid': destination_valid,
        'origin_name': ISO3_TO_COUNTRY.get(origin),
        'destination_name': ISO3_TO_COUNTRY.get(destination),
    }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Completeness Checker Test")
    print("=" * 60)
    
    state = ConversationState()
    
    # Test 1: Empty state
    result = check_completeness(state)
    print(f"\n1. Empty state:")
    print(f"   Result: {result}")
    print(f"   Suggestion: {result.suggestion}")
    
    # Test 2: Only destination
    state.destination = "ARE"
    state.destination_name = "United Arab Emirates"
    result = check_completeness(state)
    print(f"\n2. Only destination (Dubai):")
    print(f"   Result: {result}")
    print(f"   Suggestion: {result.suggestion}")
    
    # Test 3: Both present
    state.origin = "PAK"
    state.origin_name = "Pakistan"
    result = check_completeness(state)
    print(f"\n3. Both present:")
    print(f"   Result: {result}")
    print(f"   Can query: {result.can_query}")
    
    # Test validity
    print(f"\n4. Query validity check:")
    validity = check_query_validity("PAK", "ARE")
    print(f"   PAK → ARE: {validity}")

