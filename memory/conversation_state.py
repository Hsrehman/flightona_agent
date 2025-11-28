"""
Conversation State - Tracks entities ACROSS messages

This is YOUR component (not LLM) that maintains state across the conversation.
It tracks:
- Origin country (nationality/passport)
- Destination country
- Conversation history for context

This allows us to:
1. Not call the retrieval tool until we have BOTH pieces of info
2. Remember context from earlier messages
3. Skip LLM for "need more info" responses
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_processing.entity_extractor import extract_countries_from_text


@dataclass
class ConversationState:
    """
    Tracks conversation state across multiple messages.
    
    This is YOUR state management - not LLM memory.
    Fast, predictable, testable.
    """
    
    # Core visa query info
    origin: Optional[str] = None          # ISO3 code (e.g., "PAK")
    origin_name: Optional[str] = None     # Human name (e.g., "Pakistan")
    destination: Optional[str] = None     # ISO3 code (e.g., "SGP")
    destination_name: Optional[str] = None  # Human name (e.g., "Singapore")
    
    # Conversation context
    has_visa_context: bool = False        # Are we in a visa-related conversation?
    message_count: int = 0                # Number of messages processed
    last_intent: Optional[str] = None     # Last classified intent
    
    # Pending clarification - when a country is ambiguous
    pending_clarification: Optional[str] = None      # ISO3 code of ambiguous country
    pending_clarification_name: Optional[str] = None # Name of ambiguous country
    
    # History for debugging/evaluation
    history: List[Dict] = field(default_factory=list)
    
    # Rolling window size for LLM context
    max_history_for_llm: int = 5
    
    def add_response(self, response: str):
        """Add the bot's response to the last history entry."""
        if self.history:
            self.history[-1]['response'] = response
    
    def get_conversation_history(self, max_turns: int = None) -> List[Dict]:
        """
        Get recent conversation history for LLM context.
        
        This implements the ROLLING WINDOW memory - only the last N turns.
        
        Args:
            max_turns: Max number of turns to return. If None, uses max_history_for_llm
        
        Returns:
            List of {'role': 'user'/'assistant', 'content': str}
        """
        if max_turns is None:
            max_turns = self.max_history_for_llm
        
        # Get last N history entries that have both message and response
        recent = []
        for entry in self.history[-(max_turns * 2):]:  # Get more, then filter
            if 'message' in entry:
                recent.append({'role': 'user', 'content': entry['message']})
            if 'response' in entry:
                recent.append({'role': 'assistant', 'content': entry['response']})
        
        # Return only the last max_turns * 2 messages (user + assistant)
        return recent[-(max_turns * 2):]
    
    def update(self, message: str, intent: str = None) -> Dict:
        """
        Update state based on a new message.
        Extracts entities and updates tracking.
        
        Args:
            message: The user's message
            intent: Pre-classified intent (optional)
        
        Returns:
            Dict with what was extracted/updated, including:
            - 'needs_clarification': True if we need to ask which the country is
            - 'clarification_country': The country name to ask about
        """
        self.message_count += 1
        
        # First, check if this is a response to a pending clarification
        if self.pending_clarification:
            clarification_result = self._handle_clarification_response(message)
            if clarification_result:
                # Update context flags
                if intent:
                    self.last_intent = intent
                    if intent in ['visa_query', 'follow_up']:
                        self.has_visa_context = True
                
                # Track history
                self.history.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message,
                    'intent': intent,
                    'origin': self.origin,
                    'destination': self.destination,
                    'clarification_resolved': True,
                })
                
                return clarification_result
        
        # Extract entities from this message
        extracted = extract_countries_from_text(message)
        
        updates = {
            'message': message,
            'extracted': extracted,
            'origin_updated': False,
            'destination_updated': False,
            'needs_clarification': False,
        }
        
        new_origin = extracted.get('origin')
        new_destination = extracted.get('destination')
        
        # CASE 1: Both origin and destination found clearly
        if new_origin and new_destination:
            self.origin = new_origin
            self.origin_name = extracted.get('origin_name', new_origin)
            self.destination = new_destination
            self.destination_name = extracted.get('destination_name', new_destination)
            updates['origin_updated'] = True
            updates['destination_updated'] = True
        
        # CASE 2: We already have origin, only destination missing, and got ambiguous country
        elif self.origin and not self.destination and new_origin and not new_destination:
            # User probably meant destination
            self.destination = new_origin
            self.destination_name = extracted.get('origin_name', new_origin)
            updates['destination_updated'] = True
        
        # CASE 3: We already have destination, only origin missing, and got ambiguous country
        elif not self.origin and self.destination and new_origin and not new_destination:
            # User probably meant origin
            self.origin = new_origin
            self.origin_name = extracted.get('origin_name', new_origin)
            updates['origin_updated'] = True
        
        # CASE 4: We have NOTHING and got only one country
        elif not self.origin and not self.destination and new_origin and not new_destination:
            # Check if this is a nationality word (e.g., "pakistani", "american")
            # Nationality words are NOT ambiguous - they're clearly the origin!
            origin_is_nationality = extracted.get('origin_is_nationality', False)
            
            if origin_is_nationality:
                # User said a nationality word - it's definitely their origin
                self.origin = new_origin
                self.origin_name = extracted.get('origin_name', new_origin)
                updates['origin_updated'] = True
            elif self.has_visa_context:
                # In visa context with ambiguous country name (e.g., "dubai") - ask for clarification
                self.pending_clarification = new_origin
                self.pending_clarification_name = extracted.get('origin_name', new_origin)
                updates['needs_clarification'] = True
                updates['clarification_country'] = self.pending_clarification_name
            else:
                # Not in visa context, just store as origin (default)
                self.origin = new_origin
                self.origin_name = extracted.get('origin_name', new_origin)
                updates['origin_updated'] = True
        
        # CASE 5: Clear destination found
        elif new_destination and not new_origin:
            self.destination = new_destination
            self.destination_name = extracted.get('destination_name', new_destination)
            updates['destination_updated'] = True
        
        # CASE 6: Clear origin found
        elif new_origin and not new_destination:
            self.origin = new_origin
            self.origin_name = extracted.get('origin_name', new_origin)
            updates['origin_updated'] = True
        
        # Update context flags
        if intent:
            self.last_intent = intent
            if intent in ['visa_query', 'follow_up']:
                self.has_visa_context = True
        
        # Track history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'intent': intent,
            'origin': self.origin,
            'destination': self.destination,
            'extracted': extracted,
            'needs_clarification': updates.get('needs_clarification', False),
        })
        
        return updates
    
    def _handle_clarification_response(self, message: str) -> Optional[Dict]:
        """
        Handle a response to a clarification question.
        
        Args:
            message: The user's response to "Is X your nationality or destination?"
        
        Returns:
            Dict with updates, or None if not a valid clarification response
        """
        message_lower = message.lower().strip()
        
        updates = {
            'message': message,
            'origin_updated': False,
            'destination_updated': False,
            'clarification_resolved': True,
        }
        
        country_code = self.pending_clarification
        country_name = self.pending_clarification_name
        
        # FIRST: Check if user provided BOTH pieces of info in their response
        # (This must come before indicator checks!)
        extracted = extract_countries_from_text(message)
        if extracted.get('origin') and extracted.get('destination'):
            self.origin = extracted['origin']
            self.origin_name = extracted.get('origin_name', extracted['origin'])
            self.destination = extracted['destination']
            self.destination_name = extracted.get('destination_name', extracted['destination'])
            updates['origin_updated'] = True
            updates['destination_updated'] = True
            self.pending_clarification = None
            self.pending_clarification_name = None
            return updates
        
        # Check for clear indicators
        is_origin_indicators = [
            'nationality', 'citizen', 'passport', 'i am from', 
            "i'm from", 'my nationality', 'where i am from', 'my country'
        ]
        is_destination_indicators = [
            'destination', 'going to', 'travel to', 'visit', 
            'want to go', 'where i want', 'traveling to'
        ]
        
        # Check if user said it's their nationality/origin
        for indicator in is_origin_indicators:
            if indicator in message_lower:
                self.origin = country_code
                self.origin_name = country_name
                updates['origin_updated'] = True
                self.pending_clarification = None
                self.pending_clarification_name = None
                return updates
        
        # Check if user said it's their destination
        for indicator in is_destination_indicators:
            if indicator in message_lower:
                self.destination = country_code
                self.destination_name = country_name
                updates['destination_updated'] = True
                self.pending_clarification = None
                self.pending_clarification_name = None
                return updates
        
        # Check for simple "origin" or "destination" response
        if message_lower in ['origin', 'nationality', 'from there', 'from']:
            self.origin = country_code
            self.origin_name = country_name
            updates['origin_updated'] = True
            self.pending_clarification = None
            self.pending_clarification_name = None
            return updates
        
        if message_lower in ['destination', 'to', 'going there', 'there']:
            self.destination = country_code
            self.destination_name = country_name
            updates['destination_updated'] = True
            self.pending_clarification = None
            self.pending_clarification_name = None
            return updates
        
        # If we can't determine from the clarification response, 
        # clear the pending and let normal update flow handle it
        self.pending_clarification = None
        self.pending_clarification_name = None
        return None
    
    def is_complete(self) -> bool:
        """
        Check if we have all info needed to query visa requirements.
        
        Returns:
            True if we have both origin and destination
        """
        return self.origin is not None and self.destination is not None
    
    def get_missing(self) -> List[str]:
        """
        Get list of missing information.
        
        Returns:
            List of missing fields: ['origin'], ['destination'], or ['origin', 'destination']
        """
        missing = []
        if self.origin is None:
            missing.append('origin')
        if self.destination is None:
            missing.append('destination')
        return missing
    
    def get_query_params(self) -> Optional[Dict]:
        """
        Get parameters for a visa query if complete.
        
        Returns:
            Dict with origin/destination ISO3 codes, or None if incomplete
        """
        if not self.is_complete():
            return None
        
        return {
            'origin': self.origin,
            'origin_name': self.origin_name,
            'destination': self.destination,
            'destination_name': self.destination_name,
        }
    
    def reset_query(self, keep_origin: bool = True):
        """
        Reset the current query for a new destination query.
        
        Args:
            keep_origin: If True, keeps nationality for follow-up queries.
                        e.g., "What about France?" after asking about Dubai.
        """
        if not keep_origin:
            self.origin = None
            self.origin_name = None
        # Always clear destination for new query
        self.destination = None
        self.destination_name = None
        self.pending_clarification = None
        self.pending_clarification_name = None
        # Keep has_visa_context True since user is still asking about visas
    
    def reset_all(self):
        """Full reset - new conversation."""
        self.origin = None
        self.origin_name = None
        self.destination = None
        self.destination_name = None
        self.pending_clarification = None
        self.pending_clarification_name = None
        self.has_visa_context = False
        self.message_count = 0
        self.last_intent = None
        self.history = []
    
    def needs_clarification(self) -> bool:
        """Check if we're waiting for clarification about a country."""
        return self.pending_clarification is not None
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of current state."""
        parts = []
        
        if self.origin_name:
            parts.append(f"Nationality: {self.origin_name}")
        if self.destination_name:
            parts.append(f"Destination: {self.destination_name}")
        
        if not parts:
            return "No travel info collected yet"
        
        return ", ".join(parts)
    
    def __repr__(self):
        return (
            f"ConversationState("
            f"origin={self.origin_name or self.origin}, "
            f"destination={self.destination_name or self.destination}, "
            f"complete={self.is_complete()}, "
            f"messages={self.message_count})"
        )


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Conversation State Test")
    print("=" * 60)
    
    state = ConversationState()
    
    # Simulate a conversation
    messages = [
        ("hi", "casual"),
        ("do i need a visa", "visa_query"),
        ("i want to go to dubai", "follow_up"),
        ("i'm pakistani", "follow_up"),
    ]
    
    for msg, intent in messages:
        print(f"\nUser: {msg}")
        updates = state.update(msg, intent)
        print(f"  State: {state}")
        print(f"  Complete: {state.is_complete()}")
        print(f"  Missing: {state.get_missing()}")
        
        if state.is_complete():
            params = state.get_query_params()
            print(f"  Ready to query: {params}")

