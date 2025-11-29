"""
Travel Agent Chatbot with Knowledge Graph - OPTIMIZED VERSION

This version uses YOUR components to control the flow:
1. Intent Classifier - decides what type of message
2. Conversation State - tracks entities across messages
3. Completeness Checker - determines if we can query
4. Templates - fast responses for predictable cases
5. KG Retrieval - fast graph lookup
6. LLM - ONLY for casual chat and formatting results

Flow:
User Input → Intent Classifier → 
  ├─ Casual → LLM (direct response)
  └─ Visa Query → State Update → Completeness Check →
      ├─ Incomplete → Template Response (NO LLM!)
      └─ Complete → KG Retrieval → LLM (format only)

Supports both BLOCKING and STREAMING modes.
"""

import sys
import time
from typing import Optional, Generator
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Google Generative AI for LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# YOUR components (FYP contribution!)
from query_processing import classify_intent, check_completeness, init_classifier
from query_processing.intent_classifier import is_coming_soon_intent, get_coming_soon_response
from memory.conversation_state import ConversationState
from conversation.templates import (
    get_template_response, 
    get_welcome_message, 
    get_goodbye_message,
    get_clarification_question,
    format_visa_result,
)
from retrieval.knowledge_graph import TravelKnowledgeGraph

load_dotenv()

# ============================================================================
# KNOWLEDGE GRAPH SINGLETON
# ============================================================================

_kg_instance = None

def get_knowledge_graph() -> TravelKnowledgeGraph:
    """Get or create the Knowledge Graph singleton."""
    global _kg_instance
    
    if _kg_instance is None:
        print("Loading Knowledge Graph...")
        _kg_instance = TravelKnowledgeGraph()
        csv_path = Path(__file__).parent.parent / "data" / "dataset" / "passport-index-tidy-iso3.csv"
        _kg_instance.build_from_csv(str(csv_path))
    
    return _kg_instance


# ============================================================================
# LLM SETUP (minimal role)
# ============================================================================

# System prompt - LLM only handles casual chat and formatting
LLM_SYSTEM_PROMPT = """You are James, a friendly travel agent at Rehman Travels.

CONVERSATION STYLE:
- Be warm but direct. No fluff or filler phrases.
- Keep responses SHORT (1-2 sentences max).
- Sound like a real person, not a customer service bot.

HANDLING UNCERTAINTY (when user says "idk", "not sure", etc.):
- If they don't know their DESTINATION: That's fine, they can come back when they've decided.
- If they don't know their NATIONALITY or something you need: Be direct - explain you need that info to help them.
- If they're confused about a question you asked: Rephrase it simply.

HANDLING DISPUTES (when user says "but I heard...", "another agent said...", etc.):
- Acknowledge their concern without dismissing them.
- Stand by your information but suggest they verify with official sources.
- Example: "Our records show you need a visa. Requirements can change - I'd recommend confirming with the embassy."
- Don't argue or repeat the same answer robotically.

FORMATTING VISA RESULTS:
When given visa information, rephrase it naturally and conversationally.
"""


def get_llm():
    """Get the LLM for casual chat and formatting."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.7,
    )


# ============================================================================
# BLOCKING MODE - Returns complete response
# ============================================================================

def process_message(
    user_input: str,
    state: ConversationState,
    llm: ChatGoogleGenerativeAI,
) -> dict:
    """
    Process a user message using YOUR components (BLOCKING mode).
    
    Returns:
        dict with 'response', 'timing', and 'metadata'
    """
    start_time = time.time()
    timing = {}
    
    # Track turns for follow-up window
    state.increment_turn()
    
    # STEP 1: Classify intent
    intent_start = time.time()
    intent = classify_intent(user_input, state.has_visa_context)
    timing['intent_classification'] = (time.time() - intent_start) * 1000
    
    # STEP 2: Update state
    state_start = time.time()
    state.update(user_input, intent)
    timing['state_update'] = (time.time() - state_start) * 1000
    
    # STEP 3: Route based on intent
    
    # Check for "coming soon" features (booking, ticket_change, flight_info)
    if is_coming_soon_intent(intent):
        template_start = time.time()
        response = get_coming_soon_response(intent)
        timing['coming_soon_response'] = (time.time() - template_start) * 1000
        
    elif intent == 'casual':
        response = handle_casual_chat(user_input, state, llm)
        timing['llm_response'] = (time.time() - start_time) * 1000 - timing['intent_classification'] - timing['state_update']
        
    elif intent in ['visa_query', 'follow_up']:
        completeness = check_completeness(state)
        
        if completeness.complete:
            response, retrieval_time = handle_visa_query(state, llm)
            timing['retrieval'] = retrieval_time
        elif completeness.suggestion == 'clarify_country':
            # Ambiguous country - ask for clarification
            template_start = time.time()
            response = get_clarification_question(completeness.clarification_country)
            timing['template_response'] = (time.time() - template_start) * 1000
        elif state.is_in_followup_window():
            # Missing info but we recently answered a query - likely a follow-up question
            # e.g., "do i need visa or evisa?" after asking about Turkey
            # Let LLM handle it with conversation context
            response = handle_casual_chat(user_input, state, llm)
            timing['llm_followup'] = (time.time() - start_time) * 1000
        else:
            template_start = time.time()
            response = get_template_response(completeness.suggestion)
            timing['template_response'] = (time.time() - template_start) * 1000
    else:
        response = handle_casual_chat(user_input, state, llm)
        timing['llm_response'] = (time.time() - start_time) * 1000
    
    # Store response in history
    state.add_response(response)
    
    total_time = (time.time() - start_time) * 1000
    timing['total'] = total_time
    
    return {
        'response': response,
        'timing': timing,
        'metadata': {
            'intent': intent,
            'state': str(state),
            'complete': state.is_complete(),
        }
    }


def handle_casual_chat(user_input: str, state: ConversationState, llm: ChatGoogleGenerativeAI) -> str:
    """Handle casual conversation using LLM (BLOCKING)."""
    messages = _build_llm_messages(user_input, state)
    response = llm.invoke(messages)
    return response.content


def handle_visa_query(state: ConversationState, llm: ChatGoogleGenerativeAI) -> tuple:
    """Handle visa query using Knowledge Graph."""
    params = state.get_query_params()
    kg = get_knowledge_graph()
    
    retrieval_start = time.time()
    result = kg.query(params['origin'], params['destination'])
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if result['found']:
        response = format_visa_result(
            origin_name=params['origin_name'],
            destination_name=params['destination_name'],
            requirement_type=result['requirement_type'],
            days=result.get('days_allowed'),
        )
    else:
        response = get_template_response('error_not_found')
    
    state.reset_query()
    return response, retrieval_time


# ============================================================================
# STREAMING MODE - Yields chunks as they arrive
# ============================================================================

def process_message_stream(
    user_input: str,
    state: ConversationState,
    llm: ChatGoogleGenerativeAI,
) -> Generator[dict, None, None]:
    """
    Process a user message using YOUR components (STREAMING mode).
    
    Yields:
        dict with either:
        - {'chunk': str} for each text chunk
        - {'done': True, 'timing': dict, 'metadata': dict} when complete
    """
    start_time = time.time()
    timing = {}
    
    # Track turns for follow-up window
    state.increment_turn()
    
    # STEP 1: Classify intent
    intent_start = time.time()
    intent = classify_intent(user_input, state.has_visa_context)
    timing['intent_classification'] = (time.time() - intent_start) * 1000
    
    # STEP 2: Update state
    state_start = time.time()
    state.update(user_input, intent)
    timing['state_update'] = (time.time() - state_start) * 1000
    
    full_response = ""
    ttft = None  # Time to first token
    
    # STEP 3: Route based on intent
    
    # Check for "coming soon" features (booking, ticket_change, flight_info)
    if is_coming_soon_intent(intent):
        template_start = time.time()
        response = get_coming_soon_response(intent)
        timing['coming_soon_response'] = (time.time() - template_start) * 1000
        full_response = response
        yield {'chunk': response}
        
    elif intent == 'casual':
        # Stream LLM response
        llm_start = time.time()
        for chunk in handle_casual_chat_stream(user_input, state, llm):
            if ttft is None:
                ttft = (time.time() - llm_start) * 1000
            full_response += chunk
            yield {'chunk': chunk}
        timing['llm_response'] = (time.time() - llm_start) * 1000
        timing['ttft'] = ttft
        
    elif intent in ['visa_query', 'follow_up']:
        completeness = check_completeness(state)
        
        if completeness.complete:
            response, retrieval_time = handle_visa_query(state, llm)
            timing['retrieval'] = retrieval_time
            full_response = response
            # Yield entire response at once (it's from template, very fast)
            yield {'chunk': response}
        elif completeness.suggestion == 'clarify_country':
            # Ambiguous country - ask for clarification
            template_start = time.time()
            response = get_clarification_question(completeness.clarification_country)
            timing['template_response'] = (time.time() - template_start) * 1000
            full_response = response
            yield {'chunk': response}
        elif state.is_in_followup_window():
            # Missing info but we recently answered - likely a follow-up question
            # Stream LLM response with context
            llm_start = time.time()
            for chunk in handle_casual_chat_stream(user_input, state, llm):
                if ttft is None:
                    ttft = (time.time() - llm_start) * 1000
                full_response += chunk
                yield {'chunk': chunk}
            timing['llm_followup'] = (time.time() - llm_start) * 1000
            timing['ttft'] = ttft
        else:
            template_start = time.time()
            response = get_template_response(completeness.suggestion)
            timing['template_response'] = (time.time() - template_start) * 1000
            full_response = response
            yield {'chunk': response}
    else:
        # Stream LLM response
        llm_start = time.time()
        for chunk in handle_casual_chat_stream(user_input, state, llm):
            if ttft is None:
                ttft = (time.time() - llm_start) * 1000
            full_response += chunk
            yield {'chunk': chunk}
        timing['llm_response'] = (time.time() - llm_start) * 1000
        timing['ttft'] = ttft
    
    # Store response in history
    state.add_response(full_response)
    
    total_time = (time.time() - start_time) * 1000
    timing['total'] = total_time
    
    # Yield final metadata
    yield {
        'done': True,
        'timing': timing,
        'metadata': {
            'intent': intent,
            'state': str(state),
            'complete': state.is_complete(),
        }
    }


def handle_casual_chat_stream(
    user_input: str, 
    state: ConversationState, 
    llm: ChatGoogleGenerativeAI
) -> Generator[str, None, None]:
    """
    Handle casual conversation using LLM (STREAMING).
    Yields chunks as they arrive from the LLM.
    """
    messages = _build_llm_messages(user_input, state)
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def _build_llm_messages(user_input: str, state: ConversationState) -> list:
    """Build the message list for LLM (shared by blocking and streaming)."""
    messages = [
        SystemMessage(content=LLM_SYSTEM_PROMPT),
    ]
    
    # Add conversation history (rolling window)
    history = state.get_conversation_history(max_turns=5)
    for entry in history:
        if entry['role'] == 'user':
            messages.append(HumanMessage(content=entry['content']))
        else:
            messages.append(AIMessage(content=entry['content']))
    
    # Add current message
    messages.append(HumanMessage(content=user_input))
    
    return messages


# ============================================================================
# INTERACTIVE CHATBOT
# ============================================================================

def run_kg_chatbot_interactive(show_timing: bool = True, stream: bool = False):
    """
    Run the Knowledge Graph chatbot in interactive mode.
    
    Args:
        show_timing: Whether to show timing information
        stream: Whether to use streaming mode (shows text as it's generated)
    """
    mode = "STREAMING" if stream else "BLOCKING"
    print("=" * 70)
    print(f"Travel Agent Chatbot with KNOWLEDGE GRAPH ({mode})")
    print("=" * 70)
    print("YOUR components control the flow - LLM only for casual chat")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing...")
    init_classifier()  # Pre-load semantic intent classifier
    kg = get_knowledge_graph()
    llm = get_llm()
    state = ConversationState()
    print()
    
    # Welcome message
    print(f"Travel Agent: {get_welcome_message()}")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "end"]:
                print(f"\n{get_goodbye_message()}")
                break
            
            if not user_input:
                continue
            
            if stream:
                # STREAMING MODE
                print("Travel Agent: ", end="", flush=True)
                timing = {}
                
                for result in process_message_stream(user_input, state, llm):
                    if 'chunk' in result:
                        print(result['chunk'], end="", flush=True)
                    elif 'done' in result:
                        timing = result['timing']
                
                print()  # Newline after streaming
                
                # Show timing
                if show_timing:
                    timing_parts = []
                    if 'ttft' in timing:
                        timing_parts.append(f"TTFT: {timing['ttft']:.0f}ms")
                    if 'retrieval' in timing:
                        timing_parts.append(f"KG: {timing['retrieval']:.2f}ms")
                    if 'template_response' in timing:
                        timing_parts.append(f"template: {timing['template_response']:.1f}ms")
                    if 'llm_response' in timing:
                        timing_parts.append(f"LLM: {timing['llm_response']:.0f}ms")
                    
                    timing_str = ", ".join(timing_parts)
                    print(f"[Total: {timing['total']:.0f}ms | {timing_str}]")
            else:
                # BLOCKING MODE
                result = process_message(user_input, state, llm)
                print(f"Travel Agent: {result['response']}")
                
                if show_timing:
                    timing = result['timing']
                    timing_parts = []
                    
                    if 'intent_classification' in timing:
                        timing_parts.append(f"intent: {timing['intent_classification']:.1f}ms")
                    if 'retrieval' in timing:
                        timing_parts.append(f"KG: {timing['retrieval']:.2f}ms")
                    if 'template_response' in timing:
                        timing_parts.append(f"template: {timing['template_response']:.1f}ms")
                    if 'llm_response' in timing:
                        timing_parts.append(f"LLM: {timing['llm_response']:.0f}ms")
                    
                    timing_str = ", ".join(timing_parts)
                    print(f"[Total: {timing['total']:.0f}ms | {timing_str}]")
            
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{get_goodbye_message()}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again.\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KG Chatbot")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--no-timing", action="store_true", help="Hide timing info")
    args = parser.parse_args()
    
    run_kg_chatbot_interactive(show_timing=not args.no_timing, stream=args.stream)
