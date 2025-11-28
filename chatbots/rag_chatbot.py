"""
Travel Agent Chatbot with RAG (Vector Search) - OPTIMIZED VERSION

This version uses YOUR components to control the flow:
1. Intent Classifier - decides what type of message
2. Conversation State - tracks entities across messages
3. Completeness Checker - determines if we can query
4. Templates - fast responses for predictable cases
5. RAG Retrieval - semantic vector search
6. LLM - ONLY for casual chat and formatting results

Flow:
User Input → Intent Classifier → 
  ├─ Casual → LLM (direct response)
  └─ Visa Query → State Update → Completeness Check →
      ├─ Incomplete → Template Response (NO LLM!)
      └─ Complete → RAG Retrieval → LLM (format only)

NOTE: This is IDENTICAL to kg_chatbot.py except for the retrieval method.
      This allows fair performance comparison between KG and RAG.

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
from query_processing import classify_intent, check_completeness
from memory.conversation_state import ConversationState
from conversation.templates import (
    get_template_response, 
    get_welcome_message, 
    get_goodbye_message,
    get_clarification_question,
    format_visa_result,
)
from retrieval.rag_retriever import create_visa_knowledge_base

load_dotenv()

# ============================================================================
# RAG RETRIEVER SINGLETON
# ============================================================================

_rag_retriever = None

def get_rag_retriever():
    """Get or create the RAG retriever singleton."""
    global _rag_retriever
    
    if _rag_retriever is None:
        print("Loading RAG retriever...")
        _, _rag_retriever = create_visa_knowledge_base(force_recreate=False)
        print("RAG retriever loaded!")
    
    return _rag_retriever


# ============================================================================
# LLM SETUP (minimal role)
# ============================================================================

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
    
    # STEP 1: Classify intent
    intent_start = time.time()
    intent = classify_intent(user_input, state.has_visa_context)
    timing['intent_classification'] = (time.time() - intent_start) * 1000
    
    # STEP 2: Update state
    state_start = time.time()
    state.update(user_input, intent)
    timing['state_update'] = (time.time() - state_start) * 1000
    
    # STEP 3: Route based on intent
    if intent == 'casual':
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
    """Handle visa query using RAG (Vector Search)."""
    params = state.get_query_params()
    retriever = get_rag_retriever()
    
    search_query = f"visa requirements for {params['origin_name']} citizens traveling to {params['destination_name']}"
    
    retrieval_start = time.time()
    docs = retriever.invoke(search_query)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if docs:
        result_text = docs[0].page_content
        requirement_type = parse_requirement_from_rag(result_text)
        days = parse_days_from_rag(result_text)
        
        response = format_visa_result(
            origin_name=params['origin_name'],
            destination_name=params['destination_name'],
            requirement_type=requirement_type,
            days=days,
        )
    else:
        response = get_template_response('error_not_found')
    
    state.reset_query()
    return response, retrieval_time


def parse_requirement_from_rag(text: str) -> str:
    """Parse requirement type from RAG result text."""
    text_lower = text.lower()
    
    if 'visa-free' in text_lower or 'visa free' in text_lower:
        return 'visa_free'
    elif 'e-visa' in text_lower or 'evisa' in text_lower:
        return 'e_visa'
    elif 'visa on arrival' in text_lower or 'voa' in text_lower:
        return 'visa_on_arrival'
    elif 'eta' in text_lower or 'electronic travel' in text_lower:
        return 'eta'
    elif 'no admission' in text_lower or 'not permitted' in text_lower:
        return 'no_admission'
    else:
        return 'visa_required'


def parse_days_from_rag(text: str) -> Optional[int]:
    """Try to extract number of days from RAG result."""
    import re
    match = re.search(r'(\d+)\s*days?', text.lower())
    if match:
        return int(match.group(1))
    return None


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
    if intent == 'casual':
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
            yield {'chunk': response}
        elif completeness.suggestion == 'clarify_country':
            # Ambiguous country - ask for clarification
            template_start = time.time()
            response = get_clarification_question(completeness.clarification_country)
            timing['template_response'] = (time.time() - template_start) * 1000
            full_response = response
            yield {'chunk': response}
        else:
            template_start = time.time()
            response = get_template_response(completeness.suggestion)
            timing['template_response'] = (time.time() - template_start) * 1000
            full_response = response
            yield {'chunk': response}
    else:
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
    """Handle casual conversation using LLM (STREAMING)."""
    messages = _build_llm_messages(user_input, state)
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def _build_llm_messages(user_input: str, state: ConversationState) -> list:
    """Build the message list for LLM (shared by blocking and streaming)."""
    messages = [
        SystemMessage(content=LLM_SYSTEM_PROMPT),
    ]
    
    history = state.get_conversation_history(max_turns=5)
    for entry in history:
        if entry['role'] == 'user':
            messages.append(HumanMessage(content=entry['content']))
        else:
            messages.append(AIMessage(content=entry['content']))
    
    messages.append(HumanMessage(content=user_input))
    
    return messages


# ============================================================================
# INTERACTIVE CHATBOT
# ============================================================================

def run_rag_chatbot_interactive(show_timing: bool = True, stream: bool = False):
    """
    Run the RAG chatbot in interactive mode.
    
    Args:
        show_timing: Whether to show timing information
        stream: Whether to use streaming mode
    """
    mode = "STREAMING" if stream else "BLOCKING"
    print("=" * 70)
    print(f"Travel Agent Chatbot with RAG ({mode})")
    print("=" * 70)
    print("YOUR components control the flow - LLM only for casual chat")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 70)
    print()
    
    print("Initializing...")
    retriever = get_rag_retriever()
    llm = get_llm()
    state = ConversationState()
    print()
    
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
                
                print()
                
                if show_timing:
                    timing_parts = []
                    if 'ttft' in timing:
                        timing_parts.append(f"TTFT: {timing['ttft']:.0f}ms")
                    if 'retrieval' in timing:
                        timing_parts.append(f"RAG: {timing['retrieval']:.1f}ms")
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
                        timing_parts.append(f"RAG: {timing['retrieval']:.1f}ms")
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
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--no-timing", action="store_true", help="Hide timing info")
    args = parser.parse_args()
    
    run_rag_chatbot_interactive(show_timing=not args.no_timing, stream=args.stream)
