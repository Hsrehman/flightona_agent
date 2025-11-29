#!/usr/bin/env python3
"""
Comprehensive Test Suite for Travel Agent Chatbot
==================================================

Tests all components we've built:
- Intent classification (semantic + exact match)
- Entity extraction (with fuzzy matching for typos)
- Conversation state management
- Follow-up window logic
- Full pipeline (KG chatbot)

Run with: python -m tests.test_comprehensive
"""

import sys
import time
import os

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_processing.intent_classifier import classify_intent, get_intent_confidence, init_classifier
from query_processing.entity_extractor import extract_countries_from_text
from memory.conversation_state import ConversationState
from chatbots.kg_chatbot import process_message, get_knowledge_graph, get_llm


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print('='*80)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n{'-'*60}")
    print(f" {title}")
    print('-'*60)


def test_intent_classification():
    """Test the intent classifier with various inputs."""
    print_header("INTENT CLASSIFICATION TESTS")
    
    test_categories = {
        'CASUAL - Greetings': [
            ('hi', False, 'casual'),
            ('hello', False, 'casual'),
            ('hey there', False, 'casual'),
            ('good morning', False, 'casual'),
            ('whats up', False, 'casual'),
        ],
        
        'CASUAL - Farewells': [
            ('bye', False, 'casual'),
            ('goodbye', False, 'casual'),
            ('see you later', False, 'casual'),
            ('take care', False, 'casual'),
        ],
        
        'CASUAL - Thanks': [
            ('thanks', False, 'casual'),
            ('thank you', False, 'casual'),
            ('appreciate it', False, 'casual'),
            ('thats helpful', False, 'casual'),
        ],
        
        'CASUAL - Complaints': [
            ('you are the worst', False, 'casual'),
            ('terrible service', False, 'casual'),
            ('i heard you are bad', False, 'casual'),
            ('worst travel agency ever', False, 'casual'),
            ('your competitor is better', False, 'casual'),
        ],
        
        'CASUAL - Help requests (NOT visa)': [
            ('i want to create a app for visa applications can u help me', False, 'casual'),
            ('can you help me build something', False, 'casual'),
            ('i need help with my project', False, 'casual'),
            ('how do i make a website', False, 'casual'),
            ('can you code for me', False, 'casual'),
        ],
        
        'CASUAL - Random questions': [
            ('whats your name', False, 'casual'),
            ('who made you', False, 'casual'),
            ('are you a robot', False, 'casual'),
            ('how does this work', False, 'casual'),
            ('ur ahmed right', False, 'casual'),
            ('is this real', False, 'casual'),
        ],
        
        'CASUAL - Uncertainty': [
            ('idk', True, 'casual'),
            ('not sure', True, 'casual'),
            ('i dont know', True, 'casual'),
            ('havent decided yet', True, 'casual'),
            ('maybe later', True, 'casual'),
            ('let me think', True, 'casual'),
        ],
        
        'CASUAL - Disputes': [
            ('but i heard i dont need visa', True, 'casual'),
            ('another agent said different', True, 'casual'),
            ('are you sure about that', True, 'casual'),
            ('i dont think thats right', True, 'casual'),
            ('but i was told otherwise', True, 'casual'),
            ('thats wrong', True, 'casual'),
        ],
        
        'CASUAL - Weird follow-up questions with countries': [
            ('is pakistan a good country to live in', True, 'casual'),
            ('why is dubai so expensive', True, 'casual'),
            ('i heard japan has good food', True, 'casual'),
            ('what is the capital of france', True, 'casual'),
            ('tell me about singapore history', True, 'casual'),
        ],
        
        'VISA_QUERY - Direct questions': [
            ('do i need a visa', False, 'visa_query'),
            ('visa requirements', False, 'visa_query'),
            ('what visa do i need', False, 'visa_query'),
            ('is visa required', False, 'visa_query'),
            ('can i travel without visa', False, 'visa_query'),
        ],
        
        'VISA_QUERY - Travel intent': [
            ('i want to go to dubai', False, 'visa_query'),
            ('planning to visit singapore', False, 'visa_query'),
            ('going to japan', False, 'visa_query'),
            ('trip to thailand', False, 'visa_query'),
            ('traveling to france next month', False, 'visa_query'),
        ],
        
        'VISA_QUERY - Requirement questions': [
            ('what do i need to enter usa', False, 'visa_query'),
            ('entry requirements for uk', False, 'visa_query'),
            ('can pakistani visit dubai', False, 'visa_query'),
            ('do americans need visa for japan', False, 'visa_query'),
        ],
        
        'FOLLOW_UP - Short nationality (with context)': [
            ('pakistani', True, 'follow_up'),
            ('indian', True, 'follow_up'),
            ('american', True, 'follow_up'),
            ('british', True, 'follow_up'),
        ],
        
        'FOLLOW_UP - Longer nationality (with context)': [
            ('im pakistani', True, 'follow_up'),
            ('i am from pakistan', True, 'follow_up'),
            ('indian passport', True, 'follow_up'),
            ('american citizen', True, 'follow_up'),
            ('yes i am a citizen of dubai', True, 'follow_up'),
        ],
        
        'FOLLOW_UP - Short destination (with context)': [
            ('dubai', True, 'follow_up'),
            ('singapore', True, 'follow_up'),
            ('japan', True, 'follow_up'),
            ('turkey', True, 'follow_up'),
        ],
        
        'FOLLOW_UP - What about questions (with context)': [
            ('what about turkey', True, 'follow_up'),
            ('and for germany', True, 'follow_up'),
            ('how about usa', True, 'follow_up'),
        ],
        
        'FOLLOW_UP - Typos (with context) - CRITICAL FIX': [
            ('pakisatni', True, 'follow_up'),
            ('dubaii', True, 'follow_up'),
            ('singapoer', True, 'follow_up'),
            ('turky', True, 'follow_up'),
        ],
        
        'EDGE CASE - Nonsense': [
            ('asdfghjkl', False, 'casual'),
            ('lol wut', False, 'casual'),
            ('???', False, 'casual'),
            ('hmmm', False, 'casual'),
            ('123456', False, 'casual'),
        ],
        
        'EDGE CASE - Mixed intent': [
            ('hi i want to go to dubai', False, 'visa_query'),
            ('ok what about singapore', True, 'follow_up'),
        ],
    }
    
    total = 0
    passed = 0
    failed_cases = []
    
    for category, test_cases in test_categories.items():
        print_subheader(category)
        
        for text, has_context, expected in test_cases:
            result = classify_intent(text, has_context)
            details = get_intent_confidence(text)
            
            status = "✅" if result == expected else "❌"
            ctx = "✓" if has_context else "✗"
            
            print(f"  {status} [{ctx}] {details['confidence']:.2f} | {result:<12} | \"{text}\"")
            
            if result == expected:
                passed += 1
            else:
                failed_cases.append((text, has_context, expected, result))
            
            total += 1
    
    print_subheader("INTENT CLASSIFICATION SUMMARY")
    print(f"  Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if failed_cases:
        print(f"\n  FAILED CASES:")
        for text, ctx, expected, got in failed_cases:
            print(f"    - \"{text}\" (ctx={ctx}): expected {expected}, got {got}")
    
    return passed, total


def test_entity_extraction():
    """Test the entity extractor with various inputs including typos."""
    print_header("ENTITY EXTRACTION TESTS")
    
    test_cases = [
        # (input, expected_origin, expected_destination)
        ("im pakistani", "Pakistan", None),
        ("i am from pakistan", "Pakistan", None),
        ("going to dubai", None, "United Arab Emirates"),
        ("im pakistani going to dubai", "Pakistan", "United Arab Emirates"),
        ("from india to japan", "India", "Japan"),
        
        # Typos - CRITICAL
        ("pakisatni", "Pakistan", None),
        ("dubaii", None, "United Arab Emirates"),
        ("singapoer", None, "Singapore"),
        ("turky", None, "Turkey"),
        ("im pakisatni going to dubaii", "Pakistan", "United Arab Emirates"),
        
        # Nationalities
        ("indian passport", "India", None),
        ("american citizen", "United States", None),
        ("british national", "United Kingdom", None),
        
        # Complex
        ("i want to travel from pakistan to singapore via dubai", "Pakistan", "Singapore"),
    ]
    
    passed = 0
    total = 0
    
    for text, exp_origin, exp_dest in test_cases:
        entities = extract_countries_from_text(text)
        origin = entities.get('origin')
        dest = entities.get('destination')
        
        origin_ok = origin == exp_origin
        dest_ok = dest == exp_dest
        status = "✅" if origin_ok and dest_ok else "❌"
        
        print(f"  {status} \"{text}\"")
        print(f"      Origin: {origin} (expected: {exp_origin}) {'✓' if origin_ok else '✗'}")
        print(f"      Dest:   {dest} (expected: {exp_dest}) {'✓' if dest_ok else '✗'}")
        
        if origin_ok and dest_ok:
            passed += 1
        total += 1
    
    print_subheader("ENTITY EXTRACTION SUMMARY")
    print(f"  Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    return passed, total


def test_conversation_state():
    """Test conversation state management."""
    print_header("CONVERSATION STATE TESTS")
    
    passed = 0
    total = 0
    
    # Test 1: Basic flow
    print_subheader("Test 1: Basic visa query flow")
    state = ConversationState()
    
    state.update("im pakistani")
    test1a = state.origin == "Pakistan"
    print(f"  {'✅' if test1a else '❌'} After 'im pakistani': origin={state.origin}")
    
    state.update("going to dubai")
    test1b = state.destination == "United Arab Emirates"
    print(f"  {'✅' if test1b else '❌'} After 'going to dubai': dest={state.destination}")
    
    test1c = state.is_complete()
    print(f"  {'✅' if test1c else '❌'} State is complete: {state.is_complete()}")
    
    passed += sum([test1a, test1b, test1c])
    total += 3
    
    # Test 2: Typos
    print_subheader("Test 2: Typos handled")
    state = ConversationState()
    
    state.update("pakisatni")
    test2a = state.origin == "Pakistan"
    print(f"  {'✅' if test2a else '❌'} After 'pakisatni': origin={state.origin}")
    
    state.update("dubaii")
    test2b = state.destination == "United Arab Emirates"
    print(f"  {'✅' if test2b else '❌'} After 'dubaii': dest={state.destination}")
    
    passed += sum([test2a, test2b])
    total += 2
    
    # Test 3: Follow-up window
    print_subheader("Test 3: Follow-up window")
    state = ConversationState()
    state.origin = "Pakistan"
    state.destination = "United Arab Emirates"
    state.reset_query(keep_origin=True)  # Simulate answering a query
    
    test3a = state.is_in_followup_window()
    print(f"  {'✅' if test3a else '❌'} In follow-up window after reset: {state.is_in_followup_window()}")
    
    test3b = state.origin == "Pakistan"
    print(f"  {'✅' if test3b else '❌'} Origin retained: {state.origin}")
    
    test3c = state.last_answered_destination == "United Arab Emirates"
    print(f"  {'✅' if test3c else '❌'} Last answered dest stored: {state.last_answered_destination}")
    
    passed += sum([test3a, test3b, test3c])
    total += 3
    
    # Test 4: Follow-up window expires
    print_subheader("Test 4: Follow-up window expiration")
    state = ConversationState()
    state.origin = "Pakistan"
    state.destination = "Turkey"
    state.reset_query(keep_origin=True)
    
    # Simulate 3 turns
    state.turns_since_answer = 3
    test4a = not state.is_in_followup_window()
    print(f"  {'✅' if test4a else '❌'} Window expired after 3 turns: {not state.is_in_followup_window()}")
    
    passed += 1 if test4a else 0
    total += 1
    
    print_subheader("CONVERSATION STATE SUMMARY")
    print(f"  Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    return passed, total


def test_full_pipeline():
    """Test the full chatbot pipeline with various scenarios."""
    print_header("FULL PIPELINE TESTS (KG Chatbot)")
    
    kg = get_knowledge_graph()
    llm = get_llm()
    
    scenarios = [
        {
            'name': '1. Help request (NOT visa) - CRITICAL FIX',
            'messages': [
                ('i want to create a app for visa applications can u help me', 'casual', 'LLM'),
            ]
        },
        {
            'name': '2. Normal visa query flow',
            'messages': [
                ('do i need a visa', 'visa_query', 'template'),
                ('pakistani', 'follow_up', 'template'),
                ('dubai', 'follow_up', 'retrieval'),
            ]
        },
        {
            'name': '3. Typos - CRITICAL FIX',
            'messages': [
                ('do i need visaa', 'visa_query', 'template'),
                ('pakisatni', 'follow_up', 'template'),
                ('dubaii', 'follow_up', 'retrieval'),
            ]
        },
        {
            'name': '4. Casual chat',
            'messages': [
                ('hi how are you', 'casual', 'LLM'),
                ('whats your name', 'casual', 'LLM'),
                ('are you a robot', 'casual', 'LLM'),
            ]
        },
        {
            'name': '5. Disputes after answer',
            'messages': [
                ('im pakistani going to turkey', 'visa_query', 'retrieval'),
                ('but i heard i dont need visa', 'casual', 'LLM'),
                ('are you sure about that', 'casual', 'LLM'),
            ]
        },
        {
            'name': '6. Uncertainty',
            'messages': [
                ('do i need a visa', 'visa_query', 'template'),
                ('pakistani', 'follow_up', 'template'),
                ('idk', 'casual', 'LLM'),
                ('not sure yet', 'casual', 'LLM'),
            ]
        },
        {
            'name': '7. Follow-up questions',
            'messages': [
                ('im pakistani and want to visit dubai', 'visa_query', 'retrieval'),
                ('what about turkey', 'follow_up', 'retrieval'),
                ('do i need evisa or regular visa', 'casual', 'LLM'),  # Follow-up in window
            ]
        },
        {
            'name': '8. Weird questions with country names',
            'messages': [
                ('im pakistani', 'follow_up', 'template'),
                ('is pakistan a good country to live in', 'casual', 'LLM'),
                ('why is dubai so expensive', 'casual', 'LLM'),
            ]
        },
        {
            'name': '9. Nonsense inputs',
            'messages': [
                ('asdfghjkl', 'casual', 'LLM'),
                ('???', 'casual', 'LLM'),
                ('lol wut', 'casual', 'LLM'),
            ]
        },
        {
            'name': '10. Complaints',
            'messages': [
                ('you are the worst', 'casual', 'LLM'),
                ('terrible service', 'casual', 'LLM'),
                ('your competitor is better', 'casual', 'LLM'),
            ]
        },
        {
            'name': '11. Mixed greeting + query',
            'messages': [
                ('hi i want to go to dubai', 'visa_query', 'template'),
                ('im indian', 'follow_up', 'retrieval'),
            ]
        },
        {
            'name': '12. Long nationality statements',
            'messages': [
                ('do i need a visa', 'visa_query', 'template'),
                ('yes i am a citizen of dubai', 'follow_up', 'template'),
                ('turkey', 'follow_up', 'retrieval'),
            ]
        },
        {
            'name': '13. Random off-topic questions',
            'messages': [
                ('ur ahmed right', 'casual', 'LLM'),
                ('who made you', 'casual', 'LLM'),
                ('how does this work', 'casual', 'LLM'),
            ]
        },
        {
            'name': '14. Origin retained across queries',
            'messages': [
                ('im pakistani going to dubai', 'visa_query', 'retrieval'),
                ('what about japan', 'follow_up', 'retrieval'),
                ('and singapore', 'follow_up', 'retrieval'),
            ]
        },
    ]
    
    total_messages = 0
    results = []
    
    for scenario in scenarios:
        print_subheader(scenario['name'])
        state = ConversationState()  # Fresh state per scenario
        
        for msg, expected_intent, expected_handler in scenario['messages']:
            start = time.time()
            result = process_message(msg, state, llm)
            elapsed = (time.time() - start) * 1000
            
            response = result['response']
            timing = result['timing']
            
            # Determine what handler was used
            if 'llm_response' in timing or 'llm_followup' in timing:
                handler = 'LLM'
                handler_time = timing.get('llm_response', timing.get('llm_followup', 0))
            elif 'retrieval' in timing:
                handler = 'retrieval'
                handler_time = timing['retrieval']
            elif 'template_response' in timing:
                handler = 'template'
                handler_time = timing['template_response']
            else:
                handler = 'unknown'
                handler_time = 0
            
            # Truncate response for display
            display_response = response[:70] + '...' if len(response) > 70 else response
            
            # Check if handler matches expected
            handler_match = handler == expected_handler or (expected_handler == 'LLM' and handler in ['LLM'])
            status = "✅" if handler_match else "⚠️"
            
            print(f"  {status} You: \"{msg}\"")
            print(f"     Bot: {display_response}")
            print(f"     [{handler}: {handler_time:.1f}ms | Total: {elapsed:.0f}ms]")
            
            results.append({
                'scenario': scenario['name'],
                'message': msg,
                'handler': handler,
                'expected_handler': expected_handler,
                'handler_time': handler_time,
                'total_time': elapsed,
                'matched': handler_match,
            })
            
            total_messages += 1
    
    # Summary
    print_header("FULL PIPELINE SUMMARY")
    
    matched = sum(1 for r in results if r['matched'])
    print(f"  Handler matches: {matched}/{total_messages} ({100*matched/total_messages:.1f}%)")
    
    # Timing breakdown
    llm_times = [r['handler_time'] for r in results if r['handler'] == 'LLM']
    kg_times = [r['handler_time'] for r in results if r['handler'] == 'retrieval']
    template_times = [r['handler_time'] for r in results if r['handler'] == 'template']
    
    if llm_times:
        print(f"\n  LLM responses ({len(llm_times)} total):")
        print(f"    Avg: {sum(llm_times)/len(llm_times):.0f}ms")
        print(f"    Min: {min(llm_times):.0f}ms")
        print(f"    Max: {max(llm_times):.0f}ms")
    
    if kg_times:
        print(f"\n  KG retrievals ({len(kg_times)} total):")
        print(f"    Avg: {sum(kg_times)/len(kg_times):.2f}ms")
        print(f"    Min: {min(kg_times):.2f}ms")
        print(f"    Max: {max(kg_times):.2f}ms")
    
    if template_times:
        print(f"\n  Template responses ({len(template_times)} total):")
        print(f"    Avg: {sum(template_times)/len(template_times):.2f}ms")
    
    # Warnings
    mismatches = [r for r in results if not r['matched']]
    if mismatches:
        print(f"\n  ⚠️  MISMATCHED HANDLERS:")
        for r in mismatches:
            print(f"    - \"{r['message']}\" (in {r['scenario'][:20]}...)")
            print(f"      Expected: {r['expected_handler']}, Got: {r['handler']}")
    
    return matched, total_messages


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" COMPREHENSIVE TEST SUITE - Travel Agent Chatbot")
    print(" Testing all components and edge cases")
    print("="*80)
    
    # Initialize
    print("\nInitializing...")
    init_classifier()
    
    results = []
    
    # Run tests
    results.append(('Intent Classification', test_intent_classification()))
    results.append(('Entity Extraction', test_entity_extraction()))
    results.append(('Conversation State', test_conversation_state()))
    results.append(('Full Pipeline', test_full_pipeline()))
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    total_passed = 0
    total_tests = 0
    
    for name, (passed, total) in results:
        pct = 100 * passed / total if total > 0 else 0
        status = "✅" if passed == total else "⚠️"
        print(f"  {status} {name}: {passed}/{total} ({pct:.1f}%)")
        total_passed += passed
        total_tests += total
    
    print(f"\n  {'='*40}")
    overall_pct = 100 * total_passed / total_tests if total_tests > 0 else 0
    print(f"  OVERALL: {total_passed}/{total_tests} ({overall_pct:.1f}%)")
    print(f"  {'='*40}")


if __name__ == "__main__":
    main()

