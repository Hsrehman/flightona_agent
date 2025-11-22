"""
Step 4: Intent Classification
Classifies user questions into: visa, baggage, general_travel, or off_topic
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


class IntentClassification(BaseModel):
    """Classification result for user intent"""
    intent: Literal["visa", "baggage", "general_travel", "off_topic"] = Field(
        description="The classified intent of the user's question"
    )
    confidence: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


def create_intent_classifier(model: str = "gemini-2.5-flash-lite"):
    """
    Create an intent classifier function.
    
    Args:
        model: Gemini model to use for classification
        
    Returns:
        Function that classifies user intent
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)  # Low temp for consistent classification
    
    def classify_intent(user_input: str) -> dict:
        """
        Classify user input into one of: visa, baggage, general_travel, off_topic
        
        Args:
            user_input: The user's message
            
        Returns:
            Dict with 'intent' and 'confidence' keys
        """
        system_prompt = """You are an intent classifier for a travel agency chatbot. Classify the user's question into one of these categories:

1. **visa**: Questions about visa requirements, visa-free travel, visa on arrival, e-visa, visa applications, passport requirements, entry requirements
   Examples: "Do I need a visa for Dubai?", "Visa requirements for US citizens", "Can I travel visa-free to France?"

2. **baggage**: Questions about luggage, baggage rules, carry-on restrictions, checked baggage, baggage claims, weight limits, prohibited items
   Examples: "What's the baggage allowance?", "Can I bring liquids?", "Baggage rules for international flights"

3. **general_travel**: General travel questions about destinations, packages, hotels, flights, travel tips, travel planning, booking inquiries
   Examples: "Do you offer Dubai packages?", "What's the best time to visit?", "Flight booking", "Hotel recommendations"

4. **off_topic**: Questions completely unrelated to travel (weather, sports, news, random chat, etc.)
   Examples: "What's the weather?", "Tell me a joke", "How's your day?"

Respond with ONLY a JSON object:
{"intent": "visa|baggage|general_travel|off_topic", "confidence": "brief reason"}"""

        human_prompt = f"User question: {user_input}\n\nClassify this question."

        # Use structured output if available, otherwise parse JSON
        try:
            structured_llm = llm.with_structured_output(IntentClassification)
            result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            return {
                "intent": result.intent,
                "confidence": result.confidence
            }
        except Exception:
            # Fallback: use regular LLM and parse JSON
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return {
                        "intent": result.get("intent", "general_travel"),
                        "confidence": result.get("confidence", "Unable to determine")
                    }
                except json.JSONDecodeError:
                    pass
            
            # Default fallback
            return {
                "intent": "general_travel",
                "confidence": "Could not parse classification, defaulting to general_travel"
            }
    
    return classify_intent


def route_by_intent(state: dict) -> Literal["visa_handler", "general_handler", "baggage_handler", "off_topic_handler"]:
    """
    Route to appropriate handler based on classified intent.
    
    Args:
        state: ChatState with 'intent' field
        
    Returns:
        Name of the handler node to route to
    """
    intent = state.get("intent", "general_travel")
    
    if intent == "visa":
        return "visa_handler"
    elif intent == "baggage":
        return "baggage_handler"
    elif intent == "off_topic":
        return "off_topic_handler"
    else:  # general_travel
        return "general_handler"

