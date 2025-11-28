"""Conversation management module for templates and response generation."""

from .templates import (
    get_template_response,
    get_welcome_message,
    get_goodbye_message,
    get_clarification_question,
    format_visa_result,
    TEMPLATES,
)

__all__ = [
    'get_template_response',
    'get_welcome_message', 
    'get_goodbye_message',
    'get_clarification_question',
    'format_visa_result',
    'TEMPLATES',
]

