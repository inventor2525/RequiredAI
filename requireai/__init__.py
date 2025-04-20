"""
RequiredAI - A client and server API for adding requirements to AI responses.
"""

__version__ = "0.1.0"

from .requirements import (
    requirement,
    Requirements,
    Requirement
)
from .models import ContainsRequirement, WrittenRequirement
from .model_manager import ModelManager

# Import providers to register them
from .providers import BaseModelProvider
from .providers.anthropic import AnthropicProvider
