"""
Requirement model implementations for RequiredAI.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
from .requirements import requirement, Requirement

@requirement("Contains")
@dataclass
class ContainsRequirement(Requirement):
    """Requirement that checks if the AI response contains any of the specified values."""
    
    value: List[str]
    
    def evaluate(self, messages: List[dict]) -> bool:
        """
        Checks that the last message in the passed conversation (which is presumed 
        to be from an AI), contains any of the values in value.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the last message contains any of the values, False otherwise
        """
        if not messages:
            return False
            
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        if not isinstance(content, str):
            return False
            
        return any(val in content for val in self.value)
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining how the conversation did not meet this requirement.
        """
        values_str = '", "'.join(self.value)
        return f'Your response must contain at least one of the following: "{values_str}".'


@requirement("Written")
@dataclass
class WrittenRequirement(Requirement):
    """
    Requirement that uses another model to evaluate if the response 
    follows specific writing instructions.
    """
    
    value: List[str]
    positive_examples: Optional[List[str]] = None
    negative_examples: Optional[List[str]] = None
    model: Optional[str] = None
    token_limit: int = 1024
    
    def __post_init__(self):
        """Initialize optional fields with empty lists if None."""
        if self.positive_examples is None:
            self.positive_examples = []
        if self.negative_examples is None:
            self.negative_examples = []
    
    def evaluate(self, messages: List[dict]) -> bool:
        """
        Evaluates if the response follows the writing requirements.
        This would typically call an external model to evaluate.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the requirement is met, False otherwise
        """
        # In a real implementation, this would call the specified model
        # to evaluate if the response meets the writing requirements
        # For now, we'll return a placeholder implementation
        
        # This is a simplified placeholder - in a real implementation,
        # you would send the messages, requirements, and examples to the model
        # and get back an evaluation
        
        # TODO: Implement actual evaluation logic using the specified model
        return True  # Placeholder
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining the writing requirements.
        """
        requirements_str = "; ".join(self.value)
        return f"Your response should follow these writing requirements: {requirements_str}"
