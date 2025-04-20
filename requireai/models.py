"""
Requirement model implementations for RequiredAI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from .requirements import requirement, Requirement

@requirement("Contains")
@dataclass
class ContainsRequirement(Requirement):
    """Requirement that checks if the AI response contains any of the specified values."""
    
    value: List[str]
    name: str = ""
    
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
        
        # Print for debugging
        print(f"\nEvaluating Contains requirement:")
        print(f"Response preview: {content[:100]}...")
        result = any(val in content for val in self.value)
        print(f"Looking for any of these values: {self.value}")
        print(f"Requirement satisfied: {result}")
            
        return result
    
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
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    model: Optional[str] = None
    token_limit: int = 1024
    name: str = ""
    
    def evaluate(self, messages: List[dict]) -> bool:
        """
        Evaluates if the response follows the writing requirements.
        This would typically call an external model to evaluate.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the requirement is met, False otherwise
        """
        if not messages:
            return False
            
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        if not isinstance(content, str):
            return False
        
        # Simple evaluation logic - check if the content contains phrases that suggest reasoning
        reasoning_indicators = [
            "let me think", 
            "step by step", 
            "first,", 
            "second,", 
            "third,", 
            "analyzing", 
            "consider", 
            "reasoning", 
            "analysis"
        ]
        
        # Check if any of the reasoning indicators are present
        has_reasoning = any(indicator.lower() in content.lower() for indicator in reasoning_indicators)
        
        # Check if any of the required writing styles are mentioned
        required_styles = [req.lower() for req in self.value]
        follows_style = any(style in content.lower() for style in required_styles)
        
        # Print for debugging
        print(f"\nEvaluating Written requirement:")
        print(f"Response preview: {content[:100]}...")
        print(f"Has reasoning indicators: {has_reasoning}")
        print(f"Follows required style: {follows_style}")
        print(f"Requirement satisfied: {has_reasoning}")
        
        # For now, just check for reasoning indicators
        return has_reasoning
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining the writing requirements.
        """
        requirements_str = "; ".join(self.value)
        return f"Your response should follow these writing requirements: {requirements_str}"
