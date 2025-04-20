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
        Uses another model to evaluate if the response meets the writing requirements.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the requirement is met, False otherwise
        """
        from requireai.model_manager import ModelManager
        import json
        import os
        
        print(f"\nEvaluating Written requirement:")
        if not messages:
            return False
            
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        if not isinstance(content, str):
            return False
            
        print(f"Response preview: {content[:100]}...")
        
        # If no model is specified, we can't evaluate
        if not self.model:
            raise ValueError("No model specified for WrittenRequirement evaluation")
            
        try:
            # Use the ModelManager singleton directly
            from requireai.model_manager import model_manager
            
            # Prepare the evaluation prompt
            requirements_str = "; ".join(self.value)
            
            # Prepare examples if available
            examples_text = ""
            if self.positive_examples:
                examples_text += "\n\nHere are some examples that DO meet the requirements:\n"
                for i, example in enumerate(self.positive_examples, 1):
                    examples_text += f"\nExample {i}:\n{example}\n"
                    
            if self.negative_examples:
                examples_text += "\n\nHere are some examples that DO NOT meet the requirements:\n"
                for i, example in enumerate(self.negative_examples, 1):
                    examples_text += f"\nExample {i}:\n{example}\n"
            
            # Create the evaluation messages
            eval_messages = [
                {
                    "role": "system",
                    "content": "You are an AI writing style evaluator. Your task is to determine if a given text meets specific writing requirements."
                },
                {
                    "role": "user",
                    "content": f"I need you to evaluate if the following text meets these writing requirements:\n\n{requirements_str}\n\n{examples_text}\n\nText to evaluate:\n\n{content}\n\nDoes this text meet the requirements? Answer with only 'yes' or 'no'."
                }
            ]
            
            # Estimate token count and limit examples if needed
            total_tokens = 0
            for msg in eval_messages:
                total_tokens += model_manager.estimate_tokens(msg["content"], self.model)
                
            if total_tokens > self.token_limit:
                print(f"Token limit exceeded ({total_tokens} > {self.token_limit}), truncating examples")
                # Simplify the prompt to fit within token limit
                eval_messages = [
                    {
                        "role": "system",
                        "content": "You are an AI writing style evaluator. Your task is to determine if a given text meets specific writing requirements."
                    },
                    {
                        "role": "user",
                        "content": f"I need you to evaluate if the following text meets these writing requirements:\n\n{requirements_str}\n\nText to evaluate:\n\n{content}\n\nDoes this text meet the requirements? Answer with only 'yes' or 'no'."
                    }
                ]
            
            # Get the evaluation from the model
            response = model_manager.complete_with_model(
                self.model,
                eval_messages,
                {"max_tokens": 10, "temperature": 0.0}  # Use low temperature for consistent results
            )
            
            # Parse the response
            eval_text = response.get("content", "").strip().lower()
            result = "yes" in eval_text and "no" not in eval_text
            
            print(f"Model evaluation result: {eval_text}")
            print(f"Requirement satisfied: {result}")
            
            return result
            
        except Exception as e:
            print(f"Error evaluating requirement: {str(e)}")
            # Default to True on error to avoid blocking
            return True
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining the writing requirements.
        """
        requirements_str = "; ".join(self.value)
        return f"Your response should follow these writing requirements: {requirements_str}"
