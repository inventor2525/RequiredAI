"""
Requirement model implementations for RequiredAI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import random
from .requirements import requirement, Requirement
import regex as re

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

@requirement("Regex")
@dataclass
class RegexRequirement(Requirement):
    """Requirement that checks if the AI response matches positive regexes and does not match negative regexes."""
    
    positive_regexes: List[str]
    negative_regexes: List[str]
    additional_prompt: Optional[str] = None
    name: str = ""
    
    def evaluate(self, messages: List[dict]) -> bool:
        """
        Evaluates if the response matches all positive regexes and none of the negative regexes.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if all positive regexes match and no negative regexes match, False otherwise
        """
        if not messages:
            return False
            
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        if not isinstance(content, str):
            return False
            
        print(f"\nEvaluating Regex requirement:")
        print(f"Response preview: {content[:100]}...")
        
        # Check positive regexes
        for regex in self.positive_regexes:
            try:
                if not re.search(regex, content):
                    print(f"Positive regex '{regex}' not found")
                    return False
            except re.error as e:
                print(f"Invalid positive regex '{regex}': {e}")
                return False
        
        # Check negative regexes
        for regex in self.negative_regexes:
            try:
                if re.search(regex, content):
                    print(f"Negative regex '{regex}' found")
                    return False
            except re.error as e:
                print(f"Invalid negative regex '{regex}': {e}")
                return False
        
        print("All regex requirements satisfied")
        return True
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining how the response should follow positive regexes and avoid negative ones.
        """
        positive_str = (
            "```txt\n" + "\n".join(self.positive_regexes) + "\n```"
            if self.positive_regexes else None
        )
        negative_str = (
            "```txt\n" + "\n".join(self.negative_regexes) + "\n```"
            if self.negative_regexes else None
        )
        
        prompt_parts = []
        if positive_str:
            prompt_parts.append(f"Your response must match these regex patterns:\n{positive_str}")
        if negative_str:
            prompt_parts.append(f"Your response must not match these regex patterns:\n{negative_str}")
        
        if self.additional_prompt:
            prompt_parts.append(self.additional_prompt)
        return "\n".join(prompt_parts) if prompt_parts else None

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
            
        print(f"Response: {content}...")
        
        # If no model is specified, we can't evaluate
        if not self.model:
            raise ValueError("No model specified for WrittenRequirement evaluation")
            
        try:
            # Select one random requirement from the value list
            selected_requirement = random.choice(self.value)
            
            # System message focused on task
            system_msg = "Determine if the given text meets the specified written requirement. Answer with only 'yes' or 'no'."
            
            # Base user message
            user_msg = f"Written requirement: {selected_requirement}\n\nText to evaluate:\n{content}\n\nDoes this text meet the requirement?"
            
            # Accumulate positive and negative examples separately
            positive_examples = []
            negative_examples = []
            
            # Combine all examples for random selection
            all_examples = []
            if self.positive_examples:
                for ex in self.positive_examples:
                    all_examples.append(("positive", ex))
            if self.negative_examples:
                for ex in self.negative_examples:
                    all_examples.append(("negative", ex))
            
            # Randomly select examples up to token limit
            if all_examples:
                random.shuffle(all_examples)
                
                for example_type, example in all_examples:
                    # Test adding this example
                    temp_positive = positive_examples + ([example] if example_type == "positive" else [])
                    temp_negative = negative_examples + ([example] if example_type == "negative" else [])
                    
                    # Build test message with accumulated examples
                    test_examples_text = ""
                    if temp_positive:
                        test_examples_text += "\n\nExamples that meet the requirement:\n" + "\n\n".join(temp_positive)
                    if temp_negative:
                        test_examples_text += "\n\nExamples that do NOT meet the requirement:\n" + "\n\n".join(temp_negative)
                    
                    test_user_msg = f"Written requirement: {selected_requirement}{test_examples_text}\n\nText to evaluate:\n{content}\n\nDoes this text meet the requirement?"
                    test_content = system_msg + test_user_msg
                    
                    # Check token count
                    current_tokens = ModelManager.singleton().estimate_tokens(test_content, self.model)
                    
                    if current_tokens <= self.token_limit:
                        # Accept this example
                        if example_type == "positive":
                            positive_examples.append(example)
                        else:
                            negative_examples.append(example)
                    else:
                        # Stop adding examples
                        break
            
            # Build final examples text
            examples_text = ""
            if positive_examples:
                examples_text += "\n\nExamples that meet the requirement:\n" + "\n\n".join(positive_examples)
            if negative_examples:
                examples_text += "\n\nExamples that do NOT meet the requirement:\n" + "\n\n".join(negative_examples)
            
            # Create final evaluation messages
            final_user_msg = f"Written requirement: {selected_requirement}{examples_text}\n\nText to evaluate:\n```txt\n{content}\n```\nDoes this text meet the requirement?"
            
            eval_messages = [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user", 
                    "content": final_user_msg
                }
            ]
            
            # Get the evaluation from the model
            response = ModelManager.singleton().complete_with_model(
                self.model,
                eval_messages,
                {"max_tokens": 1, "temperature": 0.0}  # Use low temperature for consistent results
            )
            
            print(f"Evaluating: {eval_messages}")
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
        Returns a string explaining the written requirements.
        """
        requirements_str = "; ".join(self.value)
        return f"Your response should follow these written requirements: {requirements_str}"