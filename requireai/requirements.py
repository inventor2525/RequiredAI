"""
Core requirements functionality for RequiredAI.
"""

from typing import Any, Callable, Dict, List, Type, TypeVar
from abc import ABC, abstractmethod

T = TypeVar("T")

# Registry to store requirement types
_REQUIREMENT_REGISTRY: Dict[str, Type] = {}

class Requirement(ABC):
    """Base abstract class for all requirements."""
    
    # Class variable for model name
    model_name: str = None
    
    @abstractmethod
    def evaluate(self, messages: List[dict]) -> bool:
        """
        Evaluate if the requirement is met in the given messages.
        
        Args:
            messages: List of message dictionaries, with the last one being the AI response
            
        Returns:
            bool: True if requirement is met, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def prompt(self) -> str:
        """
        Returns a string explaining how the conversation in the last call to 
        evaluate that returned false did not meet this requirement.
        """
        pass


def requirement(name: str) -> Callable[[T], T]:
    """
    Decorator to register a requirement class with the specified name.
    
    Args:
        name: The name to register the requirement under
        
    Returns:
        The decorated class
    """
    def decorator(cls: T) -> T:
        setattr(cls, "__web_name__", name)
        _REQUIREMENT_REGISTRY[name] = cls
        return cls
    return decorator


class Requirements:
    """Utility class for handling requirements."""
    
    @staticmethod
    def to_json(requirement_instance: Requirement) -> dict:
        """
        Convert a requirement instance to a JSON-serializable dictionary.
        
        Args:
            requirement_instance: The requirement instance to convert
            
        Returns:
            dict: JSON-serializable representation of the requirement
        """
        from dataclasses import asdict
        
        requirement_type = getattr(requirement_instance.__class__, "__web_name__", None)
        if not requirement_type:
            raise ValueError(f"Requirement class {requirement_instance.__class__.__name__} is not registered")
        
        # Use dataclass asdict to get all fields
        result = asdict(requirement_instance)
        result["type"] = requirement_type
                
        return result
    
    @staticmethod
    def from_json(j: dict) -> Any:
        """
        Create a requirement instance from a JSON dictionary.
        
        Args:
            j: JSON dictionary representing a requirement
            
        Returns:
            An instance of the appropriate requirement class
        """
        from dataclasses import fields
        
        requirement_type = j.get("type")
        if not requirement_type:
            raise ValueError("Requirement JSON must include a 'type' field")
            
        if requirement_type not in _REQUIREMENT_REGISTRY:
            raise ValueError(f"Unknown requirement type: {requirement_type}")
            
        # Create an instance of the requirement class
        requirement_class = _REQUIREMENT_REGISTRY[requirement_type]
        
        # Get valid field names for this dataclass
        valid_fields = {f.name for f in fields(requirement_class)}
        
        # Filter the input dict to only include valid fields
        kwargs = {k: v for k, v in j.items() if k != "type" and k in valid_fields}
        return requirement_class(**kwargs)
