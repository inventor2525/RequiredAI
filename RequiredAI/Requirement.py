"""
Core requirements functionality for RequiredAI.
"""

from typing import Any, Callable, Dict, List, Type, TypeVar, ClassVar, Optional, Tuple, get_origin, Set
from abc import ABC, abstractmethod
from dataclasses import asdict, fields, dataclass, field

T = TypeVar("T")

# Registry to store requirement types
_REQUIREMENT_REGISTRY: Dict[str, Type] = {}

@dataclass
class RequirementResult:
    passed_eval:bool
    evaluation_log:Optional[dict] = field(default=None)
    
    def __bool__(self):
        return self.passed_eval
    
    @staticmethod
    def construct(requirement:'Requirement', passed:bool, extra_log_fields:Optional[Dict[str,Any]]={}) -> 'RequirementResult':
        return RequirementResult(
            passed_eval=passed,
            evaluation_log={
                "requirement_type": type(requirement).__web_name__,
                "requirement_name": requirement.name,
                "passed":passed,
                **extra_log_fields
            }
        )
    
class Requirement(ABC):
    """Base abstract class for all requirements."""
    
    # Class variable for web name
    __web_name__: ClassVar[str]
    
    # Name for the requirement instance
    name: str = ""
    revision_model: Optional[str] = None
    
    @abstractmethod
    def evaluate(self, messages: List[dict]) -> RequirementResult:
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
        # Set the __web_name__ class variable
        cls.__web_name__ = name
        _REQUIREMENT_REGISTRY[name] = cls
        return cls
    return decorator


# Custom dict_factory for serialization
def set_to_list_dict_factory(items):
    result = {}
    for key, value in items:
        if isinstance(value, set):
            result[key] = list(value)  # Convert set to list for JSON
        elif isinstance(value, list) and any(hasattr(item, '__dataclass_fields__') for item in value):
            # Handle nested dataclasses in lists (e.g., nodes, children)
            result[key] = [
                asdict(item, dict_factory=set_to_list_dict_factory) if hasattr(item, '__dataclass_fields__') else item
                for item in value
            ]
        else:
            result[key] = value
    return result

class Requirements:
    """Utility class for handling requirements."""
    
    @staticmethod
    def to_json(requirement_instance: Requirement | List[Requirement]) -> dict | List[dict]:
        """
        Convert a requirement instance to a JSON-serializable dictionary.
        
        Args:
            requirement_instance: The requirement instance to convert
            
        Returns:
            dict: JSON-serializable representation of the requirement
        """
        def _to_json_single(req: Requirement) -> dict:
            requirement_type = getattr(req.__class__, "__web_name__", None)
            if not requirement_type:
                raise ValueError(f"Requirement class {req.__class__.__name__} is not registered")
            
            # Use asdict with custom dict_factory to handle sets
            result = asdict(req, dict_factory=set_to_list_dict_factory)
            result["type"] = requirement_type
                    
            return result
        
        if isinstance(requirement_instance, List):
            return [_to_json_single(req) for req in requirement_instance]
        return _to_json_single(requirement_instance)
    
    @staticmethod
    def from_json(j: dict | List[dict]) -> Requirement | List[Requirement]:
        """
        Create a requirement instance from a JSON dictionary.
        
        Args:
            j: JSON dictionary representing a requirement
            
        Returns:
            An instance of the appropriate requirement class
        """
        def _from_json_single(json_dict: dict) -> Requirement:
            requirement_type = json_dict.get("type")
            if not requirement_type:
                raise ValueError("Requirement JSON must include a 'type' field")
                
            if requirement_type not in _REQUIREMENT_REGISTRY:
                raise ValueError(f"Unknown requirement type: {requirement_type}")
                
            # Create an instance of the requirement class
            requirement_class = _REQUIREMENT_REGISTRY[requirement_type]
            
            # Get valid field names for this dataclass
            valid_fields = {f.name: f for f in fields(requirement_class)}
            
            # Convert fields based on their type annotations
            kwargs = {}
            for k, v in json_dict.items():
                if k == "type" or k not in valid_fields:
                    continue
                field_type = valid_fields[k].type
                origin_type = get_origin(field_type) or field_type
                
                if origin_type in (set, Set) and isinstance(v, list):
                    kwargs[k] = set(v)  # Convert list back to set
                elif k in ("nodes", "children") and isinstance(v, list):
                    # Recursively convert nested dataclasses
                    kwargs[k] = [Requirements.from_json(item) if isinstance(item, dict) else item for item in v]
                else:
                    kwargs[k] = v
            
            return requirement_class(**kwargs)
        
        if isinstance(j, List):
            return [_from_json_single(json_dict) for json_dict in j]
        return _from_json_single(j)