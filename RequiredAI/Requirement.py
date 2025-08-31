"""
Core requirements functionality for RequiredAI.
"""

from typing import Any, Callable, Dict, List, Type, TypeVar, ClassVar, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .json_dataclass import *
from .helpers import *

T = TypeVar("T")

# Registry to store requirement types
_REQUIREMENT_REGISTRY: Dict[str, Type] = {}

@json_dataclass
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
                "requirement_type": type(requirement).__requirement_type__,
                "requirement_name": requirement.name,
                "passed":passed,
                **extra_log_fields
            }
        )

@dataclass
class typed_requirement:
    __requirement_type__: str = field(init=False)
    
class Requirement(typed_requirement):
    """Base abstract class for all requirements."""
    
    # Name for the requirement instance
    name: str = ""
    revision_model: Optional[str] = None
    
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
        # Set the __requirement_type__ class variable
        cls.__requirement_type__ = name
        _REQUIREMENT_REGISTRY[name] = cls
        return cls
    return decorator

ReqDict = Dict[str, Any]
class Requirements:
    '''
    Handles the polymorphic serialization and
    deserialization of requirement objects.
    '''
    @staticmethod
    def to_dict(requirements:Requirement|List[Requirement]) -> ReqDict|List[ReqDict]:
        if requirements is None:
            return None
        if isinstance(requirements, list):
            return [req.to_dict() for req in requirements]
        return requirements.to_dict()
    
    @staticmethod
    def from_dict(requirement_dicts:ReqDict|List[ReqDict]) -> Requirement|List[Requirement]:
        def inner(req_dict:ReqDict) -> Requirement:
            requirement_cls = _REQUIREMENT_REGISTRY[req_dict['__requirement_type__']]
            return requirement_cls.from_dict(req_dict)
        if isinstance(requirement_dicts, list):
            return [inner(req_dict) for req_dict in requirement_dicts]
        return inner(requirement_dicts)