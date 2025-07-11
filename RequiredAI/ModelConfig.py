from typing import List, Dict, Any, Optional
from .requirements import Requirements, Requirement
from dataclasses import dataclass, field, asdict

@dataclass
class ContextOriginConfig:
    """
    Configuration for how conversation context is presented to an evaluation or revision model.
    """
    include_original_system_message: bool = False
    messages_to_include: Optional[int] = 0 # 0 for last AI message only, N for last N messages (AI + N-1 preceding), None for all.
    custom_system_message: Optional[str] = None # A specific system message for this model if provided.

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the context origin configuration to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representing the context origin configuration.
        """
        result = asdict(self)
        # Remove keys with None values for cleaner JSON output
        return {k: v for k, v in result.items() if v is not None}

    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> 'ContextOriginConfig':
        """
        Create a ContextOriginConfig instance from a JSON dictionary.
        
        Args:
            json_dict: JSON dictionary representing a context origin configuration.
            
        Returns:
            A ContextOriginConfig instance.
        """
        return ContextOriginConfig(
            include_original_system_message=json_dict.get("include_original_system_message", False),
            messages_to_include=json_dict.get("messages_to_include", 0),
            custom_system_message=json_dict.get("custom_system_message", None)
        )

    @staticmethod
    def default() -> 'ContextOriginConfig':
        """
        Returns the default context origin configuration.
        This configuration dictates that only the last AI response (the one being evaluated/revised)
        is included as context, and no original system message or custom system message is used.
        The calling code (e.g., a WrittenRequirement) is expected to provide any necessary system message.
        """
        return ContextOriginConfig(
            include_original_system_message=False,
            messages_to_include=0, # Only the last AI message in the conversation
            custom_system_message=None # No custom system message defined by default
        )

@dataclass
class ModelConfig:
    """Represents a model configuration with serialization support."""
    name: str
    provider: str
    provider_model: str
    api_key_env: Optional[str] = None
    requirements: Optional[List[Requirement]] = field(default=None)
    # New field for context origin configuration, defaults to the standard behavior
    context_origin_config: ContextOriginConfig = field(default_factory=ContextOriginConfig.default)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the model configuration to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representing the model configuration.
        """
        result = {
            "name": self.name,
            "provider": self.provider,
            "provider_model": self.provider_model
        }
        if self.api_key_env is not None:
            result["api_key_env"] = self.api_key_env
        if self.requirements:
            result["requirements"] = Requirements.to_json(self.requirements)
        
        # Only include context_origin_config in the JSON if it's not the default
        default_config = ContextOriginConfig.default()
        if self.context_origin_config != default_config:
            result["context_origin_config"] = self.context_origin_config.to_json()
        return result
    
    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create a ModelConfig instance from a JSON dictionary.
        
        Args:
            json_dict: JSON dictionary representing a model configuration.
            
        Returns:
            A ModelConfig instance.
        """
        
        requirements = None
        requires_dict = json_dict.get("requirements", None)
        if requires_dict:
            requirements = Requirements.from_json(requires_dict)
        
        # Handle context_origin_config, default to standard if not provided in JSON
        context_origin_config = ContextOriginConfig.default()
        context_origin_dict = json_dict.get("context_origin_config", None)
        if context_origin_dict:
            context_origin_config = ContextOriginConfig.from_json(context_origin_dict)
            
        return ModelConfig(
            name=json_dict.get("name", ""),
            provider=json_dict.get("provider", ""),
            provider_model=json_dict.get("provider_model", ""),
            api_key_env=json_dict.get("api_key_env", None),
            requirements=requirements,
            context_origin_config=context_origin_config
        )