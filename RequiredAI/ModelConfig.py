from typing import List, Dict, Any, Optional
from .requirements import Requirements, Requirement
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Represents a model configuration with serialization support."""
    name: str
    provider: str
    provider_model: str
    api_key_env: Optional[str] = None
    requirements: Optional[List[Requirement]] = field(default=None)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the model configuration to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representing the model configuration
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
        return result
    
    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create a ModelConfig instance from a JSON dictionary.
        
        Args:
            json_dict: JSON dictionary representing a model configuration
            
        Returns:
            A ModelConfig instance
        """
        
        requirements = None
        requires_dict = json_dict.get("requirements", None)
        if requires_dict:
            requirements = Requirements.from_json(requires_dict)
        
        return ModelConfig(
            name=json_dict.get("name", ""),
            provider=json_dict.get("provider", ""),
            provider_model=json_dict.get("provider_model", ""),
            api_key_env=json_dict.get("api_key_env", None),
            requirements=requirements
        )