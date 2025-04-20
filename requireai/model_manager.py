"""
Model manager for RequiredAI.
"""

from typing import Dict, Any, List, Optional
from .providers import BaseModelProvider

# Import providers to register them
from .providers.anthropic import AnthropicProvider

class ModelManager:
    """Manager for model providers."""
    
    _instance = None
    
    @staticmethod
    def singleton():
        """Get the singleton instance of ModelManager."""
        return ModelManager._instance
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: The server configuration
        """
        self.config = config
        self.models_config = config.get("models", {})
        self.provider_instances = {}
        ModelManager._instance = self
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The model configuration
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not configured")
        return self.models_config[model_name]
    
    def get_provider(self, model_name: str) -> BaseModelProvider:
        """
        Get or create a provider for the specified model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The provider instance
        """
        if model_name in self.provider_instances:
            return self.provider_instances[model_name]
        
        model_config = self.get_model_config(model_name)
        provider_name = model_config.get("provider")
        
        if not provider_name:
            raise ValueError(f"Provider not specified for model {model_name}")
        
        provider = BaseModelProvider.create_provider(provider_name, model_config)
        self.provider_instances[model_name] = provider
        return provider
    
    def complete_with_model(self, model_name: str, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using the specified model.
        
        Args:
            model_name: The name of the model
            messages: The conversation messages
            params: Additional parameters for the request
            
        Returns:
            The model's response message
        """
        provider = self.get_provider(model_name)
        return provider.complete(messages, params)
    
    def estimate_tokens(self, text: str, model_name: str) -> int:
        """
        Estimate the number of tokens in a string.
        
        Args:
            text: The text to estimate tokens for
            model_name: The model to use for estimation
            
        Returns:
            Estimated token count
        """
        provider = self.get_provider(model_name)
        return provider.estimate_tokens(text, model_name)
