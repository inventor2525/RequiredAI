"""
Model manager for RequiredAI.
"""

from typing import Dict, Any, List, Optional
from .providers import BaseModelProvider
from .ModelConfig import ModelConfig

class ModelManager:
	"""Manager for model providers."""
	
	_instance = None
	
	@staticmethod
	def singleton():
		"""Get the singleton instance of ModelManager."""
		return ModelManager._instance
	
	def __init__(self, model_configs: List[ModelConfig]):
		"""
		Initialize the model manager.
		
		Args:
			config: The server configuration
		"""
		self.model_configs = {config.name:config for config in model_configs}
		self.provider_instances = {}
		ModelManager._instance = self
	
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
		
		model_config = self.model_configs.get(model_name,None)
		if not model_config:
			raise ValueError(f"The provided model '{model_name}' is not listed in the configuration!")
		
		if not model_config.provider:
			raise ValueError(f"Provider not specified for model {model_name}")
		
		provider_class = BaseModelProvider.get_provider(model_config.provider)
		provider = provider_class(model_config)
		
		self.provider_instances[model_name] = provider
		return provider
	
	def complete_with_model(self, model_name: str, messages: List[Dict[str, Any]], params: Dict[str, Any]={}) -> Dict[str, Any]:
		"""
		Generate a completion using the specified model.
		
		Args:
			model_name: The name of the model
			messages: The conversation messages
			params: Additional parameters for the request (note
				that these will override [by key] any in the model
				config's 'default_params', for this request.)
			
		Returns:
			The model's response message
		"""
		provider = self.get_provider(model_name)
		if params and provider.config.default_params:
			p = dict(provider.config.default_params)
			p.update(params)
		else:
			p = provider.config.default_params or params
		return provider.complete(messages, p)
	
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
