"""
Provider system for RequiredAI.
"""

from typing import Dict, Type, List, Any, Optional, TypeVar, Callable, Union
from ..ModelConfig import ModelConfig
import json
class ProviderException(Exception):
	def __init__(self, provider:str, exception: Exception, response_dict: Optional[dict] = None):
		self.provider = provider
		self.exception = exception
		self.response_dict = response_dict
		extra_str = ''
		if response_dict:
			extra_str = f"\n\nBut it did generate the following output:\n```json\n{json.dumps(response_dict, indent=4)}\n```"
		super().__init__(f"API provider '{provider}' failed to generate response with exception:\n```txt\n{str(exception)}\n```{extra_str}")

T = TypeVar('T')
class BaseModelProvider:
	"""Base class for all model providers."""
	
	# Registry to store provider types
	_PROVIDER_REGISTRY: Dict[str, Type["BaseModelProvider"]] = {}
	
	@classmethod
	def get_provider(cls, provider_name: str) -> Type["BaseModelProvider"]:
		"""Get a provider class by name."""
		if provider_name not in cls._PROVIDER_REGISTRY:
			raise ValueError(f"Unknown provider: {provider_name}")
		return cls._PROVIDER_REGISTRY[provider_name]
			
	def __init__(self, config: ModelConfig):
		"""Initialize the provider with configuration."""
		self.config = config
		
	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion for the given messages.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message
		"""
		raise NotImplementedError("Subclasses must implement this method")
	
	def estimate_tokens(self, text: str) -> int:
		"""
		Estimate the number of tokens in a string.
		
		Args:
			text: The text to estimate tokens for
			model: The model to use for estimation
			
		Returns:
			Estimated token count
		"""
		# Simple estimation based on character count
		return int(len(text) / 4.3)

def provider(provider_name:str) -> Callable[[T],T]:
	'''
	Decorator to register a model provider class.
	'''
	def inner(c:T, provider_name:str=provider_name) -> T:
		BaseModelProvider._PROVIDER_REGISTRY[provider_name] = c
		c.provider_name = provider_name
		return c
	return inner