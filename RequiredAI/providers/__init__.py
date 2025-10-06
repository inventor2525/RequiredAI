"""
Provider system for RequiredAI.
"""

from typing import Dict, Type, List, Any, Optional, TypeVar, Callable, Union
from ..ModelConfig import ModelConfig

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
	def inner(c:T, name:str=provider_name) -> T:
		BaseModelProvider._PROVIDER_REGISTRY[name] = c
		return c
	return inner