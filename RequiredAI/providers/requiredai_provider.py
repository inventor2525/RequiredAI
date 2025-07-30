import os
from typing import Dict, List, Any, Optional
from ..system import RequiredAISystem
from ..Requirement import Requirements
from . import BaseModelProvider, provider

@provider('RequiredAI')
class RequiredAIProvider(BaseModelProvider):
	"""Provider for RequiredAI's system."""
	
	def __init__(self, config: Dict[str, Any]):
		"""Initialize the RequiredAI provider."""
		super().__init__(config)
		self.model_name:str = config["provider_model"]
		self.requirements:list = config["requirements"]
		
	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using RequiredAI's system.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message
		"""
		
		return RequiredAISystem.singleton.chat_completions(self.model_name, self.requirements, messages, params)