import os
from typing import Dict, List, Any, Optional
from ..system import RequiredAISystem
from ..Requirement import Requirements
from . import BaseModelProvider, provider, ProviderException
from ..ModelConfig import ModelConfig

@provider('RequiredAI')
class RequiredAIProvider(BaseModelProvider):
	"""Provider for RequiredAI's system."""
	
	def __init__(self, config: ModelConfig):
		"""Initialize the RequiredAI provider."""
		super().__init__(config)
		
	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using RequiredAI's system.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message
		"""
		response_dict = None
		try:
			response_dict = RequiredAISystem.singleton.chat_completions(self.config.provider_model, self.config.requirements, messages, params)
			response_dict['choices'][0]['message']['tags'] = list(self.config.output_tags)
			return response_dict
		except Exception as e:
			raise ProviderException(RequiredAIProvider.provider_name, e, response_dict)