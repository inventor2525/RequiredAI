"""
Groq provider for RequiredAI.
"""

import os
from typing import Dict, List, Any, Optional
from groq import Groq
from ..ModelConfig import ModelConfig

from . import BaseModelProvider, provider, ProviderException

@provider('groq')
class GroqProvider(BaseModelProvider):
	"""Provider for Groq's API."""
	
	def __init__(self, config: ModelConfig):
		"""Initialize the Groq provider."""
		super().__init__(config)
		api_key = config.get_api_key("GROQ_API_KEY")
		if not api_key:
			raise ValueError(f"API key for Groq model named '{config.name}' not set!")
		self.client = Groq(api_key=api_key)
	
	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using Groq's API.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message
		"""
		# Extract parameters
		provider_model = self.config.provider_model
		
		# Format messages for Groq API (same format as OpenAI)
		groq_messages = []
		
		for msg in messages:
			# Groq uses the same message format as OpenAI
			groq_messages.append({
				"role": msg["role"],
				"content": msg["content"]
			})
		
		# Create the request parameters
		request_params = {
			"model": provider_model,
			"messages": groq_messages,
			**params
		}
		
		# Make the API call
		response_dict = None
		try:
			response = self.client.chat.completions.create(**request_params)
			response_dict = response.dict()
			response_dict['tags'] = list(self.config.output_tags)
			#Ensure there is a message:
			msg = response_dict['choices'][0]['message']['content']
			return response_dict
		except Exception as e:
			raise ProviderException(GroqProvider.provider_name, e, response_dict)