"""
Anthropic provider for RequiredAI.
"""

import os
from typing import Dict, List, Any, Optional
import anthropic

from ..ModelConfig import ModelConfig
from . import BaseModelProvider, provider, ProviderException

@provider('anthropic')
class AnthropicProvider(BaseModelProvider):
	"""Provider for Anthropic's Claude API."""
	
	def __init__(self, config: ModelConfig):
		"""Initialize the Anthropic provider."""
		super().__init__(config)
		api_key = config.get_api_key("ANTHROPIC_API_KEY")
		if not api_key:
			raise ValueError(f"API key for Anthropic model named '{config.name}' not set!")
		self.client = anthropic.Anthropic(api_key=api_key)
	
	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using Anthropic's Claude API.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message
		"""
		# Extract parameters
		provider_model = self.config.provider_model
		
		# Format messages for Anthropic API
		anthropic_messages = []
		system_content = None
		
		for msg in messages:
			if msg["role"] == "system":
				system_content = msg["content"]
				continue
				
			# Map roles to Anthropic's expected format
			role = msg["role"]
			if role == "assistant":
				anthropic_role = "assistant"
			else:
				anthropic_role = "user"
				
			# Format content as expected by Anthropic
			anthropic_messages.append({
				"role": anthropic_role,
				"content": msg["content"]
			})
		
		# Create the request parameters
		request_params = {
			"model": provider_model,
			"messages": anthropic_messages,
			**params
		}
		
		# Only add system parameter if it exists
		if system_content:
			request_params["system"] = system_content
		
		# Make the API call
		response_dict = None
		try:
			response = self.client.messages.create(**request_params)
			response_dict = response.dict()
			response_dict['tags'] = list(self.config.output_tags)
			return response_dict
		except Exception as e:
			raise ProviderException(AnthropicProvider.provider_name, e, response_dict)

