"""
Gemini provider for RequiredAI.
"""

import os
import uuid
import time
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from ..ModelConfig import ModelConfig

from . import BaseModelProvider, provider, ProviderException

@provider('gemini')
class GeminiProvider(BaseModelProvider):
	"""Provider for Google Gemini API."""

	def __init__(self, config: ModelConfig):
		"""
		Initialize the Gemini provider.
		
		Args:
			config: Configuration for the model.
		"""
		super().__init__(config)
		api_key = config.get_api_key("GEMINI_API_KEY")
		if not api_key:
			raise ValueError(f"API key for Gemini model named '{config.name}' not set!")
		self.client = genai.Client(api_key=api_key)

	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using Google Gemini API.

		Args:
			messages: The conversation messages in OpenAI-like format.
			params: Additional parameters for the request (e.g., max_tokens, temperature).

		Returns:
			The model's response message in OpenAI-like dictionary format.
		"""
		# Extract parameters
		provider_model_name = self.config.provider_model
		max_tokens = params.get("max_tokens", 1024)
		temperature = params.get("temperature", 0.7)

		# Extract initial system instructions
		system_instruction = ""
		i = 0
		while i < len(messages) and messages[i].get("role") == "system":
			if system_instruction:
				system_instruction += "\n\n"
			system_instruction += messages[i]["content"]
			i += 1

		# Process remaining messages
		gemini_contents = []
		for msg in messages[i:]:
			role = msg.get("role")
			content = msg["content"]
			if role == "system":
				role = "user"
				content = "from system:\n" + content
			elif role == "assistant":
				role = "model"
			else:
				role = "user"
			gemini_contents.append({"role": role, "parts": [{"text": content}]})

		# Prepare config
		generation_config = types.GenerateContentConfig(
			max_output_tokens=max_tokens,
			temperature=temperature,
			system_instruction=system_instruction if system_instruction else None
		)

		# Make the API call
		response = None
		try:
			response = self.client.models.generate_content(
				model=provider_model_name,
				contents=gemini_contents,
				config=generation_config
			)
			
			# Convert Gemini response to OpenAI-like format
			if response.candidates:
				candidate = response.candidates[0]
				content_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
				
				finish_reason = str(candidate.finish_reason).split('.')[-1].lower()

				response_id = str(uuid.uuid4())
				response_created = int(time.time())
				
				return {
					"id": f"gemini-chatcmpl-{response_id}",
					"object": "chat.completion",
					"created": response_created,
					"model": provider_model_name,
					"choices": [{
						"message": {
							"role": "assistant",
							"content": content_text
						},
						"finish_reason": finish_reason
					}],
					'raw':response.model_dump(),
					"tags": list(self.config.output_tags)
				}
			else:
				# Handle cases where no candidates are returned (e.g., safety block)
				if response.prompt_feedback and response.prompt_feedback.block_reason:
					raise ValueError(f"Gemini API blocked response due to: {response.prompt_feedback.block_reason.name}")
				raise ValueError("Gemini API returned no candidates or an empty response")

		except Exception as e:
			raise ProviderException(GeminiProvider.provider_name, e, None if not response else response.model_dump())
