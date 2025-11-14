from typing import List, Dict, Any, Optional
from ..ModelConfig import FallbackModel, ModelRetryParameters
from ..Requirement import Requirements
from ..system import RequiredAISystem
from . import BaseModelProvider, provider, ProviderException
import time

@provider('Fallback')
class FallbackProvider(BaseModelProvider):
	"""Provider for FallbackModel configurations."""

	def __init__(self, config: FallbackModel):
		super().__init__(config)
		self.config = config
		self.current_index = 0

	def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate a completion using the fallback mechanism.
		
		Args:
			messages: The conversation messages
			params: Additional parameters for the request
			
		Returns:
			The model's response message, wrapped with all attempts
		"""
		attempts = []

		def wrap_response(successful_response: Dict[str, Any]) -> Dict[str, Any]:
			# Wrap the successful response as the main choice, and add attempts & fallback tags
			wrapped = successful_response
			choice = wrapped.get('choices', [{}])[0]
			msg = choice.get('message', {})
			if 'tags' not in msg:
				msg['tags'] = []
			msg['tags'].extend(self.config.output_tags)
			wrapped['attempts'] = attempts
			return wrapped

		start_index = self.current_index

		for offset in range(len(self.config.models)):
			idx = (start_index + offset) % len(self.config.models)
			retry_params = self.config.models[idx]
			model_name = retry_params.model_name

			for attempt in range(retry_params.max_retry):
				try:
					inner_response = RequiredAISystem.singleton.chat_completions(
						model_name,
						self.config.requirements,
						messages,
						params
					)
					attempts.append(inner_response)

					# Check if the response is valid (done, no errors, valid finish_reason)
					if inner_response.get('done', False) and 'errors' not in inner_response.get('choices', [{}])[0] and inner_response.get('choices', [{}])[0].get('finish_reason') not in ['error', 'Stopped by client']:
						self.current_index = idx  # Update current index to this successful model
						return wrap_response(inner_response)
					
					# Retry delay if not the last attempt
					if attempt < retry_params.max_retry-1:
						time.sleep(retry_params.delay_between_retry)
				except Exception as e:
					error_response = {'error': str(e), 'model': model_name}
					attempts.append(error_response)
					if attempt < retry_params.max_retry-1:
						time.sleep(retry_params.delay_between_retry)

		# If all attempts fail, raise an error response with attempts
		raise ProviderException(FallbackProvider.provider_name, e, {
			"id": "fallback-error",
			"object": "chat.completion",
			"created": int(time.time()),
			"model": self.config.name,
			"choices": [],
			"attempts": attempts
		})