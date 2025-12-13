"""
Client API for RequiredAI.
"""

from typing import List, Dict, Any, Optional
from .ModelConfig import ModelConfig, FallbackModel, all_model_configs
from .Requirement import Requirement, Requirements
import requests
import json

class RequiredAIClient:
	"""Client for making requests to a RequiredAI server."""
	
	def __init__(self, base_url: str):
		"""
		Initialize the RequiredAI client.
		
		Args:
			base_url: The base URL of the RequiredAI server
		"""
		self.base_url = base_url.rstrip('/')
		self.session = requests.Session()
		
		for model_name, model_config in all_model_configs.items():
			if isinstance(model_config, ModelConfig):
				self.add_model(model_config)
			elif isinstance(model_config, FallbackModel):
				self.add_fallback_model(model_config)
	
	def create_completion(
		self,
		model: str,
		messages: List[Dict[str, Any]],
		requirements: List[Requirement]=[],
		key: Optional[str] = None,
		initial_response: Optional[Dict[str, Any]] = None,
		**kwargs
	) -> Dict[str, Any]:
		"""
		Create a completion with requirements.
		
		Args:
			model: The model to use for the completion
			messages: The conversation messages
			requirements: The requirements to apply to the response
				(These will be in addition to any the model has built
				in and will take priority over them)
			key: Optional key for tracking/persisting the completion
			initial_response: Optional initial response to continue from.
				The last prospect in this will be used as the current prospect
				that we will re-evaluate the requirements for. Only use this
				if you have requirements that may have a stochastic return
				that you want to re-run, like asking a llm if a requirement
				is met.
			**kwargs: Additional parameters to pass to the API
			
		Returns:
			The API response as a dictionary
		"""
		endpoint = f"{self.base_url}/v1/chat/completions"
		
		payload = {
			"model": model,
			"messages": messages,
			"requirements": Requirements.to_dict(requirements)
		}
		if key is not None:
			payload["key"] = key
		if initial_response is not None:
			payload["initial_response"] = initial_response
		if kwargs:
			payload.update(kwargs)
		
		response = self.session.post(endpoint, json=payload)
		response.raise_for_status()
		
		return response.json()
	
	def get_completion_status(self, key: str) -> Dict[str, Any]:
		"""
		Get the status of a completion by key.
		
		Args:
			key: The key of the completion to check
			
		Returns:
			The status response as a dictionary
		"""
		endpoint = f"{self.base_url}/v1/chat/completion/status/{key}"
		
		response = self.session.get(endpoint)
		response.raise_for_status()
		
		return response.json()
	
	def stop_completion(self, key: str) -> Dict[str, Any]:
		"""
		Stop a running completion by key.
		
		Args:
			key: The key of the completion to stop
			
		Returns:
			The API response as a dictionary
		"""
		endpoint = f"{self.base_url}/v1/chat/completion/stop/{key}"
		
		response = self.session.post(endpoint)
		response.raise_for_status()
		
		return response.json()
	
	def add_model(self, model: ModelConfig) -> Dict[str, Any]:
		"""
		Send a model configuration to the server to be added to the ModelManager.
		
		Args:
			model: The Model instance containing the configuration
			
		Returns:
			The API response as a dictionary
		"""
		model.client = self
		endpoint = f"{self.base_url}/v1/models/add"
		
		response = self.session.post(endpoint, json=model.to_dict())
		response.raise_for_status()
		
		return response.json()
	
	def add_fallback_model(self, fallback: FallbackModel) -> Dict[str, Any]:
		"""
		Send a fallback model configuration to the server to be added to the ModelManager.
		
		Args:
			fallback: The FallbackModel instance containing the configuration
			
		Returns:
			The API response as a dictionary
		"""
		fallback.client = self
		endpoint = f"{self.base_url}/v1/models/fallback/add"
		
		response = self.session.post(endpoint, json=fallback.to_dict())
		response.raise_for_status()
		
		return response.json()