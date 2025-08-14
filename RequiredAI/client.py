"""
Client API for RequiredAI.
"""

from typing import List, Dict, Any, Optional
from .ModelConfig import ModelConfig
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
    
    def create_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        requirements: List[Dict[str, Any]]=[],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a completion with requirements.
        
        Args:
            model: The model to use for the completion
            messages: The conversation messages
            requirements: The requirements to apply to the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response as a dictionary
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "requirements": requirements,
            **kwargs
        }
        
        response = self.session.post(endpoint, json=payload)
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
        endpoint = f"{self.base_url}/v1/models/add"
        
        response = self.session.post(endpoint, json=model.to_dict())
        response.raise_for_status()
        
        return response.json()