"""
Client API for RequiredAI.
"""

from typing import List, Dict, Any, Optional
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
        requirements: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a completion with requirements.
        
        Args:
            model: The model to use for the completion
            messages: The conversation messages
            requirements: The requirements to apply to the response
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response as a dictionary
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "requirements": requirements,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()
        
        return response.json()
