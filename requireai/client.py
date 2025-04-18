"""
Client API for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import requests
import json

class RequiredAIClient:
    """Client for making requests to a RequiredAI server."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the RequiredAI client.
        
        Args:
            base_url: The base URL of the RequiredAI server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
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
