"""
Anthropic provider for RequiredAI.
"""

import os
from typing import Dict, List, Any, Optional
import anthropic

from . import BaseModelProvider

class AnthropicProvider(BaseModelProvider):
    """Provider for Anthropic's Claude API."""
    
# Register the provider
BaseModelProvider.register_provider("anthropic", AnthropicProvider)
    
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
        provider_model = self.config["provider_model"]
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.7)
        
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
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages
        }
        
        # Only add system parameter if it exists
        if system_content:
            request_params["system"] = system_content
        
        # Make the API call
        try:
            response = self.client.messages.create(**request_params)
            
            # Return the response in the expected format
            return {
                "role": "assistant",
                "content": response.content[0].text
            }
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            raise

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Anthropic provider."""
        super().__init__(config)
        # Register this provider
        BaseModelProvider.register_provider("anthropic", AnthropicProvider)
        
        api_key = os.environ.get(config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError(f"API key environment variable {config.get('api_key_env')} not set")
        self.client = anthropic.Anthropic(api_key=api_key)
