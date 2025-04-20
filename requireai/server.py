"""
Server implementation for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import json
import os
from flask import Flask, request, jsonify
import anthropic
from .requirements import Requirements

class RequiredAIServer:
    """Server for handling RequiredAI requests."""
    
    def __init__(self, config_path: str):
        """
        Initialize the RequiredAI server.
        
        Args:
            config_path: Path to the server configuration file
        """
        self.app = Flask(__name__)
        self.config = self._load_config(config_path)
        self.revise_prompt_template = self.config.get("revise_prompt_template", 
            "Your previous response did not meet the following requirement: {requirement_prompt}. "
            "Please revise your response to meet this requirement.")
        
        self._setup_routes()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the server configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The loaded configuration as a dictionary
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_routes(self):
        """Set up the Flask routes."""
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            data = request.json
            
            # Extract requirements and remove from request to the model
            requirements = data.pop("requirements", [])
            
            # Get the model to use
            model_name = data.get("model")
            model_config = self._get_model_config(model_name)
            
            if not model_config:
                return jsonify({"error": f"Model {model_name} not configured"}), 400
            
            # Send the initial request to the model
            chat = data.get("messages", [])
            prospective_response = self._complete_with_model(model_config, chat, data)
            
            # Process requirements
            revision_history = []
            if requirements:
                prospective_response, revision_history = self._process_requirements(
                    requirements, 
                    chat, 
                    prospective_response, 
                    model_config
                )
            
            # Construct the choices array
            choices = [
                {
                    "index": 0,
                    "message": prospective_response,
                    "finish_reason": "stop"
                }
            ]
            
            # Add revision history to choices in reverse order (newest to oldest, excluding the final one)
            for i, revision in enumerate(revision_history[:-1][::-1], 1):
                if revision["failed_requirement"] is not None:
                    choices.append({
                        "index": i,
                        "message": revision["message"],
                        "finish_reason": "failed_requirement",
                        "requirement_name": revision["failed_requirement"],
                        "revision_prompt": revision["revision_prompt"]
                    })
            
            # Construct the final response
            response = {
                "id": "reqai-" + self._generate_id(),
                "object": "chat.completion",
                "created": self._get_timestamp(),
                "model": model_name,
                "choices": choices
            }
            
            return jsonify(response)
    
    def _get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a specific model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The model configuration or None if not found
        """
        models = self.config.get("models", {})
        return models.get(model_name)
    
    def _complete_with_model(
        self, 
        model_config: Dict[str, Any], 
        messages: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a completion request to the specified model.
        
        Args:
            model_config: Configuration for the model
            messages: The conversation messages
            params: Additional parameters for the request
            
        Returns:
            The model's response message
        """
        provider = model_config.get("provider", "").lower()
        
        if provider == "anthropic":
            return self._complete_with_anthropic(model_config, messages, params)
        elif provider == "openai":
            return self._complete_with_openai(model_config, messages, params)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    def _complete_with_anthropic(
        self,
        model_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a completion request to Anthropic's Claude API.
        
        Args:
            model_config: Configuration for the model
            messages: The conversation messages
            params: Additional parameters for the request
            
        Returns:
            The model's response message
        """
        # Get API key from environment variable
        api_key = os.environ.get(model_config.get("api_key_env", "ANTHROPIC_API_KEY"))
        if not api_key:
            raise ValueError(f"API key environment variable {model_config.get('api_key_env')} not set")
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Extract parameters
        provider_model = model_config.get("provider_model", "claude-3-5-haiku-latest")
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.7)
        
        # Format messages for Anthropic API
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Skip system messages as they'll be handled separately
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
        
        # Extract system message if present
        system_content = None
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
                break
        
        # Make the API call
        try:
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
            
            print(f"Using Anthropic model: {provider_model}")
            
            # Make the API call
            response = client.messages.create(**request_params)
            
            # Return the response in the expected format
            return {
                "role": "assistant",
                "content": response.content[0].text
            }
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            print(f"Request details: model={provider_model}")
            
            # If it's a model not found error, try with a fallback model
            if "not_found_error" in str(e) and "model:" in str(e):
                try:
                    print(f"Trying fallback model: claude-3-haiku-20240307")
                    request_params["model"] = "claude-3-haiku-20240307"
                    response = client.messages.create(**request_params)
                    return {
                        "role": "assistant",
                        "content": response.content[0].text
                    }
                except Exception as fallback_error:
                    print(f"Fallback model also failed: {str(fallback_error)}")
            
            # For debugging, but avoid printing full messages for privacy
            print(f"Number of messages: {len(anthropic_messages)}")
            print(f"System content exists: {system_content is not None}")
            
            # Return a simple error response instead of raising
            return {
                "role": "assistant",
                "content": f"I encountered an error when trying to process your request. Please try again or contact support if the issue persists."
            }
            
    def _complete_with_openai(
        self,
        model_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a completion request to OpenAI's API.
        
        Args:
            model_config: Configuration for the model
            messages: The conversation messages
            params: Additional parameters for the request
            
        Returns:
            The model's response message
        """
        # This is a placeholder for OpenAI implementation
        # Will be implemented in a future update
        raise NotImplementedError("OpenAI integration not yet implemented")
    
    def _process_requirements(
        self,
        requirements: List[Dict[str, Any]],
        chat: List[Dict[str, Any]],
        prospective_response: Dict[str, Any],
        default_model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process requirements and revise the response if needed.
        
        Args:
            requirements: List of requirement specifications
            chat: The original conversation
            prospective_response: The current response from the model
            default_model_config: The default model configuration to use
            
        Returns:
            A tuple containing:
            - The final response message that meets all requirements
            - A list of revision attempts with their prompts and failed requirements
        """
        # Convert JSON requirements to requirement objects
        requirement_objects = [Requirements.from_json(req) for req in requirements]
        
        # Track revision history
        revision_history = []
        
        # Add the initial response to the history
        revision_history.append({
            "message": prospective_response,
            "failed_requirement": None,
            "revision_prompt": None
        })
        
        while True:  # No iteration limit - continue until all requirements are met
            print("Checking requirements...")
            
            # Check if all requirements are met
            all_met = True
            failed_req = None
            
            # Create a conversation with the prospective response
            conversation = chat + [prospective_response]
            
            for req in requirement_objects:
                if not req.evaluate(conversation):
                    all_met = False
                    failed_req = req
                    break
            
            # If all requirements are met, we're done
            if all_met:
                print("All requirements met!")
                break
                
            print(f"Failed requirement: {failed_req.__class__.__web_name__}")
                
            # Get the model to use for revision
            model_name = getattr(failed_req, "model", None)
            model_config = self._get_model_config(model_name) if model_name else default_model_config
            
            # Create a revision prompt
            revision_prompt = {
                "role": "user",
                "content": self.revise_prompt_template.format(requirement_prompt=failed_req.prompt)
            }
            
            # Get a new response
            revision_conversation = conversation + [revision_prompt]
            new_response = self._complete_with_model(
                model_config, 
                revision_conversation,
                {}  # No additional params for revision
            )
            
            # Add this revision attempt to the history
            revision_history.append({
                "message": prospective_response,
                "failed_requirement": failed_req.__class__.__web_name__,
                "revision_prompt": revision_prompt
            })
            
            # Update the prospective response for the next iteration
            prospective_response = new_response
        
        # Return the final response and the revision history
        return prospective_response, revision_history
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the response."""
        import uuid
        return str(uuid.uuid4())
    
    def _get_timestamp(self) -> int:
        """Get the current timestamp."""
        import time
        return int(time.time())
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """
        Run the Flask server.
        
        Args:
            host: The host to run on
            port: The port to run on
            debug: Whether to run in debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
