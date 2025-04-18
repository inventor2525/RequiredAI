"""
Server implementation for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import json
from flask import Flask, request, jsonify
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
                        "requirement_name": revision["failed_requirement"]["name"],
                        "requirement_type": revision["failed_requirement"]["type"],
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
        # This would be implemented to route to the appropriate AI provider
        # based on the model_config
        # For now, return a placeholder
        return {
            "role": "assistant",
            "content": "This is a placeholder response. In a real implementation, this would be the response from the AI model."
        }
    
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
        
        while True:
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
                break
                
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
                "failed_requirement": {
                    "type": failed_req.__class__.__web_name__,
                    "name": failed_req.name or failed_req.__class__.__web_name__
                },
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
