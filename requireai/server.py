"""
Server implementation for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import json
import os
from flask import Flask, request, jsonify
from .requirements import Requirements
from .model_manager import ModelManager, model_manager

class RequiredAIServer:
    """Server for handling RequiredAI requests."""
    
    def __init__(self, config_path: str):
        """
        Initialize the RequiredAI server.
        
        Args:
            config_path: Path to the server configuration file
        """
        self.app = Flask(__name__)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.revise_prompt_template = self.config.get("revise_prompt_template", 
            "Your previous response did not meet the following requirement: {requirement_prompt}. "
            "Please revise your response to meet this requirement.")
        
        # Initialize the model manager singleton
        if model_manager is None:
            ModelManager(self.config)
        
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
            
            try:
                # Verify the model exists
                model_config = model_manager.get_model_config(model_name)
            except ValueError:
                return jsonify({"error": f"Model {model_name} not configured"}), 400
            
            # Send the initial request to the model
            chat = data.get("messages", [])
            prospective_response = model_manager.complete_with_model(model_name, chat, data)
            
            # Process requirements
            choices = []
            if requirements:
                choices = self._process_requirements(
                    requirements, 
                    chat, 
                    prospective_response, 
                    model_config,
                    data
                )
            else:
                # If no requirements, just add the response as a single choice
                choices = [
                    {
                        "index": 0,
                        "message": prospective_response,
                        "finish_reason": "stop"
                    }
                ]
            
            # Construct the final response
            response = {
                "id": "reqai-" + self._generate_id(),
                "object": "chat.completion",
                "created": self._get_timestamp(),
                "model": model_name,
                "choices": choices
            }
            
            return jsonify(response)
    
    
    def _process_requirements(
        self,
        requirements: List[Dict[str, Any]],
        chat: List[Dict[str, Any]],
        prospective_response: Dict[str, Any],
        default_model_config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process requirements and revise the response if needed.
        
        Args:
            requirements: List of requirement specifications
            chat: The original conversation
            prospective_response: The current response from the model
            default_model_config: The default model configuration to use
            
        Returns:
            A list of choices with the final response and revision history
        """
        # Convert JSON requirements to requirement objects
        requirement_objects = [Requirements.from_json(req) for req in requirements]
        
        # Track choices directly
        choices = []
        
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
                # Add the successful response as a choice
                choices.append({
                    "message": prospective_response,
                    "finish_reason": "stop"
                })
                break
                
            print(f"Failed requirement: {failed_req.__class__.__web_name__}")
            
            # Create a revision prompt
            revision_prompt = {
                "role": "user",
                "content": self.revise_prompt_template.format(requirement_prompt=failed_req.prompt)
            }
            
            # Add the failed attempt to choices
            choices.append({
                "message": prospective_response,
                "finish_reason": "failed_requirement",
                "requirement_name": failed_req.__class__.__web_name__,
                "revision_prompt": revision_prompt
            })
                
            # Use the model from the requirement or fall back to the original model
            model_name = getattr(failed_req, "model", None) or data.get("model")
            
            # Get a new response using ModelManager
            revision_conversation = conversation + [revision_prompt]
            new_response = model_manager.complete_with_model(
                model_name,
                revision_conversation,
                {}  # No additional params needed
            )
            
            # Update the prospective response for the next iteration
            prospective_response = new_response
        
        # Reverse the choices and add indices
        choices.reverse()
        for i, choice in enumerate(choices):
            choice["index"] = i
            
        return choices
    
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
