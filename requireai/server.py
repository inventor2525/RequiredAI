"""
Server implementation for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import json
import os
from flask import Flask, request, jsonify
from .requirements import Requirements
from .model_manager import ModelManager
from .system import RequiredAISystem

# Import providers to register them
from .providers import BaseModelProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.groq_provider import GroqProvider
from .providers.requiredai_provider import RequiredAIProvider

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
        self.config = RequiredAISystem.load_config(self.config_path)
        self.system = RequiredAISystem(self.config)
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up the Flask routes."""
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            data = request.json
            
            # Extract parameters
            model_name = data.get("model")
            requirements = data.get("requirements", [])
            messages = data.get("messages", [])
            params = {k: v for k, v in data.items() if k not in set(["model", "requirements", "messages"])}
            
            response = self.system.chat_completions(model_name, requirements, messages, params)
            if "error" in response:
                return jsonify(response), 400
            return jsonify(response)
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """
        Run the Flask server.
        
        Args:
            host: The host to run on
            port: The port to run on
            debug: Whether to run in debug mode
        """
        self.app.run(host=host, port=port, debug=debug)
