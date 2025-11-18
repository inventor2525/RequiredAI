"""
Server implementation for RequiredAI.
"""

from typing import List, Dict, Any, Optional
import json
import os
from flask import Flask, request, jsonify
from .Requirement import *
from .RequirementTypes import *
from .ModelManager import ModelManager
from .ModelConfig import ModelConfig, FallbackModel
from .system import RequiredAISystem

# Import providers to register them
from .providers import BaseModelProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.groq_provider import GroqProvider
from .providers.requiredai_provider import RequiredAIProvider
from .providers.gemini_provider import GeminiProvider
from .providers.fallback_provider import FallbackProvider

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
		try:
			with open(config_path, 'r') as f:
				self.config = json.load(f)
			if "models" not in self.config:
				self.config["models"] = []
			if "fallback_models" not in self.config:
				self.config["fallback_models"] = []
		except:
			self.config = {"models":[], "fallback_models":[]}
		self.system = RequiredAISystem(self.config)
		
		self._setup_routes()
	
	def _setup_routes(self):
		"""Set up the Flask routes."""
		
		@self.app.route('/v1/chat/completions', methods=['POST'])
		def chat_completions():
			data = request.json
			
			# Extract parameters
			model_name = data.get("model")
			requirements = Requirements.from_dict(data.get("requirements", []))
			messages = data.get("messages", [])
			key = data.get("key", None)
			initial_response = data.get("initial_response", None)
			params = {k: v for k, v in data.items() if k not in set(["model", "requirements", "messages", "key", "initial_response"])}
			
			response = self.system.chat_completions(model_name, requirements, messages, params, key, initial_response)
			if "error" in response:
				return jsonify(response), 400
			return jsonify(response)
		
		@self.app.route('/v1/chat/completion/status/<key>', methods=['GET'])
		def chat_completion_status(key):
			try:
				response = self.system.chat_completion_status(key)
				if "error" in response:
					return jsonify(response), 404
				return jsonify(response)
			except Exception as e:
				return jsonify({"error": str(e)}), 500
		
		@self.app.route('/v1/chat/completion/stop/<key>', methods=['POST'])
		def stop_chat_completion(key):
			try:
				self.system.stop_chat_completion(key)
				return jsonify({"message": f"Stopped completion {key}"})
			except Exception as e:
				return jsonify({"error": str(e)}), 500
		
		@self.app.route('/v1/models/add', methods=['POST'])
		def add_model():
			"""
			Add or override a model configuration in the ModelManager.
			Expects a JSON payload with the model configuration.
			"""
			data = request.json
			if not data:
				return jsonify({"error": "No configuration provided"}), 400
			
			try:
				model_name = data.get("name")
				if not model_name:
					return jsonify({"error": "Model name is required"}), 400
				
				# Update ModelManager's configuration
				model_manager = ModelManager.singleton()
				model_manager.model_configs[model_name] = ModelConfig.from_dict(data)
				# Clear any existing provider instance to force reinitialization
				model_manager.provider_instances.pop(model_name, None)
				
				index_found = None
				for model_indx, model_config in enumerate(self.config["models"]):
					if model_config.get('name', None) == data['name']:
						index_found = model_indx
						break
					
				if index_found is None:
					self.config["models"].append(data)
				else:
					self.config["models"][index_found] = data
				
				# Save updated configuration to disk
				with open(self.config_path, 'w') as f:
					json.dump(self.config, f, indent=4)
				
				return jsonify({"message": f"Model {model_name} added or updated successfully"})
			except Exception as e:
				return jsonify({"error": f"Failed to add model: {str(e)}"}), 500
		
		@self.app.route('/v1/models/fallback/add', methods=['POST'])
		def add_fallback_model():
			"""
			Add or override a fallback model configuration in the ModelManager.
			Expects a JSON payload with the fallback model configuration.
			"""
			data = request.json
			if not data:
				return jsonify({"error": "No configuration provided"}), 400
			
			try:
				model_name = data.get("name")
				if not model_name:
					return jsonify({"error": "Model name is required"}), 400
				
				# Update ModelManager's configuration
				model_manager = ModelManager.singleton()
				model_manager.model_configs[model_name] = FallbackModel.from_dict(data)
				# Clear any existing provider instance to force reinitialization
				model_manager.provider_instances.pop(model_name, None)
				
				index_found = None
				for model_indx, model_config in enumerate(self.config.get("fallback_models", [])):
					if model_config.get('name', None) == data['name']:
						index_found = model_indx
						break
					
				if index_found is None:
					self.config["fallback_models"].append(data)
				else:
					self.config["fallback_models"][index_found] = data
				
				# Save updated configuration to disk
				with open(self.config_path, 'w') as f:
					json.dump(self.config, f, indent=4)
				
				return jsonify({"message": f"Fallback model {model_name} added or updated successfully"})
			except Exception as e:
				return jsonify({"error": f"Failed to add fallback model: {str(e)}"}), 500
	
	def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
		"""
		Run the Flask server.
		
		Args:
			host: The host to run on
			port: The port to run on
			debug: Whether to run in debug mode
		"""
		self.app.run(host=host, port=port, debug=debug)