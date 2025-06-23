from typing import List, Dict, Any, Optional, ClassVar
import json
from .requirements import Requirements
from .model_manager import ModelManager

class RequiredAISystem:
	"""System for handling RequiredAI chat completions."""
	singleton:ClassVar['RequiredAISystem']
	
	def __init__(self, config: Dict[str, Any]):
		"""
		Initialize the RequiredAI server.
		
		Args:
			config: System configuration
		"""
		self.config = config
		self.revise_prompt_template = self.config.get("revise_prompt_template", 
			"Your previous response did not meet the following requirement: {requirement_prompt}. "
			"Please revise your response to meet this requirement.")
		
		ModelManager(self.config)
		RequiredAISystem.singleton = self
		
	@staticmethod
	def load_config(config_path: str) -> Dict[str, Any]:
		"""
		Load the server configuration from a file.
		
		Args:
			config_path: Path to the configuration file
			
		Returns:
			The loaded configuration as a dictionary
		"""
		with open(config_path, 'r') as f:
			return json.load(f)
	
	def chat_completions(self, model_name:str, requirements:list, messages:List[dict], params:dict) -> dict:
		prospective_response = ModelManager.singleton().complete_with_model(model_name, messages, params)
		
		# Process requirements
		choices = []
		if requirements:
			requirement_objects = Requirements.from_json(requirements)
		
			# Track choices directly
			chat = list(messages)
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
				corrector_model_name = getattr(failed_req, "model", None) or model_name
				
				# Get a new response using ModelManager
				revision_conversation = conversation + [revision_prompt]
				new_response = ModelManager.singleton().complete_with_model(
					corrector_model_name,
					revision_conversation,
					{}  # No additional params needed
				)
				
				# Update the prospective response for the next iteration
				prospective_response = new_response
			
			# Reverse the choices and add indices
			choices.reverse()
			for i, choice in enumerate(choices):
				choice["index"] = i
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
		
		return response
	
	def _generate_id(self) -> str:
		"""Generate a unique ID for the response."""
		import uuid
		return str(uuid.uuid4())
	
	def _get_timestamp(self) -> int:
		"""Get the current timestamp."""
		import time
		return int(time.time())