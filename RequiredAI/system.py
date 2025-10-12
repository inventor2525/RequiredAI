from typing import List, Dict, Any, Optional, ClassVar, Tuple
import json
from .Requirement import Requirements, Requirement, RequirementResult
from .ModelConfig import InputConfig, ModelConfigs
from .ModelManager import ModelManager
from .helpers import *

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
		self.revise_prompt_template = "Your previous response did not meet the following requirement: {requirement_prompt} Please revise your response to meet this requirement."
		
		ModelManager(ModelConfigs.from_dict(self.config["models"]))
		RequiredAISystem.singleton = self
	
	def chat_completions(self, model_name:str, requirements:List[Requirement], messages:List[dict], params:dict={}) -> dict:
		print("Generating prospect...")
		prospective_response = ModelManager.singleton().complete_with_model(model_name, messages, params)
		def prospect_to_choice(prospect:dict, prospects:list) -> dict:
			return {
				"id":get_id(prospect),
				"message": get_msg(prospect),
				"finish_reason": get_finish_reason(prospect),
				"prospects":prospects
			}
		
		def prospect_to_prospect_info(prospect:dict, log_element:Optional[dict]=None) -> Tuple[dict, list]:
			eval_log = [log_element] if log_element else []
			return {
				'prospect':prospect,
				'eval_log':eval_log
			}, eval_log
		
		def eval_log_end(eval_log:List[dict], requirements_met:bool, last_element_fields:Optional[Dict[str,Any]]={}) -> None:
			eval_log.append({
				"requirements_met":requirements_met,
				**last_element_fields
			})
		
		# Process requirements
		prospect_info, eval_log = prospect_to_prospect_info(prospective_response)
		prospective_responses = [prospect_info]
		
		# Track choices directly
		chat = list(messages)
		all_requirements_met = False
		while not all_requirements_met:
			# Create a conversation with the prospective response
			conversation = chat + [get_msg(prospective_response)]
			
			failed_req = None
			all_requirements_met = True
			for req in requirements:
				print(f"Evaluating {req.name}")
				req_evaluation = req.evaluate(conversation)
				eval_log.append(req_evaluation.evaluation_log)
				if not req_evaluation:
					print(f"{req.name} failed!")
					all_requirements_met = False
					failed_req = req
					break
			
			# If all requirements are met, we're done
			if all_requirements_met:
				eval_log_end(eval_log, True)
				break
			
			# Create a revision prompt
			revision_prompt = {
				"role": "user",
				"content": self.revise_prompt_template.format(requirement_prompt=failed_req.prompt)
			}
			
			# Use the model from the requirement or fall back to the original model
			corrector_model_name = getattr(failed_req, "revision_model", None) or model_name
			#get from chat using 'revision_model''s  input_config.   default input_config to a new function that returns a config for 'all' (un-touched messages)
			
			revision_model = ModelManager.singleton().get_provider(corrector_model_name)
			conversation = InputConfig.select_with(chat, revision_model.config.input_config) + [get_msg(prospective_response)]
			
			# Get a new response using ModelManager
			revision_conversation = conversation + [revision_prompt]
			revision_input = {
				"model_name": corrector_model_name,
				"messages": revision_conversation,
				"params": params
			}
			new_response = ModelManager.singleton().complete_with_model(**revision_input)
			
			eval_log_end(eval_log, False, {
				"revision_input":revision_input,
				"revision_id":get_id(new_response)
			})
			
			# Update the prospective response for the next iteration
			prospective_response = new_response
			prospect_info, eval_log = prospect_to_prospect_info(prospective_response)
			prospective_responses.append(prospect_info)
		
		# Construct the final response
		response = {
			"id": "reqai-" + self._generate_id(),
			"object": "chat.completion",
			"created": self._get_timestamp(),
			"model": model_name,
			"choices": [prospect_to_choice(prospective_response, prospective_responses)]
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