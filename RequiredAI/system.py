from typing import List, Dict, Any, Optional, ClassVar, Tuple
import json
from .Requirement import Requirements, Requirement, RequirementResult
from .ModelConfig import InputConfig, ModelConfigs, FallbackModel
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
		
		ModelManager(ModelConfigs.from_dict(self.config["models"]) + [FallbackModel.from_dict(fbm) for fbm in self.config["fallback_models"]])
		RequiredAISystem.singleton = self
		self.response_map: Dict[str, Dict[str, Any]] = {}
	
	def chat_completions(self, model_name:str, requirements:List[Requirement], messages:List[dict], params:dict={}, key: Optional[str] = None, initial_response: Optional[Dict[str, Any]] = None) -> dict:
		if initial_response is None:
			response = {
				"id": "reqai-" + self._generate_id(),
				"object": "chat.completion",
				"created": self._get_timestamp(),
				"model": model_name,
				"choices": [{
					"prospects": []
				}],
				"model_config": ModelManager.singleton().model_configs[model_name].as_dict(),
				"done":False
			}
		else:
			response = dict(initial_response)
			response.update({
				"id": "reqai-" + self._generate_id(),
				"created": self._get_timestamp(),
				"model": model_name,
				"model_config": ModelManager.singleton().model_configs[model_name].as_dict(),
				'initial_draft_response':initial_response['id'],
				"done":False
			})
		
		prospective_responses:List[dict] = response["choices"][0]["prospects"]
		
		if key:
			self.response_map[key] = response
		
		def errors(response:dict=response) -> list:
			c = response['choices'][0]
			if 'errors' not in c:
				c['errors'] = []
			return c['errors']
		def add_eval_log_to(prospect:dict) -> list:
			eval_log = []
			prospect['requirements_evaluation_log'] = eval_log
			return eval_log
		
		def end_prospects_eval_log(eval_log:List[dict], requirements_met:bool, last_element_fields:Optional[Dict[str,Any]]={}) -> None:
			eval_log.append({
				"requirements_met":requirements_met,
				**last_element_fields
			})
		
		def set_choice(prospect:dict, response:dict=response) -> None:
			choice = response["choices"][0]
			choice["id"] = get_id(prospect)
			choice["message"] = get_msg(prospect)
			choice["finish_reason"] = get_finish_reason(prospect)
		
		def stop(response:dict=response) -> bool:
			if response.get('should_stop', False):
				choice = response["choices"][0]
				choice["finish_reason"] = 'Stopped by client'
				return True
			return False
		
		if len(prospective_responses) == 0:
			# Generate a first draft response that we'll
			# check the requirements against after:
			print("Generating prospect...")
			try:
				completion_model = ModelManager.singleton().get_provider(model_name)
				prospective_response = ModelManager.singleton().complete_with_model(
					model_name,
					InputConfig.select_with(messages, completion_model.config.input_config),
					params
				)
			except Exception as e:
				response["choices"][0]["finish_reason"] = f"Error generating prospect"
				errors().append({
					'exception':e,
					'exception_type':type(e).__name__,
					'response':prospective_response
				})
				return response
			
			# Start a log for the current prospective message
			# that will be attached to it as a audit trail for
			# the evaluation of each requirement we check it for:
			eval_log = add_eval_log_to(prospective_response)
			prospective_responses.append(prospective_response)
			set_choice(prospective_response)
		else:
			# Duplicate the last prospective response and
			# replace the evaluation log with a new one:
			prospective_response = dict(response["choices"][0]["prospects"][-1])
			del prospective_response['requirements_evaluation_log']
			prospective_responses.append(prospective_response)
			eval_log = add_eval_log_to(prospective_response)
			set_choice(prospective_response)
		
		# Iteratively re-draft the response until all requirements are met:
		# (The only time this should ever stop is if the user stops it!)
		chat = list(messages)
		all_requirements_met = False
		while not all_requirements_met:
			# Create a conversation with the prospective response so that
			# any evaluations can optionally work with the whole conversation:
			conversation = chat + [get_msg(prospective_response)]
			
			# Evaluate each requirement (until one returns False):
			failed_req = None
			all_requirements_met = True
			for req in requirements:
				if stop(): # Stop though if the client told us to.
					end_prospects_eval_log(eval_log, False, {
						'checked_all_requirements':False
					})
					if key in self.response_map:
						del self.response_map[key]
					return response
				
				print(f"Evaluating {req.name}")
				try:
					req_evaluation = req.evaluate(conversation)
					eval_log.append(req_evaluation.evaluation_log)
					if not req_evaluation:
						print(f"{req.name} failed!")
						all_requirements_met = False
						failed_req = req
						break
				except Exception as e:
					errors().append({
						'exception':e,
						'exception_type':type(e).__name__,
						'requirement':req.name
					})
					response["choices"][0]["finish_reason"] = f"Error evaluating requirement"
					return response
			
			# If all requirements are met, we're done
			if all_requirements_met:
				end_prospects_eval_log(eval_log, True, {
					'checked_all_requirements':True
				})
				break
			
			if stop(): # Stop if the client told us to.
				if key in self.response_map:
					del self.response_map[key]
				return response
			
			# Else, Create a response revision prompt:
			revision_prompt = {
				"role": "user",
				"content": self.revise_prompt_template.format(requirement_prompt=failed_req.prompt)
			}
			
			# Use the revision model specified in the requirement 
			# or fall back to the original model if none was specified:
			corrector_model_name = getattr(failed_req, "revision_model", None) or model_name
			
			# Select from the chat what messages the revision
			# model is interested in in order to draft a revision
			# to the prospective message (this could include other
			# mid conversation system messages meant to aid or
			# further instruct in the revision process):
			revision_model = ModelManager.singleton().get_provider(corrector_model_name)
			conversation = InputConfig.select_with(chat, revision_model.config.input_config) + [get_msg(prospective_response)]
			
			# Generate a new candidate for the current prospective
			# message (which we will later again test requirements against):
			revision_conversation = conversation + [revision_prompt]
			revision_input = {
				"model_name": corrector_model_name,
				"messages": revision_conversation,
				"params": params
			}
			try:
				new_response = ModelManager.singleton().complete_with_model(**revision_input)
			except Exception as e:
				response["choices"][0]["finish_reason"] = f"Error generating prospect"
				errors().append({
					'exception':e,
					'exception_type':type(e).__name__,
					'response':prospective_response
				})
				return response
			
			# Update the prospective response for the next
			# iteration, keeping an audit trail of our attempts:
			end_prospects_eval_log(eval_log, False, {
				'checked_all_requirements':True,
				"revision_input":revision_input,
				"revision_id":get_id(new_response)
			})
			prospective_response = new_response
			eval_log = add_eval_log_to(prospective_response)
			prospective_responses.append(prospective_response)
			set_choice(prospective_response)
		
		response["done"] = True
		if key in self.response_map:
			del self.response_map[key]
		return response
	
	def chat_completion_status(self, key: str) -> Dict[str, Any]:
		if key in self.response_map:
			return self.response_map[key]
		else:
			return {"error": "key not found"}
	
	def stop_chat_completion(self, key: str):
		if key in self.response_map:
			self.response_map[key]['should_stop'] = True
	
	def _generate_id(self) -> str:
		"""Generate a unique ID for the response."""
		import uuid
		return str(uuid.uuid4())
	
	def _get_timestamp(self) -> int:
		"""Get the current timestamp."""
		import time
		return int(time.time())