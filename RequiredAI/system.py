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
		# Generate a first draft response that we'll
		# check the requirements against after:
		print("Generating prospect...")
		prospective_response = ModelManager.singleton().complete_with_model(model_name, messages, params)
		
		# Start a log for the current prospective message
		# that will be attached to it as a audit trail for
		# the evaluation of each requirement we check it for:
		def add_eval_log_to(prospect:dict) -> list:
			eval_log = []
			prospect['requirements_evaluation_log'] = eval_log
			return eval_log
		
		def end_prospects_eval_log(eval_log:List[dict], requirements_met:bool, last_element_fields:Optional[Dict[str,Any]]={}) -> None:
			eval_log.append({
				"requirements_met":requirements_met,
				**last_element_fields
			})
		
		eval_log = add_eval_log_to(prospective_response)
		prospective_responses = [prospective_response]
		
		# Iteratively re-draft the response until all requirements are met:
		# (The only time this should ever stop is if the user stops it!)
		
		# TODO: implement a means for the client to stop a given chat completion
		# for instances such as conflicting requirements causing infinite loop
		# or long response time.
		#
		# There should not be a server side stopping condition.
		#
		# json for the return obj should be stored to ram disk at the eval of 
		# each requirement or re-draft, and the client should pass in a key
		# that the client can call cancel on via separate route or get
		# a status with or reload a response from ram disk after a server 
		# application crash (possibly another thread or process could move
		# to disk periodic). -- Client should also be able to continue from
		# a response object provided in the same format as we return (that way
		# they can continue to generate new candidate prospective messages
		# until again it is stopped early or the requirements are all met).
		# 
		# The client passing the key, means that it doesn't need to
		# wait for the server to provide it a one, and doesn't need a separate
		# async response function for chat completions. It can simply provide
		# an optional key to chat completion and then stop it or ask for status.
		# Those other functions, even if received earlier on the server than the
		# chat completion it-self, could simply wait for the chat completion to start.
		#
		# Clients however need to additionally have their host name passed into chat
		# completion so as to dis-ambiguity multiple clients with the same key passed.
		#
		# It will be client error if they pass same key concurrently with ambiguity.
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
				end_prospects_eval_log(eval_log, True)
				break
			
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
			new_response = ModelManager.singleton().complete_with_model(**revision_input)
			
			# Update the prospective response for the next
			# iteration, keeping an audit trail of our attempts:
			end_prospects_eval_log(eval_log, False, {
				"revision_input":revision_input,
				"revision_id":get_id(new_response)
			})
			prospective_response = new_response
			eval_log = add_eval_log_to(prospective_response)
			prospective_responses.append(prospective_response)
		
		# Construct and return the final response
		response = {
			"id": "reqai-" + self._generate_id(),
			"object": "chat.completion",
			"created": self._get_timestamp(),
			"model": model_name,
			"choices": [{
				"id":get_id(prospective_response),
				"message": get_msg(prospective_response),
				"finish_reason": get_finish_reason(prospective_response),
				"prospects":prospective_responses
			}],
			"model_config": ModelManager.singleton().model_configs[model_name].as_dict()
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