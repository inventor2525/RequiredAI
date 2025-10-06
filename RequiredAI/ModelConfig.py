from typing import List, Dict, Any, Optional, Tuple, Union
from .Requirement import Requirements, Requirement
from .json_dataclass import *
from .helpers import *
import os

@json_dataclass
class InputConfig:
	"""
	Configuration for how conversation context is presented to an evaluation or revision model.
	"""
	messages_to_include: None | int | Dict[str,str] | Tuple[int, int] | List[int | Dict[str,str] | Tuple[int, int]] = -1
	'''Indexes or range's of messages to include from a source conversation, and or new message dictionaries.'''
	
	filter_roles: None | List[str] | Dict[str, bool] = field(default=None)
	'''
	Message roles to optionally filter in or out of a source conversation.
	
	A list of strings will be taken as roles to include.
	
	A dictionary[string, bool] will be taken as roles to include or exclude (true, false).
	'''
	
	filter_tags: None | List[None | str] | Dict[None|str, bool] = field(default=None)
	'''
	Tags by which incoming messages will be optionally filtered.
	
	A list of strings will be taken as tags to include.
	
	A dictionary[string, bool] will be taken as tags to include or exclude (true, false).
	
	If none is in the list or dictionary, it will match messages that do not have tags attached.
	'''
	
	@staticmethod
	def all():
		'''
		Returns a input config that returns the original conversation
		that is passed to it, basically not effecting it at all.
		'''
		return InputConfig((0,-1))
	
	@staticmethod
	def select_with(messages: List[Dict[str, str]], context_configs:Union[None, 'InputConfig', List['InputConfig']]) -> List[Dict[str, str]]:
		if context_configs is None:
			return messages
		
		if isinstance(context_configs, InputConfig):
			return context_configs.select(messages)
		
		new_messages = []
		for context_config in context_configs:
			new_messages.extend(context_config.select(messages))
		return new_messages
	
	def select(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
		"""
		Creates a new list of messages based on this input configuration.

		This method processes an input list of messages, potentially including multiple
		system messages throughout or messages from other roles and first filters them
		by role (if filter_roles is configured), then filters by tags (if filter_tags)
		is configured, and then gets messages_to_include by index (or range of indices)
		from that filtered list of messages. Note that it is that filtered list of messages
		that the indices & range's operate on.
		
		messages_to_include is used to select any number of indexes or ranges of messages from
		the filtered conversation, or to inject new message dictionaries anywhere in that set
		of messages. - If None, all role & tag filtered messages will be returned.

		Args:
			messages: The full list of original conversation messages, where each message
					is a dictionary with at least 'role' and 'content' keys.

		Returns:
			A new set of messages, selected from messages and possibly including new messages
			from messages_to_include, if it had any message dictionaries in it.
		"""
		if self.filter_roles:
			filter_roles = self.filter_roles
			if isinstance(filter_roles, list):
				filter_roles = {role:True for role in filter_roles}
				all_roles_include = True
			else:
				all_roles_include = all(filter_roles.values())
			role_filtered_messages = []
			for message in messages:
				role = message.get('role',None)
				if all_roles_include:
					if role in filter_roles:
						role_filtered_messages.append(message)
				elif filter_roles.get(role, True):
					role_filtered_messages.append(message)
			messages = role_filtered_messages
				
		if self.filter_tags:
			filter_tags = self.filter_tags
			if isinstance(filter_tags, list):
				filter_tags = {tag:True for tag in filter_tags}
			some_tags_include = any(filter_tags.values())
			all_tags_include = all(filter_tags.values())
			tag_filtered_messages = []
			for message in messages:
				# figure out if to include the message based on how tags are configured:
				tags = message.get('tags',[])
				if not tags:
					if all_tags_include:
						include_msg = None in filter_tags
					else:
						include_msg = filter_tags.get(None, True)
				
				elif all_tags_include:
					include_msg = False
					for tag in tags:
						if tag in filter_tags:
							include_msg = True
							break
				elif some_tags_include:
					include_msg = False
					for tag in tags:
						tag_val = filter_tags.get(tag, None)
						if tag_val:
							include_msg = True
						if tag_val == False:
							include_msg = False
							break
				else:
					include_msg = True
					for tag in tags:
						tag_val = filter_tags.get(tag, True)
						if not tag_val:
							include_msg = False
							break
				
				if include_msg:
					tag_filtered_messages.append(message)
			messages = tag_filtered_messages
		
		def _inner_get_messages(index_msg_or_range: int | Dict[str,str] | Tuple[int, int]) -> List[Dict[str, str]]:
			"""Helper function to extract messages based on a single index or a range."""
			selected_messages = []
			conv_len = len(messages)

			if isinstance(index_msg_or_range, int):
				idx = index_msg_or_range
				if idx < 0:
					idx = conv_len + idx
				
				if 0 <= idx < conv_len:
					selected_messages.append(messages[idx])
			elif isinstance(index_msg_or_range, dict):
				selected_messages.append(index_msg_or_range)
			elif (isinstance(index_msg_or_range, tuple) or isinstance(index_msg_or_range, list)) and len(index_msg_or_range) == 2:
				start_orig, end_orig = index_msg_or_range

				start_idx = conv_len + start_orig if start_orig < 0 else start_orig
				end_idx = conv_len + end_orig if end_orig < 0 else end_orig
				
				if start_idx <= end_idx: # Forward iteration
					for i in range(start_idx, end_idx + 1):
						if 0 <= i < conv_len:
							selected_messages.append(messages[i])
				else: # Backward iteration
					for i in range(start_idx, end_idx - 1, -1):
						if 0 <= i < conv_len:
							selected_messages.append(messages[i])
			return selected_messages
		
		if not self.messages_to_include:
			return messages
		
		new_conversation_messages: List[Dict[str, str]] = []
		if isinstance(self.messages_to_include, list):
			for item in self.messages_to_include:
				new_conversation_messages.extend(_inner_get_messages(item))
		elif self.messages_to_include is not None:
			new_conversation_messages.extend(_inner_get_messages(self.messages_to_include))
		
		return new_conversation_messages

from dataclasses_json import config

all_model_configs:Dict[str, 'ModelConfig'] = {}

@json_dataclass
class ModelConfig:
	'''
	Configuration for a Large Language Model and optionally additional input/output filters.
	'''
	
	name: str
	'''Name this model will be referred to by the RequiredAI package.'''
	
	provider: str
	'''Company/framework name that facilitates this model. eg, groq, RequiredAI, gemini, anthropic, etc.'''
	
	provider_model: str
	'''Name the provider refers to this model as.'''
	
	api_key_env: Optional[str] = None
	'''Each provider has a default api key environment variable, but if you wish to specify separate keys for different sets of models then you can set a different environment variable here.'''
	
	requirements: Optional[List[Requirement]] = field(default=None, metadata=config(
			decoder=Requirements.from_dict, encoder=Requirements.to_dict
		)
	)
	'''Any requirements you wish this model to follow. Any response that does not follow any requirement will be re-drafted and all requirements checked again.'''
	
	input_config: None | InputConfig | List[InputConfig] = field(default=None)
	'''This controls how this model selects from it's input. Filter by role, tag, message index or slicing.'''
	
	output_tags: List[str] = field(default_factory=list)
	'''A list of tags that will be added to each message produced by the model.'''
	
	def __post_init__(self):
		all_model_configs[self.name] = self
	
	def get_api_key(self, default_env_var:Optional[str]=None) -> Optional[str]:
		env_var = self.api_key_env or default_env_var
		if env_var:
			return os.environ.get(env_var)
		return None

def InheritedModel(name:str, base_model:ModelConfig, requirements:List[Requirement]=None, input_config:InputConfig=None, output_tags:List[str]=[]) -> ModelConfig:
	'''
	Produces a model that filters the input, passes it to base model,
	and ensures all requirements are met for any output.
	'''
	return ModelConfig(
		name=name,
		provider='RequiredAI',
		provider_model=base_model.name,
		requirements=requirements,
		input_config=input_config,
		output_tags=output_tags
	)

class ModelConfigs:
	"""
	Handles serialization and deserialization of n ModelConfig objects.
	"""
	@staticmethod
	def to_dict(configs: Union[ModelConfig, List[ModelConfig]]) -> Dict[str, Any] | List[Dict[str, Any]]:
		if isinstance(configs, list):
			return [config.to_dict() for config in configs]
		return configs.to_dict()
	
	@staticmethod
	def from_dict(config_dicts: Dict[str, Any] | List[Dict[str, Any]]) -> ModelConfig | List[ModelConfig]:
		if isinstance(config_dicts, list):
			return [ModelConfig.from_dict(config_dict) for config_dict in config_dicts]
		return ModelConfig.from_dict(config_dicts)