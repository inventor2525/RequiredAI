from typing import List, Dict, Any, Optional, Tuple, Union
from .Requirement import Requirements, Requirement
from dataclasses import dataclass, field, asdict

@dataclass
class ContextOriginConfig:
    """
    Configuration for how conversation context is presented to an evaluation or revision model.
    """
    include_original_system_message: bool = False
    messages_to_include: Optional[Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]]] = -1
    custom_system_message: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the context origin configuration to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representing the context origin configuration.
        """
        result = asdict(self)
        # Remove keys with None values for cleaner JSON output
        return {k: v for k, v in result.items() if v is not None}

    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> 'ContextOriginConfig':
        """
        Create a ContextOriginConfig instance from a JSON dictionary.
        
        Args:
            json_dict: JSON dictionary representing a context origin configuration.
            
        Returns:
            A ContextOriginConfig instance.
        """
        return ContextOriginConfig(
            include_original_system_message=json_dict.get("include_original_system_message", False),
            messages_to_include=json_dict.get("messages_to_include", -1),
            custom_system_message=json_dict.get("custom_system_message", None)
        )
    
    def create_messages_from(self, messages: List[Dict[str, str]], initial_system_message: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Creates a new list of conversation messages based on the configured context origin.

        This method processes an input list of messages, potentially including multiple
        system messages throughout and a selection of other messages but it treats the first
        one as the 'original system message'.
        
        messages_to_include is used to select any number of indexes or ranges of messages from
        the source conversation, indexed from the message after any first system message.

        Args:
            initial_system_message: This can be supplied if the messages in the conversation are
                                    treated as separate from the system message by the model used
                                    to create the source conversation. If one is not supplied it
                                    will be searched for in first element of messages.
            messages: The full list of original conversation messages, where each message
                      is a dictionary with 'role' and 'content' keys.

        Returns:
            A new list of messages representing the constructed conversation context.
        """
        new_conversation_messages: List[Dict[str, str]] = []
        
        effective_conversation_start_idx = 0
        if initial_system_message is None:
            if messages and messages[0].get("role") == "system":
                initial_system_message = messages[0]
                effective_conversation_start_idx = 1
        
        conversation_messages = messages[effective_conversation_start_idx:]

        # Handle system message inclusion
        if self.custom_system_message:
            if self.include_original_system_message and initial_system_message:
                combined_content = (
                    f"{self.custom_system_message}\n\n"
                    "Original model's system message:\n"
                    "```txt\n"
                    f"{initial_system_message['content']}"
                    "```"
                )
                new_conversation_messages.append({"role": "system", "content": combined_content})
            else:
                new_conversation_messages.append({"role": "system", "content": self.custom_system_message})
        elif self.include_original_system_message and initial_system_message:
            new_conversation_messages.append(initial_system_message)

        def _inner_get_messages(index_or_range: Union[int, Tuple[int, int]]) -> List[Dict[str, str]]:
            """Helper function to extract messages based on a single index or a range."""
            selected_messages = []
            conv_len = len(conversation_messages)

            if isinstance(index_or_range, int):
                idx = index_or_range
                if idx < 0:
                    idx = conv_len + idx
                
                if 0 <= idx < conv_len:
                    selected_messages.append(conversation_messages[idx])
            elif isinstance(index_or_range, tuple) and len(index_or_range) == 2:
                start_orig, end_orig = index_or_range

                start_idx = conv_len + start_orig if start_orig < 0 else start_orig
                end_idx = conv_len + end_orig if end_orig < 0 else end_orig
                
                if start_idx <= end_idx: # Forward iteration
                    for i in range(start_idx, end_idx + 1):
                        if 0 <= i < conv_len:
                            selected_messages.append(conversation_messages[i])
                else: # Backward iteration
                    for i in range(start_idx, end_idx - 1, -1):
                        if 0 <= i < conv_len:
                            selected_messages.append(conversation_messages[i])
            return selected_messages

        if isinstance(self.messages_to_include, list):
            for item in self.messages_to_include:
                new_conversation_messages.extend(_inner_get_messages(item))
        elif self.messages_to_include is not None: # Can be int or tuple
            new_conversation_messages.extend(_inner_get_messages(self.messages_to_include))
        
        return new_conversation_messages

@dataclass
class ModelConfig:
    """Represents a model configuration with serialization support."""
    name: str
    provider: str
    provider_model: str
    api_key_env: Optional[str] = None
    requirements: Optional[List[Requirement]] = field(default=None)
    context_origin_config: ContextOriginConfig = field(default_factory=ContextOriginConfig.default)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the model configuration to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representing the model configuration.
        """
        result = {
            "name": self.name,
            "provider": self.provider,
            "provider_model": self.provider_model
        }
        if self.api_key_env is not None:
            result["api_key_env"] = self.api_key_env
        if self.requirements:
            result["requirements"] = Requirements.to_json(self.requirements)
        
        # Only include context_origin_config in the JSON if it's not the default
        default_config = ContextOriginConfig.default()
        if self.context_origin_config != default_config:
            result["context_origin_config"] = self.context_origin_config.to_json()
        return result
    
    @staticmethod
    def from_json(json_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create a ModelConfig instance from a JSON dictionary.
        
        Args:
            json_dict: JSON dictionary representing a model configuration.
            
        Returns:
            A ModelConfig instance.
        """
        
        requirements = None
        requires_dict = json_dict.get("requirements", None)
        if requires_dict:
            requirements = Requirements.from_json(requires_dict)
        
        # Handle context_origin_config, default to standard if not provided in JSON
        context_origin_config = ContextOriginConfig.default()
        context_origin_dict = json_dict.get("context_origin_config", None)
        if context_origin_dict:
            context_origin_config = ContextOriginConfig.from_json(context_origin_dict)
            
        return ModelConfig(
            name=json_dict.get("name", ""),
            provider=json_dict.get("provider", ""),
            provider_model=json_dict.get("provider_model", ""),
            api_key_env=json_dict.get("api_key_env", None),
            requirements=requirements,
            context_origin_config=context_origin_config
        )