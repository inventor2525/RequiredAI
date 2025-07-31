"""
Requirement model implementations for RequiredAI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple
import random
from .helpers import *
from .Requirement import requirement, Requirement, RequirementResult
import re

@requirement("Contains")
@dataclass
class ContainsRequirement(Requirement):
    """Requirement that checks if the AI response contains any of the specified values."""
    
    value: List[str]
    name: str = ""
    revision_model: Optional[str] = None
    
    def evaluate(self, messages: List[dict]) -> RequirementResult:
        """
        Checks that the last message in the passed conversation (which is presumed 
        to be from an AI), contains any of the values in value.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the last message contains any of the values, False otherwise
        """
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        # Print for debugging
        result = any(val in content for val in self.value)
        
        return RequirementResult.construct(self, result)
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining how the conversation did not meet this requirement.
        """
        values_str = '", "'.join(self.value)
        return f'Your response must contain at least one of the following: "{values_str}".'

@requirement("Regex")
@dataclass
class RegexRequirement(Requirement):
    """Requirement that checks if the AI response matches positive regexes and does not match negative regexes."""
    
    positive_regexes: List[str]
    negative_regexes: List[str]
    additional_prompt: Optional[str] = None
    name: str = ""
    revision_model: Optional[str] = None
    
    def evaluate(self, messages: List[dict]) -> RequirementResult:
        """
        Evaluates if the response matches all positive regexes and none of the negative regexes.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if all positive regexes match and no negative regexes match, False otherwise
        """
        last_message = messages[-1]
        content = last_message.get("content", "")
        
        # Check positive regexes
        for regex in self.positive_regexes:
            try:
                if not re.search(regex, content):
                    return RequirementResult.construct(self, False, {
                        "pattern_type":"positive",
                        "pattern":regex
                    })
            except re.error as e:
                return RequirementResult.construct(self, False, {
                    "error":f"Invalid positive regex '{regex}': {e}"
                })
        
        # Check negative regexes
        for regex in self.negative_regexes:
            try:
                if re.search(regex, content):
                    return RequirementResult.construct(self, False, {
                        "pattern_type":"negative",
                        "pattern":regex
                    })
            except re.error as e:
                return RequirementResult.construct(self, False, {
                    "error":f"Invalid negative regex '{regex}': {e}"
                })
        
        return RequirementResult.construct(self, True)
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining how the response should follow positive regexes and avoid negative ones.
        """
        positive_str = (
            "```txt\n" + "\n".join(self.positive_regexes) + "\n```"
            if self.positive_regexes else None
        )
        negative_str = (
            "```txt\n" + "\n".join(self.negative_regexes) + "\n```"
            if self.negative_regexes else None
        )
        
        prompt_parts = []
        if positive_str:
            prompt_parts.append(f"Your response must match these regex patterns:\n{positive_str}")
        if negative_str:
            prompt_parts.append(f"Your response must not match these regex patterns:\n{negative_str}")
        
        if self.additional_prompt:
            prompt_parts.append(self.additional_prompt)
        return "\n".join(prompt_parts) if prompt_parts else None

@requirement("Written")
@dataclass
class WrittenRequirement(Requirement):
    """
    Requirement that uses another model to evaluate if the response 
    follows specific writing instructions.
    """
    
    evaluation_model: str
    value: List[str]
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    token_limit: int = 1024
    name: str = ""
    revision_model: Optional[str] = None
    
    def evaluate(self, messages: List[dict]) -> RequirementResult:
        """
        Evaluates if the response follows the writing requirements.
        Uses another model to evaluate if the response meets the writing requirements.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            bool: True if the requirement is met, False otherwise
        """
        from RequiredAI.ModelManager import ModelManager, BaseModelProvider
        from RequiredAI.ModelConfig import ContextOriginConfig
        
        evaluation_model = ModelManager.singleton().get_provider(self.evaluation_model)
        last_message = messages[-1]
        text_to_evaluate = last_message.get("content", "")
        
        context_config:ContextOriginConfig = getattr(evaluation_model, "context_origin_config", None)
        extra_context:str = None
        extra_system_msg:str = None
        if context_config:
            og_system_msg, extra_system_msg, context_messages = context_config.create_messages_from(messages)
            if og_system_msg:
                context_messages = [{'role':'system', 'content':og_system_msg}] + context_messages
            
            msg_xmls = []
            worth_including = False
            for msg in context_messages:
                role = msg['role']
                content = msg['content']
                if role.lower() != "assistant" or content!=text_to_evaluate:
                    worth_including = True
                    
                role_str = f"From__{role}"
                msg_xmls.append(f"<{role_str}>\n{indent_text(content)}\n</{role_str}>")
                
            if worth_including or len(msg_xmls)>1:
                extra_context = '\n'.join(msg_xmls)
            
        try:
            # Select one random requirement from the value list
            selected_requirement = random.choice(self.value)
            
            # Combine all examples for random selection
            all_examples = []
            if self.positive_examples:
                for ex in self.positive_examples:
                    all_examples.append(("positive", ex))
            if self.negative_examples:
                for ex in self.negative_examples:
                    all_examples.append(("negative", ex))
            
            def construct_msgs(requirement:str, positive_examples:List[str], negative_examples:List[str], extra_context:str) -> Tuple[str, str]:
                # System Message Construction:
                system_msg = "# Goal\n\nDetermine if the given text meets the following written requirement. Answer with only 'yes' or 'no'."
                if extra_system_msg:
                    system_msg += " (Unless told to do more under 'Additionally'.)"
                system_msg += "\n\n> Note, for clarity: All requirement, example, and content text given to you are wrapped in markdown code blocks like this '```txt\\n{text}\\n```'."
                system_msg += f"\n\n# Written Requirement:\n{code_block_text(requirement)}"
                
                def examples_to_str(examples:List[str], prefix:str):
                    return "\n\n".join([f"## {prefix} Example {i+1}\n{code_block_text(e)}" for i,e in enumerate(examples)])
                
                if negative_examples:
                    system_msg += f"\n\n# Examples that do *NOT* meet the requirement:\n" + examples_to_str(negative_examples, "Bad")
                if positive_examples:
                    system_msg += "\n\n# Examples that *DO* meet the requirement:\n" + examples_to_str(positive_examples, "Good")
                
                if extra_system_msg:
                    system_msg += "\n\n# Additionally\nIn this case, the application would also like to tell you:\n"
                    system_msg += code_block_text(extra_system_msg)
                    system_msg += "\nPlease follow any additional instructions that it may have given you there as well."
                
                # User Message Construction:
                user_msg = ""
                if extra_context:
                    user_msg += "# Extra Context\nThe text you are suppose to evaluate in this case comes from a conversation with another AI. For context, here is the conversation that it was responding to in xml that has been indented over for clarity:\n"
                    extra_context_xml = f"<Other_Conversation>\n{indent_text(extra_context)}\n</Other_Conversation>"
                    user_msg += code_block_text(extra_context_xml, 'xml') + "\n\n"
                    
                user_msg += f"# Text to evaluate:\n{code_block_text(text_to_evaluate)}"
                user_msg += "\n\n# Question\nDoes this '# Text to evaluate' meet the '# Written Requirement'?"
                return system_msg, user_msg
                
            # Randomly select examples up to token limit
            positive_examples = []
            negative_examples = []
            if all_examples:
                random.shuffle(all_examples)
                
                for example_type, example in all_examples:
                    # Test adding this example
                    temp_positive_examples = positive_examples + ([example] if example_type == "positive" else [])
                    temp_negative_examples = negative_examples + ([example] if example_type == "negative" else [])
                    
                    # Build test message with accumulated examples
                    system_msg, user_msg = construct_msgs(selected_requirement, temp_positive_examples, temp_negative_examples, extra_context)
                    
                    # Check token count
                    current_tokens = evaluation_model.estimate_tokens(system_msg + user_msg)
                    
                    if current_tokens <= self.token_limit:
                        positive_examples = temp_positive_examples
                        negative_examples = temp_negative_examples
                    else:
                        break
            
            # Build final examples text
            system_msg, user_msg = construct_msgs(selected_requirement, positive_examples, negative_examples, extra_context)
            
            eval_messages = [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user", 
                    "content": user_msg
                }
            ]
            
            # Get the evaluation from the model
            eval_args = {
                "model_name":self.evaluation_model,
                "messages":eval_messages,
                "params":{"max_tokens": 1, "temperature": 0.0}  # Use low temperature for consistent results
            }
            response = ModelManager.singleton().complete_with_model(**eval_args)
            
            # Parse the response
            eval_text = get_msg_content(response).strip().lower()
            result = "yes" in eval_text and "no" not in eval_text
            
            return RequirementResult.construct(self, result, {
                "evaluation":eval_args,
                "eval_result":result,
                "response":response
            })
            
        except Exception as e:
            return RequirementResult.construct(self, False, {
                "error":f"Error evaluating written requirement '{self.name}': {str(e)}"
            })
    
    @property
    def prompt(self) -> str:
        """
        Returns a string explaining the written requirements.
        """
        requirements_str = "; ".join(self.value)
        return f"Your response should follow these written requirements: {requirements_str}"