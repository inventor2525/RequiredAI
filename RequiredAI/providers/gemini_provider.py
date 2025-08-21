import os
import uuid
import time
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from ..ModelConfig import ModelConfig

from . import BaseModelProvider, provider

@provider('gemini')
class GeminiProvider(BaseModelProvider):
    """Provider for Google Gemini API."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the Gemini provider.
        
        Args:
            config: Configuration for the model.
        """
        super().__init__(config)
        api_key = config.get_api_key("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(f"API key for Gemini model named '{config.name}' not set!")
        genai.configure(api_key=api_key)

    def complete(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using Google Gemini API.

        Args:
            messages: The conversation messages in OpenAI-like format.
            params: Additional parameters for the request (e.g., max_tokens, temperature).

        Returns:
            The model's response message in OpenAI-like dictionary format.
        """
        # Extract parameters
        provider_model_name = self.config.provider_model
        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.7)

        gemini_messages = []
        system_instruction_content = None

        # Extract system message if present (similar to AnthropicProvider)
        # The first message with role 'system' is treated as a system instruction for Gemini model initialization.
        conversation_messages = []
        if messages and messages[0].get("role") == "system":
            system_instruction_content = messages[0]["content"]
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages

        for msg in conversation_messages:
            role = msg["role"]
            content = msg["content"]
            # Gemini expects 'user' and 'model' roles. 'assistant' maps to 'model'.
            gemini_role = "user" if role == "user" else "model"
            gemini_messages.append({"role": gemini_role, "parts": [content]})
        
        # Prepare generation configuration
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Instantiate GenerativeModel with system_instruction for this call.
        # This allows dynamic system instructions if they are passed within the 'messages' list.
        model_client = genai.GenerativeModel(
            provider_model_name,
            system_instruction=system_instruction_content if system_instruction_content else None
        )

        # Make the API call
        try:
            response = model_client.generate_content(
                gemini_messages,
                generation_config=generation_config
            )
            
            # Convert Gemini response to OpenAI-like format
            if response.candidates:
                candidate = response.candidates[0]
                content_text = ""
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            content_text += part.text
                
                finish_reason = "stop" # Default
                if candidate.finish_reason:
                    # Map Gemini's FinishReason enum to a lowercase string
                    finish_reason = str(candidate.finish_reason).replace("FinishReason.", "").lower()

                # Generate unique ID and timestamp as Gemini response objects may not always provide them directly
                response_id = str(uuid.uuid4())
                response_created = int(time.time())

                return {
                    "id": f"gemini-chatcmpl-{response_id}",
                    "object": "chat.completion",
                    "created": response_created,
                    "model": provider_model_name,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content_text
                        },
                        "finish_reason": finish_reason
                    }],
                    "tags": list(self.config.output_tags)
                }
            else:
                # Handle cases where no candidates are returned (e.g., safety block)
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    raise ValueError(f"Gemini API blocked response due to: {response.prompt_feedback.block_reason.name}")
                raise ValueError(f"Gemini API returned no candidates or an empty response: {response}")

        except Exception as e:
            # Log the error and re-raise
            print(f"Error calling Gemini API: {str(e)}")
            raise
