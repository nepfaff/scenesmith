import logging

from typing import Any

from openai import OpenAI, Timeout

console_logger = logging.getLogger(__name__)


class VLMService:
    """
    Wrapper for OpenAI's vision models to analyze and critique 3D scenes.

    Handles both text-only and multimodal (text+image) requests, automatically
    routing to the appropriate API (Responses API for reasoning models like GPT-5,
    Chat API for standard models) based on model capabilities.
    """

    def __init__(
        self, service_tier: str | None = None, timeout: Timeout | None = None
    ) -> None:
        """Initialize OpenAI client.

        Args:
            service_tier: Optional service tier for API processing priority.
                Valid values: "default", "flex", "priority", or None to use
                project default.
            timeout: Optional timeout for OpenAI API requests.
        """
        self.client = OpenAI(timeout=timeout) if timeout is not None else OpenAI()
        # Cache for model type detection.
        self._reasoning_models = {"gpt-5", "gpt-5.2", "o3", "o4"}
        self.service_tier = service_tier

    def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        verbosity: str,
        response_format: dict[str, str] | None = None,
        vision_detail: str = "auto",
    ) -> str:
        """Create completion using appropriate API based on model type.

        Uses Responses API for reasoning models and Chat Completions API
        for standard models.

        Args:
            model: Model name (e.g., "gpt-5", "gpt-4o-mini").
            messages: List of message dictionaries.
            reasoning_effort: Effort level for reasoning models.
            verbosity: Output verbosity level ("low", "medium", "high").
            response_format: Optional response format (e.g., {"type": "json_object"}).
            vision_detail: Image resolution detail ("low", "high", "auto").

        Returns:
            Response content as string.
        """
        # Check if model supports reasoning.
        if model in self._reasoning_models:
            # Use Responses API with reasoning for reasoning-capable models.
            input_messages = self._convert_to_responses_format(
                messages=messages, vision_detail=vision_detail
            )

            # Add JSON instruction if needed.
            if response_format and response_format.get("type") == "json_object":
                input_messages = self._add_json_instruction(input_messages)

            kwargs = {
                "model": model,
                "input": input_messages,
                "reasoning": {"effort": reasoning_effort},
                "text": {"verbosity": verbosity},
            }
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier
            response = self.client.responses.create(**kwargs)

            # Raise with diagnostic details if output_text is empty.
            if not response.output_text:
                # Extract refusal messages if present.
                refusals = []
                for output in response.output:
                    if output.type == "message":
                        for content in output.content:
                            if getattr(content, "type", None) == "refusal":
                                refusals.append(content.refusal)

                refusal_info = f", Refusals: {refusals}" if refusals else ""
                raise RuntimeError(
                    f"Empty response from {model}. "
                    f"Status: {getattr(response, 'status', 'N/A')}, "
                    f"Error: {response.error}, "
                    f"Incomplete: {getattr(response, 'incomplete_details', 'N/A')}, "
                    f"Output types: {[item.type for item in response.output]}"
                    f"{refusal_info}"
                )

            return response.output_text
        else:
            # Use Chat Completions API for standard models.
            messages = self._add_vision_detail_to_messages(
                messages=messages, vision_detail=vision_detail
            )
            kwargs = {"model": model, "messages": messages}

            # Add response format if specified.
            if response_format:
                kwargs["response_format"] = response_format

            # Add service tier if configured.
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

            # Validate response content.
            if not content or not content.strip():
                raise RuntimeError(
                    f"Empty response from {model} (Chat API). "
                    f"Finish reason: {response.choices[0].finish_reason}, "
                    f"Content type: {type(content).__name__}, "
                    f"Content repr: {repr(content)}"
                )

            return content

    def _convert_to_responses_format(
        self, messages: list[dict[str, Any]], vision_detail: str = "auto"
    ) -> list[dict[str, Any]]:
        """Convert chat format to Responses API format.

        Args:
            messages: Chat messages in standard format.
            vision_detail: Image resolution detail ("low", "high", "auto").

        Returns:
            Messages converted for Responses API format.
        """
        converted = []
        for msg in messages:
            new_msg = {"role": msg["role"]}

            if isinstance(msg["content"], str):
                new_msg["content"] = msg["content"]
            else:
                # Convert multimodal content.
                new_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        new_content.append({"type": "input_text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        new_content.append(
                            {
                                "type": "input_image",
                                "image_url": item["image_url"]["url"],
                                "detail": vision_detail,
                            }
                        )
                new_msg["content"] = new_content

            converted.append(new_msg)
        return converted

    def _add_vision_detail_to_messages(
        self, messages: list[dict[str, Any]], vision_detail: str
    ) -> list[dict[str, Any]]:
        """Add vision detail parameter to image_url objects in messages.

        Args:
            messages: Chat messages with potential image_url content.
            vision_detail: Image resolution detail ("low", "high", "auto").

        Returns:
            Messages with detail parameter added to image_url objects.
        """
        updated_messages = []
        for msg in messages:
            new_msg = {"role": msg["role"]}

            if isinstance(msg["content"], str):
                new_msg["content"] = msg["content"]
            else:
                # Process multimodal content to add detail parameter.
                new_content = []
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        # Add detail parameter to image_url.
                        new_item = {
                            "type": "image_url",
                            "image_url": {
                                "url": item["image_url"]["url"],
                                "detail": vision_detail,
                            },
                        }
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content

            updated_messages.append(new_msg)
        return updated_messages

    def _add_json_instruction(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add JSON format instruction to messages.

        Args:
            messages: List of message dictionaries.

        Returns:
            Messages with JSON instruction added.
        """
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n\nPlease respond with valid JSON format."
            else:
                messages[-1]["content"].append(
                    {
                        "type": "input_text",
                        "text": "Please respond with valid JSON format.",
                    }
                )
        return messages
