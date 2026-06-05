from __future__ import annotations

import os
from dotenv import load_dotenv
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import Anthropic

load_dotenv()

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False

@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str = "end_turn"

class ModelProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[Any] | None = None) -> ModelResponse: ...

def _to_anthropic_tools(tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
        for tool in tools
    ]

def _parse_tool_input(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

def _content_block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block.dict(exclude_none=True)
    data = {"type": block.type}
    for name in ("text", "id", "name", "input", "thinking", "signature"):
        if hasattr(block, name):
            data[name] = getattr(block, name)
    return data

class AnthropicProvider:
    def __init__(
        self,
        model: str = "Qwen3-8B-Q5_K_M",
        max_tokens: int = 1024,
        base_url: str | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "ollama")
        if not api_key:
            raise ValueError("Anthropic API key is required. Set it in the code or via environment variable.")
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", base_url)
        self.client = Anthropic(api_key=api_key, base_url=self.base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        response = self.client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_content: list[dict[str, Any]] = []

        for block in response.content:
            # save assistant content
            assistant_content.append(_content_block_to_dict(block))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=_parse_tool_input(block.input)
                ))

        return ModelResponse(
            text = "\n".join(text_parts) or None,
            tool_calls = tool_calls or None,
            assistant_content = assistant_content or None,
            stop_reason = response.stop_reason or "end_turn"
        )
    
    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        text_parts: list[str] = []

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                text_parts.append(text)
                if on_text_delta:
                    on_text_delta(text)
            final_message = stream.get_final_message()

        assistant_content: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []

        for block in final_message.content:
            assistant_content.append(_content_block_to_dict(block))
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    args=_parse_tool_input(block.input)
                ))
        
        return ModelResponse(
            text = "".join(text_parts) or None,
            tool_calls = tool_calls or None,
            assistant_content = assistant_content or None,
            stop_reason = final_message.stop_reason or "end_turn"
        )

class MockProvider:
    def complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        last_message = messages[-1]

        if last_message["role"] == "user":
            text = last_message["content"].replace("use echo say", "").strip() or last_message["content"]
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_echo_1",
                        name="echo",
                        args={"text": text}
                    )
                ],
                stop_reason="tool_used"
            )
        
        if last_message["role"] == "tool":
            return ModelResponse(
                text=f"Echoed: {last_message['content']}"
            )
        
        return ModelResponse(text="Now I can only echo.")
    
def create_provider(
    name: str,
    model: str,
    base_url: str | None = None,
) -> ModelProvider:
    if name == "anthropic":
        return AnthropicProvider(model=model, base_url=base_url)
    elif name == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")