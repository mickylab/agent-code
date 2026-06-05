from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ModelProvider, ModelResponse
from .tools import ToolContext, ToolRegistry

@dataclass
class AgentResult:
    final_response: str
    trace: list[str]
    messages: list[dict[str, Any]]

def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    if response.assistant_content:
        return {"role": "assistant", "content": response.assistant_content}
    
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for tool_call in response.tool_calls or []:
        content.append({
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.name,
            "input": tool_call.args
        })
    return {"role": "assistant", "content": content}

def _tool_result_message(tool_call_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
                "is_error": is_error
            }
        ]
    }

# multi-step loop: model -> tool_use -> tool -> tool_result -> model -> ...
# stopping conditions:
# 1. The model returned no tool_calls — final answer ready.
# 2. step reaches max_steps — the harness force-stops to prevent infinite tool calls.

def run_agent(
    prompt: str, 
    provider: ModelProvider, 
    tools: ToolRegistry,
    max_steps: int = 8,
    cwd: Path | None = None,
    stream: bool = False,
    on_text_delta = None
    ) -> AgentResult:
    resolved_cwd = cwd.resolve() if cwd else Path.cwd()
    ctx = ToolContext(cwd = resolved_cwd)

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: list[str] = []

    for step in range(max_steps):
        if stream and hasattr(provider, "stream_complete"):
            response = provider.stream_complete(
                messages, 
                tools.list(),
                on_text_delta = on_text_delta or (lambda text: print(text, end="", flush=True))
            )
        else:
            response = provider.complete(messages, tools.list())

        messages.append(_assistant_message(response))

        if not response.tool_calls:
            final_response = response.text or ""
            trace.append(f"Final response: {final_response}")
            return AgentResult(final_response=final_response, trace=trace, messages=messages)
        
        for tool_call in response.tool_calls or []:
            trace.append(f"Calling tool: {tool_call.name} with args {tool_call.args}")
            tool_result = tools.run(tool_call, ctx)
            trace.append(f"Observation: {tool_result.content} ({'' if tool_result.is_error else 'no'} error)")
            messages.append(_tool_result_message(tool_result.tool_call_id, tool_result.content, tool_result.is_error))

    final_response = f"Stopped after reaching max steps: {max_steps}"
    trace.append(f"Final response: {final_response}")
    return AgentResult(final_response = final_response, trace = trace, messages = messages)