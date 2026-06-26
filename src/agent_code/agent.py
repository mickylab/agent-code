from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from rich.console import Console

from .model import ModelProvider, ModelResponse, ToolResult
from .tools import ToolContext, ToolRegistry
from .fs_safety import SkipPolicy, load_gitignore, resolve_in_cwd, apply_single_replace, check_mtime_conflict, ensure_read_before_edit
from .diff_ui import confirm_edit, render_diff

console = Console()

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

# one result per message, keep it or delete it
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
    resolved_cwd = cwd or Path.cwd()
    ctx = ToolContext(
        cwd = resolved_cwd,
        skip_policy = SkipPolicy.default(load_gitignore(resolved_cwd))
    )

    def emit(line: str) -> None:
        # stream trace
        trace.append(line) # for testing
        console.print(line) # for user

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
            emit(f"Final response: {final_response}")
            return AgentResult(final_response=final_response, trace=trace, messages=messages)
        
        for tool_call in response.tool_calls or []:
            """ The Anthropic Messages API requires: every tool_use in one assistant message must have its matching tool_result in the very next user message. 
            """
            tool_result_blocks: list[dict[str, Any]] = []
            for tool_call in response.tool_calls:
                emit(f"Calling tool: {tool_call.name} with args {tool_call.args}")
                # Intercept file_write / file_edit through the harness: run pre-validation first, then render the diff, and finally ask the user for confirmation.
                if tool_call.name in ("file_write", "file_edit"):
                    path_str = tool_call.args.get("file_path", "")
                    # 1. resolve path, raise error if outside cwd
                    try:
                        path = resolve_in_cwd(ctx.cwd, path_str)
                    except (ValueError, OSError) as e:
                        tool_result = ToolResult(tool_call.id, f"Error resolving path: {e}", is_error=True)
                        emit(f"Observation: {tool_result.content}")
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result.tool_call_id,
                                "content": tool_result.content,
                                "is_error": True
                            }
                        )
                        continue

                    old_content = path.read_text(encoding="utf-8") if path.exists() else ""

                    # 2. check read before write, mtime conflict
                    validation_error: str | None = None
                    if tool_call.name == "file_write":
                        if path.exists():
                            validation_error = (
                                ensure_read_before_edit(ctx.read_state, path)
                                or check_mtime_conflict(ctx.read_state, path)
                            )
                    else:
                        if not path.exists():
                            validation_error = f"Error: File does not exist at {path_str}."
                        else:
                            validation_error = (
                                ensure_read_before_edit(ctx.read_state, path)
                                or check_mtime_conflict(ctx.read_state, path)
                            )
                    
                    # 3. file_write use content directly, file_edit trial run in memory
                    new_content: str | None = None
                    if tool_call.name == "file_write":
                        new_content = tool_call.args.get("content", "")
                    elif tool_call.name == "file_edit" and validation_error is None:
                        new_content, replace_error = apply_single_replace(
                            old_content,
                            tool_call.args.get("old_str", ""),
                            tool_call.args.get("new_str", ""),
                            tool_call.args.get("replace_all", False)
                        )
                        if replace_error is not None:
                            validation_error = replace_error

                    # 4. if validation failed, raise error
                    if validation_error is not None:
                        tool_result = ToolResult(tool_call.id, validation_error, is_error=True)
                        emit(f"Observation: {tool_result.content}")
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result.tool_call_id,
                                "content": tool_result.content,
                                "is_error": True
                            }
                        )
                        continue

                    # 5. validation passed, render diff and user confirmation
                    if new_content is not None:
                        diff_text = render_diff(old_content, new_content, path_str)
                        console.print(f"\n[bold]Diff for {path_str}:[/bold]")
                        console.print(diff_text)
                        if not confirm_edit(path_str):
                            tool_result = ToolResult(tool_call.id, "Error: Edit not confirmed by user.", is_error=True)
                            emit(f"Observation: {tool_result.content}")
                            tool_result_blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_result.tool_call_id,
                                    "content": tool_result.content,
                                    "is_error": True
                                }
                            )
                            continue
                
                tool_result = tools.run(tool_call, ctx)
                emit(f"Observation: {tool_result.content}")

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": tool_result.content,
                        "is_error": tool_result.is_error
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})

    final_response = f"Stopped after reaching max steps: {max_steps}"
    emit(f"Final response: {final_response}")
    return AgentResult(final_response = final_response, trace = trace, messages = messages)