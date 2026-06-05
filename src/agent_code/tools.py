from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable

from .fs_safety import (
    ReadFileState,
    SkipPolicy,
    ensure_text_file,
    ensure_within_size,
    resolve_in_cwd,
    should_skip_path,
    truncate_output,
)
from .model import ToolCall, ToolResult

@dataclass
class ToolContext:
    cwd: Path
    skip_policy: SkipPolicy
    read_state: ReadFileState = field(default_factory=ReadFileState)

ToolFunc = Callable[[dict[str, Any]], str]

@dataclass
class Tool:
    name: str
    description: str
    run: ToolFunc

    parameters: dict[str, Any] = field(
        default_factory = lambda: {
            "type": "object",
            "properties": {},
            "required": []
        }
    )

def echo(args: dict[str, Any], ctx: ToolContext) -> str:
    return str(args.get("text", ""))

def system_time(args: dict[str, Any], ctx: ToolContext) -> str:
    tz_name = args.get("timezone", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return f"Invalid timezone '{tz_name}'. Please provide a valid IANA timezone name."
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path")
    if not path_str:
        return "Error: 'path' argument is required."
    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
        ensure_text_file(path)
        ensure_within_size(path)
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, ValueError) as e:
        return f"Error reading file: {str(e)}"
    ctx.read_state.record(path, text)
    return truncate_output(text)

def list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    path_str = args.get("path", ".")
    try:
        base_path = resolve_in_cwd(ctx.cwd, path_str)
    except ValueError as e:
        return f"Error: {str(e)}"
    if not base_path.is_dir():
        return f"Error: '{path_str}' is not a directory."
    entries: list[str] = []
    # dir first, then files
    for child in sorted(base_path.iterdir(), key = lambda p: (not p.is_dir(), p.name)):
        rel_path = child.relative_to(ctx.cwd)
        if should_skip_path(rel_path, ctx.skip_policy):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    return truncate_output("\n".join(entries)) or "[empty directory]"

class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list(self) -> list[Tool]:
        return list(self.tools.values())
    
    def run(self, tool_call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id = tool_call.id,
                content = f"Unknown tool '{tool_call.name}' not found.",
                is_error = True
            )
        return ToolResult(
            tool_call_id = tool_call.id,
            content = tool.run(tool_call.args, ctx),
        )
        
def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(Tool(
        name="echo",
        description="Echoes the input text.",
        run=echo,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"}
            },
            "required": ["text"]
        }
    ))
    registry.register_tool(Tool(
        name="system_time",
        description="Get current time in a specific timezone",
        run=system_time,
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone name (e.g. 'America/New_York')"}
            },
            "required": []
        }
    ))
    registry.register_tool(Tool(
        name="read_file",
        description="Read the content of a text file within the current working directory.",
        run=read_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file to read"}
            },
            "required": ["path"]
        }
    ))
    registry.register_tool(Tool(
        name="list_files",
        description="List files and directories in a given directory, relative to the current working directory.",
        run=list_files,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the directory to list, defaults to current directory", "default": "."}
            },
            "required": []
        }
    ))
    return registry