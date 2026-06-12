from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import html2text
import httpx

from .fs_safety import (
    MAX_READ_BYTES,
    ReadFileState,
    SkipPolicy,
    ensure_text_file,
    ensure_within_size,
    resolve_in_cwd,
    should_skip,
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
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required."
    
    offset = int(args.get("offset", 0))
    limit = args.get("limit")

    if offset < 0:
        return "Error: 'offset' must be non-negative."
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            return "Error: 'limit' must be a positive integer."

    try:
        path = resolve_in_cwd(ctx.cwd, path_str)
        ensure_text_file(path)
        file_size = path.stat().st_size

        if file_size > MAX_READ_BYTES and limit is None:
            return (
                f"Error: file is too large to read all at once "
                f"({file_size} bytes > {MAX_READ_BYTES} bytes). "
                f"Use offset and limit to read it in chunks, for example: "
                f'{{"path": "{path_str}", "offset": 0, "limit": 200}}'
            )
        
        lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if limit is not None and len(lines) >= limit:
                    break
                lines.append(line)
        text = "".join(lines)
    except (FileNotFoundError, IsADirectoryError, ValueError) as e:
        return f"Error reading file: {str(e)}"
    ctx.read_state.record(path, text)
    return text or "[empty]"

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
        if should_skip(rel_path, ctx.skip_policy):
            continue
        entries.append(f"{child.name}/" if child.is_dir() else child.name)
    return truncate_output("\n".join(entries)) or "[empty directory]"

def glob(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: 'pattern' argument is required."
    matches: list[Path] = []
    try:
        for path in ctx.cwd.rglob(pattern):
            rel_path = path.relative_to(ctx.cwd)
            if should_skip(rel_path, ctx.skip_policy):
                continue
            matches.append(path)
    except NotImplementedError as e:
        return f"Error: Glob not implemented: {e}"
    matches.sort(key = lambda p: p.stat().st_mtime, reverse=True)  # most recently modified first
    matches = matches[:200]  # limit to 200 matches
    lines = [str(p.relative_to(ctx.cwd)) for p in matches]
    return truncate_output("\n".join(lines)) or "[no matches]"

def grep(args: dict[str, Any], ctx: ToolContext) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: 'pattern' argument is required."
    path_arg = args.get("path", ".")
    glob_arg = args.get("glob")
    ignore_case = bool(args.get("ignore_case", False))
    try:
        base_path = resolve_in_cwd(ctx.cwd, path_arg)
    except ValueError as e:
        return f"Error: {str(e)}"
    # ripgrep is much faster than Python, fall back to Python if not available
    if shutil.which("rg"):
        return _grep_ripgrep(pattern, base_path, glob_arg, ignore_case, ctx)
    else:
        return _grep_python(pattern, base_path, glob_arg, ignore_case, ctx)

def _grep_ripgrep(pattern: str, base_path: Path, glob_arg: str | None, ignore_case: bool, ctx: ToolContext) -> str:
    args: list[str] = ["rg", "--line-number", "--no-heading", "--max-columns", "500"]
    if ignore_case:
        args.append("--ignore-case")
    for name in ctx.skip_policy.skip_dirs:
        args.extend(["--glob", f"!{name}/**"])
    if glob_arg:
        args.extend(["--glob", glob_arg])
    args.append(pattern)
    args.append(str(base_path))
    try:
        process = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error running ripgrep: {str(e)}"
    if process.returncode not in (0, 1):  # 0 = matches found, 1 = no matches, other = error
        return f"Error running ripgrep: {process.stderr.strip() or process.returncode}"
    return truncate_output(_relative_rg_output(process.stdout.strip(), base_path) or "[no matches]")

def _relative_rg_output(stdout: str, cwd: Path) -> str:
    # ripgrep outputs absolute paths when run with a non-default cwd, convert them back to relative
    cwd_prefix = f"{cwd}/"
    lines = [
        line[len(cwd_prefix):] if line.startswith(cwd_prefix) else line
        for line in stdout.splitlines()
    ]
    return "\n".join(lines).strip()

def _grep_python(pattern: str, base_path: Path, glob_arg: str | None, ignore_case: bool, ctx: ToolContext) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex pattern: {str(e)}"
    if base_path.is_file():
        candidates: list[Path] = [base_path]
    else:
        candidates = []
        for path in base_path.rglob(glob_arg or "*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(ctx.cwd)
            if should_skip(rel_path, ctx.skip_policy):
                continue
            candidates.append(path)
    matches: list[str] = []
    for path in candidates:
        try:
            ensure_text_file(path)
        except ValueError:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel_path = path.relative_to(ctx.cwd)
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel_path}:{i}:{line}")
    return truncate_output("\n".join(matches) or "[no matches]")

def project_tree(args: dict[str, Any], ctx: ToolContext) -> str:
    max_depth = int(args.get("max_depth", 3))
    max_nodes = 200
    lines: list[str] = [f"{ctx.cwd.name}/"]
    nodes = 0

    def walk(path: Path, depth: int = 0) -> None:
        nonlocal nodes
        if depth > max_depth:
            return
        children = sorted(
            (
                p for p in path.iterdir()
                if not should_skip(p.relative_to(ctx.cwd), ctx.skip_policy)
            )
            , key = lambda p: (not p.is_dir(), p.name)
        )
        for child in children:
            if nodes >= max_nodes:
                if nodes == max_nodes:
                    lines.append("  " * depth + "...[truncated]")
                    nodes += 1
                return
            suffix = "/" if child.is_dir() else ""
            lines.append("  " * depth + child.name + suffix)
            nodes += 1
            if child.is_dir():
                walk(child, depth + 1)
        
    walk(ctx.cwd, 1)
    return truncate_output("\n".join(lines))

# Hard constraints for web tools live here, like fs_safety constants — never leak to callers.
WEB_USER_AGENT = "agent-code/0.1 (+https://example.com/agent-code)"
WEB_FETCH_MAX_BYTES = 10 * 1024 * 1024
WEB_FETCH_MAX_CHARS = 20_000
WEB_URL_MAX_LENGTH = 2000
WEB_FETCH_TIMEOUT_S = 30.0
WEB_SEARCH_TIMEOUT_S = 15.0
WEB_FETCH_CACHE_TTL_S = 15 * 60  # 15 minutes
WEB_FETCH_CACHE_MAX_SIZE = 32

_web_fetch_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

def _validate_url(url: str) -> bool:
    # the first gateway for web fetch tool
    if len(url) > WEB_URL_MAX_LENGTH:
        raise ValueError(f"URL length exceeds maximum of {WEB_URL_MAX_LENGTH} characters.")
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("URLs with embedded credentials are not allowed.")
    if not parsed_url.hostname or "." not in parsed_url.hostname:
        raise ValueError(f"Invalid hostname in URL: {parsed_url.hostname}.")

def _html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = True
    converter.ignore_emphasis = True
    return converter.handle(html).strip()

def web_fetch(args: dict[str, Any], ctx: ToolContext) -> str:
    url = args.get("url", "")
    if not url:
        return "Error: 'url' argument is required."
    try:
        _validate_url(url)
    except ValueError as e:
        return f"Error: {str(e)}"

    # Cache lookup
    now = time.monotonic()
    if url in _web_fetch_cache:
        ts, cached_body = _web_fetch_cache[url]
        if now - ts < WEB_FETCH_CACHE_TTL_S:
            _web_fetch_cache.move_to_end(url)
            return cached_body
        else:
            del _web_fetch_cache[url]

    headers = {"User-Agent": WEB_USER_AGENT, "Accept": "text/html,text/*;q=0.9,*/*;q=0.5"}
    try:
        with httpx.Client(timeout=WEB_FETCH_TIMEOUT_S, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except httpx.RequestError as e:
        return f"Error fetching URL: {str(e)}"
    if len(resp.content) > WEB_FETCH_MAX_BYTES:
        return f"Error: Fetched content exceeds maximum size of {WEB_FETCH_MAX_BYTES} bytes."
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        body = _html_to_markdown(resp.text)
    elif content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
        body = resp.text
    else:
        return f"Error: Unsupported Content-Type '{content_type or 'unknown'}'."
    result = truncate_output(body, WEB_FETCH_MAX_CHARS)

    # Cache store + eviction
    _web_fetch_cache[url] = (now, result)
    _web_fetch_cache.move_to_end(url)
    while len(_web_fetch_cache) > WEB_FETCH_CACHE_MAX_SIZE:
        _web_fetch_cache.popitem(last=False)

    return result

def _unwrap_duckduckgo_url(href: str) -> str:
    # DuckDuckGo search results often have URLs wrapped in a redirect like "https://duckduckgo.com/l/?kh=-1&uddg=https%3A%2F%2Fexample.com"
    if "/l/" not in href:
        return href
    parsed = urlparse(href if href.startswith("http") else f"https:{href}")
    params = parse_qs(parsed.query)
    if "uddg" in params:
        return unquote(params["uddg"][0])
    return href

def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    headers = {"User-Agent": WEB_USER_AGENT}
    with httpx.Client(timeout=WEB_SEARCH_TIMEOUT_S, follow_redirects=True) as client:
        resp = client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
        )
        resp.raise_for_status()
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for href, title_html in pattern.findall(resp.text):
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        url = _unwrap_duckduckgo_url(href)
        if not title or not url:
            continue
        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break
    return results

def web_search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = args.get("query", "")
    if not query:
        return "Error: 'query' argument is required."
    max_results = max(1, min(int(args.get("max_results", 5)), 10))
    try:
        results = _duckduckgo_search(query, max_results)
    except httpx.RequestError as e:
        return f"Error performing web search: {str(e)}"
    if not results:
        return "[no results]"
    lines = [f"- {r['title']}\n  {r['url']}" for r in results]
    return truncate_output("\n".join(lines))

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
        description="""
        Read a text file.
        Supports line-based pagination using offset and limit.
        Required argument:
        - path: relative file path (example: pyproject.toml)
        Optional arguments:
        - offset: line offset to start reading from (default: 0)
        - limit: maximum number of lines to read (default: None, meaning no limit)
        """,
        run=read_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string", 
                    "description": "Relative path inside cwd"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line offset to start reading from, 0-based",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "default": None
                }
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
    registry.register_tool(Tool(
        name="glob",
        description="Find files matching a glob pattern. Example pattern: '**/*.py'",
        run=glob,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to search for (example: '**/*.py')"}
            },
            "required": ["pattern"]
        }
    ))
    registry.register_tool(Tool(
        name="grep",
        description="Search for a regex pattern in files. Example usage: pattern='def ', path='src/', ignore_case=true",
        run=grep,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Relative path to file or directory to search, defaults to current directory", "default": "."},
                "glob": {"type": "string", "description": "Optional glob pattern to filter files (example: '**/*.py')"},
                "ignore_case": {"type": "boolean", "description": "Whether to ignore case when matching the regex pattern", "default": False}
            },
            "required": ["pattern"]
        }
    ))
    registry.register_tool(Tool(
        name="project_tree",
        description="Show a tree view of the project files up to a certain depth. Example usage: max_depth=2",
        run=project_tree,
        parameters={
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer", 
                    "description": "Maximum depth to show in the tree, defaults to 3", 
                    "default": 3
                }
            },
            "required": []
        }
    ))
    registry.register_tool(Tool(
        name="web_fetch",
        description="Fetch the content of a web page. Example usage: url='https://www.example.com'",
        run=web_fetch,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The HTTP or HTTPS URL of the web page to fetch"}
            },
            "required": ["url"]
        }
    ))
    registry.register_tool(Tool(
        name="web_search",
        description="Perform a web search and return the top results. Example usage: query='latest news on AI', max_results=5",
        run=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Maximum number of search results to return (1-10), defaults to 5", "default": 5}
            },
            "required": ["query"]
        }
    ))

    return registry