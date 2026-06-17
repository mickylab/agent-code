from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console

from .agent import run_agent
from .tools import create_default_tool_registry
from .model import create_provider

console = Console()
app = typer.Typer(add_completion=False)

def render_header(cwd: Path, provider: str, model: str, base_url: str | None) -> None:
    console.print(f"[bold green]Agent Code CLI[/bold green]")
    console.print(f"  [bold]Working Directory:[/bold] {cwd}")
    console.print(f"  [bold]Model Provider:[/bold] {provider}")
    console.print(f"  [bold]Model:[/bold] {model}")
    if base_url:
        console.print(f"  [bold]Base URL:[/bold] {base_url}")
    console.print()

def handle_slash_commands(command: str) -> bool:
    # For future extension: parse and handle slash commands like /help, /reset, etc.
    if command == "/help":
        console.print("[bold]Available Commands:[/bold]")
        console.print("  /help - Show this help message")
        console.print("  /exit - Exit the CLI")
        return True
    return False

def run_once(
    prompt: str,
    cwd: Path,
    provider_name: str,
    model: str,
    base_url: str | None,
    max_steps: int,
) -> None:
    provider = create_provider(provider_name, model, base_url) #TODO: use slash command to change provider/model on the fly
    default_tools = create_default_tool_registry()
    result = run_agent(prompt, provider, default_tools, max_steps = max_steps, cwd = cwd, stream = False)
    for line in result.trace:
        console.print(line)

@app.callback(invoke_without_command=True)
def main_command(
    prompt: str = typer.Argument("", help="The prompt to send to the agent"),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", "-c", help="The working directory for the agent"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="The model provider to use"),
    model: str = typer.Option("Qwen3.5-9B-MLX-4bit", "--model", "-m", help="The model to use"),
    base_url: str | None = typer.Option(None, "--base-url", "-b", help="The base URL for the model provider"),
    max_steps: int = typer.Option(8, "--max-steps", "-s", help="The maximum number of steps for the agent to take")
    ) -> None:
    resolved_cwd = cwd.resolve()
    text = prompt.strip()
    if text:
        run_once(text, resolved_cwd, provider, model, base_url, max_steps)
        return
    
    # if no prompt provided, enter interactive mode
    console.print("[bold blue]Entering interactive mode. Type your prompt and press Enter.[/bold blue]")
    render_header(resolved_cwd, provider, model, base_url)
    console.print("Type [bold]/help[/bold] for available commands. Press Ctrl+C to exit.")
    
    while True:
        try:
            user_input = console.input("[bold green]>>> [/bold green]").strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                if user_input == "/exit":
                    console.print("[bold red]Exiting...[/bold red]")
                    break
                elif handle_slash_commands(user_input):
                    continue
                else:
                    console.print(f"[bold red]Unknown command: {user_input}[/bold red]")
                    continue
            run_once(user_input, resolved_cwd, provider, model, base_url, max_steps)
        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting...[/bold red]")
            break

def main() -> None:
    app()