import random

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

# MTG-flavored spinner titles shown while waiting on a response, picked at
# random each time instead of a single generic "Working..."
_THINKING_MESSAGES = [
    "Resolving the stack…",
    "Counterspelling…",
    "Checking state-based actions…",
    "Passing priority…",
    "Paying the mana cost…",
    "Untapping lands…",
    "Declaring attackers…",
    "Assigning combat damage…",
    "Searching the library…",
    "Shuffling up…",
    "Scrying…",
    "Reading the fine print…",
    "Consulting the Comprehensive Rules…",
    "Triggering an ability…",
    "Putting it on the stack…",
    "Checking for hexproof…",
    "Casting at instant speed…",
    "Untapping for value…",
    "Racking up +1/+1 counters…",
    "Milling for value…",
    "Tapping out…",
    "Holding up interaction…",
    "Blocking (or not)…",
    "Cracking a fetchland…",
    "Waiting for the smoke to clear…",
]


class TerminalChatView:
    def __init__(self) -> None:
        self.console = Console()

    def show_banner(self, model: str) -> None:
        """Displays text describing a chat session initiation."""
        self.console.print(
            f'[dim]Chat session started with "{model}", '
            'type "exit" or "quit" to end session'
        )

    def show_divider(self) -> None:
        """Render a simple horizontal rule."""
        self.console.print(Rule(style="dim"))

    def show_recognized_card(self, name: str) -> None:
        """Displays the recognised card's name so the user gets
        instant confirmation that recognition worked."""
        self.console.print()
        self.console.print(f"[bold green]Recognised:[/] [bold]{name}[/]")

    def thinking(self) -> Status:
        """Return a rich context manager to continually render a loading spinner
        with a randomly chosen, MTG-flavored title."""
        message = random.choice(_THINKING_MESSAGES)
        return self.console.status(f"[dim magenta]{message}[/]")

    def show_reply(self, text: str) -> None:
        """Displays an LLM response message in it's own dedicated panel."""
        self.console.print(
            Panel(
                Markdown(text),
                title="[bold magenta]RESOLVER BOT[/]",
                title_align="center",
                border_style="magenta",
            )
        )

    def show_tool_usage(self, text: str) -> None:
        """Displays text about agentic tool uses."""
        self.console.print(f"[dim magenta]{text}[/]")

    def get_user_input(self) -> str | None:
        """Displays text prompting a user for input, and returns the captured input."""
        # prompt user for input and check for termination signal
        user_input = self.console.input("[bold cyan]>>>[/] ")
        if user_input.lower().strip() in ("q", "quit", "exit"):
            return None
        return user_input

    def show_interrupt(self) -> None:
        """Display a KeyboardInterrupt message."""
        self.console.print("\n[yellow]Session interrupted by user. Exiting...[/]")

    def show_error_message(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"\n[bold red]An error occurred:[/] {message}\n")
