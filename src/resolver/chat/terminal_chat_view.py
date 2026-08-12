from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich.text import Text


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

    def thinking(self) -> Status:
        """Return a rich context manager to continually render a loading spinner."""
        return self.console.status("[dim magenta]Working...[/]")

    def show_reply(self, text: str) -> None:
        """Displays an LLM response message in it's own dedicated panel."""
        # parse the input text
        body = Text()
        for line in text.splitlines():
            # is this a heading (short, all caps), or a normal line?
            stripped = line.strip()
            if stripped and stripped.isupper() and len(stripped) <= 30:
                body.append(stripped, style="bold cyan")
            else:
                body.append(line)

            # add separation space between lines
            body.append("\n")

        # display the reply text in a paneled box with to differentiate
        # it from user's input text
        self.console.print(
            Panel(
                body,
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
