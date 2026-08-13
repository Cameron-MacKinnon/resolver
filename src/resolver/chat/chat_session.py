import json
import os
from typing import cast

from dotenv import load_dotenv
from openrouter import OpenRouter

from ..paths import PROJECT_ROOT
from .agent_tools import AgentTools
from .conversation import Conversation
from .terminal_chat_view import TerminalChatView


class MissingApiKeyError(Exception):
    """Raised when the user doesn't have an API key in their .env"""


class ChatSession:
    """Orchestrates an agentic chat session with an LLM and manages tool usage."""

    def __init__(
        self,
        conversation: Conversation,
        view: TerminalChatView,
        tools: AgentTools,
        model: str,
    ) -> None:
        self.conversation = conversation
        self.view = view
        self.tools = tools
        self.model = model

        # set / validate API key
        load_dotenv(PROJECT_ROOT / ".env")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if self.api_key is None:
            raise MissingApiKeyError(
                'API key not set -add it to ".env" at project root'
            )

    def _get_reply(self, client: OpenRouter) -> str:
        """Prompt the model in a loop until it gives a plain-text answer.

        A single call to send() can come back one of two ways:
          - a normal answer (message.content is the reply, tool_calls empty)
          - a request to use one or more tools (message.tool_calls populated)

        In the second case we have to satisfy every tool call before the
        model will produce a real answer, so this loops: dispatch whatever
        was asked for, feed the results back in, and call send() again. This
        can happen several times in a row (a tool result might prompt the
        model to ask for another tool), which is why this is a loop and not
        a single if/else, we don't know in advance how many round-trips a
        single human message will need.
        """
        while True:
            # console.status()-style spinner wraps just the network call,
            # so the view shows "thinking" for exactly as long as we're
            # actually waiting on a response
            with self.view.thinking():
                response = client.chat.send(
                    model=self.model,
                    messages=self.conversation.memory,
                    tools=self.tools.schemas(),
                )
            message = response.choices[0].message

            # no tool calls means this is a genuine answer - we're done
            if not message.tool_calls:
                return str(message.content)

            # the model wants to use one or more tools. Per the API's
            # protocol, the assistant's request to call tools has to be
            # recorded in memory first (as its own message, carrying
            # tool_calls), before any of the individual tool results - the
            # model needs to see its own request alongside the responses to
            # stay coherent on subsequent turns
            self.conversation.add_to_memory(
                role="assistant",
                content=str(message.content) if message.content else None,
                tool_calls=[call.model_dump() for call in message.tool_calls],
            )

            # now actually run each requested tool and record its result.
            # tool_call_id ties each result back to the specific call that
            # asked for it, since a single turn can request several at once
            for call in message.tool_calls:
                self.view.show_tool_usage(f"{self.tools.label_for(call.function.name)}...")
                result = self.tools.dispatch(
                    call.function.name, call.function.arguments
                )
                self.conversation.add_to_memory(
                    role="tool", content=result, tool_call_id=call.id
                )
            # loop back around: send() gets called again with the tool
            # results now in memory, and the model decides what to do next

    def launch_chat_session(self, payload: dict[str, dict | list | int]) -> None:
        """Run a full terminal chat session: an optional first turn seeded
        with the recognised card's data, then a human-in-the-loop exchange
        until the user signals they're done."""
        self.view.show_banner(self.model)

        with OpenRouter(api_key=self.api_key) as client:
            try:
                # the very first turn is special: instead of waiting on
                # typed user input, we seed the conversation with the
                # card payload from the recognition pipeline and let the
                # model open with a summary
                if self.conversation.first_message:
                    card_details = cast(dict, payload["card_details"])
                    card_name = cast(str, card_details["name"])
                    self.view.show_recognized_card(card_name)
                    self.conversation.add_to_memory(
                        role="user", content=json.dumps(payload)
                    )
                    reply = self._get_reply(client)
                    self.view.show_reply(reply)
                    self.conversation.add_to_memory(role="assistant", content=reply)
                    self.conversation.first_message = False

                # ordinary human turns: prompt, check for an exit signal
                # (get_user_input returns None to signal termination. Then run
                # the same send-and-maybe-use-tools cycle as above
                while True:
                    self.view.show_divider()
                    user_input = self.view.get_user_input()
                    if user_input is None:
                        break

                    self.conversation.add_to_memory(role="user", content=user_input)
                    reply = self._get_reply(client)
                    self.view.show_reply(reply)
                    self.conversation.add_to_memory(role="assistant", content=reply)
            except KeyboardInterrupt:
                self.view.show_interrupt()
            except Exception as e:
                self.view.show_error_message(str(e))
