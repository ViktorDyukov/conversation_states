from langgraph.graph import add_messages, MessagesState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    AIMessageChunk,
    HumanMessageChunk,
    ChatMessageChunk,
    SystemMessageChunk,
    FunctionMessageChunk,
    ToolMessageChunk,
    BaseMessage,
    AnyMessage,
    RemoveMessage,
)
from typing import Literal, Optional, Annotated, Union
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken
from langgraph.types import StreamWriter
from langchain_core.messages import BaseMessage


ActionType = Literal["image", "gif", "voice", "reaction",
                     "sticker", "system-message", "system-notification"]
Reaction = Literal[
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉", "🤩", "🤮",
    "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡",
    "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻",
    "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄",  "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡"
]


class Action(BaseModel):
    type: ActionType
    value: str


class ActionSender:
    writer: StreamWriter

    def __init__(self, writer: StreamWriter):
        self.writer = writer

    def send_action(self, action: Action):
        self.writer({"actions": [action.dict()]})

    def send_reaction(self, reaction: Reaction):
        action = Action(
            type="reaction",
            value=reaction
        )
        self.send_action(action)


class Human(BaseModel):
    messenger_id: str
    first_name: str
    last_name: str
    preffered_name: str
    preferences: str


class OverallState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str] = None
    actions: list[Action] = Field(default_factory=list)
    users: list[Human] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     if not hasattr(self, "messages"):
    #         self.messages = []

    def clear_state(self):
        removed = [RemoveMessage(id=m.id)
                   for m in self.messages if hasattr(m, "id") and m.id]
        self.messages = removed
        self.summary = ""
        self.actions = []
        self.users = []
        return

    def count_tokens(self, text: str) -> int:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        return len(tokenizer.encode(text))

    def get_role(self, msg: BaseMessage) -> str:
        if isinstance(msg, HumanMessage):
            return "human"
        elif isinstance(msg, AIMessage):
            return "ai"
        elif isinstance(msg, ToolMessage):
            return "tool"
        elif isinstance(msg, SystemMessage):
            return "system"
        else:
            return "unknown"

    def summarize_overall_state(self) -> str:
        # 1. Users
        user_lines = []
        for u in self.users:
            name_line = f"{u.first_name} {u.last_name} (@{u.messenger_id})"
            user_lines.append(
                f"- {name_line}\n"
                f"  - preferred_name: {u.preffered_name or 'not provided'}\n"
                f"  - preferences: {u.preferences or 'not provided'}"
            )
        users_block = "👤 Users:\n"
        "\n".join(user_lines) if user_lines else "👤 Users: none"

        # 2. Messages
        messages_block = []
        total_msg_tokens = 0
        for msg in self.messages:
            tokens = self.count_tokens(msg.content)
            total_msg_tokens += tokens
            text = msg.content.strip().replace("\n", " ")
            preview = (text[:100] + "...") if len(text) > 100 else text
            messages_block.append(
                f"- {'User' if self.get_role(msg) == 'human' else 'Assistant'}: {preview} ({tokens} tokens)")

        # 3. Summary
        if self.summary:
            summary_text = self.summary
            summary_tokens = self.count_tokens(summary_text)
        else:
            summary_text = "(No summary provided)"
            summary_tokens = 0
        summary_block = f"📝 Summary ({summary_tokens} tokens):\n{summary_text}"

        return (
            f"{users_block}\n\n"
            f"Messages: {len(self.messages)} total, {total_msg_tokens} tokens\n"
            + "\n".join(messages_block)
            + "\n\n"
            + summary_block
        )

    def summarize_last_turn(self) -> str:
        if not self.messages:
            return "No messages available."

        # Группируем сообщения в цепочки
        turns = []
        current_turn = []

        for msg in self.messages:
            if msg.type == "human":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                if current_turn:
                    current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        if not turns:
            return "No complete turn found."

        last_turn = turns[-1]

        lines = []
        total_tokens = 0

        for msg in last_turn:
            role = self.get_role(msg)
            text = msg.content.strip().replace("\n", " ")
            tokens = self.count_tokens(msg.content)
            total_tokens += tokens

            preview = (text[:200] + "...") if len(text) > 200 else text
            prefix = {
                "human": "👤 User",
                "ai": "🤖 Assistant",
                "tool": f"🛠 Tool ({getattr(msg, 'name', 'unknown')})",
                "function": "🧮 Function",
                "system": "⚙️ System"
            }.get(role, f"🔹 {role}")

            lines.append(f"{prefix} ({tokens} tokens):\n{preview}")

        return (
            f"🧵 Last turn ({len(last_turn)} messages, {total_tokens} tokens):\n\n"
            + "\n\n".join(lines)
        )

    def remove_last_message(self):
        last = self.messages[-1]
        self.messages = [RemoveMessage(id=last.id)]
        return
