from langgraph.graph import add_messages, MessagesState
from operator import add
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
from typing import Literal, Optional, Annotated, Union, Dict
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
    username: str
    first_name: str
    last_name: Optional[str] = None
    preffered_name: Optional[str] = None
    information: Dict[str, str] = Field(default_factory=dict)


class OverallState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str] = None
    actions: list[Action] = Field(default_factory=list)
    users: Annotated[list[Human], add] = Field(default_factory=list)

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
            name_line = f"{u.first_name} {u.last_name} ({u.username})"
            user_lines.append(
                f"- {name_line}\n"
                f"  - preferred_name: {u.preffered_name or 'not provided'}\n"
                f"  - info: {u.information or 'not provided'}"
            )
        users_block = "👤 Users:\n" + \
            "\n".join(user_lines) if user_lines else "👤 Users: none"

        # 2. Messages
        messages_block = []
        total_msg_tokens = 0

        for msg in self.messages:
            role = self.get_role(msg)
            tokens = self.count_tokens(msg.content)
            total_msg_tokens += tokens

            if role == "ai" and "tool_calls" in msg.additional_kwargs:
                for call in msg.additional_kwargs["tool_calls"]:
                    func_name = call.get("function", {}).get("name", "unknown")
                    args = call.get("function", {}).get("arguments", "{}")
                    messages_block.append(
                        f"- Assistant called tool: `{func_name}` with `{args}`")
                if msg.content.strip():  # Если есть текст, тоже покажем
                    text = msg.content.strip().replace("\n", " ")
                    preview = (text[:100] + "...") if len(text) > 100 else text
                    messages_block.append(
                        f"- Assistant: {preview} ({tokens} tokens)")
            elif role == "tool":
                tool_name = getattr(msg, "name", "unknown")
                text = msg.content.strip().replace("\n", " ")
                preview = (text[:100] + "...") if len(text) > 100 else text
                messages_block.append(
                    f"- Tool ({tool_name}): {preview} ({tokens} tokens)")
            else:
                role_label = "User" if role == "human" else "Assistant"
                text = msg.content.strip().replace("\n", " ")
                preview = (text[:100] + "...") if len(text) > 100 else text
                messages_block.append(
                    f"- {role_label}: {preview} ({tokens} tokens)")

        # 3. Summary
        summary_text = self.summary.strip() if self.summary else "(No summary provided)"
        summary_tokens = self.count_tokens(summary_text)
        summary_block = f"📝 Summary ({summary_tokens} tokens):\n{summary_text}"

        # Final output
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

        # Группировка в ходы
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
            tokens = self.count_tokens(msg.content)
            total_tokens += tokens

            prefix = {
                "human": "👤 User",
                "ai": "🤖 Assistant",
                "tool": f"🛠 Tool ({getattr(msg, 'name', 'unknown')})",
                "function": "🧮 Function",
                "system": "⚙️ System"
            }.get(role, f"🔹 {role}")

            # Показываем вызов тула отдельно
            if role == "ai" and "tool_calls" in msg.additional_kwargs:
                for call in msg.additional_kwargs["tool_calls"]:
                    func_name = call.get("function", {}).get("name", "unknown")
                    args = call.get("function", {}).get("arguments", "{}")
                    lines.append(
                        f"🤖 Assistant called tool: `{func_name}` with `{args}`")
                # Если сам контент пустой — пропускаем его
                if msg.content.strip():
                    preview = msg.content.strip().replace("\n", " ")
                    preview = (preview[:200] +
                               "...") if len(preview) > 200 else preview
                    lines.append(f"{prefix} ({tokens} tokens):\n{preview}")
            elif role == "tool":
                tool_output = msg.content.strip().replace("\n", " ")
                lines.append(f"{prefix} ({tokens} tokens):\n{tool_output}")
            else:
                text = msg.content.strip().replace("\n", " ")
                preview = (text[:200] + "...") if len(text) > 200 else text
                lines.append(f"{prefix} ({tokens} tokens):\n{preview}")

        return (
            f"🧵 Last turn ({len(last_turn)} messages, {total_tokens} tokens):\n\n"
            + "\n\n".join(lines)
        )

    def remove_last_message(self):
        last = self.messages[-1]
        self.messages = [RemoveMessage(id=last.id)]
        return

    def get_last_message(self):
        return self.messages[-1]

    def has_user_with_username(self, username) -> bool:
        return any(u.username == username for u in self.users)
