from langgraph.graph import add_messages, MessagesState
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, RemoveMessage
from typing import Literal, Optional, Annotated
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken



ContentType = Literal["text", "code", "alert"]
ActionType = Literal["image", "gif", "audio", "voice", "reaction", "sticker"]
Reaction = Literal[
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉", "🤩", "🤮",
    "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡",
    "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻",
    "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄",  "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡"
]
Status = Literal["pending"]


class Action(BaseModel):
    type: ActionType
    value: str


class Human(BaseModel):
    messenger_id: str
    first_name: str
    last_name: str
    preffered_name: str
    preferences: str


class OverallState(BaseModel):
    messages: Annotated[list, add_messages]
    summary: Optional[str] = None
    actions: list[Action] = Field(default_factory=list)
    users: list[Human] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "messages"):
            self.messages = []

    def add_reaction(self, reaction: Reaction, message_id: str):
        self.actions.append(Action(
            type="reaction",
            value=reaction,
            timestamp=datetime.now(),
            message_id=message_id
        ))

    def clear_state(self):
        removed = [RemoveMessage(id=m.id)
                   for m in self.messages if hasattr(m, "id") and m.id]
        self.messages = removed
        self.summary = ""
        self.actions = []
        self.users = []
        return

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    def summarize_overall_state(state: OverallState) -> str:
        # 1. Users
        user_lines = []
        for u in state.users:
            name_line = f"{u.first_name} {u.last_name} (@{u.messenger_id})"
            user_lines.append(
                f"- {name_line}\n"
                f"  - preferred_name: {u.preffered_name or 'not provided'}\n"
                f"  - preferences: {u.preferences or 'not provided'}"
            )
        users_block = "👤 Users:\n" + \
            "\n".join(user_lines) if user_lines else "👤 Users: none"

        # 2. Messages
        messages_block = []
        total_msg_tokens = 0
        for msg in state.messages:
            tokens = count_tokens(msg.content)
            total_msg_tokens += tokens
            text = msg.content.strip().replace("\n", " ")
            preview = (text[:100] + "...") if len(text) > 100 else text
            messages_block.append(
                f"- {'User' if msg.role == 'human' else 'Assistant'}: {preview} ({tokens} tokens)")

        # 3. Summary
        summary_text = state.summary or "(No summary provided)"
        summary_tokens = count_tokens(summary_text)
        summary_block = f"📝 Summary ({summary_tokens} tokens):\n{summary_text}"

        return (
            f"{users_block}\n\n"
            f"💬 Messages: {len(state.messages)} total, {total_msg_tokens} tokens\n"
            + "\n".join(messages_block)
            + "\n\n"
            + summary_block
        )
