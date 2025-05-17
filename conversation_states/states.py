from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, RemoveMessage, trim_messages, AnyMessage
from langgraph.graph import add_messages, MessagesState

import tiktoken
from .humans import Human
from .messages import MessageAPI
from .utils.reducers import add_user, add_summary


class InternalState(BaseModel):
    reasoning_messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list)
    external_messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list)
    last_external_message: BaseMessage
    users: Annotated[list[Human], add_user] = Field(default_factory=list)
    last_sender: Human
    summary: str = ""

    @property
    def reasoning_messages_api(self) -> MessageAPI:
        return MessageAPI(self, "reasoning_messages")

    @property
    def external_messages_api(self) -> MessageAPI:
        return MessageAPI(self, "external_messages")

    @classmethod
    def from_external(cls, external: "ExternalState") -> "InternalState":
        last_message = external.messages_api.last()
        sender = external.messages_api.sender(external.users)

        return cls(
            reasoning_messages=[],
            summary=external.summary,
            users=list(external.users),
            external_messages=external.messages,
            last_external_message=last_message,
            last_sender=sender
        )


class ExternalState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list)
    users: Annotated[list[Human], add_user] = Field(
        default_factory=list)
    summary: str = ""
    last_internal_state: Optional[InternalState] = None

    @property
    def messages_api(self) -> MessageAPI:
        return MessageAPI(self, "messages")

    @classmethod
    def from_internal(cls, internal: "InternalState", assistant_message: "AIMessage") -> "ExternalState":
        return cls(
            messages=[*internal.external_messages_api.items, assistant_message],
            users=list(internal.users),
            summary=internal.summary
            # last_internal_state=internal
        )

    def clear_state(self):
        removed = [RemoveMessage(id=m.id)
                   for m in self.messages if hasattr(m, "id") and m.id]
        self.messages = removed
        self.summary = ""
        self.users = []
        return

    def summarize_overall_state(self) -> str:
        # 1. Users
        user_lines = []
        for u in self.users:
            name_line = f"{u.first_name} {u.last_name} ({u.username})"
            user_lines.append(
                f"- {name_line}\n"
                f"  - preferred_name: {u.preferred_name or 'not provided'}\n"
                f"  - info: {u.information or 'not provided'}"
            )
        if user_lines:
            users_block = "👤 Users:\n" + "\n".join(user_lines)
        else:
            users_block = "👤 Users: none"

        # 2. Messages (with formatting function)
        messages = self.messages
        messages_block = self.format_messages_block(
            messages=messages,
            technical_details=True,
            truncate_chars=100
        )

        # 3. Summary
        if self.summary:
            summary_text = self.summary.strip()
            summary_tokens = self.count_tokens(summary_text)
        else:
            summary_text = "(No summary provided)"
            summary_tokens = 0
            summary_block = f"📝 Summary ({summary_tokens} tokens):\n{summary_text}"

        return f"{users_block}\n\n{messages_block}\n\n{summary_block}"

    def show_last_reasoning(self) -> str:
        if not self.messages:
            return "No messages available."

        # Group into turns
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

        formatted = self.format_messages_block(
            messages=last_turn,
            technical_details=True,
            truncate_chars=100
        )

        return f"🧵 Last turn:\n\n{formatted}"

    # from internal
