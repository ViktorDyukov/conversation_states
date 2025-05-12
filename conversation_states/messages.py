from typing import Literal, Optional, List, Dict, Callable, Tuple, Union, Annotated
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    trim_messages
)
from langgraph.graph import add_messages
import tiktoken
from .humans import Human


RoleLiteral = Literal["human", "ai", "tool", "system", "unknown"]


def count_tokens(self) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    content = getattr(self, "content", "")
    return len(tokenizer.encode(content or ""))


def get_role(msg: BaseMessage) -> RoleLiteral:
    if isinstance(msg, HumanMessage):
        return "human"
    elif isinstance(msg, AIMessage):
        return "ai"
    elif isinstance(msg, ToolMessage):
        return "tool"
    elif isinstance(msg, SystemMessage):
        return "system"
    elif hasattr(msg, "type"):
        msg_type = getattr(msg, "type", None)
        if msg_type in ("human", "ai", "tool", "system"):
            return msg_type  # type: ignore
    return "unknown"


MessageTypes = Annotated[
    Union[HumanMessage, AIMessage, ToolMessage, SystemMessage],
    Field(discriminator="type")
]


class MessageHistory(BaseModel):
    items: Annotated[list[MessageTypes], add_messages]

    @classmethod
    def from_msg(cls, items: List[BaseMessage]) -> "MessageHistory":
        # принудительная проверка на нужные подклассы
        for i, msg in enumerate(items):
            if not isinstance(msg, (HumanMessage, AIMessage, ToolMessage, SystemMessage)):
                raise TypeError(
                    f"Item {i} has invalid type: {type(msg).__name__}")
        return cls(items=items)

    @field_validator("items")
    @classmethod
    def check_all_messages_valid(cls, v):
        for i, msg in enumerate(v):
            if not isinstance(msg, (HumanMessage, AIMessage, ToolMessage, SystemMessage)):
                raise TypeError(
                    f"Invalid message type at index {i}: {type(msg).__name__}")
        return v

    def as_pretty(
        self,
        technical: bool = False,
        truncate: Optional[int] = None
    ) -> str:
        total_tokens = 0
        lines = []

        for msg in self.items:
            role = msg.type
            name = getattr(msg, "name", None)
            at_name = f"@{name}" if name else ""

            prefix = {
                "human": f"👤 User {at_name}",
                "ai": f"🤖 Assistant {at_name}",
                "tool": f"🛠 Tool ({name or 'unknown'})",
                "function": f"🧮 Function",
                "system": f"⚙️ System"
            }.get(role, f"🔹 {role} {at_name}")

            content = (msg.content or "").strip().replace("\n", " ")
            if truncate:
                content = content[:truncate] + \
                    "..." if len(content) > truncate else content

            tokens = msg.count_tokens()
            total_tokens += tokens

            if role == "ai" and "tool_calls" in msg.additional_kwargs:
                for call in msg.additional_kwargs["tool_calls"]:
                    func = call.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    lines.append(
                        f"🤖 Assistant called tool: `{tool_name}` with `{args}`")
                if not content:
                    continue

            line = f"{prefix}: {content}"
            if technical:
                line += f" ({tokens} tokens)"
            lines.append(line)

        header = f"💬 Messages: {len(self.items)}"
        if technical:
            header += f", {total_tokens} tokens"

        return header + "\n" + "\n".join(lines)

    def last(self, role: Optional[RoleLiteral] = None) -> Optional[BaseMessage]:
        if role is None:
            return self.items[-1] if self.items else None
        for msg in reversed(self.items):
            if get_role(msg) == role:
                return msg
        return None

    def remove_last(self):
        if self.items:
            self.items.pop()

    def trim(self, first_tokens: int = 50, last_tokens: int = 250) -> List[BaseMessage]:
        trimmed_first = trim_messages(
            self.items,
            max_tokens=first_tokens,
            strategy="first",
            token_counter=count_tokens,
            end_on=("ai", "tool"),
            allow_partial=True
        )
        trimmed_last = trim_messages(
            self.items,
            max_tokens=last_tokens,
            strategy="last",
            token_counter=count_tokens,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=True
        )
        return trimmed_first + trimmed_last

    def sender(self, users) -> Optional[Human]:
        last_human = self.last("human")
        last_any = self.last()
        if not last_human or not hasattr(last_human, "name"):
            return None

        username = getattr(last_human, "name", None)
        if not username:
            return None

        for user in users:
            if user.username == username:
                return user
        return None
