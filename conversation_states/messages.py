from typing import Literal, Optional, List
from pydantic import BaseModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    AnyMessage,
    trim_messages,
    RemoveMessage
)
import tiktoken
from .humans import Human


RoleLiteral = Literal["human", "ai", "tool", "system", "unknown"]


def count_tokens(msg) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    return len(tokenizer.encode(content))


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


class MessageAPI:
    def __init__(self, state: BaseModel, field_name: str):
        self._state = state
        self._field_name = field_name

    @property
    def items(self) -> List[AnyMessage]:
        return getattr(self._state, self._field_name)

    def as_pretty(self, technical: bool = False, truncate: Optional[int] = None) -> str:
        total_tokens = 0
        lines = []

        for msg in self.items:
            role = msg.type
            name = getattr(msg, "name", None)
            at_name = f"@{name}" if name else ""

            prefix = {
                "human": f"👤 User {at_name}",
                "ai": f"🤖 Assistant ({name})",
                "tool": f"🛠 Tool ({name or 'unknown'})",
                "function": f"🧮 Function",
                "system": f"⚙️ System"
            }.get(role, f"🔹 {role} {at_name}")

            content = (msg.content or "").strip().replace("\n", " ")
            if truncate:
                content = content[:truncate] + \
                    "..." if len(content) > truncate else content

            tokens = count_tokens(msg)
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

            if technical:
                prefix += f" ({tokens} tokens)"
            line = f"{prefix}: <blockquote>{content}</blockquote>\n"
            lines.append(line)

        header = f"Messages: {len(self.items)}"
        if technical:
            header += f", {total_tokens} tokens"
        header += "\n"

        return header + "\n" + "\n".join(lines)

    def last(
        self,
        role: Optional[RoleLiteral] = None,
        name: Optional[str] = None,
        count: CountType = None
    ) -> list[BaseMessage]:

        # Приведение count к числу
        if count is None:
            count = 1

        # Без фильтра: просто берем из self.items
        if role is None and name is None:
            if not self.items:
                return []
            if count == "all":
                return list(self.items)
            return self.items[-count:]

        # С фильтром: собираем подходящие
        filtered = []
        for msg in reversed(self.items):
            if role is not None and get_role(msg) == role:
                filtered.append(msg)
            elif name is not None and getattr(msg, "name", None) == name:
                filtered.append(msg)
            if count != "all" and len(filtered) >= count:
                break

        return list(reversed(filtered))

    def remove_last(self):
        for msg in reversed(self.items):
            if hasattr(msg, "id") and msg.id:
                self.items.append(RemoveMessage(id=msg.id))
                return

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
        last_human = self.last(role="human")
        if not last_human or not hasattr(last_human, "name"):
            return None
        username = getattr(last_human, "name", None)
        if not username:
            return None
        for user in users:
            if user.username == username:
                return user
        return None
