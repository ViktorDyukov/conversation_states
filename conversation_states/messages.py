from typing import Literal, Optional, List, Dict, Callable, Tuple, Union, Annotated, Any, Sequence, TypedDict
from pydantic import BaseModel, Field, model_validator, TypeAdapter
from langchain_core.messages.utils import _get_type
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    AnyMessage,
    trim_messages
)
from langgraph.graph import add_messages, MessagesState
import tiktoken
from .humans import Human
from pydantic import Discriminator, Field, Tag


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
    Union[
        Annotated[AIMessage, Tag(tag="ai")],
        Annotated[HumanMessage, Tag(tag="human")],
        Annotated[SystemMessage, Tag(tag="system")],
        Annotated[ToolMessage, Tag(tag="tool")],
    ],
    Field(discriminator=Discriminator(_get_type)),
]


# class MessageHistory(BaseModel):
#     items: Annotated[list[AnyMessage], add_messages]

#     model_config = {
#         "exclude_none": True,
#         "arbitrary_types_allowed": True
#     }

#     def as_pretty(
#         self,
#         technical: bool = False,
#         truncate: Optional[int] = None
#     ) -> str:
#         total_tokens = 0
#         lines = []

#         for msg in self.items:
#             role = msg.type
#             name = getattr(msg, "name", None)
#             at_name = f"@{name}" if name else ""

#             prefix = {
#                 "human": f"👤 User {at_name}",
#                 "ai": f"🤖 Assistant {at_name}",
#                 "tool": f"🛠 Tool ({name or 'unknown'})",
#                 "function": f"🧮 Function",
#                 "system": f"⚙️ System"
#             }.get(role, f"🔹 {role} {at_name}")

#             content = (msg.content or "").strip().replace("\n", " ")
#             if truncate:
#                 content = content[:truncate] + \
#                     "..." if len(content) > truncate else content

#             tokens = msg.count_tokens()
#             total_tokens += tokens

#             if role == "ai" and "tool_calls" in msg.additional_kwargs:
#                 for call in msg.additional_kwargs["tool_calls"]:
#                     func = call.get("function", {})
#                     tool_name = func.get("name", "unknown")
#                     args = func.get("arguments", "{}")
#                     lines.append(
#                         f"🤖 Assistant called tool: `{tool_name}` with `{args}`")
#                 if not content:
#                     continue

#             line = f"{prefix}: {content}"
#             if technical:
#                 line += f" ({tokens} tokens)"
#             lines.append(line)

#         header = f"💬 Messages: {len(self.items)}"
#         if technical:
#             header += f", {total_tokens} tokens"

#         return header + "\n" + "\n".join(lines)

#     def last(self, role: Optional[RoleLiteral] = None) -> Optional[BaseMessage]:
#         if role is None:
#             return self.items[-1] if self.items else None
#         for msg in reversed(self.items):
#             if get_role(msg) == role:
#                 return msg
#         return None

#     def remove_last(self):
#         if self.items:
#             self.items.pop()

#     def trim(self, first_tokens: int = 50, last_tokens: int = 250) -> List[BaseMessage]:
#         trimmed_first = trim_messages(
#             self.items,
#             max_tokens=first_tokens,
#             strategy="first",
#             token_counter=count_tokens,
#             end_on=("ai", "tool"),
#             allow_partial=True
#         )
#         trimmed_last = trim_messages(
#             self.items,
#             max_tokens=last_tokens,
#             strategy="last",
#             token_counter=count_tokens,
#             start_on="human",
#             end_on=("human", "tool"),
#             include_system=True,
#             allow_partial=True
#         )
#         return trimmed_first + trimmed_last

#     def sender(self, users) -> Optional[Human]:
#         last_human = self.last("human")
#         last_any = self.last()
#         if not last_human or not hasattr(last_human, "name"):
#             return None

#         username = getattr(last_human, "name", None)
#         if not username:
#             return None

#         for user in users:
#             if user.username == username:
#                 return user
#         return None
