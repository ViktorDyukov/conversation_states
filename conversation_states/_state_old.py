from langgraph.graph import add_messages, MessagesState
from operator import add
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    RemoveMessage,
)
from typing import Literal, Optional, Annotated, Union, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken
from langgraph.types import StreamWriter
from langchain_core.messages import BaseMessage

from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# TODO
# split into internal and external states

# external
# - full user list
# - Should have only assistant and human messages
# - previous internal iteration (without ext message history)
# - full message history
# - external.from_internal()

# internal
# - full user list
# - Should have all iterations between the models with the names
# - Only trimmed messages that are needed to use
# - Summary
# - previous internal iteration (without ext message history)
# - internal.from_external()

# flow
# - TG uses ExternalState
# - messenger -> first node - convert ext -> int - update user list, take the trimmed message history, take the last summary from the previous internal state saved
# - preprocessing - generate instruction (user list, history, summary as a part of system message)
# - stream text assistant response
# - post processing - generate new summary (less trimmed messages), update user profile (last 3 messages from trimmed as a part of system message)
# - internal -> external - add text assistant response, save internal as last turn internal, update user data





#####
#####
#####
#####
#####

class OverallState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Annotated[str, add_summary] = ""
    users: Annotated[list[Human], add_user] = Field(
        default_factory=list)

    model_config = {
        "exclude_none": True,
        "arbitrary_types_allowed": True
    }

    def clear_state(self):
        removed = [RemoveMessage(id=m.id)
                   for m in self.messages if hasattr(m, "id") and m.id]
        self.messages = removed
        self.summary = ""
        self.users = []
        return

    def count_tokens(self, msg) -> int:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        content = getattr(msg, "content", "")
        return len(tokenizer.encode(content or ""))

    def get_role(self, msg: BaseMessage) -> RoleLiteral:
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

    def format_messages_block(
        self,
        messages: List[BaseMessage],
        technical_details: bool = False,
        truncate_chars: Optional[int] = None
    ) -> str:
        total_tokens = 0
        lines = []

        for msg in messages:
            role = msg.type  # 'human', 'ai', 'tool', etc.
            name = getattr(msg, "name", None)
            at_name = f"@{name}" if name else ""

            # Emoji-based prefix
            prefix = {
                "human": f"👤 User {at_name}",
                "ai": f"🤖 Assistant {at_name}",
                "tool": f"🛠 Tool ({name or 'unknown'})",
                "function": f"🧮 Function",
                "system": f"⚙️ System"
            }.get(role, f"🔹 {role} {at_name}")

            content = msg.content.strip().replace("\n", " ")
            if truncate_chars:
                content = (
                    content[:truncate_chars] + "...") if len(content) > truncate_chars else content

            tokens = self.count_tokens(msg.content)
            total_tokens += tokens

            # Show tool_calls separately if present
            if role == "ai" and "tool_calls" in msg.additional_kwargs:
                for call in msg.additional_kwargs["tool_calls"]:
                    func = call.get("function", {})
                    tool_name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    lines.append(
                        f"🤖 Assistant called tool: `{tool_name}` with `{args}`")
                if not content:
                    continue  # Skip if AI message has only tool calls

            line = f"{prefix}: {content}"
            if technical_details:
                line += f" ({tokens} tokens)"
            lines.append(line)

        header = f"💬 Messages: {len(messages)}"
        if technical_details:
            header += f", {total_tokens} tokens"

        return header + "\n" + "\n".join(lines)

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
        users_block = "👤 Users:\n" + \
            "\n".join(user_lines) if user_lines else "👤 Users: none"

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

    def summarize_last_turn(self) -> str:
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

    def remove_last_message(self):
        last = self.messages[-1]
        self.messages = [RemoveMessage(id=last.id)]
        return

    def get_last_message(self, role: Optional[RoleLiteral] = None) -> Optional[BaseMessage]:
        if role is None:
            return self.messages[-1] if self.messages else None

        for msg in reversed(self.messages):
            if self.get_role(msg) == role:
                return msg
        return None

    def get_trimmed_messages(self, first_tokens: int = 50, last_tokens: int = 250) -> List[BaseMessage]:

        trimmed_first = trim_messages(
            self.messages,
            max_tokens=first_tokens,
            strategy="first",
            token_counter=self.count_tokens,
            end_on=("ai", "tool"),
            allow_partial=True
        )

        trimmed_last = trim_messages(
            self.messages,
            max_tokens=last_tokens,
            strategy="last",
            token_counter=self.count_tokens,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=True
        )
        return trimmed_first + trimmed_last

    def get_sender(self) -> Optional[Human]:
        last_message = self.get_last_message("human")
        if not last_message or not hasattr(last_message, "name"):
            return None

        sender_username = getattr(last_message, "name", None)
        if not sender_username:
            return None

        for user in self.users:
            if user.username == sender_username:
                return user

        return None  # Sender not found in user list

    def has_user_with_username(self, username) -> bool:
        return any(u.username == username for u in self.users)
