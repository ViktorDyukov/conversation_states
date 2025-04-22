from langgraph.graph import add_messages, MessagesState
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, RemoveMessage
from typing import Literal, Optional, Annotated
from datetime import datetime
from pydantic import BaseModel, Field


ContentType = Literal["text", "code", "alert"]
ActionType = Literal["image", "gif", "audio", "voice", "reaction", "sticker"]
Reaction = Literal[
    "👍", "❤️", "😂", "🔥", "🥰",
    "👏", "😢", "😮", "🤔", "🎉",
    "🤯", "💩", "🙏", "💯", "😎",
    "👀", "🤡", "😐", "🤬", "🥺"
]
Status = Literal["pending"]


class Action(BaseModel):
    type: ActionType
    value: str
    timestamp: datetime
    status: Status = "pending"
    message_id: Optional[str] = None


class Human(BaseModel):
    messenger_id: str
    first_name: str
    last_name: str
    preffered_name: str
    preferences: str


class OverallState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
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