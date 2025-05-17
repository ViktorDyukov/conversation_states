from typing import Annotated, Union, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    trim_messages
)


MessageType = Annotated[
    Union[HumanMessage, AIMessage],
    Field(discriminator="type")
]


class MessageHistory(BaseModel):
    items: Annotated[
        List[MessageType],
        add_messages
    ]


# ✅ Валидация
data = {
    "items": [
        {"type": "human", "content": "Hi"},
        {"type": "ai", "content": "Hello!"}
    ]
}

msg = MessageHistory.model_validate(data)
print(msg)
