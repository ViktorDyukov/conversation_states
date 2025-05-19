from pydantic import TypeAdapter, BaseModel
from langchain_core.messages import AnyMessage, HumanMessage
from typing import Any, Optional, Callable
from langgraph.graph.schema_utils import SchemaCoercionMapper

data = {
    "type": "system",
    "content": "This is a system message"
}


class MyWrapper(BaseModel):
    message: AnyMessage


# Создаем маппер для AnyMessage
mapper = SchemaCoercionMapper(MyWrapper)

# Вызываем маппер с данными
parsed = mapper(data)

print(parsed)
print(type(parsed))
