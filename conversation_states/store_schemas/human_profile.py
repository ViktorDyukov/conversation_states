from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID
from conversation_states.humans import Human
from .instruction import InstructionList


class TemporaryStateItem(BaseModel):
    value: List[str]  # e.g. ["do not disturb", "exam at 12:30"]
    recorded_at: datetime  # when the status was recorded
    # until when this is expected to be relevant
    relevant_until: Optional[datetime] = None
    note: Optional[str] = None  # e.g. "finals week, needs quiet"


class MemoryItem(BaseModel):
    key: str  # e.g. "location"
    value: List[str]  # e.g. ["New York"]


class MemoryList(BaseModel):
    category: str  # e.g. "main"
    memories: List[MemoryItem] = Field(default_factory=list)


class HumanProfile(BaseModel):
    # e.g. (локация, работа, предпоч имя, зона, время и тд)
    main: MemoryList = Field(
        default_factory=lambda: MemoryList(category="main"))
    # e.g.  (предпочтения в общении, предпочтения, в целом, стиль общения) вкл нравится ли что бот первый пишет, нравится ли общаться с ботом
    preferences: MemoryList = Field(
        default_factory=lambda: MemoryList(category="preferences"))
    instructions: InstructionList = Field(
        default_factory=list)  # e.g. "greet with emoji"
    temporary_intents: List[TemporaryStateItem] = Field(
        default_factory=list)  # e.g. "feeling overwhelmed today"
    other_info: List[MemoryList] = Field(
        default_factory=list)  # e.g. hobbies, favorite food
    thread_id: UUID
    user: Human

    def to_prompt(self) -> str:
        # Returns user state formatted for LLM prompt
        pass

    def update(self, **kwargs) -> None:
        # Updates specific fields of the state in-place
        pass

    def diff(self, other: 'HumanProfile') -> dict:
        # Compares current state with another and returns a diff dict
        pass

    def explain_state(self) -> str:
        # Returns a natural language explanation of the state
        pass
