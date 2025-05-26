from pydantic import BaseModel, Field
from typing import List
from uuid import UUID


class ExampleItem(BaseModel):
    input: str  # e.g. "I hate small talk."
    output: str  # e.g. "[Bot avoids small talk.]"


class InstructionItem(BaseModel):
    key: str  # e.g. "greeting_style"
    value: str  # e.g. "friendly but concise"
    examples: List[ExampleItem] = Field(default_factory=list)
    anti_examples: List[ExampleItem] = Field(default_factory=list)
    weight: int = 5  # 0–10
    condition: str  # e.g. "Only if user is not tired"


class InstructionList(BaseModel):
    items: List[InstructionItem] = Field(
        default_factory=list)  # e.g., [InstructionItem(...)]


class ThreadInstructionList(InstructionList):
    thread_id: UUID


class GlobalInstructionList(InstructionList):
    # TODO needed?
    pass
