# file: task_models.py
from pydantic import BaseModel
from typing import Optional, Literal, List
from datetime import datetime

from sqlalchemy import Boolean
from conversation_states import Human


class ActionItem(BaseModel):
    type: Literal["remind", "research", "ask_user", "check_profile"]
    instruction: str
    context: Optional[str] = None  # used by LLM


class ApproximateDateTime(BaseModel):
    target: datetime
    user_timezone: Optional[str] = None  # used by LLM
    hr_before: int = 0
    hr_after: int = 0

    def get_range(self) -> tuple[datetime, datetime]:
        # Returns start and end of time range window
        pass

    def in_range(self, date: datetime) -> bool:
        # Checks if given date falls within the time range window
        pass


class Task(BaseModel):
    time: ApproximateDateTime
    requested_by: Optional[Human]
    reply_to: Optional[Human]
    action: ActionItem
    completed_at: datetime


class TaskList(BaseModel):
    thread_id: str
    tasks: List[Task]

    def get_by_date(self, from_date: datetime, to_date: datetime) -> List[Task]:
        # Returns tasks within a date range
        pass

    def add(self, tasks: List[Task]) -> Boolean:
        # Adds new tasks to the list
        pass

    def remove_by_id(self, task_ids: List[str]) -> Boolean:
        # Removes tasks by ID
        pass
