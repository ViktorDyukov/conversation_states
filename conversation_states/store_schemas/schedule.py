# file: schedule_models.py
from pydantic import BaseModel, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from .task import ActionItem, Task
from conversation_states import Human
from uuid import UUID
from croniter import croniter, CroniterBadCronError


class CronExpression(BaseModel):
    type: Literal["cron"]
    expression: str  # e.g., "0 9 * * *"

    @field_validator("expression")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        try:
            croniter(v)
        except CroniterBadCronError:
            raise ValueError(f"Invalid cron expression: {v}")
        return v


class Schedule(BaseModel):
    frequency: CronExpression
    reply_to: Optional[Human] = None  # ID or name of the human to notify
    requested_by: Optional[Human] = None  # additional info
    action: ActionItem
    ends_at: Optional[datetime] = None


class ScheduleList(BaseModel):
    thread_id: UUID
    schedules: List[Schedule]

    def add(self, schedules: List[Schedule]) -> None:
        # Adds schedules to the manager
        pass

    def get():
        pass

    def remove_by_id(self, schedule_ids: List[str]) -> None:
        # Removes schedules by their ID
        pass

    def generate_tasks(self, date: Optional[datetime] = None) -> List[Task]:
        # Generates tasks for the given date (defaults to now) based on active schedules
        pass
