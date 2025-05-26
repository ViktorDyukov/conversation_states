# file: versioned.py
from pydantic import BaseModel
from typing import Generic, TypeVar, List, Callable
from datetime import datetime

T = TypeVar("T")


class VersionRecord(BaseModel, Generic[T]):
    data: T
    timestamp: datetime
    changed_by: Optional[str] = None
    comment: Optional[str] = None


class VersionedObject(BaseModel, Generic[T]):
    object_id: str
    current: T
    versions: List[VersionRecord[T]]

    def add_version(self, new_data: T, changed_by: Optional[str] = None, comment: Optional[str] = None) -> None:
        # Saves new version to versions list and updates current
        pass

    def get_latest(self) -> T:
        # Returns the latest version (i.e., current)
        pass

    def get_at(self, timestamp: datetime) -> Optional[T]:
        # Returns the version active at a given time, if any
        pass

    def compare_versions(self, index1: int, index2: int) -> dict:
        # Returns field-by-field diff between two saved versions
        pass

