from typing import Optional, Dict
from pydantic import BaseModel, Field

class Human(BaseModel):
    username: str
    first_name: str
    last_name: Optional[str] = None
    preferred_name: Optional[str] = None
    information: Dict = Field(default_factory=dict)

    def update_info(self, updates: dict[str, str] | list[dict[str, str]]) -> None:
        if isinstance(updates, dict):
            updates = [updates]

        for pair in updates:
            for key, value in pair.items():
                if value:
                    self.information[key] = value  # Add or update
                elif key in self.information:
                    # Remove existing if value is empty
                    del self.information[key]
                # else: ignore non-existent empty key
