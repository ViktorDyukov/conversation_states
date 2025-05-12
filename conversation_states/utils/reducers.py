from typing import Optional, List
from conversation_states.humans import Human


def add_summary(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    return b


def add_user(left: list["Human"], right: list["Human"]) -> list["Human"]:
    right = [u if isinstance(u, Human) else Human(**u)
             for u in right or []]

    existing_ids = {u.username for u in left}
    return left + [u for u in right if u.username not in existing_ids]
