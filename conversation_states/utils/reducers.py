from typing import Optional, Union, Any
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


def manage_state(
    a: Optional[Union["InternalState", list[Any]]],
    b: Optional[Union["InternalState", list[Any]]]
) -> Optional[Union["InternalState", list[Any]]]:
    def is_empty(val):
        return val is None or val == []
    if is_empty(a) and is_empty(b):
        return None
    if is_empty(a):
        return b
    if is_empty(b):
        return a
    return b
