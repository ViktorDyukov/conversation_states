# file: tracker.py
from copy import deepcopy


class Tracked:
    def __init__(self, obj: T, on_change: Callable[[T, T], None]):
        self._obj = obj
        self._on_change = on_change

    def update(self, **kwargs):
        # Updates fields and triggers on_change with old and new object
        pass

    def get(self) -> T:
        # Returns current state of the object
        pass
