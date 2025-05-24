from typing import Literal, Dict
from pydantic import BaseModel
from langgraph.types import StreamWriter

ActionType = Literal["image", "gif", "voice", "reaction",
                     "sticker", "system-message", "system-notification"]
Reaction = Literal[
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉", "🤩", "🤮",
    "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡",
    "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻",
    "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄",  "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡"
]


class Action(BaseModel):
    type: ActionType
    value: str


class ActionSender:
    writer: StreamWriter

    def __init__(self, writer: StreamWriter):
        self.writer = writer

    def send_action(self, action: Action):
        self.writer({"actions": [action.dict()]})

    def send_reaction(self, reaction: Reaction):
        action = Action(
            type="reaction",
            value=reaction
        )
        self.send_action(action)
