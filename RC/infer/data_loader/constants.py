from enum import Enum, unique


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

IGNORE_INDEX = -100