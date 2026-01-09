from dataclasses import dataclass
from typing import Any


@dataclass
class TextCommand:
    """
    Represents a text command.
    """

    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextCommand":
        return cls(text=str(data["text"]))


@dataclass
class PointCommand:
    """
    Represents a point command.
    Coordinates are typically normalized (0.0 to 1.0).
    """

    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointCommand":
        if "point" in data:
            parts = data["point"].split(",")
            return cls(x=float(parts[0]), y=float(parts[1]))
        else:
            return cls(x=float(data["x"]), y=float(data["y"]))
