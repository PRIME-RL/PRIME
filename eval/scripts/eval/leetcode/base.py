from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract class for a tool."""

    name: str
    signature: str
    description: str

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> str:
        """Execute the tool with the given args and return the output."""
        # execute tool with abitrary args
        pass

    def reset(self) -> None:
        """Reset the tool to its initial state."""
        pass
