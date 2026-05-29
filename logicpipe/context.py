from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LogicPipeRuntime:
    """Holds runtime objects shared across online stages."""

    args: "RuntimeLaunchArgs"
    config: Any
    model: Any
    tokenizer: Any


class RuntimeLaunchArgs(Protocol):
    """Minimal runtime arguments contract evaluated during startup."""

    rank: int
    world: int
    config_file: str
    load_in_8bit: bool
    load_in_4bit: bool
