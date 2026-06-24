from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    handler: Callable[..., Any]
    description: str = ""


class AlgorithmRegistry:
    def __init__(self):
        self._algorithms: Dict[str, AlgorithmSpec] = {}

    def register(self, name: str, handler: Callable[..., Any], description: str = "") -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Algorithm name must be a non-empty string")
        if name in self._algorithms:
            raise ValueError(f"Algorithm already registered: {name}")
        if not callable(handler):
            raise TypeError("Algorithm handler must be callable")
        self._algorithms[name] = AlgorithmSpec(name=name, handler=handler, description=description)

    def list_names(self) -> List[str]:
        return sorted(self._algorithms)

    def describe(self, name: str) -> str:
        return self._algorithms[name].description

    def run(self, name: str, data: Any, **kwargs) -> Any:
        return self._algorithms[name].handler(data, **kwargs)
