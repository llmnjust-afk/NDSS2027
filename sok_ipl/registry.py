"""Registry pattern so attacks and defenses are discovered by string name.

This lets the YAML experiment config reference `defense: struq` and
`attack: adaptive_single` without the runner needing to import every module.
Registrations happen at import time of the leaf modules, so the runner only
needs to `import sok_ipl.attacks` and `import sok_ipl.defenses` (which their
`__init__.py` already do eagerly).
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {"attack": {}, "defense": {}, "principle": {}}


def register(category: str, name: str):
    """Class decorator that records the class in a named registry.

    Usage:
        @register("defense", "struq")
        class StruQDefense(BaseDefense): ...
    """

    def deco(cls: type[T]) -> type[T]:
        if name in _REGISTRY[category]:
            raise ValueError(f"duplicate {category} registration: {name}")
        _REGISTRY[category][name] = cls
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get(category: str, name: str) -> type:
    if name not in _REGISTRY[category]:
        raise KeyError(f"unknown {category}: {name!r}; known={list(_REGISTRY[category])}")
    return _REGISTRY[category][name]


def list_names(category: str) -> list[str]:
    return sorted(_REGISTRY[category].keys())
